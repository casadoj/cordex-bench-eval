"""
CORDEX-ML-Bench Scorecard  --  RMSE edition
============================================
Three scorecard styles, light->dark grey performance colormap.

  Style 1 -- "all_regions":   per-metric x per-domain (ALPS/NZ/SA/AV) detail
  Style 2 -- "regions_avg":   per-metric, AV column only
  Style 3 -- "overall_rank":  Generative (left) | Deterministic (right),
              4-level column headers, background shading, rotated side labels,
              dashed separator, triangle legend -- matches reference layout.

Fixes
-----
  1. Unique ranks  -- method="average" per-metric, method="first" for display
  2. n_global      -- counts only models with >=1 finite metric value
  3. Grey colormap -- #f5f5f5 (highest/lightest) to #1a1a1a (lowest/darkest)
  4. Side-by-side  -- Generative | Deterministic in overall rank
  5. Bigger titles -- suptitle 17pt, panel titles 15pt
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import rcParams
from matplotlib.path import Path as MplPath

warnings.filterwarnings("ignore")

rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size":       8.0,
    "axes.linewidth":  0.5,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DIAG_ROOT       = "/lustre/gmeteo/WORK/rampaln/cordex-ml-bench/cordex-bench-eval/diagnostics/outputs/netcdf"
DOMAINS_DEFAULT = ["ALPS_Domain", "NZ_Domain", "SA_Domain"]
OROG_SUFFIX     = "-orog"

# Canonical names used in scorecard rows.
MODEL_CANONICAL_ALIASES: Dict[str, str] = {
    "deepsensor_bas":        "DeepSensor",
    "deepsensor_orog_bas":   "DeepSensor-orog",

}
# "DetUNet_orog": "DetUNet-orog",
# "spaGAN_orog": "spaGAN-orog",
# "Prithvi-UNet-orog_old": "Prithvi-UNet-orog",
# "Prithvi-UNet_orog_old": "Prithvi-UNet-orog",

# Candidate on-disk directory names for each canonical model.
MODEL_DISK_PRIORITY: Dict[str, List[str]] = {
    "DeepSensor":        ["deepsensor_bas"],
    "DeepSensor-orog":   ["deepsensor_orog_bas"],
}
# "DetUNet-orog": ["DetUNet-orog", "DetUNet_orog"],
# "spaGAN-orog": ["spaGAN-orog", "spaGAN_orog"],
# "Prithvi-UNet-orog": ["Prithvi-UNet-orog", "Prithvi-UNet-orog_old"]
SKIP_DIRS       = {".ipynb_checkpoints", "__pycache__", ".git", ".DS_Store",
                   "tmp", "temp"}

GENERATIVE_MODELS = [
    "ResGAN", "ResGAN-v2", "FlowMatching-v1", "CorrDiff-TW1_NTNU-CWA",
    "spaGAN", "ViT-IFCAv1", "EnScale",
    "ParamDiffusion-orog", "RCMGEM-mv-orog", "RCMFlow-orog",
    "UiBCorrDiff-orog", "EnScale-linex",
]
DETERMINISTIC_MODELS = [
    "DetUNet", "DetUNet-v2", "Rossby-UNet", "Prithvi-UNet",
    "DeepESDcrps_IFCAv2", "DeepESD-IFCAv1", "ANN", "CNRM-UNeT",
    "DeepESD_IDL", "ParamUNET", "DeepSensor", "XGBoost_IDL", "GNN4CD-orog",
]
DISPLAY_LABELS: Dict[str, str] = {
    "Prithvi-UNet":      "Prithvi-UNet",
    "Prithvi-UNet-orog": "Prithvi-UNet-orog",
}

METRICS_PR = {
    "rmse":                "RMSE",
    "mean_bias":           "Mean Bias",
    "sdii_bias":           "SDII",
    "rx1day_bias":         "Rx1day",
    "lag1_corr_bias":      "Lag-1",
    "ralsd":               "RALSD",
    "lhd":                 "LHD",
}
METRICS_TASMAX = {
    "rmse":                "RMSE",
    "mean_bias":           "Mean Bias",
    "txx_bias":            "TXx",
    "lag1_corr_bias":      "Lag-1",
    "ralsd":               "RALSD",
    "perkins_skill_score": "PSS",
    "interannual_var_bias":"IAV",
}

# ---------------------------------------------------------------------------
# COLOURS  -- light->dark grey; grey section headers
# ---------------------------------------------------------------------------
WHITE      = "#ffffff"
BLACK      = "#1a1a1a"
GOLD       = "#d4a017"

HEADER_GEN = "#4a4a4a"   # medium-dark grey for Generative band
HEADER_DET = "#2a2a2a"   # dark grey for Deterministic band
HEADER_OTH = "#222222"

BG_GEN     = "#f2f7f2"   # very pale green tint for generative rows
BG_GEN2    = "#e8f0e8"   # alternating row tint
BG_DET     = "#f4f4f6"   # very pale cool grey for deterministic rows
BG_DET2    = "#ebebf0"   # alternating row tint

LBL_GEN    = "#2d6a2d"   # dark green model labels (generative)
LBL_DET    = "#2a3a6a"   # dark blue model labels (deterministic)

GREY_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "grey_perf",
    ["#f5f5f5", "#e0e0e0", "#cccccc", "#b0b0b0", "#909090",
     "#707070", "#505050", "#383838", "#282828", "#1a1a1a"],
    N=256,
)


def _gc(rank_norm) -> tuple:
    """0 = lightest (highest)  ->  1 = darkest (lowest)."""
    if rank_norm is None or not np.isfinite(float(rank_norm)):
        return (0.82, 0.82, 0.82, 1.0)
    return GREY_CMAP(float(np.clip(rank_norm, 0.0, 1.0)))


def _tc(rgba) -> str:
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return WHITE if lum < 0.55 else BLACK


def _fmt(v: float) -> str:
    if not np.isfinite(v):
        return ""
    a = abs(v)
    if a < 0.1:  return f"{v:.3f}"
    if a < 10:   return f"{v:.2f}"
    if a < 100:  return f"{v:.1f}"
    return f"{v:.0f}"


# ---------------------------------------------------------------------------
# DATA HELPERS
# ---------------------------------------------------------------------------
def discover_models(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    found = [
        canonicalize_model_name(d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
        and d not in SKIP_DIRS and not d.startswith(".")
    ]
    # Remove duplicate aliases (e.g. spaGAN_orog/spaGAN-orog).
    return sorted(dict.fromkeys(found), key=lambda x: x.lower())


def canonicalize_model_name(name: str) -> str:
    if name in MODEL_CANONICAL_ALIASES:
        return MODEL_CANONICAL_ALIASES[name]
    if name.endswith("-orog_old"):
        return name[:-4]
    if name.endswith("_orog_old"):
        return name[:-9] + "-orog"
    if name.endswith("_orog"):
        return name[:-5] + "-orog"
    return name


def _disk_model_candidates(model: str) -> List[str]:
    cands = list(MODEL_DISK_PRIORITY.get(model, [model]))
    if model.endswith("-orog"):
        base = model[:-5]
        cands.extend([f"{base}_orog", f"{base}-orog_old", f"{base}_orog_old"])
    out: List[str] = []
    for c in cands:
        if c not in out:
            out.append(c)
    return out


def resolve_metric_file(root, model, domain, experiment, variable, period, metric):
    for disk_name in _disk_model_candidates(model):
        nc = os.path.join(root, disk_name, domain, experiment, variable, period,
                          f"metric_{metric}.nc")
        if os.path.exists(nc):
            return nc
    return None


def resolve_model_path(root, model, domain, experiment, variable, period, metric=None):
    if metric is not None:
        nc = resolve_metric_file(root, model, domain, experiment, variable, period, metric)
        if nc is not None:
            return os.path.dirname(nc)
    fallback = None
    for disk_name in _disk_model_candidates(model):
        p = os.path.join(root, disk_name, domain, experiment, variable, period)
        if not os.path.isdir(p):
            continue
        if fallback is None:
            fallback = p
        if metric is None:
            return p
        if os.path.exists(os.path.join(p, f"metric_{metric}.nc")):
            return p
    return fallback


def model_has_complete_metrics(
    root: str,
    model: str,
    domains: Sequence[str],
    variable: str,
    metric_keys: Sequence[str],
    scenarios: Sequence[Tuple[Optional[str], Optional[str]]],
) -> bool:
    for exp, period in scenarios:
        if not exp or not period:
            continue
        for dom in domains:
            for metric in metric_keys:
                if resolve_metric_file(root, model, dom, exp, variable, period, metric) is None:
                    return False
    return True


def load_scalar(path: str, metric: str) -> float:
    nc = os.path.join(path, f"metric_{metric}.nc")
    if not os.path.exists(nc):
        return np.nan
    try:
        ds = xr.open_dataset(nc)
        if metric in ("psd", "ralsd"):
            return float(ds.RALSD.values)
        if metric == "lhd":
            return float(ds["lhd"].values)
        if metric == "perkins_skill_score":
            return float(ds["perkins_skill_score"].values)
        vals = ds[list(ds.data_vars)[0]].values.astype(float).ravel()
        f = vals[np.isfinite(vals)]
        return float(np.sqrt(np.mean(f ** 2))) if f.size > 0 else np.nan
    except Exception:
        return np.nan


def build_df(root, experiment, variable, period, domains, model_names,
             per_domain=False):
    md       = METRICS_PR if variable == "pr" else METRICS_TASMAX
    ds_short = [d.split("_")[0] for d in domains]

    if not per_domain:
        rec = {}
        for m in model_names:
            row = {}
            for mk, ml in md.items():
                vals = []
                for d in domains:
                    p = resolve_model_path(
                        root, m, d, experiment, variable, period, metric=mk)
                    if p is not None:
                        vals.append(load_scalar(p, mk))
                finite = [v for v in vals if np.isfinite(v)]
                row[ml] = float(np.mean(finite)) if finite else np.nan
            rec[m] = row
        return pd.DataFrame.from_dict(rec, orient="index")

    cols = []
    for ml in md.values():
        for ds in ds_short:
            cols.append((ml, ds))
        cols.append((ml, "AV"))
    mc   = pd.MultiIndex.from_tuples(cols, names=["Metric", "Domain"])
    data = np.full((len(model_names), len(cols)), np.nan)
    for i, m in enumerate(model_names):
        for mi, (mk, ml) in enumerate(md.items()):
            base = mi * (len(ds_short) + 1)
            dv   = []
            for di, d in enumerate(domains):
                p = resolve_model_path(
                    root, m, d, experiment, variable, period, metric=mk)
                v = np.nan if p is None else load_scalar(p, mk)
                data[i, base + di] = v
                if np.isfinite(v):
                    dv.append(v)
            data[i, base + len(ds_short)] = float(np.mean(dv)) if dv else np.nan
    return pd.DataFrame(data, index=list(model_names), columns=mc)


# ---------------------------------------------------------------------------
# RANKING  (Fix 1: unique integers via method="first")
# ---------------------------------------------------------------------------
def _norm_rank(mat: np.ndarray) -> np.ndarray:
    out = np.full_like(mat, np.nan, dtype=float)
    for j in range(mat.shape[1]):
        col = mat[:, j].astype(float)
        r   = pd.Series(col).rank(method="average", ascending=True)
        mr  = np.nanmax(r.values)
        out[:, j] = np.where(
            np.isfinite(col),
            0.0 if (not np.isfinite(mr) or mr <= 1) else (r.values - 1) / (mr - 1),
            np.nan,
        )
    return out


def compute_global_ranks(all_model_names, df_a, df_b, metric_labels):
    def _pmr(df):
        if df is None:
            return pd.DataFrame(np.nan, index=list(all_model_names),
                                columns=list(metric_labels))
        sub = df.reindex(all_model_names)
        return pd.DataFrame({
            c: (sub[c].rank(method="average", ascending=True, na_option="bottom")
                if c in sub.columns
                else pd.Series(np.nan, index=sub.index))
            for c in metric_labels
        })

    pm_a = _pmr(df_a)
    pm_b = _pmr(df_b)

    sum_a = pm_a.mean(axis=1, skipna=True)
    sum_b = (pm_b.mean(axis=1, skipna=True)
             if df_b is not None
             else pd.Series(np.nan, index=sum_a.index))

    ra = sum_a.rank(method="first", ascending=True, na_option="bottom")
    rb = (sum_b.rank(method="first", ascending=True, na_option="bottom")
          if df_b is not None
          else pd.Series(np.nan, index=ra.index))

    combined = (sum_a + sum_b) / 2.0 if df_b is not None else sum_a
    ov = combined.rank(method="first", ascending=True, na_option="bottom")

    gr: Dict[str, Optional[int]] = {}
    pr: Dict[str, Optional[Tuple[int, int]]] = {}
    for m in all_model_names:
        o = ov.get(m, np.nan)
        if np.isfinite(o):
            gr[m] = int(o)
            a, b  = ra.get(m, np.nan), rb.get(m, np.nan)
            pr[m] = (int(a) if np.isfinite(a) else None,
                     int(b) if np.isfinite(b) else None)
        else:
            gr[m] = None
            pr[m] = None
    return gr, pr


# ---------------------------------------------------------------------------
# ROW LAYOUT
# ---------------------------------------------------------------------------
SECTION_H = 0.50
MODEL_H   = 1.0


def build_sorted_rows(available, global_ranks):
    avail = set(available)

    def _rs(m):
        r = global_ranks.get(m) or global_ranks.get(m + OROG_SUFFIX)
        return r or 9999

    def _sg(lst):
        rows = []
        for m in sorted(lst, key=_rs):
            rows.append(dict(kind="model", label=DISPLAY_LABELS.get(m, m),
                             model_name=m, is_orog=False, has_data=(m in avail)))
            orog = m + OROG_SUFFIX
            if not m.endswith(OROG_SUFFIX) and orog in avail:
                rows.append(dict(kind="model",
                                 label=DISPLAY_LABELS.get(orog, orog),
                                 model_name=orog, is_orog=True, has_data=True))
        return rows

    rows  = [dict(kind="section", label="Generative Models",
                  color=HEADER_GEN, model_name=None,
                  is_orog=False, has_data=True)]
    rows += _sg(GENERATIVE_MODELS)
    rows += [dict(kind="section", label="Deterministic Models",
                  color=HEADER_DET, model_name=None,
                  is_orog=False, has_data=True)]
    rows += _sg(DETERMINISTIC_MODELS)

    shown  = {r["model_name"] for r in rows if r["kind"] == "model"}
    extras = sorted(m for m in available if m not in shown)
    if extras:
        rows += [dict(kind="section", label="Other", color=HEADER_OTH,
                      model_name=None, is_orog=False, has_data=True)]
        rows += [dict(kind="model", label=DISPLAY_LABELS.get(m, m),
                      model_name=m, is_orog=False, has_data=True)
                 for m in extras]
    return rows


def _split_by_group(rows):
    gen, det, other = [], [], []
    cur = None
    for r in rows:
        if r["kind"] == "section":
            cur = r["label"]
            continue
        if cur and "Generative" in cur:      gen.append(r)
        elif cur and "Deterministic" in cur: det.append(r)
        else:                                other.append(r)
    det += other
    return gen, det


def row_y_positions(rows):
    total = sum(SECTION_H if r["kind"] == "section" else MODEL_H for r in rows)
    y     = total
    ym    = {}
    for r in rows:
        h = SECTION_H if r["kind"] == "section" else MODEL_H
        y -= h
        if r["kind"] == "model":
            ym[r["model_name"]] = y
    return ym, total


def _flat_y(model_rows):
    total = len(model_rows) * MODEL_H
    ym    = {r["model_name"]: total - (i + 1) * MODEL_H
             for i, r in enumerate(model_rows)}
    return ym, total


# ---------------------------------------------------------------------------
# DRAWING PRIMITIVES
# ---------------------------------------------------------------------------
def _tri(ax, x, y, vn, upper, best=False):
    c     = _gc(vn)
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.CLOSEPOLY]
    verts = ([(x, y+1), (x+1, y+1), (x, y), (x, y+1)] if upper
             else [(x+1, y), (x+1, y+1), (x, y), (x+1, y)])
    path  = MplPath(verts, codes)
    ax.add_patch(mpatches.PathPatch(path, fc=c, ec=WHITE, lw=0.3, zorder=2))
    if best:
        ax.add_patch(mpatches.PathPatch(
            path, fc="none", ec=GOLD, lw=1.6,
            joinstyle="round", capstyle="round", zorder=3))
    return c


def _best_mask(nm: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(nm, dtype=bool)
    for j in range(nm.shape[1]):
        col = nm[:, j]
        if np.any(np.isfinite(col)):
            mask[:, j] = (col == np.nanmin(col))
    return mask


def _section_bands(ax, rows, ym, total_w):
    for idx, r in enumerate(rows):
        if r["kind"] != "section":
            continue
        mems = []
        for rr in rows[idx + 1:]:
            if rr["kind"] == "section":
                break
            mems.append(rr["model_name"])
        tops = [ym[m] + MODEL_H for m in mems if m in ym]
        if not tops:
            continue
        yt = max(tops)
        ax.add_patch(plt.Rectangle(
            [0, yt], total_w, SECTION_H,
            fc=r.get("color", HEADER_OTH), ec="none",
            clip_on=False, zorder=4))
        ax.text(0.25, yt + SECTION_H * 0.5, r["label"],
                ha="left", va="center", fontsize=8.5,
                color=WHITE, fontweight="bold", zorder=5, clip_on=False)


def _rlabel(ax, x, y, label, is_orog, no_data, col=None):
    if is_orog:
        ax.text(x, y + 0.5, label, ha="right", va="center",
                fontsize=6.0, color="#888888", style="italic")
    else:
        ax.text(x, y + 0.5, label, ha="right", va="center",
                fontsize=7.2, fontweight="semibold",
                color="#bbbbbb" if no_data else (col or BLACK))


# ---------------------------------------------------------------------------
# STYLE 1 / 2  -- DETAIL PANEL
# ---------------------------------------------------------------------------
def _detail_panel(ax, df_a, df_b, rows, metric_labels, ds_short, per_domain,
                  title, show_labels, ym, total_h, pranks, n_global,
                  lbl_up, lbl_dn, show_rank=True):

    n_m = len(metric_labels)
    sp  = (len(ds_short) + 1) if per_domain else 1
    nc  = n_m * sp
    rw  = 0.95 if show_rank else 0.0

    mr  = [r for r in rows if r["kind"] == "model"]
    mn  = [r["model_name"] for r in mr]

    def _mat(df):
        mat = np.full((len(mn), nc), np.nan)
        if df is None:
            return mat
        for i, m in enumerate(mn):
            if m not in df.index:
                continue
            for mi, ml in enumerate(metric_labels):
                if per_domain:
                    for di, ds in enumerate(ds_short):
                        col = (ml, ds)
                        if col in df.columns:
                            mat[i, mi * sp + di] = df.loc[m, col]
                    ac = (ml, "AV")
                    if ac in df.columns:
                        mat[i, mi * sp + len(ds_short)] = df.loc[m, ac]
                else:
                    if ml in df.columns:
                        mat[i, mi] = df.loc[m, ml]
        return mat

    va = _mat(df_a)
    vb = _mat(df_b) if df_b is not None else None
    na = _norm_rank(va)
    nb = _norm_rank(vb) if vb is not None else None
    ba = _best_mask(na)
    bb = _best_mask(nb) if nb is not None else None

    tw = nc + 0.3 + (rw + 0.3 if show_rank else 0)
    _section_bands(ax, rows, ym, tw)

    for i, r in enumerate(mr):
        m  = r["model_name"]
        if m not in ym:
            continue
        y  = ym[m]
        nd = not r.get("has_data", True)

        for j in range(nc):
            if vb is not None:
                ca = _tri(ax, j, y, na[i, j] if not nd else None,
                          upper=True,  best=bool(ba[i, j]) and not nd)
                cb = _tri(ax, j, y, nb[i, j] if not nd else None,
                          upper=False, best=bool(bb[i, j]) and not nd)
                if not nd and np.isfinite(va[i, j]):
                    ax.text(j + 0.27, y + 0.72, _fmt(va[i, j]),
                            ha="center", va="center", fontsize=4.8,
                            color=_tc(ca), fontfamily="monospace")
                if not nd and np.isfinite(vb[i, j]):
                    ax.text(j + 0.73, y + 0.28, _fmt(vb[i, j]),
                            ha="center", va="center", fontsize=4.8,
                            color=_tc(cb), fontfamily="monospace")
            else:
                c = _gc(na[i, j] if not nd else None)
                ax.add_patch(plt.Rectangle([j, y], 1, MODEL_H,
                                           fc=c, ec=WHITE, lw=0.3, zorder=2))
                if not nd and ba[i, j]:
                    ax.add_patch(plt.Rectangle([j, y], 1, MODEL_H,
                                               fc="none", ec=GOLD, lw=1.6,
                                               joinstyle="round", zorder=3))
                if not nd and np.isfinite(va[i, j]):
                    ax.text(j + 0.5, y + 0.5, _fmt(va[i, j]),
                            ha="center", va="center", fontsize=5.8,
                            color=_tc(c), fontfamily="monospace")

    # Rank column
    if show_rank:
        rx = nc + 0.25
        ma = max((p[0] for p in pranks.values() if p and p[0]), default=1)
        mb = max((p[1] for p in pranks.values() if p and p[1]), default=1)
        for r in mr:
            m = r["model_name"]
            if m not in ym:
                continue
            y    = ym[m]
            pair = pranks.get(m)
            if not pair:
                continue
            a, b = pair
            an   = (a - 1) / max(ma - 1, 1) if a else None
            bn   = (b - 1) / max(mb - 1, 1) if b else None
            if df_b is not None:
                ca = _tri(ax, rx, y, an, upper=True)
                cb = _tri(ax, rx, y, bn, upper=False)
                if a:
                    ax.text(rx + 0.27, y + 0.72, str(a),
                            ha="center", va="center", fontsize=5.6,
                            color=_tc(ca), fontfamily="monospace",
                            fontweight="bold")
                if b:
                    ax.text(rx + 0.73, y + 0.28, str(b),
                            ha="center", va="center", fontsize=5.6,
                            color=_tc(cb), fontfamily="monospace",
                            fontweight="bold")
            else:
                c = _gc(an)
                ax.add_patch(plt.Rectangle([rx, y], rw, MODEL_H,
                                           fc=c, ec=WHITE, lw=0.3, zorder=2))
                if a:
                    ax.text(rx + rw / 2, y + 0.5, str(a),
                            ha="center", va="center", fontsize=6.0,
                            color=_tc(c), fontfamily="monospace",
                            fontweight="bold")

    # Row labels
    if show_labels:
        cc = BLACK
        for r in rows:
            if r["kind"] == "section":
                cc = r.get("color", BLACK)
                continue
            m = r["model_name"]
            if m not in ym:
                continue
            _rlabel(ax, -0.15, ym[m], r["label"], r["is_orog"],
                    not r.get("has_data", True), col=cc)

    # Column headers
    if per_domain:
        for mi in range(n_m):
            base = mi * sp
            for di, ds in enumerate(ds_short):
                ax.text(base + di + 0.5, total_h + 0.02, ds,
                        ha="center", va="bottom", fontsize=5.8, color="#444444")
            ax.text(base + len(ds_short) + 0.5, total_h + 0.02, "AV",
                    ha="center", va="bottom", fontsize=5.8, color="#444444",
                    fontweight="bold")
        for mi, ml in enumerate(metric_labels):
            mid = mi * sp + sp / 2
            ax.text(mid, total_h + SECTION_H + 0.05, ml,
                    ha="center", va="bottom", fontsize=7.2, color=BLACK,
                    fontweight="bold")
            if mi > 0:
                ax.plot([mi * sp, mi * sp], [0, total_h + SECTION_H],
                        color="#cccccc", lw=0.4, zorder=1)
    else:
        for mi, ml in enumerate(metric_labels):
            ax.text(mi + 0.5, total_h + 0.05, ml,
                    ha="left", va="bottom", fontsize=7.2, color=BLACK,
                    fontweight="bold", rotation=45, rotation_mode="anchor")

    if show_rank:
        ax.text(nc + 0.25 + rw / 2, total_h + 0.05, "Rank",
                ha="center", va="bottom", fontsize=7.2, color=BLACK,
                fontweight="bold")

    # Triangle direction legend
    if df_b is not None:
        leg_x, leg_y = 0.10, total_h + SECTION_H + 0.72
        sw = 0.55   # swatch size in data units
        codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.CLOSEPOLY]
        ax.add_patch(mpatches.PathPatch(
            MplPath([(leg_x, leg_y + sw), (leg_x + sw, leg_y + sw),
                     (leg_x, leg_y), (leg_x, leg_y + sw)], codes),
            fc=_gc(0.12), ec=WHITE, lw=0.3, zorder=5, clip_on=False))
        ax.add_patch(mpatches.PathPatch(
            MplPath([(leg_x + sw, leg_y), (leg_x + sw, leg_y + sw),
                     (leg_x, leg_y), (leg_x + sw, leg_y)], codes),
            fc=_gc(0.78), ec=WHITE, lw=0.3, zorder=5, clip_on=False))
        ax.text(leg_x + sw + 0.15, leg_y + sw * 0.72, f"▲  {lbl_up}",
                ha="left", va="center", fontsize=7.5, color=BLACK,
                clip_on=False)
        ax.text(leg_x + sw + 0.15, leg_y + sw * 0.28, f"▽  {lbl_dn}",
                ha="left", va="center", fontsize=7.5, color=BLACK,
                clip_on=False)
 
    ax.set_title(title, fontsize=15, color=BLACK, pad=10, fontweight="bold")
    ax.set_xlim(-0.15, tw)
    ax.set_ylim(-0.15, total_h + SECTION_H + 2.20)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.set_facecolor(WHITE)


# ---------------------------------------------------------------------------
# STYLE 3  -- helpers for the reference-style overall rank layout
# ---------------------------------------------------------------------------

_COL_HDR_H = 2.70   # total vertical space (data units) above cells for headers


def _draw_col_headers(ax, col_meta: List[dict], total_h: float,
                      accent_color: str):
    """
    4-level column headers matching the reference:
      1. Variable group label + coloured underline
      2. Variable abbreviation
      3. Experiment name
      4. ▲ CV  ▽ Extrap./Interp.
    """
    # Group consecutive columns by variable
    groups: List[Tuple[str, List]] = []
    cur_v, cur_c = None, []
    for ci, meta in enumerate(col_meta):
        v = meta["variable"]
        if v != cur_v:
            if cur_c:
                groups.append((cur_v, cur_c))
            cur_v, cur_c = v, []
        cur_c.append((ci, meta))
    if cur_c:
        groups.append((cur_v, cur_c))

    for var, cols in groups:
        xs     = [ci for ci, _ in cols]
        x_lo   = min(xs)
        x_hi   = max(xs) + 1
        x_mid  = (x_lo + x_hi) / 2.0
        lbl    = ("Precipitation (pr)"
                  if var == "pr" else "Max. Temp. (tasmax)")

        # Level 1 — group label + underline
        y1 = total_h + _COL_HDR_H - 0.18
        ax.text(x_mid, y1, lbl, ha="center", va="bottom",
                fontsize=8.0, color=accent_color, fontweight="bold",
                clip_on=False)
        ax.plot([x_lo + 0.06, x_hi - 0.06], [y1 - 0.04, y1 - 0.04],
                color=accent_color, lw=1.2, clip_on=False)

        # Per-column levels 2-4
        for ci, meta in cols:
            cx = ci + 0.5
            ax.text(cx, total_h + 1.85, meta["variable"],
                    ha="center", va="bottom", fontsize=9.8,
                    color=BLACK, fontweight="bold", clip_on=False)
            ax.text(cx, total_h + 1.18, meta["experiment"],
                    ha="center", va="bottom", fontsize=9.8,
                    color="#444444", clip_on=False)
            ax.text(cx, total_h + 0.55,
                    f"▲ CV  ▽ {meta['lower_label']}",
                    ha="center", va="bottom", fontsize=7,
                    color="#666666", clip_on=False)

        # Dashed separator before each new variable group (except first)
        if groups.index((var, cols)) > 0:
            ax.plot([x_lo, x_lo], [0, total_h + _COL_HDR_H],
                    color="#bbbbbb", lw=0.5, ls="--", zorder=1,
                    clip_on=False)


def _rank_half(ax, model_rows, ym, pair_list, max_rank,
               col_meta, show_labels,
               lbl_color, hdr_color, hdr_label,
               total_h, bg_a, bg_b):
    nc = len(col_meta)
    tw = nc + 0.15

    # Background
    ax.add_patch(plt.Rectangle([0, 0], tw, total_h,
                                fc=bg_a, ec="none", zorder=0))
    for i, r in enumerate(model_rows):
        m = r["model_name"]
        if m not in ym:
            continue
        if i % 2 == 1:
            ax.add_patch(plt.Rectangle([0, ym[m]], tw, MODEL_H,
                                       fc=bg_b, ec="none", zorder=1))

    # Header band
    ax.add_patch(plt.Rectangle([0, total_h], tw, SECTION_H,
                                fc=hdr_color, ec="none",
                                clip_on=False, zorder=4))
    ax.text(tw / 2, total_h + SECTION_H * 0.5, hdr_label,
            ha="center", va="center", fontsize=9.5,
            color=WHITE, fontweight="bold", zorder=5, clip_on=False)

    # Cells
    for ci, pd_ in enumerate(pair_list):
        for r in model_rows:
            m = r["model_name"]
            if m not in ym:
                continue
            y    = ym[m]
            pair = pd_.get(m)
            if pair is None:
                ax.add_patch(plt.Rectangle([ci, y], 1, MODEL_H,
                                           fc=(0.85, 0.85, 0.85, 1),
                                           ec=WHITE, lw=0.3, zorder=2))
                continue
            a, b = pair
            an   = (a - 1) / max(max_rank - 1, 1) if a else None
            bn   = (b - 1) / max(max_rank - 1, 1) if b else None
            ca   = _tri(ax, ci, y, an, upper=True)
            cb   = _tri(ax, ci, y, bn, upper=False)
            if a:
                ax.text(ci + 0.27, y + 0.72, str(a),
                        ha="center", va="center", fontsize=8,
                        color=_tc(ca), fontfamily="monospace",
                        fontweight="bold")
            if b:
                ax.text(ci + 0.73, y + 0.28, str(b),
                        ha="center", va="center", fontsize=8,
                        color=_tc(cb), fontfamily="monospace",
                        fontweight="bold")

    # Vertical grid lines
    for ci in range(1, nc):
        ax.plot([ci, ci], [0, total_h], color="#d8d8d8", lw=0.3, zorder=1)

    # Row labels
    if show_labels:
        for r in model_rows:
            m = r["model_name"]
            if m not in ym:
                continue
            _rlabel(ax, -0.15, ym[m], r["label"], r["is_orog"],
                    not r.get("has_data", True), col=lbl_color)

    # 4-level column headers
    _draw_col_headers(ax, col_meta, total_h, accent_color=hdr_color)

    # Axes
    ax.set_xlim(-0.15, tw)
    ax.set_ylim(-0.15, total_h + SECTION_H + _COL_HDR_H + 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.set_facecolor(WHITE)


# ---------------------------------------------------------------------------
# Fix 2  -- n_global
# ---------------------------------------------------------------------------
def _n_with_data(*dfs) -> int:
    s = set()
    for df in dfs:
        if df is None:
            continue
        for m in df.index:
            if np.any(np.isfinite(df.loc[m].values.astype(float))):
                s.add(m)
    return max(len(s), 1)


# ---------------------------------------------------------------------------
# COLORBARS
# ---------------------------------------------------------------------------
def _cbar_metric(fig, x, y, w, h):
    cax = fig.add_axes([x, y, w, h])
    sm  = plt.cm.ScalarMappable(cmap=GREY_CMAP, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cb  = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["Highest", "Lowest"])
    cb.ax.tick_params(labelsize=7, length=2, pad=1)
    cb.outline.set_edgecolor("#cccccc")
    cb.outline.set_linewidth(0.4)
    cax.set_title("Metric performance", fontsize=12, pad=2, color=BLACK)


def _cbar_rank(fig, x, y, w, h, n: int):
    cax = fig.add_axes([x, y, w, h])
    sm  = plt.cm.ScalarMappable(cmap=GREY_CMAP, norm=mcolors.Normalize(1, n))
    sm.set_array([])
    cb  = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([1, n])
    cb.set_ticklabels(["1 (highest rank)", f"{n} (lowest rank)"])
    cb.ax.tick_params(labelsize=15, length=2, pad=1)
    cb.outline.set_edgecolor("#cccccc")
    cb.outline.set_linewidth(0.4)
    cax.set_title("Overall rank", fontsize=13, pad=2, color=BLACK)


# ---------------------------------------------------------------------------
# DATA GATHERING
# ---------------------------------------------------------------------------
def _gather(diag_root, variable, domains,
            esd_exp, esd_pa, esd_pb,
            emu_exp, emu_pa, emu_pb, per_domain):
    discovered = discover_models(diag_root)
    md         = METRICS_PR if variable == "pr" else METRICS_TASMAX
    mls        = list(md.values())
    registry   = GENERATIVE_MODELS + DETERMINISTIC_MODELS
    metric_keys = list(md.keys())
    scenarios = [
        (esd_exp, esd_pa),
        (esd_exp, esd_pb),
        (emu_exp, emu_pa),
        (emu_exp, emu_pb),
    ]
    candidates = list(dict.fromkeys(registry + discovered))
    all_names = [
        m for m in candidates
        if model_has_complete_metrics(
            diag_root, m, domains, variable, metric_keys, scenarios)
    ]
    allowed = set(all_names)
    available = [m for m in discovered if m in allowed]

    if not all_names:
        raise ValueError(
            f"No complete models found for variable={variable!r} "
            f"using scenarios={scenarios} and metrics={metric_keys}"
        )

    def _bdf(exp, p, pd_=per_domain):
        if not (exp and p):
            return None
        z1 = build_df(diag_root, exp, variable, p, domains, all_names,
                        per_domain=pd_)
        return z1.loc[all_names]
        
        #     def bdf(exp, p):
#         z1 = build_df(diag_root, exp, variable, p, domains, all_names,
#                         per_domain=per_domain) if exp and p else None
#         return z1.loc[all_model_names]
#     def bdf_f(exp, p):
#         z1 = build_df(diag_root, exp, variable, p, domains, all_names) \
#             if exp and p else None
#         return z1.loc[all_model_names]

    da_a = _bdf(esd_exp, esd_pa);  da_b = _bdf(esd_exp, esd_pb)
    du_a = _bdf(emu_exp, emu_pa);  du_b = _bdf(emu_exp, emu_pb)

    if per_domain:
        fa_a = _bdf(esd_exp, esd_pa, False)
        fa_b = _bdf(esd_exp, esd_pb, False)
        fu_a = _bdf(emu_exp, emu_pa, False)
        fu_b = _bdf(emu_exp, emu_pb, False)
    else:
        fa_a, fa_b, fu_a, fu_b = da_a, da_b, du_a, du_b

    _none = {m: None for m in all_names}
    ge, pe = (compute_global_ranks(all_names, fa_a, fa_b, mls)
              if fa_a is not None else (_none.copy(), _none.copy()))
    gu, pu = (compute_global_ranks(all_names, fu_a, fu_b, mls)
              if fu_a is not None else (_none.copy(), _none.copy()))

    return dict(
        available=available,
        all_model_names=all_names,
        metric_labels=mls,
        df_esd_a=da_a, df_esd_b=da_b,
        df_emu_a=du_a, df_emu_b=du_b,
        grank_esd=ge, prank_esd=pe,
        grank_emu=gu, prank_emu=pu,
        n_global=_n_with_data(fa_a, fa_b, fu_a, fu_b),
    )


# ---------------------------------------------------------------------------
# MAIN PLOT FUNCTIONS
# ---------------------------------------------------------------------------
def _dual_panels(data, variable, domains, per_domain,
                 save_path, figsize, style_label):
    mls = data["metric_labels"]
    ds  = [d.split("_")[0] for d in domains]
    sr  = (data["grank_esd"]
           if any(v is not None for v in data["grank_esd"].values())
           else data["grank_emu"])
    rows = build_sorted_rows(data["available"], sr)
    ym, th = row_y_positions(rows)

    sp  = (len(ds) + 1) if per_domain else 1
    nc  = len(mls) * sp
    cw  = 0.50 if per_domain else 0.75
    lw  = 2.4;  pw = nc * cw + 1.8;  gw = 0.7
    he  = data["df_esd_a"] is not None
    hu  = data["df_emu_a"] is not None
    np_ = int(he) + int(hu)
    tw  = lw + pw * np_ + gw * max(np_ - 1, 0) + 0.3
    th_ = th * 0.36 + 3.8

    fig = plt.figure(figsize=figsize or (tw, th_), facecolor=WHITE)
    fig.patch.set_facecolor(WHITE)

    lf = lw / tw;  pf = pw / tw;  gf = gw / tw
    bot, top = 0.07, 0.88
    #top = 1.38
    cx = lf

    if he:
        _detail_panel(fig.add_axes([cx, bot, pf, top - bot]),
                      data["df_esd_a"], data["df_esd_b"],
                      rows, mls, ds, per_domain,
                      "ESD", True, ym, th,
                      data["prank_esd"], data["n_global"],
                      "Cross-validation", "Extrapolation")
        cx += pf + gf

    if hu:
        _detail_panel(fig.add_axes([cx, bot, pf, top - bot]),
                      data["df_emu_a"], data["df_emu_b"],
                      rows, mls, ds, per_domain,
                      "Emulator Hist-Future", not he, ym, th,
                      data["prank_emu"], data["n_global"],
                      "Cross-validation", "Interpolation")

    vl = "Precipitation (pr)" if variable == "pr" else "Max Temperature (tasmax)"
    fig.suptitle(
        f"CORDEX-ML-Bench Scorecard  ·  RMSE  |  {vl} ",
        fontsize=20, color=BLACK, fontweight="bold", y =0.955)

    #_cbar_metric(fig, 0.20, 0.028, 0.27, 0.011)
    _cbar_rank(fig,   0.2, 0.028, 0.6, 0.011, data["n_global"])
    fig.text(0.50, 0.007,
             f"Gold = highest metric performance  ·  "
             f"Rank across {data['n_global']} models with data  ·  "
             "RMSE = sqrt(mean($x^2$))",
             ha="center", va="center", fontsize=9.5,
             color="#555555", style="italic")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=WHITE)
        print(f"Saved -> {save_path}")
    return fig


def _overall_rank(datasets, save_path, figsize):
    """
    Generative (left) | Deterministic (right) overall rank summary.
    Matches reference layout: 4-level column headers, background shading,
    rotated side labels, dashed separator, triangle legend.
    """
    primary = datasets[0][1]
    avail_sets = [set(d["available"]) for _, d in datasets]
    avail = sorted(set.intersection(*avail_sets)) if avail_sets else []
    if not avail:
        raise ValueError("No common complete models across requested datasets.")
    sr      = (primary["grank_esd"]
               if any(v is not None for v in primary["grank_esd"].values())
               else primary["grank_emu"])
    rows    = build_sorted_rows(avail, sr)
 
    pair_list: List[dict] = []
    col_meta:  List[dict] = []
    ng = 1
    for var, d in datasets:
        if any(v is not None for v in d["prank_esd"].values()):
            pair_list.append(d["prank_esd"])
            col_meta.append(dict(variable=var, experiment="ESD",
                                 lower_label="Extrap."))
        if any(v is not None for v in d["prank_emu"].values()):
            pair_list.append(d["prank_emu"])
            col_meta.append(dict(variable=var, experiment="Emulator",
                                 lower_label="Interp."))
        ng = max(ng, d["n_global"])
 
    if not col_meta:
        raise ValueError("No experiments with data for overall rank summary.")
 
    max_rank = max(
        (r for pd_ in pair_list for v in pd_.values()
         if v for r in v if r),
        default=1,
    )
 
    gen, det = _split_by_group(rows)
    gy, gh   = _flat_y(gen)
    dy, dh   = _flat_y(det)
 
    nc      = len(col_meta)
    cell_w  = 0.85          # slightly narrower when 4 cols to keep figure width sane
    lbl_w   = 2.00    # space for row labels
    panel_w = nc * cell_w + 0.25
    gap_w   = 1.20    # space between the two halves
 
    fig_w = lbl_w * 2 + panel_w * 2 + gap_w + 0.8
    fig_h = max(gh, dh) * 0.32 + 5.0
 
    fig = plt.figure(figsize=figsize or (fig_w, fig_h), facecolor=WHITE)
    fig.patch.set_facecolor(WHITE)
    fig.suptitle("CORDEX-ML-Bench  —  Overall Rank Summary",
                 fontsize=17, color=BLACK, fontweight="bold", y=0.972)
 
    bot, top = 0.13, 0.88
 
    # Axes in figure-fraction coordinates
    #   Generative: labels on left, cells next
    #   Deterministic: gap, then labels, cells
    gen_lbl_f  = lbl_w / fig_w
    gen_cell_f = panel_w / fig_w
    det_left_f = (lbl_w + panel_w + gap_w) / fig_w
    det_cell_f = panel_w / fig_w
 
    ax_g = fig.add_axes([gen_lbl_f,                 bot, gen_cell_f, top - bot])
    ax_d = fig.add_axes([det_left_f + lbl_w/fig_w,  bot, det_cell_f, top - bot])
 
    _rank_half(ax_g, gen, gy, pair_list, max_rank, col_meta,
               show_labels=True,
               lbl_color=LBL_GEN, hdr_color=HEADER_GEN,
               hdr_label="Generative Models",
               total_h=gh, bg_a=BG_GEN, bg_b=BG_GEN2)
 
    _rank_half(ax_d, det, dy, pair_list, max_rank, col_meta,
               show_labels=True,
               lbl_color=LBL_DET, hdr_color=HEADER_DET,
               hdr_label="Deterministic Models",
               total_h=dh, bg_a=BG_DET, bg_b=BG_DET2)
 
    # Rotated section labels on outer edges
    fig.text(0.010, (bot + top) / 2, "GENERATIVE",
             rotation=90, ha="center", va="center",
             fontsize=9.5, color=HEADER_GEN, fontweight="bold")
    fig.text(det_left_f - 0.012, (bot + top) / 2, "DETERMINISTIC",
             rotation=90, ha="center", va="center",
             fontsize=9.5, color=HEADER_DET, fontweight="bold")
 
    # Dashed vertical separator
    sep_x = (gen_lbl_f + gen_cell_f + det_left_f) / 2
    fig.add_artist(plt.Line2D(
        [sep_x, sep_x], [bot + 0.01, top - 0.01],
        transform=fig.transFigure,
        color="#999999", lw=1.0, ls="--", zorder=10))
 
    # Colorbar (bottom-left)
    _cbar_rank(fig, 0.05, 0.045, 0.38, 0.015, ng)
 
    # Legend with triangle swatch (bottom-right)
    la = fig.add_axes([0.56, 0.022, 0.40, 0.072])
    la.set_xlim(0, 1);  la.set_ylim(0, 1);  la.axis("off")
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.CLOSEPOLY]
    sw = 0.075
    la.add_patch(mpatches.PathPatch(
        MplPath([(0.01, 0.90), (sw + 0.01, 0.90), (0.01, 0.10), (0.01, 0.90)],
                codes),
        fc=_gc(0.08), ec=WHITE, lw=0.6, transform=la.transAxes, zorder=2))
    la.add_patch(mpatches.PathPatch(
        MplPath([(sw + 0.01, 0.90), (sw + 0.01, 0.10), (0.01, 0.10),
                 (sw + 0.01, 0.90)], codes),
        fc=_gc(0.62), ec=WHITE, lw=0.6, transform=la.transAxes, zorder=2))
    la.text(sw + 0.06, 0.70,
            "▲  upper-left  =  Cross-validation rank",
            ha="left", va="center", fontsize=8.5, color=BLACK,
            transform=la.transAxes)
    la.text(sw + 0.06, 0.28,
            "▽  lower-right  =  Extrapolation / Interpolation rank",
            ha="left", va="center", fontsize=8.5, color=BLACK,
            transform=la.transAxes)
 
    fig.text(0.50, 0.006,
             f"Rank across {ng} models with data  ·  "
             "method='first': ties broken by list order  ·  "
             "RMSE = sqrt(mean(x^2))",
             ha="center", va="center", fontsize=6.5,
             color="#555555", style="italic")
 
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=WHITE)
        print(f"Saved -> {save_path}")
    return fig

# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------
def plot_scorecard_all_regions(
    variable="pr", diag_root=DIAG_ROOT, domains=None,
    esd_experiment="ESD_pseudo_reality",
    esd_period_a="Perfect_cross_validation",
    esd_period_b="Perfect_extrapolation",
    emu_experiment="Emulator_hist_future",
    emu_period_a="Perfect_cross_validation",
    emu_period_b="Perfect_interpolation",
    save_path=None, figsize=None,
) -> plt.Figure:
    """STYLE 1 -- per-domain detail (ALPS / NZ / SA / AV)."""
    domains = domains or DOMAINS_DEFAULT
    print(f"[Style 1: all regions] variable={variable!r}")
    d = _gather(diag_root, variable, domains,
                esd_experiment, esd_period_a, esd_period_b,
                emu_experiment, emu_period_a, emu_period_b,
                per_domain=True)
    return _dual_panels(d, variable, domains, True, save_path, figsize,
                        "")


def plot_scorecard_regions_avg(
    variable="pr", diag_root=DIAG_ROOT, domains=None,
    esd_experiment="ESD_pseudo_reality",
    esd_period_a="Perfect_cross_validation",
    esd_period_b="Perfect_extrapolation",
    emu_experiment="Emulator_hist_future",
    emu_period_a="Perfect_cross_validation",
    emu_period_b="Perfect_interpolation",
    save_path=None, figsize=None,
) -> plt.Figure:
    """STYLE 2 -- region-averaged (one AV value per metric)."""
    domains = domains or DOMAINS_DEFAULT
    print(f"[Style 2: regions avg] variable={variable!r}")
    d = _gather(diag_root, variable, domains,
                esd_experiment, esd_period_a, esd_period_b,
                emu_experiment, emu_period_a, emu_period_b,
                per_domain=False)
    return _dual_panels(d, variable, domains, False, save_path, figsize,
                        " ")


def plot_scorecard_overall_rank(
    variable="pr", diag_root=DIAG_ROOT, domains=None,
    esd_experiment="ESD_pseudo_reality",
    esd_period_a="Perfect_cross_validation",
    esd_period_b="Perfect_extrapolation",
    emu_experiment="Emulator_hist_future",
    emu_period_a="Perfect_cross_validation",
    emu_period_b="Perfect_interpolation",
    save_path=None, figsize=None,
) -> plt.Figure:
    """STYLE 3 -- side-by-side Generative | Deterministic overall rank."""
    domains = domains or DOMAINS_DEFAULT
    print(f"[Style 3: overall rank] variable={variable!r}")
    d = _gather(diag_root, variable, domains,
                esd_experiment, esd_period_a, esd_period_b,
                emu_experiment, emu_period_a, emu_period_b,
                per_domain=False)
    return _overall_rank([(variable, d)], save_path=save_path, figsize=figsize)

def plot_scorecard_combined_rank(
    diag_root=DIAG_ROOT, domains=None,
    esd_experiment="ESD_pseudo_reality",
    esd_period_a="Perfect_cross_validation",
    esd_period_b="Perfect_extrapolation",
    emu_experiment="Emulator_hist_future",
    emu_period_a="Perfect_cross_validation",
    emu_period_b="Perfect_interpolation",
    save_path=None, figsize=None,
) -> plt.Figure:
    """STYLE 3 (combined) -- pr + tasmax in one figure, 4 columns per half.
 
    Matches the reference layout exactly: Precipitation (pr) and
    Max. Temp. (tasmax) columns grouped side-by-side within each half.
    """
    domains = domains or DOMAINS_DEFAULT
    print("[Style 3: combined rank] pr + tasmax")
    d_pr     = _gather(diag_root, "pr", domains,
                       esd_experiment, esd_period_a, esd_period_b,
                       emu_experiment, emu_period_a, emu_period_b,
                       per_domain=False)
    d_tasmax = _gather(diag_root, "tasmax", domains,
                       esd_experiment, esd_period_a, esd_period_b,
                       emu_experiment, emu_period_a, emu_period_b,
                       per_domain=False)
    return _overall_rank([("pr", d_pr), ("tasmax", d_tasmax)],
                         save_path=save_path, figsize=figsize)
# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pathlib as _pl
    out = _pl.Path(__file__).resolve().parent
    out.mkdir(exist_ok=True)

    for var in ("pr", "tasmax"):
        plot_scorecard_all_regions(
            variable=var,
            save_path=str(out / f"scorecard_{var}_all_regions.pdf"))
        plt.close("all")

        plot_scorecard_regions_avg(
            variable=var,
            save_path=str(out / f"scorecard_{var}_regions_avg.pdf"))
        plt.close("all")

    plot_scorecard_combined_rank(
        save_path=str(out / "scorecard_combined_overall_rank.pdf"))
    plt.close("all")
        

# plot_scorecard_overall_rank(
#     variable=var,
#     save_path=str(out / f"scorecard_{var}_overall_rank.pdf"))
#plt.close("all")
