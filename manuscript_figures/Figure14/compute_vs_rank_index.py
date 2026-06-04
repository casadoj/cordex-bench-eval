"""
Compute cost vs rank/index figures for Figure 14.

This replaces the metric-unit y axes from the notebook with:
  1. overall rank for each variable and experiment;
  2. a skill index derived from the same overall ranks.

The compute metadata are loaded from ``compute_vs_skillful_final.ipynb`` so the
published table remains the single source for training/inference times.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

try:
    from adjustText import adjust_text

    _HAS_ADJUST = True
except ImportError:
    _HAS_ADJUST = False


SCRIPT_DIR = Path(__file__).resolve().parent
MANUSCRIPT_DIR = SCRIPT_DIR.parent
NOTEBOOK_PATH = SCRIPT_DIR / "compute_vs_skillful_final.ipynb"
DEFAULT_RANK_CSV = MANUSCRIPT_DIR / "scorecard_tidy_overall_rankings.csv"
DEFAULT_OUT_DIR = SCRIPT_DIR / "figures"

COST_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "cost",
    ["#fff5e6", "#ffd9a8", "#fdae6b", "#f16913", "#a63603", "#3d1300"],
    N=256,
)

ALPHA_BEST = 0.92
ALPHA_SUB = 0.24
LABEL_COLOR_BEST = "#111111"
LABEL_COLOR_SUB = "#9a9a9a"


def _setup_style() -> None:
    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11.0,
        "axes.titlesize": 13.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 13.0,
        "axes.labelweight": "bold",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


@lru_cache(maxsize=1)
def _load_notebook_defs() -> Dict[str, object]:
    nb = json.loads(NOTEBOOK_PATH.read_text())
    source = ""
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        text = "".join(cell.get("source", []))
        if "COMPUTE:" in text and "PARAMS:" in text:
            source = text
            break
    if not source:
        raise ValueError(f"Could not find compute metadata in {NOTEBOOK_PATH}")

    marker = "# =============================================================================\n# 4.  BUILD MASTER DATAFRAME"
    source = source.split(marker)[0]
    ns: Dict[str, object] = {"__name__": "figure14_notebook_defs"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(source, str(NOTEBOOK_PATH), "exec"), ns)
    return ns


def _ensure_rank_csv(path: Path) -> None:
    if path.exists():
        return

    sys.path.insert(0, str(MANUSCRIPT_DIR))
    import final_scorecard_tidy as tidy

    datasets = tidy.gather_tidy_data()
    tidy.write_overall_rankings_csv(datasets, path)


def _base_name(model: str) -> str:
    return model[:-5] if model.endswith("-orog") else model


def _size_from_params(params_m: pd.Series, s_min: float = 55, s_max: float = 520) -> np.ndarray:
    p = np.where(
        np.isfinite(params_m.values) & (params_m.values > 0),
        params_m.values,
        0.1,
    )
    lp = np.log10(p)
    lo, hi = np.nanmin(lp), np.nanmax(lp)
    if hi - lo < 1e-6:
        return np.full(len(p), (s_min + s_max) / 2, dtype=float)
    return s_min + (lp - lo) / (hi - lo) * (s_max - s_min)


def _rank_lookup(ranks: pd.DataFrame, model: str, variable: str, column: str) -> float:
    match = ranks[(ranks["model"] == model) & (ranks["variable"] == variable)]
    if match.empty:
        return np.nan
    return float(match.iloc[0][column])


def build_compute_rank_table(rank_csv: Path = DEFAULT_RANK_CSV) -> pd.DataFrame:
    _ensure_rank_csv(rank_csv)
    defs = _load_notebook_defs()
    compute = defs["COMPUTE"]
    params = defs["PARAMS"]
    gpu_tier = defs["GPU_TIER"]

    ranks = pd.read_csv(rank_csv)
    rank_cols = [
        "pr_esd_rank",
        "pr_emulator_rank",
        "tasmax_esd_rank",
        "tasmax_emulator_rank",
    ]

    rows = []
    for model, (gpu, n_gpu, train_h, infer_s, model_type, institution, var_mode) in compute.items():
        cost_factor = 2.0 if var_mode == "Uni" else 1.0
        param_val, param_est = params.get(model, (np.nan, True))
        row = {
            "model": model,
            "base": _base_name(model),
            "institution": institution,
            "type": model_type,
            "var_mode": var_mode,
            "gpu": gpu,
            "gpu_tier": gpu_tier.get(gpu, "Other"),
            "n_gpu": n_gpu,
            "train_h": train_h,
            "eff_train_h": train_h * n_gpu * cost_factor,
            "eff_infer_s": infer_s * n_gpu * cost_factor,
            "params_M": param_val,
            "params_est": param_est,
            "n_models_ranked": np.nanmax(ranks["n_models_ranked"].to_numpy(dtype=float)),
            "pr_esd_rank": _rank_lookup(ranks, model, "pr", "esd_avg_periods_overall_rank"),
            "pr_emulator_rank": _rank_lookup(
                ranks, model, "pr", "emulator_hist_future_avg_periods_overall_rank"
            ),
            "tasmax_esd_rank": _rank_lookup(ranks, model, "tasmax", "esd_avg_periods_overall_rank"),
            "tasmax_emulator_rank": _rank_lookup(
                ranks, model, "tasmax", "emulator_hist_future_avg_periods_overall_rank"
            ),
        }
        vals = [row[col] for col in rank_cols if np.isfinite(row[col])]
        row["mean_overall_rank"] = float(np.mean(vals)) if vals else np.nan
        n_ranked = max(float(row["n_models_ranked"]), 1.0)
        row["rank_skill_index"] = (
            1.0 - (row["mean_overall_rank"] - 1.0) / (n_ranked - 1.0)
            if vals and n_ranked > 1
            else np.nan
        )
        for rank_col in rank_cols:
            index_col = rank_col.replace("_rank", "_skill_index")
            rank = row[rank_col]
            row[index_col] = (
                1.0 - (rank - 1.0) / (n_ranked - 1.0)
                if np.isfinite(rank) and n_ranked > 1
                else np.nan
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    df["rank_skill_index"] = df["rank_skill_index"].clip(0.0, 1.0)
    for col in [c for c in df.columns if c.endswith("_skill_index")]:
        df[col] = df[col].clip(0.0, 1.0)
    df["is_best"] = True
    for _base, group in df.groupby("base"):
        scores = group["mean_overall_rank"].dropna()
        if len(group) < 2 or scores.empty:
            continue
        best_idx = scores.idxmin()
        df.loc[group.index.difference([best_idx]), "is_best"] = False
    return df


def _compute_norm(df: pd.DataFrame) -> mpl.colors.LogNorm:
    cost = df["eff_train_h"].replace(0, np.nan).dropna()
    vmin = max(0.05, np.nanpercentile(cost, 2))
    vmax = min(300.0, np.nanpercentile(cost, 98))
    if vmax <= vmin:
        vmax = vmin * 10.0
    return mpl.colors.LogNorm(vmin=vmin, vmax=vmax)


def _scatter_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    y_col: str,
    *,
    norm: mpl.colors.LogNorm,
    tier_marker: Dict[str, str],
    title: str,
    ylabel: Optional[str] = None,
    is_rank: bool = True,
) -> None:
    sizes = _size_from_params(df["params_M"])

    for is_best in [False, True]:
        sub = df[df["is_best"] == is_best]
        alpha = ALPHA_BEST if is_best else ALPHA_SUB
        edge = "#1a1a1a" if is_best else "#777777"
        line_w = 0.8 if is_best else 0.35
        zorder = 3 if is_best else 2

        for tier, marker in tier_marker.items():
            t = sub[sub["gpu_tier"] == tier]
            t = t[t["eff_infer_s"].notna() & t[y_col].notna()]
            if t.empty:
                continue
            ax.scatter(
                t["eff_infer_s"],
                t[y_col],
                s=sizes[t.index],
                c=t["eff_train_h"],
                cmap=COST_CMAP,
                norm=norm,
                marker=marker,
                edgecolors=edge,
                linewidths=line_w,
                alpha=alpha,
                zorder=zorder,
            )

    ax.set_xscale("log")
    ax.set_xlim(0.08, 10000)
    if is_rank:
        ax.set_ylim(-1, 48)
        ax.set_yticks([1, 10, 20, 30, 40])
        ax.axhline(1.0, ls ='--', alpha =0.2)
    else:
        ax.set_ylim(0.0, 1.04)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    ax.grid(True, which="major", color="#d7d7d7", lw=0.6, zorder=0)
    ax.grid(True, which="minor", color="#eeeeee", lw=0.35, zorder=0)
    ax.set_axisbelow(True)


def _label_points(ax: plt.Axes, df: pd.DataFrame, y_col: str, label_suboptimal: bool = True) -> None:
    texts = []
    for _, row in df.iterrows():
        x, y = row["eff_infer_s"], row[y_col]
        if not (np.isfinite(float(x)) and np.isfinite(float(y))):
            continue
        is_best = bool(row["is_best"])
        if not is_best and not label_suboptimal:
            continue
        text = ax.text(
            x,
            y,
            row["model"],
            fontsize=7.8 if is_best else 6.6,
            color=LABEL_COLOR_BEST if is_best else LABEL_COLOR_SUB,
            alpha=1.0 if is_best else 0.58,
            ha="left",
            va="bottom",
            zorder=5 if is_best else 4,
        )
        texts.append(text)

    if _HAS_ADJUST and texts:
        adjust_text(
            texts,
            ax=ax,
            expand=(1.12, 1.25),
            arrowprops=dict(arrowstyle="-", color="#a8a8a8", lw=0.45, shrinkA=1, shrinkB=1),
            force_text=(0.35, 0.55),
            force_static=(0.15, 0.30),
            min_arrow_len=5,
        )


def _add_legends(
    fig: plt.Figure,
    df: pd.DataFrame,
    norm: mpl.colors.LogNorm,
    tier_marker: Dict[str, str],
    *,
    right_x: float = 0.985,
) -> None:
    cax = fig.add_axes([0.885, 0.565, 0.020, 0.295])
    sm = plt.cm.ScalarMappable(cmap=COST_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Training cost\n(1-GPU equiv. h)", fontsize=10.5, fontweight="bold", labelpad=10)
    cbar.ax.tick_params(labelsize=9.5)
    cbar.outline.set_edgecolor("#999999")
    cbar.outline.set_linewidth(0.5)

    gpu_handles = [
        Line2D(
            [0], [0], marker=marker, color="none",
            markerfacecolor="#cccccc", markeredgecolor="#333333",
            markeredgewidth=0.7, markersize=8.5, label=tier,
        )
        for tier, marker in tier_marker.items()
    ]
    leg_gpu = fig.legend(
        handles=gpu_handles, loc="center right", bbox_to_anchor=(right_x, 0.330),
        frameon=True, fontsize=9.3, title="GPU class", title_fontsize=10.0,
        labelspacing=0.66, handletextpad=0.55, borderpad=0.75,
    )
    leg_gpu.get_frame().set_edgecolor("#cccccc")
    leg_gpu.get_frame().set_linewidth(0.5)
    leg_gpu.get_title().set_fontweight("bold")

    pmax = df["params_M"].max(skipna=True)
    pmin = max(df["params_M"].min(skipna=True), 0.01)
    size_vals = [v for v in [1, 10, 100, pmax] if np.isfinite(v) and v <= pmax * 1.05]
    size_handles = [
        Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor="#999999", markeredgecolor="#333333",
            markeredgewidth=0.7,
            markersize=np.sqrt(_size_from_params(pd.Series([v, pmin, pmax]))[0]),
            label=f"{v:g} M",
        )
        for v in size_vals
    ]
    leg_size = fig.legend(
        handles=size_handles, loc="center right", bbox_to_anchor=(right_x, 0.115),
        frameon=True, fontsize=9.3, title="Parameters", title_fontsize=10.0,
        labelspacing=1.35, handletextpad=0.70, borderpad=0.75,
    )
    leg_size.get_frame().set_edgecolor("#cccccc")
    leg_size.get_frame().set_linewidth(0.5)
    leg_size.get_title().set_fontweight("bold")


def make_rank_figure(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    _setup_style()
    defs = _load_notebook_defs()
    tier_marker = defs["TIER_MARKER"]
    norm = _compute_norm(df)

    fig = plt.figure(figsize=(14.2, 10.4), facecolor="white")
    gs = fig.add_gridspec(
        2, 2, left=0.075, right=0.855, bottom=0.100, top=0.875,
        wspace=0.185, hspace=0.285,
    )
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)])

    panels = [
        (axes[0, 0], "pr_emulator_rank", "Precipitation | Emulator Hist-Future", "Overall rank"),
        (axes[0, 1], "pr_esd_rank", "Precipitation | ESD pseudo-reality", None),
        (axes[1, 0], "tasmax_emulator_rank", "Maximum Temperature | Emulator Hist-Future", "Overall rank"),
        (axes[1, 1], "tasmax_esd_rank", "Maximum Temperature | ESD pseudo-reality", None),
    ]
    for ax, y_col, title, ylabel in panels:
        _scatter_panel(ax, df, y_col, norm=norm, tier_marker=tier_marker, title=title, ylabel=ylabel)
        _label_points(ax, df, y_col)

    for ax in axes[1]:
        ax.set_xlabel("Effective inference time (s yr$^{-1}$; 1-GPU equivalent)")

    fig.suptitle(
        "Compute cost vs. CORDEX-ML-Bench rank",
        fontsize=17.0, fontweight="bold", y=0.965, color="#1a1a1a",
    )
    # fig.text(
    #     0.5, 0.930,
    #     "Training cost is colour; parameters are marker size; GPU class is marker shape; faded points are lower-ranked orog/non-orog variants.",
    #     fontsize=10.2, color="#555555", style="italic", ha="center",
    # )
    _add_legends(fig, df, norm, tier_marker)
    # fig.text(
    #     0.075, 0.030,
    #     "Ranks are average-period overall ranks from final_scorecard_tidy.py; lower is better.",
    #     fontsize=8.7, color="#666666", style="italic", ha="left", va="bottom",
    # )

    if save_path:
        fig.savefig(save_path, dpi=400, bbox_inches="tight", facecolor="white")
        print(f"Saved -> {save_path}")
    return fig


def make_index_figure(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    _setup_style()
    defs = _load_notebook_defs()
    tier_marker = defs["TIER_MARKER"]
    norm = _compute_norm(df)

    fig = plt.figure(figsize=(14.2, 10.4), facecolor="white")
    gs = fig.add_gridspec(
        2, 2, left=0.075, right=0.855, bottom=0.100, top=0.875,
        wspace=0.185, hspace=0.285,
    )
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)])

    panels = [
        (axes[0, 0], "pr_emulator_skill_index", "Precipitation | Emulator Hist-Future", "Skill index"),
        (axes[0, 1], "pr_esd_skill_index", "Precipitation | ESD pseudo-reality", None),
        (axes[1, 0], "tasmax_emulator_skill_index", "Maximum Temperature | Emulator Hist-Future", "Skill index"),
        (axes[1, 1], "tasmax_esd_skill_index", "Maximum Temperature | ESD pseudo-reality", None),
    ]
    for ax, y_col, title, ylabel in panels:
        _scatter_panel(
            ax, df, y_col, norm=norm, tier_marker=tier_marker,
            title=title, ylabel=ylabel, is_rank=False,
        )
        _label_points(ax, df, y_col)

    for ax in axes[1]:
        ax.set_xlabel("Effective inference time (s yr$^{-1}$; 1-GPU equivalent)")

    fig.suptitle(
        "Compute cost vs. CORDEX-ML-Bench skill index",
        fontsize=17.0, fontweight="bold", y=0.965, color="#1a1a1a",
    )
    fig.text(
        0.5, 0.930,
        "Training cost is colour; parameters are marker size; GPU class is marker shape; faded points are lower-ranked orog/non-orog variants.",
        fontsize=10.2, color="#555555", style="italic", ha="center",
    )
    _add_legends(fig, df, norm, tier_marker)

    if save_path:
        fig.savefig(save_path, dpi=400, bbox_inches="tight", facecolor="white")
        print(f"Saved -> {save_path}")
    return fig


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Create Figure14 rank/index compute plots.")
    parser.add_argument("--rank-csv", default=str(DEFAULT_RANK_CSV), help="Tidy rankings CSV path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR), help="Output directory.")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"], choices=["pdf", "png", "svg"])
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build_compute_rank_table(Path(args.rank_csv))
    master_csv = out_dir / "compute_vs_rank_index_master.csv"
    df.to_csv(master_csv, index=False)
    print(f"Saved -> {master_csv}")

    for stem, maker in [
        ("compute_vs_rank", make_rank_figure),
        ("compute_vs_skill_index", make_index_figure),
    ]:
        fig = maker(df)
        for fmt in args.formats:
            path = out_dir / f"{stem}.{fmt}"
            fig.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
            print(f"Saved -> {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
