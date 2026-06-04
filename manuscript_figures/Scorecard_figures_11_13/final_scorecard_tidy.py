"""
Tidy CORDEX-ML-Bench scorecards for compact journal figures.

This script reuses the data-loading and ranking logic from ``final_scorecard.py``
and changes only the visualization layer.  The tidy figures emphasize ranks
instead of raw metric values, keep the original greyscale/gold formal style,
hide empty registry rows by default, and use a single model-label column to
reduce width.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import final_scorecard as base

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 8.0,
    "axes.linewidth": 0.45,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


WHITE = base.WHITE
BLACK = base.BLACK
GOLD = base.GOLD
GREY_CMAP = base.GREY_CMAP
MODEL_H = base.MODEL_H
SECTION_H = base.SECTION_H

GRID = "#ffffff"
GRID_DARK = "#d2d2d2"
MISSING = "#d8d8d8"
OTHER_BG_A = "#f6f6f6"
OTHER_BG_B = "#eeeeee"
OTHER_LABEL = "#555555"

LABEL_W = 2.62
GROUP_GAP = 0.55
HIGHER_IS_BETTER = {
    "tasmax": {"PSS"},
}
EXCLUDE_INDIVIDUAL_METRICS = {"Lag-1"}


def _save(fig: plt.Figure, path: Path, dpi: int = 400) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=WHITE)
    print(f"Saved -> {path}")


def _save_formats(fig: plt.Figure, stem: Path, formats: Sequence[str]) -> None:
    for fmt in formats:
        _save(fig, stem.with_suffix(f".{fmt}"))


def _variable_label(variable: str) -> str:
    return "Precipitation (pr)" if variable == "pr" else "Max. temperature (tasmax)"


def _variable_title(variable: str) -> str:
    return "Precipitation" if variable == "pr" else "Maximum Temperature"


def _short_variable_label(variable: str) -> str:
    return "pr" if variable == "pr" else "tasmax"


def _display_metrics(metric_labels: Sequence[str]) -> List[str]:
    return [metric for metric in metric_labels if metric not in EXCLUDE_INDIVIDUAL_METRICS]


def _metric_ascending(variable: str, metric_label: str) -> bool:
    return metric_label not in HIGHER_IS_BETTER.get(variable, set())


def _clean_rank(rank: Optional[float]) -> Optional[int]:
    if rank is None:
        return None
    try:
        value = float(rank)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return int(round(value))


def _mean_rank(*ranks: Optional[float]) -> Optional[float]:
    vals = []
    for rank in ranks:
        if rank is None:
            continue
        try:
            value = float(rank)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            vals.append(value)
    return round(float(np.mean(vals)), 3) if vals else None


def _pair_rank(
    pair: Optional[Tuple[Optional[int], Optional[int]]],
    index: int,
) -> Optional[int]:
    if not pair or index >= len(pair):
        return None
    return _clean_rank(pair[index])


def _compact_rows(
    available: Sequence[str],
    global_ranks: Dict[str, Optional[int]],
    include_missing: bool = False,
) -> List[dict]:
    rows = base.build_sorted_rows(available, global_ranks)
    if include_missing:
        return rows

    compact: List[dict] = []
    i = 0
    while i < len(rows):
        row = rows[i]
        if row["kind"] != "section":
            if row.get("has_data", True):
                compact.append(row)
            i += 1
            continue

        section = row
        block: List[dict] = []
        i += 1
        while i < len(rows) and rows[i]["kind"] != "section":
            if rows[i].get("has_data", True):
                block.append(rows[i])
            i += 1
        if block:
            compact.append(section)
            compact.extend(block)
    return compact


def _model_rows(rows: Sequence[dict]) -> List[dict]:
    return [row for row in rows if row["kind"] == "model"]


def _model_names(rows: Sequence[dict]) -> List[str]:
    return [row["model_name"] for row in _model_rows(rows)]


def _row_group_map(rows: Sequence[dict]) -> Dict[str, Tuple[str, int]]:
    groups: Dict[str, Tuple[str, int]] = {}
    counts = {"gen": 0, "det": 0, "other": 0}
    current = "other"

    for row in rows:
        if row["kind"] == "section":
            label = row["label"].lower()
            if "generative" in label:
                current = "gen"
            elif "deterministic" in label:
                current = "det"
            else:
                current = "other"
            continue

        model = row["model_name"]
        groups[model] = (current, counts[current])
        counts[current] += 1
    return groups


def _row_colors(group: str, index: int) -> Tuple[str, str]:
    if group == "gen":
        return (base.BG_GEN if index % 2 == 0 else base.BG_GEN2), base.LBL_GEN
    if group == "det":
        return (base.BG_DET if index % 2 == 0 else base.BG_DET2), base.LBL_DET
    return (OTHER_BG_A if index % 2 == 0 else OTHER_BG_B), OTHER_LABEL


def _group_label(group: str) -> str:
    if group == "gen":
        return "Generative"
    if group == "det":
        return "Deterministic"
    return "Other"


def _draw_row_backgrounds(
    ax: plt.Axes,
    rows: Sequence[dict],
    ymap: Dict[str, float],
    total_w: float,
    label_w: float,
) -> None:
    groups = _row_group_map(rows)

    for row in _model_rows(rows):
        model = row["model_name"]
        if model not in ymap:
            continue
        group, idx = groups.get(model, ("other", 0))
        bg, _ = _row_colors(group, idx)
        y = ymap[model]
        ax.add_patch(plt.Rectangle(
            [-label_w, y], total_w + label_w, MODEL_H,
            fc=bg, ec="none", zorder=0,
        ))
        ax.plot([-label_w, total_w], [y, y], color=GRID, lw=0.35, zorder=1)

    i = 0
    while i < len(rows):
        row = rows[i]
        if row["kind"] != "section":
            i += 1
            continue

        section = row
        members: List[str] = []
        i += 1
        while i < len(rows) and rows[i]["kind"] != "section":
            members.append(rows[i]["model_name"])
            i += 1

        tops = [ymap[m] + MODEL_H for m in members if m in ymap]
        if not tops:
            continue
        y = max(tops)
        ax.add_patch(plt.Rectangle(
            [-label_w, y], total_w + label_w, SECTION_H,
            fc=section.get("color", base.HEADER_OTH), ec="none",
            clip_on=False, zorder=8,
        ))
        ax.text(
            -label_w + 0.08, y + SECTION_H * 0.52, section["label"],
            ha="left", va="center", fontsize=7.8, color=WHITE,
            fontweight="bold", zorder=9, clip_on=False,
        )


def _draw_model_labels(
    ax: plt.Axes,
    rows: Sequence[dict],
    ymap: Dict[str, float],
    label_w: float,
) -> None:
    groups = _row_group_map(rows)

    for row in _model_rows(rows):
        model = row["model_name"]
        if model not in ymap:
            continue

        group, idx = groups.get(model, ("other", 0))
        _, label_color = _row_colors(group, idx)
        y = ymap[model] + MODEL_H * 0.5
        if row["is_orog"]:
            ax.text(
                -0.07, y, row["label"], ha="right", va="center",
                fontsize=5.8, color="#777777", style="italic",
                zorder=7,
            )
        else:
            ax.text(
                -0.07, y, row["label"], ha="right", va="center",
                fontsize=6.7, color=label_color, fontweight="semibold",
                zorder=7,
            )


def _rank_label(rank: Optional[float]) -> str:
    if rank is None:
        return ""
    try:
        value = float(rank)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value):
        return ""
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.1f}"


def _rank_norm(rank: Optional[float], max_rank: float) -> Optional[float]:
    if rank is None:
        return None
    try:
        value = float(rank)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return 0.0 if max_rank <= 1 else (value - 1.0) / (max_rank - 1.0)


def _is_finite_rank(rank: Optional[float]) -> bool:
    try:
        return bool(np.isfinite(float(rank)))
    except (TypeError, ValueError):
        return False


def _draw_split_rank_cell(
    ax: plt.Axes,
    x: float,
    y: float,
    upper_rank: Optional[float],
    lower_rank: Optional[float],
    upper_max: float,
    lower_max: float,
    upper_best: bool = False,
    lower_best: bool = False,
    fontsize: float = 6.2,
) -> None:
    if not _is_finite_rank(upper_rank) and not _is_finite_rank(lower_rank):
        ax.add_patch(plt.Rectangle(
            [x, y], 1, MODEL_H, fc=MISSING, ec=GRID, lw=0.35, zorder=2,
        ))
        return

    cu = base._tri(
        ax, x, y, _rank_norm(upper_rank, upper_max),
        upper=True, best=upper_best,
    )
    cl = base._tri(
        ax, x, y, _rank_norm(lower_rank, lower_max),
        upper=False, best=lower_best,
    )

    ulab = _rank_label(upper_rank)
    llab = _rank_label(lower_rank)
    if ulab:
        ax.text(
            x + 0.29, y + 0.72, ulab, ha="center", va="center",
            fontsize=fontsize, color=base._tc(cu), fontfamily="monospace",
            fontweight="bold", zorder=5,
        )
    if llab:
        ax.text(
            x + 0.72, y + 0.28, llab, ha="center", va="center",
            fontsize=fontsize, color=base._tc(cl), fontfamily="monospace",
            fontweight="bold", zorder=5,
        )


def _fmt_metric_value(value: Optional[float]) -> str:
    if value is None:
        return ""
    try:
        return base._fmt(float(value))
    except (TypeError, ValueError):
        return ""


def _get_value(df: Optional[pd.DataFrame], model: str, column) -> float:
    if df is None or model not in df.index or column not in df.columns:
        return np.nan
    return float(df.loc[model, column])


def _draw_split_value_cell(
    ax: plt.Axes,
    x: float,
    y: float,
    upper_value: Optional[float],
    lower_value: Optional[float],
    upper_rank: Optional[float],
    lower_rank: Optional[float],
    upper_max: float,
    lower_max: float,
    upper_best: bool = False,
    lower_best: bool = False,
    fontsize: float = 5.0,
) -> None:
    has_upper = _fmt_metric_value(upper_value) != ""
    has_lower = _fmt_metric_value(lower_value) != ""
    if not has_upper and not has_lower:
        ax.add_patch(plt.Rectangle(
            [x, y], 1, MODEL_H, fc=MISSING, ec=GRID, lw=0.35, zorder=2,
        ))
        return

    cu = base._tri(
        ax, x, y, _rank_norm(upper_rank, upper_max),
        upper=True, best=upper_best,
    )
    cl = base._tri(
        ax, x, y, _rank_norm(lower_rank, lower_max),
        upper=False, best=lower_best,
    )

    ulab = _fmt_metric_value(upper_value)
    llab = _fmt_metric_value(lower_value)
    if ulab:
        ax.text(
            x + 0.30, y + 0.72, ulab, ha="center", va="center",
            fontsize=fontsize, color=base._tc(cu), fontfamily="monospace",
            fontweight="bold", zorder=5,
        )
    if llab:
        ax.text(
            x + 0.72, y + 0.28, llab, ha="center", va="center",
            fontsize=fontsize, color=base._tc(cl), fontfamily="monospace",
            fontweight="bold", zorder=5,
        )


def _blank_grid(ax: plt.Axes, x_positions: Iterable[float], total_h: float) -> None:
    for x in x_positions:
        ax.plot([x, x], [0, total_h], color=GRID_DARK, lw=0.35, zorder=1)


def _metric_rank_df(
    df: Optional[pd.DataFrame],
    model_names: Sequence[str],
    metric_labels: Sequence[str],
    variable: str,
) -> Optional[pd.DataFrame]:
    if df is None:
        return None

    sub = df.reindex(model_names)
    ranks = pd.DataFrame(index=list(model_names), columns=list(metric_labels), dtype=float)
    for metric in metric_labels:
        if metric in sub.columns:
            ranks[metric] = sub[metric].astype(float).rank(
                method="average", ascending=_metric_ascending(variable, metric),
            )
    return ranks


def _metric_domain_rank_df(
    df: Optional[pd.DataFrame],
    model_names: Sequence[str],
    metric_labels: Sequence[str],
    domains_short: Sequence[str],
    variable: str,
) -> Optional[pd.DataFrame]:
    if df is None:
        return None

    cols = pd.MultiIndex.from_product(
        [list(metric_labels), list(domains_short) + ["AV"]],
        names=["Metric", "Domain"],
    )
    sub = df.reindex(model_names)
    ranks = pd.DataFrame(index=list(model_names), columns=cols, dtype=float)
    for metric in metric_labels:
        ascending = _metric_ascending(variable, metric)
        for domain in list(domains_short) + ["AV"]:
            col = (metric, domain)
            if col in sub.columns:
                ranks[col] = sub[col].astype(float).rank(
                    method="average", ascending=ascending,
                )
    return ranks


def _flat_metric_df(df: Optional[pd.DataFrame], metric_labels: Sequence[str]) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    flat = pd.DataFrame(index=df.index)
    for metric in metric_labels:
        col = (metric, "AV")
        if col in df.columns:
            flat[metric] = df[col]
    return flat


def _directional_global_ranks(
    variable: str,
    all_model_names: Sequence[str],
    df_a: Optional[pd.DataFrame],
    df_b: Optional[pd.DataFrame],
    metric_labels: Sequence[str],
) -> Tuple[Dict[str, Optional[int]], Dict[str, Optional[Tuple[int, int]]]]:
    def _pmr(df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if df is None:
            return pd.DataFrame(np.nan, index=list(all_model_names), columns=list(metric_labels))
        sub = df.reindex(all_model_names)
        out = {}
        for metric in metric_labels:
            if metric in sub.columns:
                out[metric] = sub[metric].rank(
                    method="average",
                    ascending=_metric_ascending(variable, metric),
                    na_option="bottom",
                )
            else:
                out[metric] = pd.Series(np.nan, index=sub.index)
        return pd.DataFrame(out)

    pm_a = _pmr(_flat_metric_df(df_a, metric_labels))
    pm_b = _pmr(_flat_metric_df(df_b, metric_labels)) if df_b is not None else None

    sum_a = pm_a.mean(axis=1, skipna=True)
    sum_b = (
        pm_b.mean(axis=1, skipna=True)
        if pm_b is not None
        else pd.Series(np.nan, index=sum_a.index)
    )
    rank_a = sum_a.rank(method="first", ascending=True, na_option="bottom")
    rank_b = (
        sum_b.rank(method="first", ascending=True, na_option="bottom")
        if pm_b is not None
        else pd.Series(np.nan, index=rank_a.index)
    )
    combined = (sum_a + sum_b) / 2.0 if pm_b is not None else sum_a
    overall = combined.rank(method="first", ascending=True, na_option="bottom")

    grank: Dict[str, Optional[int]] = {}
    prank: Dict[str, Optional[Tuple[int, int]]] = {}
    for model in all_model_names:
        overall_rank = overall.get(model, np.nan)
        if np.isfinite(overall_rank):
            grank[model] = int(overall_rank)
            a = rank_a.get(model, np.nan)
            b = rank_b.get(model, np.nan)
            prank[model] = (
                int(a) if np.isfinite(a) else None,
                int(b) if np.isfinite(b) else None,
            )
        else:
            grank[model] = None
            prank[model] = None
    return grank, prank


def _apply_directional_ranks(variable: str, data: dict) -> dict:
    metric_labels = list(data["metric_labels"])
    model_names = list(data["all_model_names"])
    if data["df_esd_a"] is not None:
        data["grank_esd"], data["prank_esd"] = _directional_global_ranks(
            variable, model_names, data["df_esd_a"], data["df_esd_b"], metric_labels,
        )
    if data["df_emu_a"] is not None:
        data["grank_emu"], data["prank_emu"] = _directional_global_ranks(
            variable, model_names, data["df_emu_a"], data["df_emu_b"], metric_labels,
        )
    return data


def _col_max(ranks: Optional[pd.DataFrame], column: str) -> float:
    if ranks is None or column not in ranks.columns:
        return 1.0
    vals = ranks[column].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.nanmax(vals)) if vals.size else 1.0


def _is_col_best(ranks: Optional[pd.DataFrame], column: str, value: Optional[float]) -> bool:
    if ranks is None or column not in ranks.columns or not _is_finite_rank(value):
        return False
    vals = ranks[column].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return bool(vals.size and np.isclose(float(value), float(np.nanmin(vals))))


def _pair_max(pair_dicts: Sequence[Dict[str, Optional[Tuple[int, int]]]]) -> float:
    vals: List[int] = []
    for pair_dict in pair_dicts:
        for pair in pair_dict.values():
            if not pair:
                continue
            vals.extend([rank for rank in pair if rank])
    return float(max(vals)) if vals else 1.0


def _add_rank_colorbar(fig: plt.Figure, x: float, y: float, w: float, h: float, n: int) -> None:
    cax = fig.add_axes([x, y, w*0.9, h])
    sm = plt.cm.ScalarMappable(cmap=GREY_CMAP, norm=mcolors.Normalize(1, max(n, 1)))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([1, max(n, 1)])
    cb.set_ticklabels(["Highest \n performing", "Lowest \n performing"])
    cb.ax.tick_params(labelsize=8.0, length=2, pad=1)
    cb.outline.set_edgecolor("#cccccc")
    cb.outline.set_linewidth(0.4)
    cax.set_title("Overall rank", fontsize=8.2, pad=2, color=BLACK)


def _add_metric_colorbar(fig: plt.Figure, x: float, y: float, w: float, h: float) -> None:
    cax = fig.add_axes([x, y, w, h])
    sm = plt.cm.ScalarMappable(cmap=GREY_CMAP, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["Highest \n performing", "Lowest \n performing"])
    cb.ax.tick_params(labelsize=8.0, length=2, pad=1)
    cb.outline.set_edgecolor("#cccccc")
    cb.outline.set_linewidth(0.4)
    cax.set_title("Metric performance", fontsize=8.2, pad=2, color=BLACK)


def _add_triangle_note(fig: plt.Figure, x: float, y: float, w: float, h: float) -> None:
    ax = fig.add_axes([x, y, w, h])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    codes = [base.MplPath.MOVETO, base.MplPath.LINETO, base.MplPath.LINETO, base.MplPath.CLOSEPOLY]
    sw = 0.12
    ax.add_patch(mpatches.PathPatch(
        base.MplPath([(0.02, 0.88), (0.02 + sw, 0.88), (0.02, 0.16), (0.02, 0.88)], codes),
        fc=base._gc(0.12), ec=WHITE, lw=0.5, transform=ax.transAxes,
    ))
    ax.add_patch(mpatches.PathPatch(
        base.MplPath([(0.02 + sw, 0.88), (0.02 + sw, 0.16), (0.02, 0.16), (0.02 + sw, 0.88)], codes),
        fc=base._gc(0.72), ec=WHITE, lw=0.5, transform=ax.transAxes,
    ))
    ax.text(
        0.18, 0.66, "upper-left: cross-validation",
        ha="left", va="center", fontsize=6.7, color=BLACK, transform=ax.transAxes,
    )
    ax.text(
        0.18, 0.30, "lower-right: extrapolation/interpolation",
        ha="left", va="center", fontsize=6.7, color=BLACK, transform=ax.transAxes,
    )
    ax.text(
        0.68, 0.48, "gold outline: best in column",
        ha="left", va="center", fontsize=6.7, color="#555555", transform=ax.transAxes,
    )


def _setup_table_axis(
    ax: plt.Axes,
    total_w: float,
    total_h: float,
    label_w: float,
    header_h: float,
) -> None:
    ax.set_xlim(-label_w, total_w)
    ax.set_ylim(-0.15, total_h + header_h)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(WHITE)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_combined_headers(
    ax: plt.Axes,
    col_meta: Sequence[dict],
    total_h: float,
) -> None:
    groups: List[Tuple[str, List[int]]] = []
    current_var = None
    current_cols: List[int] = []
    for i, meta in enumerate(col_meta):
        variable = meta["variable"]
        if current_var is not None and variable != current_var:
            groups.append((current_var, current_cols))
            current_cols = []
        current_var = variable
        current_cols.append(i)
    if current_var is not None:
        groups.append((current_var, current_cols))

    for variable, cols in groups:
        x0, x1 = min(cols), max(cols) + 1
        color = "#4a4a4a" if variable == "pr" else "#2f3b52"
        ax.text(
            (x0 + x1) / 2, total_h + 1.36,
            "Precipitation (pr)" if variable == "pr" else "Max. temp. (tasmax)",
            ha="center", va="bottom", fontsize=8.2, color=color,
            fontweight="bold", clip_on=False,
        )
        ax.plot(
            [x0 + 0.07, x1 - 0.07], [total_h + 1.31, total_h + 1.31],
            color=color, lw=1.15, clip_on=False,
        )
        if x0 > 0:
            ax.plot([x0, x0], [0, total_h + 1.38], color="#a8a8a8",
                    lw=0.6, ls="--", zorder=1, clip_on=False)

    for ci, meta in enumerate(col_meta):
        ax.text(
            ci + 0.5, total_h + 0.78, meta["experiment"],
            ha="center", va="bottom", fontsize=8.0, color=BLACK,
            fontweight="bold", clip_on=False,
        )
        ax.text(
            ci + 0.5, total_h + 0.21, f"CV / {meta['lower_label']}",
            ha="center", va="bottom", fontsize=6.5, color="#555555",
            clip_on=False,
        )


def _draw_metric_headers(
    ax: plt.Axes,
    groups: Sequence[dict],
    metrics: Sequence[str],
    total_h: float,
) -> None:
    for group in groups:
        start = group["start"]
        width = len(metrics)
        color = group["color"]
        ax.add_patch(plt.Rectangle(
            [start, total_h + 1.50], width, 0.48,
            fc=color, ec="none", clip_on=False, zorder=8,
        ))
        ax.text(
            start + width / 2, total_h + 1.74,
            f"{group['label']}  |  CV / {group['lower_label']}",
            ha="center", va="center", fontsize=9.2, color=WHITE,
            fontweight="bold", clip_on=False, zorder=9,
        )
        for i, metric in enumerate(metrics):
            label = metric.replace("Mean Bias", "Mean\nBias")
            ax.text(
                start + i + 0.5, total_h + 0.72, label,
                ha="center", va="bottom", fontsize=6.8, color=BLACK,
                fontweight="bold", clip_on=False, linespacing=0.90,
            )
        ax.text(
            start + width / 2, total_h + 0.16,
            "upper-left = CV; lower-right = second period; numbers are metric values",
            ha="center", va="bottom", fontsize=5.9, color="#666666",
            clip_on=False,
        )


def plot_tidy_combined_rank(
    datasets: Sequence[Tuple[str, dict]],
    save_path: Optional[str | Path] = None,
    figsize: Optional[Tuple[float, float]] = None,
    include_missing: bool = False,
) -> plt.Figure:
    """Plot a compact pr + tasmax overall-rank summary."""
    if not datasets:
        raise ValueError("At least one dataset is required.")

    primary = datasets[0][1]
    avail_sets = [set(data["available"]) for _, data in datasets]
    available = sorted(set.intersection(*avail_sets)) if avail_sets else []
    if not available:
        raise ValueError("No common complete models across requested datasets.")

    sort_ranks = (
        primary["grank_esd"]
        if any(v is not None for v in primary["grank_esd"].values())
        else primary["grank_emu"]
    )
    rows = _compact_rows(available, sort_ranks, include_missing=include_missing)
    ymap, total_h = base.row_y_positions(rows)

    pair_list: List[Dict[str, Optional[Tuple[int, int]]]] = []
    col_meta: List[dict] = []
    n_global = 1
    for variable, data in datasets:
        if any(v is not None for v in data["prank_esd"].values()):
            pair_list.append(data["prank_esd"])
            col_meta.append({
                "variable": variable,
                "experiment": "ESD",
                "lower_label": "Extrap.",
            })
        if any(v is not None for v in data["prank_emu"].values()):
            pair_list.append(data["prank_emu"])
            col_meta.append({
                "variable": variable,
                "experiment": "Emulator",
                "lower_label": "Interp.",
            })
        n_global = max(n_global, int(data["n_global"]))

    total_w = float(len(col_meta))
    max_rank = _pair_max(pair_list)
    fig_w = max(6.5, 3.25 + total_w * 0.78) *0.8
    fig_h = max(4.65, total_h * 0.235 + 1.78)
    fig = plt.figure(figsize=figsize or (fig_w, fig_h), facecolor=WHITE)
    ax = fig.add_axes([0.052, 0.130, 0.915, 0.805])

    _draw_row_backgrounds(ax, rows, ymap, total_w, LABEL_W)
    _draw_model_labels(ax, rows, ymap, LABEL_W)
    _blank_grid(ax, np.arange(1, len(col_meta)), total_h)

    for ci, pair_dict in enumerate(pair_list):
        for row in _model_rows(rows):
            model = row["model_name"]
            if model not in ymap:
                continue
            pair = pair_dict.get(model)
            if not pair:
                _draw_split_rank_cell(ax, ci, ymap[model], None, None, max_rank, max_rank)
                continue
            upper, lower = pair
            _draw_split_rank_cell(
                ax, ci, ymap[model], upper, lower, max_rank, max_rank,
                upper_best=(upper == 1), lower_best=(lower == 1),
                fontsize=6.8,
            )

    _draw_combined_headers(ax, col_meta, total_h)
    _setup_table_axis(ax, total_w, total_h, LABEL_W, header_h=1.80)

    fig.suptitle(
        "CORDEX-ML-Bench Overall Rank",
        fontsize=12.4, color=BLACK, fontweight="bold", y=0.966,
    )
    _add_rank_colorbar(fig, 0.115, 0.047, 0.430, 0.014, n_global)
    _add_triangle_note(fig, 0.565, 0.020, 0.390, 0.070)
    fig.text(
        0.50, 0.006,
        "Rows are complete models only by default; lower rank is better.",
        ha="center", va="center", fontsize=6.3, color="#555555", style="italic",
    )

    if save_path is not None:
        _save(fig, Path(save_path))
    return fig


def plot_tidy_metric_ranks(
    variable: str,
    data: dict,
    save_path: Optional[str | Path] = None,
    figsize: Optional[Tuple[float, float]] = None,
    include_missing: bool = False,
    title_suffix: str = "region-averaged",
) -> plt.Figure:
    """Plot a compact region-averaged metric-value scorecard for one variable."""
    sort_ranks = (
        data["grank_esd"]
        if any(v is not None for v in data["grank_esd"].values())
        else data["grank_emu"]
    )
    rows = _compact_rows(data["available"], sort_ranks, include_missing=include_missing)
    ymap, total_h = base.row_y_positions(rows)
    models = _model_names(rows)
    metrics = _display_metrics(data["metric_labels"])

    groups: List[dict] = []
    cursor = 0.0
    if data["df_esd_a"] is not None:
        groups.append({
            "label": "ESD",
            "lower_label": "Extrap.",
            "start": cursor,
            "color": base.HEADER_GEN,
            "raw_a": data["df_esd_a"],
            "raw_b": data["df_esd_b"],
            "rank_a": _metric_rank_df(data["df_esd_a"], models, metrics, variable),
            "rank_b": _metric_rank_df(data["df_esd_b"], models, metrics, variable),
        })
        cursor += len(metrics) + GROUP_GAP
    if data["df_emu_a"] is not None:
        groups.append({
            "label": "Emulator Hist-Future",
            "lower_label": "Interp.",
            "start": cursor,
            "color": base.HEADER_DET,
            "raw_a": data["df_emu_a"],
            "raw_b": data["df_emu_b"],
            "rank_a": _metric_rank_df(data["df_emu_a"], models, metrics, variable),
            "rank_b": _metric_rank_df(data["df_emu_b"], models, metrics, variable),
        })
        cursor += len(metrics)

    if not groups:
        raise ValueError(f"No experiment data available for variable={variable!r}.")

    total_w = cursor - (GROUP_GAP if len(groups) == 1 else 0)

    fig_w = max(8.2, 2.9 + total_w * 0.50)
    fig_h = max(4.9, total_h * 0.235 + 1.95)
    fig = plt.figure(figsize=figsize or (fig_w, fig_h), facecolor=WHITE)
    ax = fig.add_axes([0.045, 0.125, 0.930, 0.795])

    _draw_row_backgrounds(ax, rows, ymap, total_w, LABEL_W)
    _draw_model_labels(ax, rows, ymap, LABEL_W)

    for group_i, group in enumerate(groups):
        start = group["start"]
        raw_a = group["raw_a"]
        raw_b = group["raw_b"]
        rank_a = group["rank_a"]
        rank_b = group["rank_b"]

        for mi, metric in enumerate(metrics):
            x = start + mi
            upper_max = _col_max(rank_a, metric)
            lower_max = _col_max(rank_b, metric)
            for row in _model_rows(rows):
                model = row["model_name"]
                if model not in ymap:
                    continue
                upper_rank = rank_a.loc[model, metric] if rank_a is not None else np.nan
                lower_rank = rank_b.loc[model, metric] if rank_b is not None else np.nan
                _draw_split_value_cell(
                    ax, x, ymap[model],
                    _get_value(raw_a, model, metric),
                    _get_value(raw_b, model, metric),
                    upper_rank, lower_rank, upper_max, lower_max,
                    upper_best=_is_col_best(rank_a, metric, upper_rank),
                    lower_best=_is_col_best(rank_b, metric, lower_rank),
                    fontsize=4.9,
                )

        x_positions = [start + i for i in range(1, len(metrics))]
        _blank_grid(ax, x_positions, total_h)
        if group_i > 0:
            ax.plot(
                [start - GROUP_GAP / 2, start - GROUP_GAP / 2],
                [0, total_h + 1.95],
                color="#999999", lw=0.7, ls="--", zorder=1, clip_on=False,
            )

    _draw_metric_headers(ax, groups, metrics, total_h)
    _setup_table_axis(ax, total_w, total_h, LABEL_W, header_h=2.15)

    fig.suptitle(
        f"CORDEX-ML-Bench {_variable_title(variable)} Scorecard ({title_suffix})",
        fontsize=12.4, color=BLACK, fontweight="bold", y=0.955,
    )
    _add_metric_colorbar(fig, 0.115, 0.047, 0.420, 0.014)
    _add_triangle_note(fig, 0.570, 0.020, 0.390, 0.070)
    fig.text(
        0.50, 0.006,
        "Cells show metric values; shading and gold outlines are based on within-column performance.",
        ha="center", va="center", fontsize=6.3, color="#555555", style="italic",
    )

    if save_path is not None:
        _save(fig, Path(save_path))
    return fig


def _draw_all_region_headers(
    ax: plt.Axes,
    groups: Sequence[dict],
    metrics: Sequence[str],
    domains_short: Sequence[str],
    total_h: float,
) -> None:
    domains = list(domains_short) + ["AV"]
    metric_w = len(domains)
    for group in groups:
        start = group["start"]
        width = len(metrics) * metric_w
        color = group["color"]
        ax.add_patch(plt.Rectangle(
            [start, total_h + 1.78], width, 0.48,
            fc=color, ec="none", clip_on=False, zorder=8,
        ))
        ax.text(
            start + width / 2, total_h + 2.02,
            f"{group['label']}  |  CV / {group['lower_label']}",
            ha="center", va="center", fontsize=9.2, color=WHITE,
            fontweight="bold", clip_on=False, zorder=9,
        )
        for mi, metric in enumerate(metrics):
            base_x = start + mi * metric_w
            ax.text(
                base_x + metric_w / 2, total_h + 1.18, metric.replace("Mean Bias", "Mean Bias"),
                ha="center", va="bottom", fontsize=6.4, color=BLACK,
                fontweight="bold", clip_on=False,
            )
            ax.plot(
                [base_x, base_x], [0, total_h + 1.45],
                color="#d0d0d0", lw=0.35, zorder=1, clip_on=False,
            )
            for di, domain in enumerate(domains):
                ax.text(
                    base_x + di + 0.5, total_h + 0.48, domain,
                    ha="center", va="bottom", fontsize=5.4,
                    color="#333333" if domain == "AV" else "#666666",
                    fontweight="bold" if domain == "AV" else "normal",
                    clip_on=False,
                )


def plot_tidy_all_regions(
    variable: str,
    data: dict,
    domains: Sequence[str],
    save_path: Optional[str | Path] = None,
    figsize: Optional[Tuple[float, float]] = None,
    include_missing: bool = False,
) -> plt.Figure:
    """Plot the full ALPS/NZ/SA/AV tidy scorecard for one variable."""
    sort_ranks = (
        data["grank_esd"]
        if any(v is not None for v in data["grank_esd"].values())
        else data["grank_emu"]
    )
    rows = _compact_rows(data["available"], sort_ranks, include_missing=include_missing)
    ymap, total_h = base.row_y_positions(rows)
    models = _model_names(rows)
    metrics = _display_metrics(data["metric_labels"])
    domains_short = [domain.split("_")[0] for domain in domains]
    domain_labels = domains_short + ["AV"]
    metric_w = len(domain_labels)

    groups: List[dict] = []
    cursor = 0.0
    if data["df_esd_a"] is not None:
        groups.append({
            "label": "ESD",
            "lower_label": "Extrap.",
            "start": cursor,
            "color": base.HEADER_GEN,
            "raw_a": data["df_esd_a"],
            "raw_b": data["df_esd_b"],
            "rank_a": _metric_domain_rank_df(data["df_esd_a"], models, metrics, domains_short, variable),
            "rank_b": _metric_domain_rank_df(data["df_esd_b"], models, metrics, domains_short, variable),
        })
        cursor += len(metrics) * metric_w + GROUP_GAP
    if data["df_emu_a"] is not None:
        groups.append({
            "label": "Emulator Hist-Future",
            "lower_label": "Interp.",
            "start": cursor,
            "color": base.HEADER_DET,
            "raw_a": data["df_emu_a"],
            "raw_b": data["df_emu_b"],
            "rank_a": _metric_domain_rank_df(data["df_emu_a"], models, metrics, domains_short, variable),
            "rank_b": _metric_domain_rank_df(data["df_emu_b"], models, metrics, domains_short, variable),
        })
        cursor += len(metrics) * metric_w

    if not groups:
        raise ValueError(f"No experiment data available for variable={variable!r}.")

    total_w = cursor - (GROUP_GAP if len(groups) == 1 else 0)

    fig_w = max(18.0, 2.8 + total_w * 0.315)
    fig_h = max(4.95, total_h * 0.235 + 2.18)
    fig = plt.figure(figsize=figsize or (fig_w, fig_h), facecolor=WHITE)
    ax = fig.add_axes([0.030, 0.125, 0.950, 0.795])

    _draw_row_backgrounds(ax, rows, ymap, total_w, LABEL_W)
    _draw_model_labels(ax, rows, ymap, LABEL_W)

    for group_i, group in enumerate(groups):
        start = group["start"]
        raw_a = group["raw_a"]
        raw_b = group["raw_b"]
        rank_a = group["rank_a"]
        rank_b = group["rank_b"]

        for mi, metric in enumerate(metrics):
            for di, domain in enumerate(domain_labels):
                x = start + mi * metric_w + di
                col = (metric, domain)
                upper_max = _col_max(rank_a, col)
                lower_max = _col_max(rank_b, col)
                for row in _model_rows(rows):
                    model = row["model_name"]
                    if model not in ymap:
                        continue
                    upper_rank = rank_a.loc[model, col] if rank_a is not None and col in rank_a.columns else np.nan
                    lower_rank = rank_b.loc[model, col] if rank_b is not None and col in rank_b.columns else np.nan
                    _draw_split_value_cell(
                        ax, x, ymap[model],
                        _get_value(raw_a, model, col),
                        _get_value(raw_b, model, col),
                        upper_rank, lower_rank, upper_max, lower_max,
                        upper_best=_is_col_best(rank_a, col, upper_rank),
                        lower_best=_is_col_best(rank_b, col, lower_rank),
                        fontsize=3.25,
                    )

        if group_i > 0:
            ax.plot(
                [start - GROUP_GAP / 2, start - GROUP_GAP / 2],
                [0, total_h + 2.25],
                color="#999999", lw=0.7, ls="--", zorder=1, clip_on=False,
            )

    _draw_all_region_headers(ax, groups, metrics, domains_short, total_h)
    for x in np.arange(1, int(np.ceil(total_w))):
        ax.plot([x, x], [0, total_h], color="#e1e1e1", lw=0.25, zorder=1)
    _setup_table_axis(ax, total_w, total_h, LABEL_W, header_h=2.45)

    fig.suptitle(
        f"CORDEX-ML-Bench {_variable_title(variable)} Scorecard (all regions)",
        fontsize=12.4, color=BLACK, fontweight="bold", y=0.955,
    )
    _add_metric_colorbar(fig, 0.085, 0.047, 0.450, 0.014)
    _add_triangle_note(fig, 0.585, 0.020, 0.365, 0.070)
    fig.text(
        0.50, 0.006,
        "Each metric is shown for ALPS, NZ, SA, and AV; cells show metric values with performance shading.",
        ha="center", va="center", fontsize=6.3, color="#555555", style="italic",
    )

    if save_path is not None:
        _save(fig, Path(save_path))
    return fig


def gather_tidy_data(
    diag_root: str = base.DIAG_ROOT,
    domains: Optional[Sequence[str]] = None,
    esd_experiment: str = "ESD_pseudo_reality",
    esd_period_a: str = "Perfect_cross_validation",
    esd_period_b: str = "Perfect_extrapolation",
    emu_experiment: str = "Emulator_hist_future",
    emu_period_a: str = "Perfect_cross_validation",
    emu_period_b: str = "Perfect_interpolation",
    per_domain: bool = False,
) -> Dict[str, dict]:
    """Load pr and tasmax data using final_scorecard.py metric-loading logic."""
    domains = list(domains or base.DOMAINS_DEFAULT)
    datasets: Dict[str, dict] = {}
    for variable in ("pr", "tasmax"):
        print(f"[tidy gather] variable={variable!r}")
        datasets[variable] = base._gather(
            diag_root, variable, domains,
            esd_experiment, esd_period_a, esd_period_b,
            emu_experiment, emu_period_a, emu_period_b,
            per_domain=per_domain,
        )
        datasets[variable] = _apply_directional_ranks(variable, datasets[variable])
    return datasets


def build_overall_rankings_table(
    datasets: Dict[str, dict],
    include_missing: bool = False,
) -> pd.DataFrame:
    """Return one CSV-ready row per variable/model overall ranking."""
    records: List[dict] = []

    for variable, data in datasets.items():
        sort_ranks = (
            data["grank_esd"]
            if any(v is not None for v in data["grank_esd"].values())
            else data["grank_emu"]
        )
        rows = _compact_rows(data["available"], sort_ranks, include_missing=include_missing)
        groups = _row_group_map(rows)

        for order, row in enumerate(_model_rows(rows), start=1):
            model = row["model_name"]
            group, _ = groups.get(model, ("other", 0))
            esd_pair = data["prank_esd"].get(model)
            emu_pair = data["prank_emu"].get(model)
            esd_cv = _pair_rank(esd_pair, 0)
            esd_extra = _pair_rank(esd_pair, 1)
            emu_cv = _pair_rank(emu_pair, 0)
            emu_interp = _pair_rank(emu_pair, 1)

            records.append({
                "variable": variable,
                "variable_label": _variable_label(variable),
                "model_group": _group_label(group),
                "model": model,
                "display_model": row["label"],
                "is_orog": bool(row["is_orog"]),
                "plot_order": order,
                "n_models_ranked": int(data["n_global"]),
                "esd_avg_periods_overall_rank": _clean_rank(data["grank_esd"].get(model)),
                "esd_cross_validation_rank": esd_cv,
                "esd_extrapolation_rank": esd_extra,
                "esd_period_rank_mean": _mean_rank(esd_cv, esd_extra),
                "emulator_hist_future_avg_periods_overall_rank": _clean_rank(
                    data["grank_emu"].get(model)
                ),
                "emulator_hist_future_cross_validation_rank": emu_cv,
                "emulator_hist_future_interpolation_rank": emu_interp,
                "emulator_hist_future_period_rank_mean": _mean_rank(emu_cv, emu_interp),
            })

    table = pd.DataFrame.from_records(records)
    if table.empty:
        return table

    int_columns = [
        "plot_order",
        "n_models_ranked",
        "esd_avg_periods_overall_rank",
        "esd_cross_validation_rank",
        "esd_extrapolation_rank",
        "emulator_hist_future_avg_periods_overall_rank",
        "emulator_hist_future_cross_validation_rank",
        "emulator_hist_future_interpolation_rank",
    ]
    for column in int_columns:
        table[column] = pd.to_numeric(table[column], errors="coerce").astype("Int64")

    return table


def write_overall_rankings_csv(
    datasets: Dict[str, dict],
    save_path: str | Path,
    include_missing: bool = False,
) -> pd.DataFrame:
    table = build_overall_rankings_table(datasets, include_missing=include_missing)
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    print(f"Saved -> {path}")
    return table


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Create tidy journal-oriented CORDEX-ML-Bench scorecards.",
    )
    parser.add_argument(
        "--diag-root", default=base.DIAG_ROOT,
        help="Diagnostics netCDF root used by final_scorecard.py.",
    )
    parser.add_argument(
        "--output-dir", default=str(Path(__file__).resolve().parent),
        help="Directory for tidy scorecard outputs.",
    )
    parser.add_argument(
        "--domains", nargs="+", default=base.DOMAINS_DEFAULT,
        help="Domain directories to average.",
    )
    parser.add_argument(
        "--formats", nargs="+", default=["pdf", "png"],
        choices=["pdf", "png", "svg"],
        help="Output formats to write.",
    )
    parser.add_argument(
        "--figures", nargs="+", default=["combined", "regions_avg", "all_regions"],
        choices=["combined", "regions_avg", "all_regions", "metrics"],
        help="Which tidy figure families to produce.",
    )
    parser.add_argument(
        "--rank-csv",
        default=None,
        help=(
            "Path for the overall rankings CSV. Defaults to "
            "<output-dir>/scorecard_tidy_overall_rankings.csv."
        ),
    )
    parser.add_argument(
        "--no-csv", action="store_true",
        help="Skip writing the overall rankings CSV.",
    )
    parser.add_argument(
        "--include-missing", action="store_true",
        help="Show registry rows without complete data, matching the denser original tables.",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    datasets = gather_tidy_data(
        diag_root=args.diag_root, domains=args.domains, per_domain=False,
    )

    if not args.no_csv:
        csv_path = Path(args.rank_csv) if args.rank_csv else out_dir / "scorecard_tidy_overall_rankings.csv"
        write_overall_rankings_csv(
            datasets, csv_path, include_missing=args.include_missing,
        )

    if "combined" in args.figures:
        fig = plot_tidy_combined_rank(
            [("pr", datasets["pr"]), ("tasmax", datasets["tasmax"])],
            include_missing=args.include_missing,
        )
        _save_formats(fig, out_dir / "scorecard_tidy_combined_overall_rank", args.formats)
        plt.close(fig)

    if "regions_avg" in args.figures or "metrics" in args.figures:
        for variable, data in datasets.items():
            fig = plot_tidy_metric_ranks(
                variable, data, include_missing=args.include_missing,
                title_suffix="region-averaged",
            )
            _save_formats(
                fig,
                out_dir / f"scorecard_tidy_{_short_variable_label(variable)}_regions_avg",
                args.formats,
            )
            plt.close(fig)

    if "all_regions" in args.figures:
        full_datasets = gather_tidy_data(
            diag_root=args.diag_root, domains=args.domains, per_domain=True,
        )
        for variable, data in full_datasets.items():
            fig = plot_tidy_all_regions(
                variable, data, args.domains, include_missing=args.include_missing,
            )
            _save_formats(
                fig,
                out_dir / f"scorecard_tidy_{_short_variable_label(variable)}_all_regions",
                args.formats,
            )
            plt.close(fig)


if __name__ == "__main__":
    main()
