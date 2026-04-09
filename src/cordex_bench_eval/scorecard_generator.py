"""
Scorecard Generator for CORDEX Benchmark Evaluation.

Generates WeatherBench-style colored scorecard tables as PDF. Each scorecard
shows models as rows and metrics as grouped columns (with one sub-column per
domain). Cells are colored by per-column min-max normalized scores so that the
best-performing model is blue and the worst is red, sharing a single colorbar
across all metrics.

When two experiment names are passed, both experiments are rendered in the same
table with each cell split diagonally: the upper-left triangle shows the first
experiment and the lower-right triangle shows the second.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .catalog import SubmissionCatalog
from .config import SubmissionWorkflowConfig
from .table_generator import TableGenerator


class ScorecardGenerator:
    """
    Generate WeatherBench-style scorecard tables as PDF.

    Wraps a :class:`TableGenerator` to load pre-computed metric data, then
    transposes the result so that models appear as rows and ``(Metric, Domain)``
    pairs as columns. Cells are colored using per-column min-max normalization
    with a shared diverging colormap.

    Parameters
    ----------
    config : SubmissionWorkflowConfig
        Configuration containing diagnostic_path.
    catalog : SubmissionCatalog
        Submission catalog to list available models.
    table_config_path : str | Path
        Path to JSON file defining table specifications.
    """

    def __init__(self, config: SubmissionWorkflowConfig, catalog: SubmissionCatalog,
                 table_config_path: str | Path):
        self.table_generator = TableGenerator(config, catalog, table_config_path)

    def generate_scorecard(self, table_name: str | List[str],
                           submission_names: Optional[List[str]] = None,
                           domains: Optional[List[str]] = None,
                           include_average: bool = True) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Generate a scorecard DataFrame with models as rows.

        Parameters
        ----------
        table_name : str or list of str
            Name of table specification from config file. If a list of
            exactly two names is given, two scorecards are generated (one
            per experiment) and returned as a list for dual-experiment
            rendering by :meth:`export_scorecard_pdf`.
        submission_names : list of str, optional
            Model names to include. If None, uses all in catalog.
        domains : list of str, optional
            Domains to include.
        include_average : bool, default True
            Include AV (average) domain column for each metric.

        Returns
        -------
        pd.DataFrame or list of pd.DataFrame
            Single DataFrame when *table_name* is a string (index = model
            names, columns = MultiIndex ``(Metric, Domain)``), or a list
            of two such DataFrames when *table_name* is a two-element list.
        """
        if isinstance(table_name, list):
            if len(table_name) != 2:
                raise ValueError(
                    "table_name list must contain exactly 2 experiment names."
                )
            scorecards = []
            for name in table_name:
                df = self.table_generator.generate_table(
                    table_name=name,
                    submission_names=submission_names,
                    domains=domains,
                    include_average=include_average,
                )
                sc = df.T
                sc.attrs = df.attrs
                scorecards.append(sc)

            if not scorecards[0].index.equals(scorecards[1].index):
                raise ValueError(
                    "Both experiments must share the same model index."
                )
            if not scorecards[0].columns.equals(scorecards[1].columns):
                raise ValueError(
                    "Both experiments must share the same (Metric, Domain) columns."
                )
            return scorecards

        df = self.table_generator.generate_table(
            table_name=table_name,
            submission_names=submission_names,
            domains=domains,
            include_average=include_average,
        )
        scorecard = df.T
        scorecard.attrs = df.attrs
        return scorecard

    def export_scorecard_pdf(self, df: pd.DataFrame | List[pd.DataFrame],
                             output_path: str | Path,
                             float_format: str = "%.2f",
                             cmap: str = "RdBu_r",
                             description: str = "",
                             normalization: Optional[str] = None,
                             average_only: bool = False,
                             n_color_levels: int = 10,
                             highlight_best: bool = True,
                             cell_text_color: Optional[str] = None,
                             best_cell_border_color: Optional[str] = None,
                             best_cell_border_linewidth: float = 2.2,
                             colorbar_worse_on_left: bool = True) -> None:
        """
        Render a scorecard DataFrame as a colored PDF table.

        Parameters
        ----------
        df : pd.DataFrame or list of pd.DataFrame
            Scorecard DataFrame (models as rows, ``(Metric, Domain)`` columns).
            If a list of two DataFrames is passed (from a dual-experiment
            :meth:`generate_scorecard` call), each cell is split diagonally:
            the upper-left triangle shows the first experiment and the
            lower-right triangle shows the second.
        output_path : str or Path
            Output PDF file path.
        float_format : str, default "%.2f"
            Format string for cell values.
        cmap : str, default "RdBu_r"
            Matplotlib colormap name (diverging recommended).
        description : str, optional
            Description text shown above the table.
        normalization : str or None, default None
            Normalization strategy for cell coloring. ``None`` uses plain
            min-max normalization.  Alternatives:

            * ``"rank"`` -- color by rank position (most robust against
              outliers, but loses magnitude information).
            * ``"percentile"`` -- clip at 5th/95th percentiles, then min-max
              normalize (balances robustness with magnitude info).
            * ``"robust_zscore"`` -- center at median, scale by IQR, clip at
              +/-2 IQR, then rescale to [0, 1].
        average_only : bool, default False
            If True, only show the AV (average) column for each metric,
            hiding the per-domain sub-columns.
        n_color_levels : int, default 10
            Number of discrete colour bins for the colorbar.
        highlight_best : bool, default True
            If True, draw a border around the best model(s) in each metric
            column. In single-experiment mode a rectangular border is used;
            in dual-experiment mode the border is drawn on the corresponding
            triangle only.
        cell_text_color : str or None, default None
            Matplotlib color for numeric cell values. If ``None``, the label
            colour is chosen automatically from the cell background (light text
            on dark fills, dark text on light fills). If set, every non-NaN cell
            uses this colour (e.g. ``\"black\"`` for print-friendly tables).
        best_cell_border_color : str or None, default None
            Border colour for best-performing cells when *highlight_best* is
            True. If ``None``, ``\"black\"`` is used.
        best_cell_border_linewidth : float, default 2.2
            Stroke width (points) for the best-cell border.
        colorbar_worse_on_left : bool, default True
            If True, the horizontal colorbar runs from worse (left) to better
            (right), matching the usual reading direction. If False, better is
            on the left and worse on the right. Cell colours are unchanged; only
            the legend gradient and end labels are flipped (using a reversed
            colormap for the colorbar so colours still match the table).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # --- Dual-experiment detection ---------------------------------------
        is_dual = isinstance(df, list)
        if is_dual:
            if len(df) != 2:
                raise ValueError("df list must contain exactly 2 DataFrames.")
            df1, df2 = df
        else:
            df1 = df
            df2 = None

        if average_only:
            av_cols = [c for c in df1.columns if c[1] == "AV"]
            if not av_cols:
                raise ValueError("No AV columns found. Set include_average=True "
                                 "in generate_scorecard().")
            df1 = df1[av_cols]
            if is_dual:
                df2 = df2[av_cols]

        # --- Normalized scores & best masks ----------------------------------
        norm_scores1 = self._compute_normalized_scores(df1, normalization=normalization)
        best_mask1 = (
            self.table_generator._compute_best_mask(df1.T).T
            if highlight_best
            else pd.DataFrame(False, index=df1.index, columns=df1.columns)
        )

        if is_dual:
            norm_scores2 = self._compute_normalized_scores(df2, normalization=normalization)
            best_mask2 = (
                self.table_generator._compute_best_mask(df2.T).T
                if highlight_best
                else pd.DataFrame(False, index=df2.index, columns=df2.columns)
            )

        colormap = plt.get_cmap(cmap, n_color_levels)
        boundaries = np.linspace(0, 1, n_color_levels + 1)
        color_norm = mcolors.BoundaryNorm(boundaries, colormap.N)

        # --- Layout ----------------------------------------------------------
        n_rows, n_cols = df1.shape
        col_width = 0.55
        row_height = 0.35
        left_margin = 2.8
        top_margin = 1.0 + (0.3 if description else 0.0)
        bottom_margin = 0.8

        fig_width = left_margin + n_cols * col_width + 0.3
        fig_height = top_margin + n_rows * row_height + bottom_margin

        fig = plt.figure(figsize=(fig_width, fig_height))

        table_left = left_margin / fig_width
        table_bottom = bottom_margin / fig_height
        table_width = (n_cols * col_width) / fig_width
        table_height = (n_rows * row_height) / fig_height

        ax = fig.add_axes([table_left, table_bottom, table_width, table_height])
        ax.set_xlim(0, n_cols)
        ax.set_ylim(n_rows, 0)
        ax.axis("off")

        # Half the highlight stroke width in data units so the stroke stays
        # *inside* the cell. Matplotlib draws strokes centered on the path; a
        # thick edge on the 1×1 cell boundary otherwise straddles the grid and
        # miter joins can spike at corners.
        tw, th = fig_width * table_width, fig_height * table_height
        half_lw_x = (best_cell_border_linewidth / 2) / 72 * n_cols / tw
        half_lw_y = (best_cell_border_linewidth / 2) / 72 * n_rows / th
        half_lw_x = min(half_lw_x, 0.45)
        half_lw_y = min(half_lw_y, 0.45)

        border_color = best_cell_border_color or "black"

        # --- Cells -----------------------------------------------------------
        if is_dual:
            shrink = max(half_lw_x, half_lw_y) * 2.5
            for i in range(n_rows):
                for j in range(n_cols):
                    val1 = df1.iloc[i, j]
                    val2 = df2.iloc[i, j]
                    score1 = norm_scores1.iloc[i, j]
                    score2 = norm_scores2.iloc[i, j]
                    is_nan1 = pd.isna(val1)
                    is_nan2 = pd.isna(val2)

                    fc1 = "#f0f0f0" if is_nan1 else colormap(color_norm(score1))
                    fc2 = "#f0f0f0" if is_nan2 else colormap(color_norm(score2))

                    # Upper-left triangle (experiment 1)
                    verts_ul = [(j, i), (j + 1, i), (j, i + 1)]
                    ax.add_patch(Polygon(
                        verts_ul, closed=True, facecolor=fc1,
                        edgecolor="none", linewidth=0, zorder=1,
                    ))

                    # Lower-right triangle (experiment 2)
                    verts_lr = [(j + 1, i), (j + 1, i + 1), (j, i + 1)]
                    ax.add_patch(Polygon(
                        verts_lr, closed=True, facecolor=fc2,
                        edgecolor="none", linewidth=0, zorder=1,
                    ))

                    # Best-cell border on the corresponding triangle
                    if bool(best_mask1.iloc[i, j]) and highlight_best:
                        cx, cy = j + 1 / 3, i + 1 / 3
                        inset = [(vx + (cx - vx) * shrink,
                                  vy + (cy - vy) * shrink)
                                 for vx, vy in verts_ul]
                        ax.add_patch(Polygon(
                            inset, closed=True, facecolor="none",
                            edgecolor=border_color,
                            linewidth=best_cell_border_linewidth,
                            joinstyle="round", zorder=4,
                        ))

                    if bool(best_mask2.iloc[i, j]) and highlight_best:
                        cx, cy = j + 2 / 3, i + 2 / 3
                        inset = [(vx + (cx - vx) * shrink,
                                  vy + (cy - vy) * shrink)
                                 for vx, vy in verts_lr]
                        ax.add_patch(Polygon(
                            inset, closed=True, facecolor="none",
                            edgecolor=border_color,
                            linewidth=best_cell_border_linewidth,
                            joinstyle="round", zorder=4,
                        ))

                    # Value labels
                    if not is_nan1:
                        tc1 = self._text_color_for_background(fc1, cell_text_color)
                        ax.text(j + 0.3, i + 0.35, float_format % val1,
                                ha="center", va="center", fontsize=5.5,
                                color=tc1, zorder=5)

                    if not is_nan2:
                        tc2 = self._text_color_for_background(fc2, cell_text_color)
                        ax.text(j + 0.7, i + 0.65, float_format % val2,
                                ha="center", va="center", fontsize=5.5,
                                color=tc2, zorder=5)
        else:
            for i in range(n_rows):
                for j in range(n_cols):
                    val = df1.iloc[i, j]
                    score = norm_scores1.iloc[i, j]
                    is_nan = pd.isna(val)

                    facecolor = "#f0f0f0" if is_nan else colormap(color_norm(score))
                    is_best = bool(best_mask1.iloc[i, j]) and highlight_best
                    if is_best:
                        rect_fill = plt.Rectangle(
                            (j, i), 1, 1, facecolor=facecolor,
                            edgecolor="none", linewidth=0, zorder=1,
                        )
                        ax.add_patch(rect_fill)
                        rect_border = plt.Rectangle(
                            (j + half_lw_x, i + half_lw_y),
                            1 - 2 * half_lw_x, 1 - 2 * half_lw_y,
                            facecolor="none", edgecolor=border_color,
                            linewidth=best_cell_border_linewidth,
                            joinstyle="round", zorder=2,
                        )
                        ax.add_patch(rect_border)
                    else:
                        rect = plt.Rectangle(
                            (j, i), 1, 1, facecolor=facecolor,
                            edgecolor="white", linewidth=0.5, joinstyle="round",
                            zorder=1,
                        )
                        ax.add_patch(rect)

                    if not is_nan:
                        text_color = self._text_color_for_background(
                            facecolor, cell_text_color)
                        ax.text(j + 0.5, i + 0.5, float_format % val,
                                ha="center", va="center", fontsize=6.5,
                                color=text_color, zorder=5)

        # --- Row labels (model names) ----------------------------------------
        for i, model in enumerate(df1.index):
            ax.text(-0.15, i + 0.5, model, ha="right", va="center", fontsize=7)

        # --- Column headers ---------------------------------------------------
        metrics = df1.columns.get_level_values("Metric")
        domains = df1.columns.get_level_values("Domain")

        if not average_only:
            for j in range(n_cols):
                ax.text(j + 0.5, -0.1, domains[j], ha="center", va="bottom",
                        fontsize=6, rotation=0)

        metric_groups = []
        current, start = None, 0
        for j, m in enumerate(metrics):
            if m != current:
                if current is not None:
                    metric_groups.append((current, start, j))
                current, start = m, j
        metric_groups.append((current, start, n_cols))

        header_y = -0.15 if average_only else -0.55
        line_y = -0.05 if average_only else -0.4
        for name, s, e in metric_groups:
            mid = (s + e) / 2.0
            ax.text(mid, header_y, name, ha="center", va="bottom",
                    fontsize=7, fontweight="bold")
            ax.plot([s + 0.05, e - 0.05], [line_y, line_y], color="black",
                    linewidth=0.7, clip_on=False)
            if s > 0:
                ax.plot([s, s], [line_y + 0.05, n_rows], color="grey",
                        linewidth=0.6, clip_on=False)

        # --- Description ------------------------------------------------------
        if description:
            pad = 16 if average_only else 28
            ax.set_title(description, fontsize=8, fontstyle="italic", pad=pad)

        # --- Colorbar ---------------------------------------------------------
        cbar_left = table_left
        cbar_bottom = 0.15 / fig_height
        cbar_width = table_width
        cbar_height = 0.18 / fig_height

        cbar_cmap = colormap.reversed() if colorbar_worse_on_left else colormap
        sm = plt.cm.ScalarMappable(cmap=cbar_cmap, norm=color_norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([cbar_left, cbar_bottom,
                                cbar_width, cbar_height])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.set_ticks([0, 1])
        if colorbar_worse_on_left:
            cbar.set_ticklabels(["Worse", "Better"])
        else:
            cbar.set_ticklabels(["Better", "Worse"])
        cbar.ax.tick_params(labelsize=7)
        cbar.outline.set_linewidth(0.4)

        fig.savefig(output_path, bbox_inches="tight", dpi=200)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _text_color_for_background(facecolor, cell_text_color: Optional[str] = None) -> str:
        """Choose a readable text colour given the cell background."""
        if cell_text_color is not None:
            return cell_text_color
        rgba = mcolors.to_rgba(facecolor)
        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        return "white" if lum < 0.45 else "black"

    def _compute_normalized_scores(self, df: pd.DataFrame,
                                    normalization: Optional[str] = None) -> pd.DataFrame:
        """
        Compute per-column normalized scores in [0, 1].

        Each column ``(Metric, Domain)`` is independently normalized so that
        0 = best and 1 = worst model. The scoring direction (lower is better
        vs. closer to zero) follows the same rules as
        :meth:`TableGenerator._score_row_for_best`.

        Parameters
        ----------
        df : pd.DataFrame
            Scorecard DataFrame (models as rows, ``(Metric, Domain)`` columns).
        normalization : str or None, default None
            Strategy used to map scores to [0, 1].  ``None`` uses plain
            min-max.  Alternatives: ``"rank"``, ``"percentile"``,
            ``"robust_zscore"`` (see :meth:`export_scorecard_pdf`).

        Returns
        -------
        pd.DataFrame
            Normalized scores in [0, 1], same shape as *df*.
        """
        _VALID_METHODS = {None, "rank", "percentile", "robust_zscore"}
        if normalization not in _VALID_METHODS:
            raise ValueError(
                f"Unknown normalization: {normalization!r}. "
                f"Choose from {_VALID_METHODS}."
            )

        norm = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
        metric_specs = df.attrs.get("metric_specs_by_name", {})
        metrics_registry = (self.table_generator.evaluator
                            .metrics_registry.get("metrics", {}))

        for col in df.columns:
            metric_name = col[0] if isinstance(col, tuple) else col
            spec = metric_specs.get(str(metric_name), {})
            reduction_fn = spec.get("reduction", "mean")
            metric_key = spec.get("key")
            metric_type = metrics_registry.get(metric_key, {}).get("type")

            values = pd.to_numeric(df[col], errors="coerce")
            scores = self.table_generator._score_row_for_best(
                row_values=values,
                metric_name=str(metric_name),
                metric_type=metric_type,
                reduction_fn=reduction_fn,
            )

            valid = scores[np.isfinite(scores)]
            if valid.empty or valid.max() == valid.min():
                norm[col] = 0.5
                norm.loc[~np.isfinite(scores), col] = np.nan
                continue

            norm[col] = self._normalize_column(scores, valid, normalization)
            norm.loc[~np.isfinite(scores), col] = np.nan

        return norm

    @staticmethod
    def _normalize_column(scores: pd.Series, valid: pd.Series,
                          method: Optional[str]) -> pd.Series:
        """Normalize a single column of scores to [0, 1]."""
        if method is None:
            return (scores - valid.min()) / (valid.max() - valid.min())

        if method == "rank":
            ranked = scores.rank(method="average", na_option="keep")
            n_valid = valid.size
            if n_valid <= 1:
                return pd.Series(0.5, index=scores.index)
            return (ranked - ranked.min()) / (ranked.max() - ranked.min())

        if method == "percentile":
            lo = float(np.percentile(valid, 5))
            hi = float(np.percentile(valid, 95))
            if hi == lo:
                return pd.Series(0.5, index=scores.index)
            clipped = scores.clip(lower=lo, upper=hi)
            return (clipped - lo) / (hi - lo)

        if method == "robust_zscore":
            median = float(valid.median())
            q75 = float(np.percentile(valid, 75))
            q25 = float(np.percentile(valid, 25))
            iqr = q75 - q25
            if iqr == 0:
                return pd.Series(0.5, index=scores.index)
            z = (scores - median) / iqr
            clipped = z.clip(lower=-2, upper=2)
            return (clipped + 2) / 4.0

        raise ValueError(f"Unknown normalization method: {method!r}")
