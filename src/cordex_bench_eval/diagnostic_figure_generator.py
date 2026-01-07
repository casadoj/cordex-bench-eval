"""
Diagnostic figure generation utilities for CORDEX benchmark evaluation.

This module produces spatial maps of computed diagnostics (e.g., RMSE, bias)
for multiple submissions. All subplots share the same colormap and colorbar
configuration, similar to the daily figure generation workflow.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import xarray as xr

from .catalog import SubmissionCatalog
from .config import SubmissionWorkflowConfig
from .evaluator import Evaluator

# Keys that describe the scientific scenario (non-plot settings) in the figure spec
_SCENARIO_KEYS = {"description", "domain", "training_experiment",
                  "evaluation_experiment", "variable", "use_orography", "metric"}

class DiagnosticFigureGenerator:
    """
    Generate spatial maps of computed diagnostics for multiple submissions.

    Parameters
    ----------
    config : SubmissionWorkflowConfig
        Workflow configuration that defines data locations.
    catalog : SubmissionCatalog
        Submission catalog used to resolve submission names.
    evaluator : Evaluator
        Evaluator instance for loading computed metrics.
    figure_config_path : str | Path
        Path to the JSON file describing diagnostic figure specifications.
    """

    def __init__(self, config: SubmissionWorkflowConfig, catalog: SubmissionCatalog,
                 evaluator: Evaluator, figure_config_path: str | Path):
        self.config = config
        self.catalog = catalog
        self.evaluator = evaluator
        self.figure_config = self._load_figure_config(figure_config_path)

    def generate_figure(self, figure_name: str, submission_names: Optional[Sequence[str]] = None,
                        output_path: str | Path | None = None,
                        output_dir: str | Path | None = None,
                        file_format: Optional[str] = None) -> Path:
        """
        Generate a single diagnostic comparison figure.

        Parameters
        ----------
        figure_name : str
            Name of the figure specification defined in the JSON config.
        submission_names : Sequence[str], optional
            Subset of submissions to include. Defaults to all catalog entries.
        output_path : str | Path, optional
            Explicit output path. If omitted, one is built from the config/output_dir.
        output_dir : str | Path, optional
            Directory where the figure will be saved when output_path is not provided.
        file_format : str, optional
            Output format (e.g., "png", "pdf"). Defaults to config value.

        Returns
        -------
        Path
            Location of the saved figure file.
        """
        spec = self._get_figure_spec(figure_name)
        plot_settings = self._build_plot_settings(spec)
        submissions = self._resolve_submission_names(submission_names)

        domain = spec["domain"]
        training_experiment = spec["training_experiment"]
        evaluation_experiment = spec["evaluation_experiment"]
        variable = spec["variable"]
        metric = spec["metric"]
        use_orography = spec.get("use_orography")

        diagnostics = self._load_diagnostics(submissions=submissions,
                                             domain=domain,
                                             training_experiment=training_experiment,
                                             evaluation_experiment=evaluation_experiment,
                                             variable=variable,
                                             metric=metric,
                                             use_orography=use_orography)

        destination = self._resolve_output_path(figure_name=figure_name,
                                                output_path=output_path,
                                                output_dir=output_dir or plot_settings.get("output_dir"),
                                                file_format=file_format or plot_settings.get("format", "png"))

        self._render_figure(diagnostics=diagnostics,
                            metric=metric,
                            variable=variable,
                            domain=domain,
                            plot_settings=plot_settings,
                            output_path=destination)

        return destination

    def generate_all_figures(self, submission_names: Optional[Sequence[str]] = None,
                             output_dir: str | Path | None = None) -> Dict[str, Path]:
        """
        Generate every diagnostic figure defined in the configuration.

        Returns
        -------
        dict[str, Path]
            Mapping of figure names to generated file paths.
        """
        outputs: Dict[str, Path] = {}
        for name in self.figure_config.get("figures", {}):
            outputs[name] = self.generate_figure(figure_name=name,
                                                 submission_names=submission_names,
                                                 output_dir=output_dir)
        return outputs

    def _load_diagnostics(self, submissions: Sequence[str], domain: str,
                          training_experiment: str, evaluation_experiment: str,
                          variable: str, metric: str,
                          use_orography: bool | None) -> List[Tuple[str, xr.DataArray]]:
        """Load the diagnostic data arrays for all submissions."""
        loaded: List[Tuple[str, xr.DataArray]] = []
        for submission in submissions:
            ds: Optional[xr.Dataset] = None
            try:
                ds = self.evaluator.load_metric(submission_name=submission,
                                               domain=domain,
                                               training_experiment=training_experiment,
                                               evaluation_experiment=evaluation_experiment,
                                               variable=variable,
                                               metric_name=metric,
                                               use_orography=use_orography)
                
                # Extract the diagnostic variable from the dataset
                # Metrics may have different variable names, so we take the first data variable
                data_vars = list(ds.data_vars)
                if not data_vars:
                    print(f"Warning: No data variables found in metric dataset for {submission}")
                    continue
                
                data_array = ds[data_vars[0]]
                loaded.append((submission, data_array))

            except FileNotFoundError as exc:
                print(f"Metric file not found for {submission}: {exc}")
            except KeyError as exc:
                print(f"Key error loading metric for {submission}: {exc}")

            finally:
                try:
                    if ds is not None:
                        ds.close()
                except Exception:
                    pass
        
        return loaded

    def _render_figure(self, diagnostics: Sequence[Tuple[str, xr.DataArray]],
                       metric: str,
                       variable: str,
                       domain: str,
                       plot_settings: Dict[str, Any],
                       output_path: Path) -> None:
        """Render the diagnostic figure."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        figsize = tuple(plot_settings.get("figsize", (12, 8)))
        dpi = plot_settings.get("dpi", 150)
        plots_per_row = max(1, int(plot_settings.get("plots_per_row", 3)))

        # Define the projection (NZ needs central_longitude=180)
        domain_key = domain.replace("_Domain", "").upper()
        central_lon = 180 if domain_key == "NZ" else 0
        projection = ccrs.PlateCarree(central_longitude=central_lon)

        fig = plt.figure(figsize=figsize, dpi=dpi)
        n_cols = plots_per_row
        n_rows = math.ceil(len(diagnostics) / n_cols)
        
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.15, wspace=0.05)

        # Prepare shared colorbar
        cmap, norm, levels = self._build_colormap(diagnostics, plot_settings)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        # Plot each diagnostic
        diagnostic_axes = []
        for idx, (submission, data_array) in enumerate(diagnostics):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col], projection=projection)
            self._draw_dataarray(ax=ax,
                                data_array=data_array,
                                cmap=cmap,
                                norm=norm)
            spatial_mean = self._compute_spatial_mean(data_array)
            title = f"{submission}\n(mean: {spatial_mean:.3f})"
            ax.set_title(title)
            diagnostic_axes.append(ax)

        # Add colorbar
        cbar_orientation = plot_settings.get("colorbar_orientation", "horizontal")
        colorbar_label = plot_settings.get("colorbar_label", metric)
        cbar = fig.colorbar(sm,
                            ax=diagnostic_axes,
                            orientation=cbar_orientation,
                            fraction=0.05 if cbar_orientation == "horizontal" else 0.035,
                            pad=0.08)
        if levels is not None:
            cbar.set_ticks(levels)
        cbar.set_label(colorbar_label)
        cbar.ax.tick_params(labelsize=6) # TODO: Remove this hard-coded parameter

        # Add title
        title = plot_settings.get("title")
        if title:
            fig.suptitle(title, y=0.98)
        else:
            fig.suptitle(f"{metric} - {variable}", y=0.98)

        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    def _draw_dataarray(self, ax: plt.Axes, data_array: xr.DataArray,
                        cmap: mcolors.Colormap, norm: mcolors.Normalize) -> None:
        """Draw the data array on the axes."""
        ax.pcolormesh(data_array['lon'].values, data_array['lat'].values,
                      data_array, cmap=cmap, norm=norm, 
                      transform=ccrs.PlateCarree())
        
        ax.coastlines(resolution='50m', color='black', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray')
        
        ax.set_xticks([])
        ax.set_yticks([])

    def _build_colormap(self, diagnostics: Sequence[Tuple[str, xr.DataArray]],
                        plot_settings: Dict[str, Any]) -> Tuple[mcolors.Colormap, mcolors.Normalize, Optional[List[float]]]:
        """Build the colormap for the figure."""
        colorbar_settings = plot_settings.get("colorbar", {})
        levels = max(2, int(colorbar_settings.get("levels", 11)))
        colormap_name = plot_settings.get("colormap", "turbo")

        data_values = [da.values for _, da in diagnostics]
        min_candidates = [np.nanmin(arr) for arr in data_values if not np.all(np.isnan(arr))]
        max_candidates = [np.nanmax(arr) for arr in data_values if not np.all(np.isnan(arr))]
        if not min_candidates or not max_candidates:
            raise ValueError("Unable to determine color range because all arrays contain NaNs.")
        data_min = float(np.nanmin(min_candidates))
        data_max = float(np.nanmax(max_candidates))
        vmin = colorbar_settings.get("vmin")
        vmax = colorbar_settings.get("vmax")
        if vmin is None:
            vmin = data_min
        if vmax is None:
            vmax = data_max
        if vmin == vmax:
            vmax = vmin + 1.0

        boundaries = np.linspace(vmin, vmax, levels + 1)
        cmap = plt.get_cmap(colormap_name, levels)
        norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap.N)
        tick_positions = list(boundaries)
        return cmap, norm, tick_positions

    def _infer_dim(self, data_array: xr.DataArray, axis: str) -> str:
        """Infer the dimension name for the given axis."""
        candidates = ("lat", "latitude", "y") if axis == "y" else ("lon", "longitude", "x")
        for dim in data_array.dims:
            lower = dim.lower()
            if any(key in lower for key in candidates):
                return dim
        raise ValueError(f"Unable to infer {'latitude' if axis == 'y' else 'longitude'} dimension names.")

    def _compute_spatial_mean(self, data_array: xr.DataArray) -> float:
        """Compute the spatial mean of a diagnostic data array.
        
        Parameters
        ----------
        data_array : xr.DataArray
            Diagnostic data array (typically 2D spatial after metric computation).
        
        Returns
        -------
        float
            Spatial mean value, computed over all array values (handles NaNs).
        """
        arr = data_array.squeeze()
        return float(np.nanmean(arr.values))

    def _resolve_submission_names(self, submission_names: Optional[Sequence[str]]) -> List[str]:
        """Resolve submission names from the catalog."""
        if submission_names is not None:
            return list(submission_names)
        return [entry.name for entry in self.catalog.all_entries()]

    def _resolve_output_path(self, figure_name: str, output_path: str | Path | None,
                             output_dir: str | Path | None, file_format: str) -> Path:
        """Resolve the output path for the figure."""
        if output_path:
            return Path(output_path).expanduser().resolve()

        target_dir = Path(output_dir or "diagnostic_figures_output").expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        suffix = file_format.lstrip(".")
        return target_dir / f"{figure_name}.{suffix}"

    def _build_plot_settings(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Build plot settings from the figure specification."""
        defaults = self.figure_config.get("defaults", {}).copy()
        plot_overrides = {k: v for k, v in spec.items() if k not in _SCENARIO_KEYS}
        defaults.update(plot_overrides)
        return defaults

    def _get_figure_spec(self, figure_name: str) -> Dict[str, Any]:
        """Get the figure specification from the configuration."""
        try:
            return self.figure_config["figures"][figure_name]
        except KeyError as exc:
            available = ", ".join(sorted(self.figure_config.get("figures", {}).keys()))
            raise KeyError(f"Unknown figure '{figure_name}'. Available: {available}") from exc

    def _load_figure_config(self, path: str | Path) -> Dict[str, Any]:
        """Load the figure configuration from a JSON file."""
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Figure config not found: {path}")
        return json.loads(path.read_text())