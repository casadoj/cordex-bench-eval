"""
Daily figure generation utilities for CORDEX benchmark evaluation.

This module produces comparison plots that place the ground-truth snapshot
alongside predictions from multiple submissions for a single day. The layout,
colorbar settings, and other plot preferences are driven by a JSON
configuration file.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import xarray as xr

from .catalog import SubmissionCatalog
from .config import SubmissionWorkflowConfig
from .data_loader import GroundTruthDataLoader, SubmissionDataLoader

# Keys that describe the scientific scenario (non-plot settings) in the figure spec
_SCENARIO_KEYS = {"description", "domain", "training_experiment",
                  "evaluation_experiment", "variable", "use_orography", "target_date"}

class FigureGenerator:
    """
    Generate daily comparison figures for multiple submissions.

    Parameters
    ----------
    config : SubmissionWorkflowConfig
        Workflow configuration that defines data locations.
    catalog : SubmissionCatalog
        Submission catalog used to resolve staged prediction paths.
    figure_config_path : str | Path
        Path to the JSON file describing figure specifications.
    """

    def __init__(self, config: SubmissionWorkflowConfig, catalog: SubmissionCatalog,
                 figure_config_path: str | Path):
        self.config = config
        self.catalog = catalog
        self.figure_config = self._load_figure_config(figure_config_path)
        self.submission_loader = SubmissionDataLoader(catalog)
        self.gt_loader = GroundTruthDataLoader(config.groundtruth_path)

    def generate_figure(self, figure_name: str, submission_names: Optional[Sequence[str]] = None,
                        target_date: str | datetime | np.datetime64 | None = None,
                        output_path: str | Path | None = None,
                        output_dir: str | Path | None = None,
                        file_format: Optional[str] = None) -> Path:
        """
        Generate a single comparison figure.

        Parameters
        ----------
        figure_name : str
            Name of the figure specification defined in the JSON config.
        submission_names : Sequence[str], optional
            Subset of submissions to include. Defaults to all catalog entries.
        target_date : str | datetime | np.datetime64, optional
            Date to visualize. Overrides the config value if provided.
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

        date_value = target_date or spec.get("target_date") or plot_settings.get("target_date")
        if date_value is None:
            raise ValueError("target_date must be provided either via parameter or figure config.")
        target_timestamp = self._parse_target_date(date_value)

        domain = spec["domain"]
        training_experiment = spec["training_experiment"]
        evaluation_experiment = spec["evaluation_experiment"]
        variable = spec["variable"]
        use_orography = spec.get("use_orography")

        ground_truth = self._load_ground_truth(domain=domain,
                                               training_experiment=training_experiment,
                                               evaluation_experiment=evaluation_experiment,
                                               variable=variable,
                                               target_date=target_timestamp)

        predictions = self._load_predictions(submissions=submissions,
                                             domain=domain,
                                             training_experiment=training_experiment,
                                             evaluation_experiment=evaluation_experiment,
                                             variable=variable,
                                             target_date=target_timestamp,
                                             use_orography=use_orography)

        destination = self._resolve_output_path(figure_name=figure_name,
                                                output_path=output_path,
                                                output_dir=output_dir or plot_settings.get("output_dir"),
                                                file_format=file_format or plot_settings.get("format", "png"))

        self._render_figure(ground_truth=ground_truth,
                            predictions=predictions,
                            target_date=target_timestamp,
                            variable=variable,
                            domain=domain,
                            plot_settings=plot_settings,
                            output_path=destination)

        return destination

    def generate_all_figures(self, submission_names: Optional[Sequence[str]] = None,
                             output_dir: str | Path | None = None) -> Dict[str, Path]:
        """
        Generate every figure defined in the configuration.

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

    def _load_ground_truth(self, domain: str, training_experiment: str,
                           evaluation_experiment: str, variable: str,
                           target_date: np.datetime64) -> xr.DataArray:
        """Load the ground truth data array for a given domain, training experiment,
           evaluation experiment, variable, and target date."""
        ds = self.gt_loader.load_test_target(domain=domain,
                                             training_experiment=training_experiment,
                                             evaluation_experiment=evaluation_experiment,
                                             variable=variable)
        try:
            data_array = ds[variable]
        except KeyError as exc:
            raise KeyError(f"Variable '{variable}' not found in ground truth dataset.") from exc
        slice_da = self._slice_day(data_array, target_date).load()
        ds.close()
        return slice_da

    def _load_predictions(self, submissions: Sequence[str], domain: str,
                          training_experiment: str, evaluation_experiment: str,
                          variable: str, target_date: np.datetime64,
                          use_orography: bool | None) -> List[Tuple[str, xr.DataArray]]:
        """Load the predictions data array"""
        loaded: List[Tuple[str, xr.DataArray]] = []
        for submission in submissions:
            ds: Optional[xr.Dataset] = None
            try:
                ds = self.submission_loader.load_dataset(submission_name=submission,
                                                         domain=domain,
                                                         training_experiment=training_experiment,
                                                         evaluation_experiment=evaluation_experiment,
                                                         variable=variable,
                                                         use_orography=use_orography)
                data_array = ds[variable]
                slice_da = self._slice_day(data_array, target_date).load()
                loaded.append((submission, slice_da))

            except FileNotFoundError as exc:
                print(f"File not found: {exc}")
            except KeyError as exc:
                print(f"Key error: {exc}")

            finally:
                try:
                    if ds is not None:
                        ds.close()
                except Exception:
                    pass
        return loaded

    def _render_figure(self, ground_truth: xr.DataArray,
                       predictions: Sequence[Tuple[str, xr.DataArray]],
                       target_date: np.datetime64,
                       variable: str,
                       domain: str,
                       plot_settings: Dict[str, Any],
                       output_path: Path) -> None:
        """Render the figure"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        figsize = tuple(plot_settings.get("figsize", (12, 10)))
        dpi = plot_settings.get("dpi", 150)
        preds_per_row = max(1, int(plot_settings.get("predictions_per_row", 3)))

        # Define the projection (NZ needs central_longitude=180)
        domain_key = domain.replace("_Domain", "").upper()
        central_lon = 180 if domain_key == "NZ" else 0
        projection = ccrs.PlateCarree(central_longitude=central_lon)

        fig = plt.figure(figsize=figsize, dpi=dpi)
        n_cols = preds_per_row
        n_rows = math.ceil(len(predictions) / n_cols)
        gs = fig.add_gridspec(n_rows + 1, n_cols, hspace=0.15, wspace=0.05)

        # Prepare shared colorbar
        cmap, norm, levels = self._build_colormap(ground_truth, predictions, plot_settings)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        # Ground truth axis positioned at top row center
        gt_col = n_cols // 2
        gt_ax = fig.add_subplot(gs[0, gt_col], projection=projection)
        self._draw_dataarray(ax=gt_ax,
                             data_array=ground_truth,
                             cmap=cmap,
                             norm=norm)
        gt_label = plot_settings.get("ground_truth_label", "Ground Truth")
        gt_ax.set_title(gt_label)

        # Hide unused top-row axes
        for col in range(n_cols):
            if col == gt_col:
                continue
            ax = fig.add_subplot(gs[0, col])
            ax.axis("off")

        prediction_axes = []
        for idx, (submission, data_array) in enumerate(predictions):
            row = 1 + idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col], projection=projection)
            self._draw_dataarray(ax=ax,
                                 data_array=data_array,
                                 cmap=cmap,
                                 norm=norm)
            ax.set_title(submission)
            prediction_axes.append(ax)

        axes_for_colorbar = [gt_ax] + prediction_axes
        cbar_orientation = plot_settings.get("colorbar_orientation", "horizontal")
        colorbar_label = plot_settings.get("colorbar_label", variable)
        cbar = fig.colorbar(sm,
                            ax=axes_for_colorbar,
                            orientation=cbar_orientation,
                            fraction=0.05 if cbar_orientation == "horizontal" else 0.035,
                            pad=0.08)
        if levels is not None:
            cbar.set_ticks(levels)
        cbar.set_label(colorbar_label)

        title = plot_settings.get("title")
        if title:
            fig.suptitle(title, y=0.98)
        else:
            formatted_date = np.datetime_as_string(target_date, unit="D")
            fig.suptitle(f"{variable} snapshot - {formatted_date}", y=0.98)

        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    def _draw_dataarray(self, ax: plt.Axes, data_array: xr.DataArray,
                        cmap: mcolors.Colormap, norm: mcolors.Normalize) -> None:
        """Draw the data array on the axes"""
        ax.pcolormesh(data_array['lon'].values, data_array['lat'].values,
                      data_array, cmap=cmap, norm=norm, 
                      transform=ccrs.PlateCarree())
        
        ax.coastlines(resolution='50m', color='black', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray')
        
        ax.set_xticks([])
        ax.set_yticks([])

    def _build_colormap(self, ground_truth: xr.DataArray,
                        predictions: Sequence[Tuple[str, xr.DataArray]],
                        plot_settings: Dict[str, Any]) -> Tuple[mcolors.Colormap, mcolors.Normalize, Optional[List[float]]]:
        """Build the colormap for the figure"""
        colorbar_settings = plot_settings.get("colorbar", {})
        levels = max(2, int(colorbar_settings.get("levels", 11)))
        colormap_name = plot_settings.get("colormap", "turbo")

        data_values = [ground_truth.values] + [da.values for _, da in predictions]
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

    def _resolve_submission_names(self, submission_names: Optional[Sequence[str]]) -> List[str]:
        """Resolve submission names from the catalog"""
        if submission_names is not None:
            return list(submission_names)
        return [entry.name for entry in self.catalog.all_entries()]

    def _resolve_output_path(self, figure_name: str, output_path: str | Path | None,
                             output_dir: str | Path | None, file_format: str) -> Path:
        """Resolve the output path for the figure"""
        if output_path:
            return Path(output_path).expanduser().resolve()

        target_dir = Path(output_dir or "figures_output").expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        suffix = file_format.lstrip(".")
        return target_dir / f"{figure_name}.{suffix}"

    def _slice_day(self, data_array: xr.DataArray, target_date: np.datetime64) -> xr.DataArray:
        """Slice the data array to the target date"""
        if "time" not in data_array.dims:
            raise ValueError("Dataset does not contain a 'time' dimension required for daily plots.")
        try:
            # Use string-based selection to match the day regardless of the hour (e.g., 00:00 vs 12:00)
            day_str = np.datetime_as_string(target_date, unit='D')
            sliced = data_array.sel(time=day_str)
            # If the selection returns multiple hours for the same day, take the first one
            if sliced.time.size > 1:
                sliced = sliced.isel(time=0)
        except KeyError as exc:
            times = data_array["time"].values
            if times.size == 0:
                raise KeyError("Dataset does not contain any timesteps.") from exc
            start = np.datetime_as_string(times[0], unit="D")
            end = np.datetime_as_string(times[-1], unit="D")
            raise KeyError(f"Date {target_date} not present in dataset (available range: {start} - {end}).") from exc
        return sliced.squeeze()

    def _infer_dim(self, data_array: xr.DataArray, axis: str) -> str:
        """Infer the dimension name for the given axis"""
        candidates = ("lat", "latitude", "y") if axis == "y" else ("lon", "longitude", "x")
        for dim in data_array.dims:
            lower = dim.lower()
            if any(key in lower for key in candidates):
                return dim
        raise ValueError(f"Unable to infer {'latitude' if axis == 'y' else 'longitude'} dimension names.")

    def _build_plot_settings(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Build plot settings from the figure specification"""
        defaults = self.figure_config.get("defaults", {}).copy()
        plot_overrides = {k: v for k, v in spec.items() if k not in _SCENARIO_KEYS}
        defaults.update(plot_overrides)
        return defaults

    def _get_figure_spec(self, figure_name: str) -> Dict[str, Any]:
        """Get the figure specification from the configuration"""
        try:
            return self.figure_config["figures"][figure_name]
        except KeyError as exc:
            available = ", ".join(sorted(self.figure_config.get("figures", {}).keys()))
            raise KeyError(f"Unknown figure '{figure_name}'. Available: {available}") from exc

    def _load_figure_config(self, path: str | Path) -> Dict[str, Any]:
        """Load the figure configuration from a JSON file"""
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Figure config not found: {path}")
        return json.loads(path.read_text())

    def _parse_target_date(self, value: str | datetime | np.datetime64) -> np.datetime64:
        """Parse the target date from a string, datetime, or numpy datetime64.
           This is required for the Alps domain."""
        if isinstance(value, np.datetime64):
            return value
        if isinstance(value, datetime):
            return np.datetime64(value.date())
        return np.datetime64(str(value))