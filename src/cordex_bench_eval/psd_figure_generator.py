"""
PSD figure generation utilities for CORDEX benchmark evaluation.

This module produces log-log power spectral density comparison plots that
overlay the ground-truth PSD with the PSD of each submission. Each figure
can contain one subplot per domain to facilitate cross-domain comparison.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .catalog import SubmissionCatalog
from .config import SubmissionWorkflowConfig
from .evaluator import Evaluator

_SCENARIO_KEYS = {"description", "training_experiment",
                  "evaluation_experiment", "variable", "domains"}


class PsdFigureGenerator:
    """
    Generate PSD comparison figures for multiple submissions across domains.

    Parameters
    ----------
    config : SubmissionWorkflowConfig
        Workflow configuration that defines data locations.
    catalog : SubmissionCatalog
        Submission catalog used to resolve submission names.
    evaluator : Evaluator
        Evaluator instance for loading computed PSD metrics.
    figure_config_path : str | Path
        Path to the JSON file describing PSD figure specifications.
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
        Generate a single PSD comparison figure.

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

        training_experiment = spec["training_experiment"]
        evaluation_experiment = spec["evaluation_experiment"]
        variable = spec["variable"]
        domains = spec["domains"]

        psd_data = self._load_psd_data(submissions=submissions,
                                       domains=domains,
                                       training_experiment=training_experiment,
                                       evaluation_experiment=evaluation_experiment,
                                       variable=variable)

        destination = self._resolve_output_path(figure_name=figure_name,
                                                output_path=output_path,
                                                output_dir=output_dir or plot_settings.get("output_dir"),
                                                file_format=file_format or plot_settings.get("format", "png"))

        self._render_figure(psd_data=psd_data,
                            domains=domains,
                            variable=variable,
                            plot_settings=plot_settings,
                            output_path=destination)

        return destination

    def generate_all_figures(self, submission_names: Optional[Sequence[str]] = None,
                             output_dir: str | Path | None = None) -> Dict[str, Path]:
        """
        Generate every PSD figure defined in the configuration.

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

    def _load_psd_data(self, submissions: Sequence[str], domains: Sequence[str],
                       training_experiment: str, evaluation_experiment: str,
                       variable: str) -> Dict[str, List[Tuple[str, xr.DataArray, xr.DataArray]]]:
        """Load PSD DataArrays for every submission in every domain.

        Returns
        -------
        dict[str, list[tuple[str, xr.DataArray, xr.DataArray]]]
            Mapping from domain name to a list of
            ``(submission_name, psd_x0, psd_x1)`` tuples.
        """
        result: Dict[str, List[Tuple[str, xr.DataArray, xr.DataArray]]] = {}
        for domain in domains:
            loaded: List[Tuple[str, xr.DataArray, xr.DataArray]] = []
            for submission in submissions:
                ds: Optional[xr.Dataset] = None
                try:
                    ds = self.evaluator.load_metric(submission_name=submission,
                                                    domain=domain,
                                                    training_experiment=training_experiment,
                                                    evaluation_experiment=evaluation_experiment,
                                                    variable=variable,
                                                    metric_name="psd")
                    psd_x0 = ds["PSD_x0"]
                    psd_x1 = ds["PSD_x1"]
                    loaded.append((submission, psd_x0, psd_x1))

                except FileNotFoundError as exc:
                    print(f"PSD metric file not found for {submission} ({domain}): {exc}")
                except KeyError as exc:
                    print(f"Key error loading PSD for {submission} ({domain}): {exc}")

                finally:
                    try:
                        if ds is not None:
                            ds.close()
                    except Exception:
                        pass

            result[domain] = loaded
        return result

    def _render_figure(self, psd_data: Dict[str, List[Tuple[str, xr.DataArray, xr.DataArray]]],
                       domains: Sequence[str],
                       variable: str,
                       plot_settings: Dict[str, Any],
                       output_path: Path) -> None:
        """Render the PSD comparison figure."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        figsize = tuple(plot_settings.get("figsize", (18, 5)))
        dpi = plot_settings.get("dpi", 150)
        subplot_title_fontsize = plot_settings.get("subplot_title_fontsize", 12)
        figure_title_fontsize = plot_settings.get("figure_title_fontsize", 14)
        axis_label_fontsize = plot_settings.get("axis_label_fontsize", 11)
        legend_fontsize = plot_settings.get("legend_fontsize", 8)
        gt_label = plot_settings.get("ground_truth_label", "Ground Truth")
        gt_linewidth = float(plot_settings.get("ground_truth_linewidth", 2.5))
        sub_linewidth = float(plot_settings.get("submission_linewidth", 1.2))
        xlabel = plot_settings.get("xlabel", "Wavenumber")
        ylabel = plot_settings.get("ylabel", "Power")

        n_domains = len(domains)
        fig, axes = plt.subplots(1, n_domains, figsize=figsize, dpi=dpi, squeeze=False)
        axes = axes.flat

        for idx, domain in enumerate(domains):
            ax = axes[idx]
            entries = psd_data.get(domain, [])

            gt_plotted = False
            for submission, psd_x0, psd_x1 in entries:
                wavenumber = np.arange(len(psd_x0))

                if not gt_plotted:
                    ax.plot(wavenumber, psd_x0.values, color="black",
                            linewidth=gt_linewidth, label=gt_label, zorder=10)
                    gt_plotted = True

                ax.plot(wavenumber, psd_x1.values, linewidth=sub_linewidth,
                        label=submission, alpha=0.85)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
            if idx == 0:
                ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)

            domain_label = domain.replace("_Domain", "").replace("_", " ")
            ax.set_title(domain_label, fontsize=subplot_title_fontsize)

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center",
                       ncol=min(len(handles), 6),
                       fontsize=legend_fontsize,
                       bbox_to_anchor=(0.5, -0.02),
                       frameon=True)

        title = plot_settings.get("title")
        if title:
            fig.suptitle(title, fontsize=figure_title_fontsize)
        else:
            fig.suptitle(f"Power Spectral Density - {variable}",
                         fontsize=figure_title_fontsize)

        fig.tight_layout(rect=[0, 0.06, 1, 0.95])
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

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

        target_dir = Path(output_dir or "psd_figures_output").expanduser().resolve()
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
