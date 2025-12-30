"""
Table Generator for CORDEX Benchmark Evaluation.

This module provides functionality to generate comparison tables across models,
domains, and experiments. Tables aggregate pre-computed metrics from the
diagnostic files into a structured format suitable for analysis and publication.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

from .catalog import SubmissionCatalog
from .config import SubmissionWorkflowConfig
from .evaluator import Evaluator

class TableGenerator:
    """
    Generate comparison tables from computed metrics.
    
    Loads pre-computed metrics for multiple models and organizes them into
    tables suitable for model comparison and analysis. Tables can be exported
    to various formats (CSV, Excel, LaTeX, HTML).
    
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
        if config.diagnostic_path is None:
            raise ValueError("diagnostic_path must be set in config.")

        self.config = config
        self.catalog = catalog
        self.table_config = self._load_table_config(table_config_path)
        self.evaluator = Evaluator(config, catalog, self.table_config.get("metrics_registry_path", "configs/metrics.json"))

    def generate_table(self, table_name: str, submission_names: Optional[List[str]] = None,
                       domains: Optional[List[str]] = None, include_average: bool = True) -> pd.DataFrame:
        """
        Generate a comparison table for specified models.
        
        Parameters
        ----------
        table_name : str
            Name of table specification from config file.
        submission_names : list of str, optional
            Model names to include. If None, uses all in catalog.
        domains : list of str, optional
            Domains to include. Default: ["NZ_Domain", "SA_Domain", "ALPS_Domain"].
        include_average : bool, default True
            Include AV (average) domain row for each metric.
        
        Returns
        -------
        pd.DataFrame
            Multi-index DataFrame with structure:
            - Index: (Metric, Domain)
            - Columns: Model names
        """
        if table_name not in self.table_config.get("tables", {}):
            raise ValueError(f"Unknown table: {table_name}")

        table_spec = self.table_config["tables"][table_name]
        
        # Default submission names: all in catalog
        if submission_names is None:
            submission_names = [entry.name for entry in self.catalog.all_entries()]
        
        # Default domains
        if domains is None:
            domains = ["NZ_Domain", "SA_Domain", "ALPS_Domain"]
        
        # Extract table parameters
        training_experiment = table_spec["training_experiment"]
        evaluation_experiment = table_spec["evaluation_experiment"]
        variable = table_spec["variable"]
        use_orography = table_spec.get("use_orography")
        metrics_list = table_spec["metrics"]
        
        # Build table data
        table_data = []
        index_tuples = []
        
        for metric_spec in metrics_list:
            metric_name = metric_spec["name"]
            metric_key = metric_spec["key"]
            reduction_fn = metric_spec.get("reduction", "mean")
            
            for domain in domains:
                domain_short = self._extract_domain_short(domain)
                row_data = {}
                
                for submission_name in submission_names:
                    try:
                        value = self._load_metric_value(submission_name=submission_name,
                                                        domain=domain,
                                                        training_experiment=training_experiment,
                                                        evaluation_experiment=evaluation_experiment,
                                                        variable=variable,
                                                        metric_key=metric_key,
                                                        reduction_fn=reduction_fn,
                                                        use_orography=use_orography)
                        row_data[submission_name] = value
                    except FileNotFoundError:
                        row_data[submission_name] = np.nan
                
                table_data.append(row_data)
                index_tuples.append((metric_name, domain_short))
            
            # Add average row if requested
            if include_average:
                avg_row = {}
                for submission_name in submission_names:
                    domain_values = []
                    for domain in domains:
                        try:
                            value = self._load_metric_value(submission_name=submission_name,
                                                            domain=domain,
                                                            training_experiment=training_experiment,
                                                            evaluation_experiment=evaluation_experiment,
                                                            variable=variable,
                                                            metric_key=metric_key,
                                                            reduction_fn=reduction_fn,
                                                            use_orography=use_orography)

                            if not np.isnan(value):
                                domain_values.append(value)
                        except FileNotFoundError:
                            pass
                    
                    avg_row[submission_name] = np.mean(domain_values) if domain_values else np.nan
                
                table_data.append(avg_row)
                index_tuples.append((metric_name, "AV"))
        
        # Create DataFrame
        df = pd.DataFrame(table_data, columns=submission_names)
        df.index = pd.MultiIndex.from_tuples(index_tuples, names=["Metric", "Domain"])
        
        return df

    def generate_all_tables(self, submission_names: Optional[List[str]] = None,
                            domains: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate all tables defined in config.
        
        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary mapping table names to DataFrames.
        """
        tables = {}
        for table_name in self.table_config.get("tables", {}).keys():
            tables[table_name] = self.generate_table(table_name=table_name,
                                                     submission_names=submission_names,
                                                     domains=domains)
        return tables

    def export_table(self, df: pd.DataFrame, output_path: str | Path, format: str = "csv",
                     **kwargs: Any) -> None:
        """
        Export table to file.
        
        Parameters
        ----------
        df : pd.DataFrame
            Table to export.
        output_path : str or Path
            Output file path.
        format : str, default "csv"
            Export format: "csv", "latex", "html".
        **kwargs
            Additional arguments passed to pandas export function.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "csv":
            df.to_csv(output_path, **kwargs)
        elif format == "latex":
            latex_str = df.to_latex(**kwargs)
            output_path.write_text(latex_str)
        elif format == "html":
            html_str = df.to_html(**kwargs)
            output_path.write_text(html_str)
        else:
            raise ValueError(f"Unknown format: {format}")

    def export_all_tables(self, output_dir: str | Path, format: str = "csv",
                          submission_names: Optional[List[str]] = None, **kwargs: Any) -> None:
        """
        Generate and export all tables.
        
        Parameters
        ----------
        output_dir : str or Path
            Directory to save tables.
        format : str, default "csv"
            Export format.
        submission_names : list of str, optional
            Models to include.
        **kwargs
            Additional arguments for export.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tables = self.generate_all_tables(submission_names=submission_names)
        
        for table_name, df in tables.items():
            extension = {"csv": ".csv", "latex": ".tex", "html": ".html"}[format]
            output_path = output_dir / f"{table_name}{extension}"
            self.export_table(df, output_path, format=format, **kwargs)

    def _load_table_config(self, config_path: str | Path) -> dict:
        """Load table configuration from JSON file."""
        path = Path(config_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Table config not found: {path}")
        return json.loads(path.read_text())

    def _load_metric_value(self, submission_name: str, domain: str,
                           training_experiment: str, evaluation_experiment: str,
                           variable: str, metric_key: str,
                           reduction_fn: str = "mean", use_orography: bool | None = None) -> float:
        """
        Load a metric value from diagnostic file and apply reduction.
        
        Parameters
        ----------
        metric_key : str
            Metric identifier (e.g., "rmse", "cdd_bias").
        reduction_fn : str
            How to reduce the metric to scalar: "mean", "median", "max", "min", "abs_mean", "std".
        
        Returns
        -------
        float
            Scalar metric value.
        """
        # Load the metric dataset
        ds = self.evaluator.load_metric(submission_name=submission_name,
                                        domain=domain,
                                        training_experiment=training_experiment,
                                        evaluation_experiment=evaluation_experiment,
                                        variable=variable,
                                        metric_name=metric_key,
                                        use_orography=use_orography)
        
        # Extract the primary variable
        # Most metrics have a single data variable
        data_vars = list(ds.data_vars)
        if len(data_vars) == 0:
            raise ValueError(f"No data variables in metric: {metric_key}")
        
        # Use first data variable (or look for specific names)
        var_name = data_vars[0]
        data_array = ds[var_name]
        
        # Apply reduction function
        if reduction_fn == "mean":
            return float(data_array.mean().values)
        elif reduction_fn == "median":
            return float(data_array.median().values)
        elif reduction_fn == "max":
            return float(data_array.max().values)
        elif reduction_fn == "min":
            return float(data_array.min().values)
        elif reduction_fn == "std":
            return float(data_array.std().values)
        elif reduction_fn == "abs_mean":
            return float(np.abs(data_array).mean().values)
        else:
            raise ValueError(f"Unknown reduction function: {reduction_fn}")

    def _extract_domain_short(self, domain: str) -> str:
        """
        Convert domain name to short form.
        
        Examples:
            "ALPS_Domain" -> "ALPS"
            "NZ_Domain" -> "NZ"
            "SA_Domain" -> "SA"
        """
        if "_Domain" in domain:
            return domain.replace("_Domain", "")
        return domain
