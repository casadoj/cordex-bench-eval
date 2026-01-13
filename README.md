# CORDEX Benchmark Evaluation

A toolkit for evaluating deep learning model predictions against ground truth data in the CORDEX-ML benchmark.

## Overview

This package provides a complete workflow for:

1. **Ingesting** submission archives (`.zip` files) from participants
2. **Loading** prediction and ground truth datasets
3. **Evaluating** predictions using configurable metrics (RMSE, bias indices, etc.)
4. **Generating tables and figures** for model comparison and analysis

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Incoming   │ ───► │   Staging    │ ───► │  Evaluation  │
│   (.zip)     │      │   (NetCDF)   │      │  (metrics)   │
└──────────────┘      └──────────────┘      └──────────────┘
                                                   │
                                                   │
                           ┌───────────────────────┴───────────────────────┐
                           │                                               │
                    ┌──────────────┐                              ┌──────────────┐
                    │    Tables    │                              │    Figures   │
                    │  (CSV/HTML)  │                              │  (PNG/PDF)   │
                    └──────────────┘                              └──────────────┘
                          │                                               │
                          ▼                                               ▼
                  Comparison Tables                                 Various Figures
```

## Installation

```bash
pip install -e .
```

## Submission Structure

Submissions should be organized following this structure:

```
<submission_name>/
├── ALPS_Domain/
│   ├── Emulator_hist_future/
│   │   ├── historical/
│   │   │   ├── perfect/
│   │   │   │   ├── Predictions_pr_tasmax_CNRM-CM5_1981-2000.nc
│   │   │   │   └── Predictions_pr_tasmax_MPI-ESM-LR_1981-2000.nc
│   │   │   └── imperfect/
│   │   │       └── ...
│   │   ├── mid_century/
│   │   │   ├── perfect/
│   │   │   └── imperfect/
│   │   └── end_century/
│   │       ├── perfect/
│   │       └── imperfect/
│   ├── ESD_pseudo_reality/
│   │   └── ...
├── NZ_Domain/
│   └── ...
└── SA_Domain/
    └── ...

For more details check the [Benchmark documentation](https://docs.google.com/document/d/1zM6Aza1glzBJ8Ewvhu0_0vcm5dvctJ65Bypt1gnN2ck/edit?usp=sharing).
```

Key points:
- **Domains**: `ALPS_Domain`, `NZ_Domain`, `SA_Domain`
- **Training Experiments**: `ESD_pseudo_reality`, `Emulator_hist_future`
- **Periods**: `historical`, `mid_century`, `end_century`
- **Conditions**: `perfect` (within the same RCM as training), `imperfect` (directly on the GCM)
- **Files**: Named as `Predictions_pr_tasmax_<GCM>_<date_range>.nc` (contains both `pr` and `tasmax` variables)
- **Orography**: Whether a submission uses orography as a predictor is indicated by the submission archive filename. For versions using orography, the archive should have a `-orog` suffix (e.g., `DeepESD-asym1-orog.zip`). This information is automatically extracted and stored in the catalog during ingestion.

## Evaluation Experiments

Based on the training experiment chosen, the following evaluation experiments are available:

### ESD_pseudo_reality

| Evaluation Experiment | Period | Condition | GCM |
|----------------------|--------|-----------|-----|
| `Perfect_cross_validation` | historical | perfect | Same as train |
| `Imperfect_cross_validation` | historical | imperfect | Same as train |
| `Perfect_extrapolation` | mid_end_century | perfect | Same as train |
| `Imperfect_extrapolation` | mid_end_century | imperfect | Same as train |
| `Perfect_extrapolation_HT` | mid_end_century | perfect | Different GCM |

### Emulator_hist_future

| Evaluation Experiment | Period | Condition | GCM |
|----------------------|--------|-----------|-----|
| `Perfect_cross_validation` | historical | perfect | Same as train |
| `Imperfect_cross_validation` | historical | imperfect | Same as train |
| `Perfect_cross_validation_HT` | historical | perfect | Different GCM |
| `Imperfect_cross_validation_HT` | historical | imperfect | Different GCM |
| `Perfect_interpolation` | mid_end_century | perfect | Same as train |
| `Imperfect_interpolation` | mid_end_century | imperfect | Same as train |
| `Perfect_interpolation_HT` | mid_end_century | perfect | Different GCM |
| `Imperfect_interpolation_HT` | mid_end_century | imperfect | Different GCM |

## Quick Start

```python
from cordex_bench_eval import (
    SubmissionCatalog,
    SubmissionDataLoader,
    GroundTruthDataLoader,
    Evaluator,
    TableGenerator,
    FigureGenerator,
    DiagnosticFigureGenerator,
)

from cordex_bench_eval.config import load_config_file

# Load configuration
config = load_config_file("configs/config.json")

# Set the submission loader
catalog = SubmissionCatalog(config.catalog_path)
loader = SubmissionDataLoader(catalog)

# Load a specific submission
ds = loader.load_dataset(submission_name="DeepESD-v1_IFCA_m1",
                         domain="ALPS_Domain",
                         training_experiment="Emulator_hist_future",
                         evaluation_experiment="Perfect_cross_validation",
                         variable="pr")

# Set the evaluator
evaluator = Evaluator(config=config,
                      catalog=catalog,
                      metrics_registry_path="configs/metrics.json",
                      ml_benchmark_path=config.mlbenchmark_path)

# Run the evaluation
# (results are saved as NetCDF files and returned as a dict of xr.Datasets)
results = evaluator.evaluate(submission_name="DeepESD-v1_IFCA_m1",
                             domain="ALPS_Domain",
                             training_experiment="Emulator_hist_future",
                             evaluation_experiment="Perfect_cross_validation",
                             variable="pr")

# Generate comparison tables
table_gen = TableGenerator(config=config,
                           catalog=catalog,
                           table_config_path="configs/tables.json")

df = table_gen.generate_table(table_name="tasmax_perfect_cross_validation",
                              submission_names=["DeepESD-v1_IFCA_m1"],
                              include_average=True)

# Generate figures
figure_gen = FigureGenerator(config=config,
                             catalog=catalog,
                             figure_config_path="configs/figures.json")

figure_path = figure_gen.generate_figure(figure_name="alps_pr_daily_snapshot",
                                         submission_names=["DeepESD-v1_IFCA_m1"])
```

## Workflow Steps

### 1. Ingest Submissions

Place `.zip` archives in the `incoming` directory, then run:

```python
from cordex_bench_eval import SubmissionHandler
from cordex_bench_eval.config import load_config_file

config = load_config_file("configs/config.json")
handler = SubmissionHandler(config)
ingested = handler.ingest()

print(f"Ingested {len(ingested)} submissions")
```

This extracts archives to `staging/` and registers them in `catalog.json`.

### 2. Run Complete Workflow (Evaluation + Table/Figures Generation)

The simplest way to use the package is to run `main.py`, which demonstrates the complete workflow:

```bash
python main.py
```

This script:
- Loads all available submissions from the catalog
- Demonstrates data loading and exploration
- Runs evaluation with metrics computation
- Generates comparison tables and export them in multiple formats (CSV, Excel, HTML)
- Generates some figures

## Configuration

### `configs/config.json`

Set standard paths to ensure the workflow functions correctly.

```json
{
  "groundtruth_path": "data/target",
  "incoming_path": "data/incoming",
  "staging_path": "data/staging",
  "catalog_path": "catalog/catalog.json",
  "diagnostic_path": "data/diagnostics",
  "mlbenchmark_path": "/path/to/ml-benchmark",
  "archive_extensions": [".zip"],
  "overwrite_existing": false
}
```

### `configs/metrics.json`

Defines available metrics and defaults per variable:

```json
{
  "metrics": {
    "rmse": { "type": "rmse" },
    "cdd_bias": { "type": "bias_index", "index_fn": "evaluation.indices.cdd" },
    ...
  },
  "defaults": {
    "pr": ["rmse", "cdd_bias", "sdii_ratio", "rx1day_bias"],
    "tasmax": ["rmse", "mean_bias", "txx_bias", "su_bias"]
  }
}
```

### `configs/tables.json`

Defines table specifications for model comparison:

```json
{
  "metrics_registry_path": "configs/metrics.json",
  "tables": {
    "esd_perfect_extrapolation": {
      "description": "ESD Experiment - Perfect Extrapolation",
      "training_experiment": "ESD_pseudo_reality",
      "evaluation_experiment": "Perfect_extrapolation",
      "variable": "tasmax",
      "metrics": [
        {
          "name": "RMSE",
          "key": "rmse",
          "reduction": "mean"
        },
        {
          "name": "TXX_bias",
          "key": "txx_bias",
          "reduction": "mean"
        }
      ]
    }
  }
}
```

### `configs/figures.json`

Defines figure specifications for plotting daily fields:

```json
{
  "defaults": {
    "figsize": [12, 10],
    "dpi": 150,
    "predictions_per_row": 3,
    "format": "pdf",
    "output_dir": "figures_output",
    "ground_truth_label": "Ground Truth",
    "colorbar_label": "Value",
    "colormap": "turbo",
    "colorbar": {
      "levels": 11,
      "vmin": null,
      "vmax": null
    }
  },
  "figures": {
    "sa_pr_daily_snapshot": {
      "description": "Daily precipitation snapshot for the South America domain.",
      "domain": "SA_Domain",
      "training_experiment": "ESD_pseudo_reality",
      "evaluation_experiment": "Perfect_cross_validation",
      "variable": "pr",
      "target_date": "1985-01-15",
      "colorbar_label": "Daily precipitation (mm/day)"
    }
  }
}
```

Each figure specification defines the scenario (domain, experiments, variable),
the target date, and any overrides for the plotting defaults (e.g., number of
columns, figure size, or colorbar label).

### `configs/diagnostic_figures.json`

Defines diagnostic figure specifications for spatial maps of computed metrics:

```json
{
  "defaults": {
    "figsize": [12, 8],
    "dpi": 150,
    "plots_per_row": 3,
    "format": "pdf",
    "output_dir": "diagnostic_figures_output",
    "colorbar_orientation": "horizontal",
    "colorbar_label": "Value",
    "colormap": "viridis",
    "colorbar": {
      "levels": 11,
      "vmin": null,
      "vmax": null
    }
  },
  "figures": {
    "sa_pr_rmse_map": {
      "description": "RMSE spatial map for precipitation in South America domain.",
      "domain": "SA_Domain",
      "training_experiment": "ESD_pseudo_reality",
      "evaluation_experiment": "Perfect_cross_validation",
      "variable": "pr",
      "metric": "rmse",
      "colorbar_label": "RMSE (mm/day)",
      "colormap": "Reds"
    }
  }
}
```

Each diagnostic figure specification defines the scenario (domain, experiments, variable),
the metric to visualize, and any overrides for the plotting defaults.

## Examples of Figure and Table generations

An example of figure and tables generation is included in `./figs` and `./tables`

## Directory Structure

```
cordex-bench-eval/
├── configs/
│   ├── config.json             # Paths configuration
│   ├── metrics.json            # Metrics definitions
│   ├── tables.json             # Table specifications
│   ├── figures.json            # Daily figure specifications
│   └── diagnostic_figures.json # Diagnostic figure specifications
├── catalog/
│   └── catalog.json            # Registry of ingested submissions
├── data/
│   ├── incoming/               # Drop .zip files here
│   ├── staging/                # Extracted submissions
│   ├── target/                 # Ground truth data
│   └── diagnostics/            # Computed metrics output
├── src/cordex_bench_eval/
│   ├── config.py                     # Configuration loading
│   ├── catalog.py                    # Submission catalog
│   ├── submission_handler.py         # Archive ingestion
│   ├── data_loader.py                # Data loading utilities
│   ├── evaluator.py                  # Metrics computation
│   ├── table_generator.py            # Table generation
│   ├── figure_generator.py           # Daily figure generation
│   └── diagnostic_figure_generator.py # Diagnostic figure generation
└── main.py                       # Complete workflow
```