# CORDEX Benchmark Evaluation

A toolkit for evaluating deep learning model predictions against ground truth data in the CORDEX-ML benchmark.

## Overview

This package provides a complete workflow for:

1. **Ingesting** submission archives (`.zip` files) from participants
2. **Loading** prediction and ground truth datasets
3. **Evaluating** predictions using configurable metrics (RMSE, bias indices, etc.)
4. **Generating tables** for model comparison and analysis
5. **Creating daily figures** that compare predictions against ground truth

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Incoming   │ ───► │   Staging    │ ───► │  Evaluation  │ ───► │    Tables    │ ───► │    Figures   │
│   (.zip)     │      │   (NetCDF)   │      │  (metrics)   │      │  (CSV/HTML) │      │  (PNG/PDF)  │
└──────────────┘      └──────────────┘      └──────────────┘      └──────────────┘      └──────────────┘
        │                    │                     │                      │                     │
        ▼                    ▼                     ▼                      ▼                     ▼
   catalog.json      SubmissionDataLoader    diagnostics/*.nc    Comparison Tables     Daily Snapshots
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
│   ├── Emulator_hist_future_OROG/
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
│   ├── Emulator_hist_future_NO_OROG/
│   │   └── ...
│   ├── ESD_pseudo_reality_OROG/
│   │   └── ...
│   └── ESD_pseudo_reality_NO_OROG/
│       └── ...
├── NZ_Domain/
│   └── ...
└── SA_Domain/
    └── ...
```

Key points:
- **Domains**: `ALPS_Domain`, `NZ_Domain`, `SA_Domain`
- **Training Experiments**: `Emulator_hist_future`, `ESD_pseudo_reality` (with `_OROG` or `_NO_OROG` suffix)
- **Periods**: `historical`, `mid_century`, `end_century`
- **Conditions**: `perfect` (same GCM as training), `imperfect` (different GCM)
- **Files**: Named as `Predictions_pr_tasmax_<GCM>_<date_range>.nc` (contains both `pr` and `tasmax` variables)

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
)
from cordex_bench_eval.config import load_config_file

# Load configuration
config = load_config_file("configs/config.json")

# Load a prediction dataset
catalog = SubmissionCatalog(config.catalog_path)
loader = SubmissionDataLoader(catalog)

ds = loader.load_dataset(submission_name="DeepESD-v1_IFCA_m1",
                         domain="ALPS_Domain",
                         training_experiment="Emulator_hist_future",
                         evaluation_experiment="Perfect_cross_validation",
                         use_orography=True,
                         variable="pr")

# Run evaluation
evaluator = Evaluator(config=config,
                      catalog=catalog,
                      metrics_registry_path="configs/metrics.json",
                      ml_benchmark_path=config.mlbenchmark_path)

results = evaluator.evaluate(submission_name="DeepESD-v1_IFCA_m1",
                             domain="ALPS_Domain",
                             training_experiment="Emulator_hist_future",
                             evaluation_experiment="Perfect_cross_validation",
                             variable="pr",
                             use_orography=True,)
# Results are saved as NetCDF files and returned as a dict of xr.Dataset

# Generate comparison tables
table_gen = TableGenerator(config=config,
                           catalog=catalog,
                           table_config_path="configs/tables.json")

df = table_gen.generate_table(table_name="esd_perfect_extrapolation",
                              submission_names=["DeepESD-v1_IFCA_m1"],
                              include_average=True)

# Generate a daily figure
figure_gen = FigureGenerator(config=config,
                             catalog=catalog,
                             figure_config_path="configs/figures.json")
figure_path = figure_gen.generate_figure(
    figure_name="sa_pr_daily_snapshot",
    submission_names=["DeepESD-v1_IFCA_m1"],
    target_date="1985-01-15"
)
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

This extracts archives to `staging/` and registers them in `catalog.json` with metadata from NetCDF attributes.

### 2. Run Complete Workflow (Evaluation + Table Generation)

The simplest way to use the package is to run `main.py`, which demonstrates the complete workflow:

```bash
python main.py
```

This script:
1. Loads all submissions from the catalog
2. Demonstrates data loading and exploration
3. Runs evaluation with metrics computation
4. Generates all available comparison tables
5. Exports tables in multiple formats (CSV, Excel, HTML)

For more fine-grained control, see the next sections.

### 3. Load Prediction Data

```python
from cordex_bench_eval import SubmissionCatalog, SubmissionDataLoader

catalog = SubmissionCatalog("catalog/catalog.json")
loader = SubmissionDataLoader(catalog)

# Explore what's available
print(loader.available_domains("DeepESD-v1_IFCA_m1"))
print(loader.available_experiments("DeepESD-v1_IFCA_m1", "ALPS_Domain"))
print(loader.available_periods("DeepESD-v1_IFCA_m1", "ALPS_Domain", "Emulator_hist_future_OROG"))
print(loader.available_conditions("DeepESD-v1_IFCA_m1", "ALPS_Domain", 
                                   "Emulator_hist_future_OROG", "historical"))

# Load a dataset using training + evaluation experiment
ds = loader.load_dataset(submission_name="DeepESD-v1_IFCA_m1",
                         domain="ALPS_Domain",
                         training_experiment="Emulator_hist_future",
                         evaluation_experiment="Perfect_cross_validation",
                         use_orography=True, # Appends _OROG to experiment name
                         variable="pr")
```

The loader automatically:
- Maps the evaluation experiment to the correct period/condition/GCM combination
- Handles `mid_end_century` periods by merging mid and end century files
- Supports case-insensitive experiment name matching

### 4. Load Ground Truth Data

```python
from cordex_bench_eval import GroundTruthDataLoader

gt_loader = GroundTruthDataLoader("data/target")

# Training data
ds_train = gt_loader.load_train_target(domain="ALPS",
                                       training_experiment="Emulator_hist_future",
                                       variable="pr")

# Test data (for evaluation)
ds_test = gt_loader.load_test_target(domain="ALPS",
                                     training_experiment="Emulator_hist_future",
                                     evaluation_experiment="Perfect_cross_validation",
                                     variable="pr")
```

### 5. Run Evaluation

```python
from cordex_bench_eval import Evaluator

evaluator = Evaluator(config=config,
                      catalog=catalog,
                      metrics_registry_path="configs/metrics.json",
                      ml_benchmark_path="/path/to/ml-benchmark")

# Compute metrics (uses defaults from metrics.json)
results = evaluator.evaluate(submission_name="DeepESD-v1_IFCA_m1",
                             domain="ALPS_Domain",
                             training_experiment="Emulator_hist_future",
                             evaluation_experiment="Perfect_cross_validation",
                             variable="pr",
                             use_orography=True)

# Or specify which metrics to compute
results = evaluator.evaluate(..., metrics=["rmse", "cdd_bias", "rx1day_bias"])

# Load a previously computed metric
rmse = evaluator.load_metric(submission_name="DeepESD-v1_IFCA_m1",
                             domain="ALPS_Domain",
                             training_experiment="Emulator_hist_future",
                             evaluation_experiment="Perfect_cross_validation",
                             variable="pr",
                             metric_name="rmse",
                             use_orography=True)
```

### 6. Generate Comparison Tables

After computing metrics for multiple models, generate comparison tables:

```python
from cordex_bench_eval import TableGenerator

table_gen = TableGenerator(config=config,
                           catalog=catalog,
                           table_config_path="configs/tables.json")

# Generate a single table
df = table_gen.generate_table(table_name="esd_perfect_extrapolation",
                              submission_names=["DeepESD-v1_IFCA_m1", "XGBoost-v2", "ML-v1"],
                              domains=["NZ_Domain", "SA_Domain", "ALPS_Domain"],
                              include_average=True)

# Export to various formats
table_gen.export_table(df, "output/table.csv", format="csv")
table_gen.export_table(df, "output/table.html", format="html")

# Generate all tables at once
all_tables = table_gen.generate_all_tables(submission_names=["DeepESD-v1_IFCA_m1", "XGBoost-v2"])

# Export all tables
table_gen.export_all_tables(output_dir="output/tables",
                            format="csv",
                            submission_names=["DeepESD-v1_IFCA_m1", "XGBoost-v2"])
```

**Key Features:**
- Tables aggregate pre-computed metrics across models and domains
- Multi-index structure: (Metric, Domain) x Models
- Automatically computes average (AV) across domains
- Easy to update with new models (just re-run with updated submission list)
- Supports multiple export formats (CSV, LaTeX, HTML)

### 7. Generate Daily Figures

After selecting a set of submissions, create daily comparison plots that place
the ground truth snapshot at the top and predictions below:

```python
from cordex_bench_eval import FigureGenerator

figure_gen = FigureGenerator(config=config,
                             catalog=catalog,
                             figure_config_path="configs/figures.json")

figure_path = figure_gen.generate_figure(
    figure_name="sa_pr_daily_snapshot",
    submission_names=["DeepESD-v1_IFCA_m1", "ML-v1"],
    target_date="1985-01-15",
    output_dir="figures_output",
    file_format="png",
)

print(f"Figure saved to {figure_path}")
```

**Figure Highlights:**
- Ground-truth panel is centered on the top row for quick reference
- All prediction panels share the same discrete turbo colormap and colorbar
- Layout, colorbar bounds, and output format are controlled via `configs/figures.json`
- Supports PNG or PDF outputs for publication-ready graphics

## Configuration

### `configs/config.json`

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
    "tasmax": ["rmse", "txx_bias", "su_bias"]
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
      "description": "ESD Experiment (with orography) - Perfect Extrapolation",
      "training_experiment": "ESD_pseudo_reality",
      "evaluation_experiment": "Perfect_extrapolation",
      "variable": "tasmax",
      "use_orography": true,
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

Each table specification defines:
- **training_experiment**: Which training experiment was used
- **evaluation_experiment**: Which evaluation scenario (e.g., Perfect_cross_validation)
- **variable**: Which variable (pr, tasmax, etc.)
- **metrics**: List of metrics to include with their display names and reduction methods

### `configs/figures.json`

Defines figure specifications for the daily snapshot plots:

```json
{
  "defaults": {
    "figsize": [12, 10],
    "dpi": 150,
    "predictions_per_row": 3,
    "format": "png",
    "output_dir": "figures_output",
    "colormap": "turbo",
    "colorbar": { "levels": 11, "vmin": null, "vmax": null }
  },
  "figures": {
    "sa_pr_daily_snapshot": {
      "description": "Daily precipitation snapshot for the South America domain.",
      "domain": "SA_Domain",
      "training_experiment": "ESD_pseudo_reality",
      "evaluation_experiment": "Perfect_cross_validation",
      "variable": "pr",
      "target_date": "1985-01-15",
      "use_orography": true,
      "colorbar_label": "Daily precipitation (mm/day)"
    }
  }
}
```

Each figure specification defines the scenario (domain, experiments, variable),
the target date, and any overrides for the plotting defaults (e.g., number of
columns, figure size, or colorbar label).

## Directory Structure

```
cordex-bench-eval/
├── configs/
│   ├── config.json          # Paths configuration
│   ├── metrics.json         # Metrics definitions
│   ├── tables.json          # Table specifications
│   └── figures.json         # Daily figure specifications
├── catalog/
│   └── catalog.json         # Registry of ingested submissions
├── data/
│   ├── incoming/            # Drop .zip files here
│   ├── staging/             # Extracted submissions
│   ├── target/              # Ground truth data
│   └── diagnostics/         # Computed metrics output
├── src/cordex_bench_eval/
│   ├── config.py            # Configuration loading
│   ├── catalog.py           # Submission catalog
│   ├── submission_handler.py # Archive ingestion
│   ├── data_loader.py       # Data loading utilities
│   ├── evaluator.py         # Metrics computation
│   ├── table_generator.py   # Table generation
│   └── figure_generator.py  # Daily figure generation
├── tables_output/           # Exported tables
├── figures_output/          # Exported figures
└── main.py                  # Complete workflow (evaluation + tables + figures)
```