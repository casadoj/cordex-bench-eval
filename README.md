# CORDEX Benchmark Evaluation

A toolkit for evaluating deep learning model predictions against ground truth data in the CORDEX-ML benchmark.

## Overview

This package provides a complete workflow for:

1. **Ingesting** submission archives (`.zip` files) from participants
2. **Loading** prediction and ground truth datasets
3. **Evaluating** predictions using configurable metrics (RMSE, bias indices, etc.)
4. **Generating tables** for model comparison and analysis

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Incoming   │ ───► │   Staging    │ ───► │  Evaluation  │ ───► │    Tables    │
│   (.zip)     │      │   (NetCDF)   │      │  (metrics)   │      │  (CSV/Excel) │
└──────────────┘      └──────────────┘      └──────────────┘      └──────────────┘
        │                    │                     │                      │
        ▼                    ▼                     ▼                      ▼
   catalog.json      SubmissionDataLoader    diagnostics/*.nc    Comparison Tables
```

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from cordex_bench_eval import (
    SubmissionCatalog,
    SubmissionDataLoader,
    GroundTruthDataLoader,
    Evaluator,
    TableGenerator,
)
from cordex_bench_eval.config import load_config_file

# Load configuration
config = load_config_file("configs/config.json")

# Load a prediction dataset
catalog = SubmissionCatalog(config.catalog_path)
loader = SubmissionDataLoader(catalog)

ds = loader.load_dataset(
    submission_name="DeepESD-v1_IFCA_m1",
    domain="ALPS_Domain",
    experiment="Emulator_Hist_Future",
    use_orography=True,
    variable="pr",
    prediction="Extrapolation_perfect",
)

# Run evaluation
evaluator = Evaluator(
    config=config,
    catalog=catalog,
    metrics_registry_path="configs/metrics.json",
    ml_benchmark_path=config.mlbenchmark_path,
)

results = evaluator.evaluate(
    submission_name="DeepESD-v1_IFCA_m1",
    domain="ALPS_Domain",
    experiment="Emulator_Hist_Future",
    variable="pr",
    prediction="Extrapolation_perfect",
    use_orography=True,
)
# Results are saved as NetCDF files and returned as a dict of xr.Dataset

# Generate comparison tables
table_gen = TableGenerator(
    config=config,
    catalog=catalog,
    table_config_path="configs/tables.json",
)

df = table_gen.generate_table(
    table_name="emul_perfect_diff_gcm",
    submission_names=["DeepESD-v1_IFCA_m1"],
    include_average=True,
)
print(df)
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
print(loader.available_variables("DeepESD-v1_IFCA_m1", "ALPS_Domain", "Emulator_Hist_Future_OROG"))
print(loader.available_predictions("DeepESD-v1_IFCA_m1", "ALPS_Domain", "Emulator_Hist_Future_OROG", "pr"))

# Load a dataset
ds = loader.load_dataset(
    submission_name="DeepESD-v1_IFCA_m1",
    domain="ALPS_Domain",
    experiment="Emulator_Hist_Future",
    use_orography=True,      # Appends _OROG to experiment name
    variable="pr",
    prediction="Extrapolation_perfect",
)
```

### 4. Load Ground Truth Data

```python
from cordex_bench_eval import GroundTruthDataLoader

gt_loader = GroundTruthDataLoader("data/target")

# Training data
ds_train = gt_loader.load_train_target(
    domain="ALPS",
    training_experiment="Emulator_hist_future",
    variable="pr",
)

# Test data (for evaluation)
ds_test = gt_loader.load_test_target(
    domain="ALPS",
    training_experiment="Emulator_hist_future",
    evaluation_experiment="Extrapolation_perfect",
    variable="pr",
)
```

### 5. Run Evaluation

```python
from cordex_bench_eval import Evaluator

evaluator = Evaluator(
    config=config,
    catalog=catalog,
    metrics_registry_path="configs/metrics.json",
    ml_benchmark_path="/path/to/ml-benchmark",
)

# Compute metrics (uses defaults from metrics.json)
results = evaluator.evaluate(
    submission_name="DeepESD-v1_IFCA_m1",
    domain="ALPS_Domain",
    experiment="Emulator_Hist_Future",
    variable="pr",
    prediction="Extrapolation_perfect",
    use_orography=True,
)

# Or specify which metrics to compute
results = evaluator.evaluate(
    ...,
    metrics=["rmse", "cdd_bias", "rx1day_bias"],
)

# Load a previously computed metric
rmse = evaluator.load_metric(
    submission_name="DeepESD-v1_IFCA_m1",
    domain="ALPS_Domain",
    experiment="Emulator_Hist_Future",
    variable="pr",
    prediction="Extrapolation_perfect",
    metric_name="rmse",
    use_orography=True,
)
```

### 6. Generate Comparison Tables

After computing metrics for multiple models, generate comparison tables:

```python
from cordex_bench_eval import TableGenerator

table_gen = TableGenerator(
    config=config,
    catalog=catalog,
    table_config_path="configs/tables.json",
)

# Generate a single table
df = table_gen.generate_table(
    table_name="emul_perfect_diff_gcm",
    submission_names=["DeepESD-v1_IFCA_m1", "XGBoost-v2", "ML-v1"],
    domains=["NZ_Domain", "SA_Domain", "ALPS_Domain"],
    include_average=True,  # Include AV (average across domains)
)

print(df)

# Export to various formats
table_gen.export_table(df, "output/table.csv", format="csv")
table_gen.export_table(df, "output/table.xlsx", format="excel")
table_gen.export_table(df, "output/table.html", format="html")

# Generate all tables at once
all_tables = table_gen.generate_all_tables(
    submission_names=["DeepESD-v1_IFCA_m1", "XGBoost-v2"],
)

# Export all tables
table_gen.export_all_tables(
    output_dir="output/tables",
    format="csv",
    submission_names=["DeepESD-v1_IFCA_m1", "XGBoost-v2"],
)
```

**Key Features:**
- Tables aggregate pre-computed metrics across models and domains
- Multi-index structure: (Metric, Domain) x Models
- Automatically computes average (AV) across domains
- Easy to update with new models (just re-run with updated submission list)
- Supports multiple export formats (CSV, Excel, LaTeX, HTML)

For more details, see [TABLE_GENERATION.md](TABLE_GENERATION.md).

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
    "esd_perfect_2041_2060": {
      "description": "ESD Experiment (with orography) - Perfect Setting (2041-2060)",
      "experiment": "ESD_Pseudo-Reality",
      "variable": "tasmax",
      "prediction": "Extrapolation_perfect",
      "use_orography": true,
      "metrics": [
        {
          "name": "TMean_historical (1981-2000)",
          "key": "rmse",
          "reduction": "mean"
        },
        {
          "name": "TXX_historical (1981-2000)",
          "key": "txx_bias",
          "reduction": "mean"
        }
      ]
    },
    "emul_perfect_diff_gcm": {
      "description": "Emul_hist_future - Perfect Evaluation, Different GCM (2041-2060)",
      "experiment": "Emulator_Hist_Future",
      "variable": "pr",
      "prediction": "Extrapolation_perfect",
      "use_orography": true,
      "metrics": [
        {
          "name": "Pr_Mean_historical (1981-2000)",
          "key": "rmse",
          "reduction": "mean"
        }
      ]
    }
  }
}
```

Each table specification defines:
- **experiment**: Which experiment to evaluate
- **variable**: Which variable (pr, tasmax, etc.)
- **prediction**: Which prediction scenario (e.g., Extrapolation_perfect)
- **metrics**: List of metrics to include with their display names and reduction methods

## Directory Structure

```
cordex-bench-eval/
├── configs/
│   ├── config.json          # Paths configuration
│   ├── metrics.json         # Metrics definitions
│   └── tables.json          # Table specifications
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
│   └── table_generator.py   # Table generation
└── main.py                  # Complete workflow (evaluation + table generation)
```
