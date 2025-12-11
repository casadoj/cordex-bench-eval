# CORDEX Benchmark Evaluation

A toolkit for evaluating deep learning model predictions against ground truth data in the CORDEX-ML benchmark.

## Overview

This package provides a complete workflow for:

1. **Ingesting** submission archives (`.zip` files) from participants
2. **Loading** prediction and ground truth datasets
3. **Evaluating** predictions using configurable metrics (RMSE, bias indices, etc.)

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Incoming   │ ───► │   Staging    │ ───► │  Evaluation  │
│   (.zip)     │      │   (NetCDF)   │      │  (metrics)   │
└──────────────┘      └──────────────┘      └──────────────┘
        │                    │                     │
        ▼                    ▼                     ▼
   catalog.json      SubmissionDataLoader    diagnostics/*.nc
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

### 2. Load Prediction Data

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

### 3. Load Ground Truth Data

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

### 4. Run Evaluation

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

## Directory Structure

```
cordex-bench-eval/
├── configs/
│   ├── config.json          # Paths configuration
│   └── metrics.json         # Metrics definitions
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
│   └── evaluator.py         # Metrics computation
└── main.py                  # Example usage
```
