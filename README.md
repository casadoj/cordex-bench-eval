# CORDEX Bench Evaluation Utilities

This repository hosts modular pieces of the CORDEX-ML benchmark evaluation workflow. The initial milestone focuses on tracking and staging model submissions that arrive as `.zip` archives.

## Submission staging workflow

1. Monitor an **incoming** directory for submission archives.
2. Extract every new archive into a configurable **staging** directory using a folder that matches the archive name.
3. Register the staged submission in a lightweight **catalog** (including submission metadata read from the NetCDF global attributes `esd_method_id`, `esd_method_version`, and `training_methodology_id`) so later workflow steps can discover it.

The logic lives inside `cordex_bench_eval.submission_handler`. Compose it inside your own orchestration code (see `main.py`) to stage submissions however you like—cron, notebooks, or ad-hoc scripts.

### Example configuration

```
{
  "incoming_dir": "/data/incoming",
  "staging_dir": "/data/staging",
  "catalog_path": "/data/catalog/submissions.json",
  "archive_extensions": [".zip"],
  "overwrite_existing": false
}
```

### Usage

```
python -m pip install -e .
python main.py
```

`main.py` is a minimal example that:

1. Loads the JSON config via `load_config_file`.
2. Instantiates `SubmissionHandler`.
3. Calls `ingest()`, which stages submissions and captures the NetCDF attributes (`esd_method_id`, `esd_method_version`, `training_methodology_id`) from the first file it finds.

Feel free to copy that snippet into your own runner so you can plug additional validation or scoring steps right after ingestion. Downstream tools can always read `catalog_path` to know where each submission lives.

## Loading staged data

Once a submission is registered you can reach any NetCDF file by domain / experiment / variable
through `SubmissionDataLoader`:

```
from cordex_bench_eval import SubmissionCatalog, SubmissionDataLoader

catalog = SubmissionCatalog("/data/catalog/submissions.json")
loader = SubmissionDataLoader(catalog)

print("Domains:", loader.available_domains("DeepESD-v1_IFCA_m1"))
print("Experiments:", loader.available_experiments("DeepESD-v1_IFCA_m1", "ALPS_Domain"))
print("Variables:", loader.available_variables("DeepESD-v1_IFCA_m1", "ALPS_Domain", "Emulator_Hist_Future_NO_OROG"))
print("Predictions:", loader.available_predictions(
    "DeepESD-v1_IFCA_m1",
    "ALPS_Domain",
    "Emulator_Hist_Future_NO_OROG",
    "pr",
))

ds = loader.load_dataset(
    submission_name="DeepESD-v1_IFCA_m1",
    domain="ALPS_Domain",
    experiment="Emulator_Hist_Future_NO_OROG",
    variable="pr",
    prediction="Extrapolation_perfect",  # omit if there is only one .nc file
)
print(ds)
```

`load_dataset()` wraps `xarray.open_dataset()` so you can pass any keyword argument it understands
(e.g. `decode_cf=False`). When a variable folder contains multiple NetCDF files, provide the
``prediction`` name without the `predictions_` prefix or `.nc` suffix. Use the helper methods above
to discover the available domains, experiments, variables, and predictions exposed by a submission.

