"""
CORDEX Benchmark Evaluation - Example Usage

This script demonstrates the complete workflow for evaluating deep learning
model predictions against ground truth data.

Workflow steps:
1. Load configuration
2. (Optional) Ingest new submissions from incoming directory
3. Load a submission from the catalog
4. Load ground truth data
5. Run evaluation metrics
"""

from cordex_bench_eval import (
    SubmissionHandler,
    SubmissionCatalog,
    SubmissionDataLoader,
    GroundTruthDataLoader,
    Evaluator,
)
from cordex_bench_eval.config import load_config_file


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

CONFIG_PATH = "/gpfs/projects/meteo/WORK/gonzabad/cordex-bench-eval/configs/config.json"
METRICS_PATH = "/gpfs/projects/meteo/WORK/gonzabad/cordex-bench-eval/configs/metrics.json"

config = load_config_file(CONFIG_PATH)


# =============================================================================
# 2. INGEST NEW SUBMISSIONS (run once when new .zip files arrive)
# =============================================================================

# handler = SubmissionHandler(config)
# ingested = handler.ingest()
# print(f"Ingested {len(ingested)} new submissions")


# =============================================================================
# 3. BROWSE THE SUBMISSION CATALOG
# =============================================================================

catalog = SubmissionCatalog(config.catalog_path)
print("=== Available Submissions ===")
for entry in catalog.all_entries():
    print(f"  - {entry.name} ({entry.status})")


# =============================================================================
# 4. LOAD A SPECIFIC SUBMISSION
# =============================================================================

# Select which submission/experiment to evaluate
SUBMISSION_NAME = "DeepESD-v1_IFCA_m1"
EXPERIMENT = "Emulator_Hist_Future"
USE_OROGRAPHY = True
DOMAIN = "ALPS_Domain"
VARIABLE = "pr"
PREDICTION = "Extrapolation_perfect"

# Create data loader
loader = SubmissionDataLoader(catalog)

# Explore available data
print(f"\n=== Available Data for {SUBMISSION_NAME} ===")
print(f"Domains: {loader.available_domains(SUBMISSION_NAME)}")
print(f"Experiments in {DOMAIN}: {loader.available_experiments(SUBMISSION_NAME, DOMAIN)}")

experiment_name = f"{EXPERIMENT}_{'OROG' if USE_OROGRAPHY else 'NO_OROG'}"
print(f"Variables in {experiment_name}: {loader.available_variables(SUBMISSION_NAME, DOMAIN, experiment_name)}")
print(f"Predictions for {VARIABLE}: {loader.available_predictions(SUBMISSION_NAME, DOMAIN, experiment_name, VARIABLE)}")

# Load the prediction dataset
prediction_ds = loader.load_dataset(
    submission_name=SUBMISSION_NAME,
    domain=DOMAIN,
    experiment=EXPERIMENT,
    use_orography=USE_OROGRAPHY,
    variable=VARIABLE,
    prediction=PREDICTION,
)
print(f"\n=== Prediction Dataset ===\n{prediction_ds}")


# =============================================================================
# 5. LOAD GROUND TRUTH DATA
# =============================================================================

gt_loader = GroundTruthDataLoader(config.groundtruth_path)

print(f"\n=== Available Ground Truth Domains ===")
print(f"Domains: {gt_loader.available_domains()}")

# Load training target
gt_train = gt_loader.load_train_target(
    domain="ALPS",
    training_experiment="Emulator_hist_future",
    variable=VARIABLE,
)
print(f"\n=== Ground Truth (Train) ===\n{gt_train}")

# Load test target
gt_test = gt_loader.load_test_target(
    domain="ALPS",
    training_experiment="Emulator_hist_future",
    evaluation_experiment=PREDICTION,
    variable=VARIABLE,
)
print(f"\n=== Ground Truth (Test) ===\n{gt_test}")


# =============================================================================
# 6. RUN EVALUATION
# =============================================================================

evaluator = Evaluator(
    config=config,
    catalog=catalog,
    metrics_registry_path=METRICS_PATH,
    ml_benchmark_path=config.mlbenchmark_path,
)

# Compute default metrics for the variable (defined in metrics.json)
print(f"\n=== Running Evaluation ===")
results = evaluator.evaluate(
    submission_name=SUBMISSION_NAME,
    domain=DOMAIN,
    experiment=EXPERIMENT,
    variable=VARIABLE,
    prediction=PREDICTION,
    use_orography=USE_OROGRAPHY,
)

print(f"Computed metrics: {list(results.keys())}")

# List saved metric files
metric_files = evaluator.available_metric_files(
    submission_name=SUBMISSION_NAME,
    domain=DOMAIN,
    experiment=EXPERIMENT,
    variable=VARIABLE,
    prediction=PREDICTION,
    use_orography=USE_OROGRAPHY,
)
print(f"Saved metric files: {metric_files}")

# Load a specific metric
rmse_ds = evaluator.load_metric(
    submission_name=SUBMISSION_NAME,
    domain=DOMAIN,
    experiment=EXPERIMENT,
    variable=VARIABLE,
    prediction=PREDICTION,
    metric_name="rmse",
    use_orography=USE_OROGRAPHY,
)
print(f"\n=== RMSE Result ===\n{rmse_ds}")