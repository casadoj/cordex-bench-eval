"""
CORDEX Benchmark Evaluation - Complete Workflow

This script demonstrates the complete workflow for evaluating deep learning
model predictions against ground truth data and generating comparison tables.

Workflow steps:
1. Load configuration
2. (Optional) Ingest new submissions from incoming directory
3. Load a submission from the catalog
4. Load ground truth data
5. Run evaluation metrics
6. Generate comparison tables from computed metrics
7. Export tables to various formats
"""

from cordex_bench_eval import (
    SubmissionHandler,
    SubmissionCatalog,
    SubmissionDataLoader,
    GroundTruthDataLoader,
    Evaluator,
    TableGenerator,
)
from cordex_bench_eval.config import load_config_file

########################################################################################
# 1. CONFIGURATION

CONFIG_PATH = "/gpfs/projects/meteo/WORK/gonzabad/cordex-bench-eval/configs/config.json"
METRICS_PATH = "/gpfs/projects/meteo/WORK/gonzabad/cordex-bench-eval/configs/metrics.json"
TABLES_CONFIG_PATH = "/gpfs/projects/meteo/WORK/gonzabad/cordex-bench-eval/configs/tables.json"

config = load_config_file(CONFIG_PATH)
########################################################################################

########################################################################################
# 2. INGEST NEW SUBMISSIONS (run once when new .zip files arrive)

# handler = SubmissionHandler(config)
# ingested = handler.ingest()
# print(f"Ingested {len(ingested)} new submissions")
########################################################################################

########################################################################################
# 3. BROWSE THE SUBMISSION CATALOG

catalog = SubmissionCatalog(config.catalog_path)
print("Available Submissions:")
for entry in catalog.all_entries():
    print(f"{entry.name} ({entry.status})")
########################################################################################

########################################################################################
# 4. LOAD A SPECIFIC SUBMISSION

# Select which submission/experiment to evaluate
SUBMISSION_NAME = "DeepESD-v1_IFCA_m1"
EXPERIMENT = "ESD_Pseudo-Reality"
USE_OROGRAPHY = True
DOMAIN = "NZ_Domain"
VARIABLE = "tasmax"
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
prediction_ds = loader.load_dataset(submission_name=SUBMISSION_NAME, domain=DOMAIN, experiment=EXPERIMENT,
                                    use_orography=USE_OROGRAPHY, variable=VARIABLE, prediction=PREDICTION)
print(f"Prediction Dataset: {prediction_ds}")
########################################################################################

########################################################################################
# 5. LOAD GROUND TRUTH DATA

gt_loader = GroundTruthDataLoader(config.groundtruth_path)

print("Available Ground Truth Domains:")
print(f"Domains: {gt_loader.available_domains()}")

# Load training target
gt_train = gt_loader.load_train_target(domain="ALPS", training_experiment="Emulator_hist_future", variable=VARIABLE)
print(f"Ground Truth (Train): {gt_train}")

# Load test target
gt_test = gt_loader.load_test_target(domain="ALPS", training_experiment="Emulator_hist_future",
                                     evaluation_experiment=PREDICTION, variable=VARIABLE)
print(f"Ground Truth (Test): {gt_test}")
########################################################################################

########################################################################################
# 6. RUN EVALUATION

evaluator = Evaluator(config=config, catalog=catalog, metrics_registry_path=METRICS_PATH,
                      ml_benchmark_path=config.mlbenchmark_path)

# Compute default metrics for the variable (defined in metrics.json)
print("Running Evaluation:")
results = evaluator.evaluate(submission_name=SUBMISSION_NAME, domain=DOMAIN, experiment=EXPERIMENT, variable=VARIABLE,
                             prediction=PREDICTION, use_orography=USE_OROGRAPHY)

print(f"Computed metrics: {list(results.keys())}")

# List saved metric files
metric_files = evaluator.available_metric_files(submission_name=SUBMISSION_NAME,
                                                domain=DOMAIN, experiment=EXPERIMENT, variable=VARIABLE,
                                                prediction=PREDICTION, use_orography=USE_OROGRAPHY)
print(f"Saved metric files: {metric_files}")

# Load a specific metric
rmse_ds = evaluator.load_metric(submission_name=SUBMISSION_NAME, domain=DOMAIN, experiment=EXPERIMENT, variable=VARIABLE,
                                prediction=PREDICTION, metric_name="rmse", use_orography=USE_OROGRAPHY)
print(f"RMSE Result: {rmse_ds}")
########################################################################################

########################################################################################
# 7. GENERATE COMPARISON TABLES

# Initialize table generator
table_gen = TableGenerator(config=config,
                           catalog=catalog,
                           table_config_path=TABLES_CONFIG_PATH)

# Get all available submissions
all_submissions = [entry.name for entry in catalog.all_entries()]
print(f"\nAvailable submissions for tables: {all_submissions}")
########################################################################################

########################################################################################
# 7a. GENERATE A SINGLE TABLE

print("\n--- Generating Single Table ---")
table_name = "esd_perfect_2041_2060"

df_single = table_gen.generate_table(table_name=table_name,
                                     submission_names=all_submissions,
                                     domains=["NZ_Domain", "SA_Domain", "ALPS_Domain"],
                                     include_average=True)

print(f"\nTable: {table_name}")
print(f"Shape: {df_single.shape}")
print("\nTable Preview:")
print(df_single)
########################################################################################

########################################################################################
# 7b. EXPORT SINGLE TABLE TO MULTIPLE FORMATS

OUTPUT_DIR = "/gpfs/projects/meteo/WORK/gonzabad/cordex-bench-eval/tables_output"


# Export to CSV
table_gen.export_table(df=df_single,
                       output_path=f"{OUTPUT_DIR}/{table_name}.csv",
                       format="csv")
print(f"Saved CSV: {OUTPUT_DIR}/{table_name}.csv")

# Export to HTML
table_gen.export_table(df=df_single,
                       output_path=f"{OUTPUT_DIR}/{table_name}.html",
                       format="html")
print(f"Saved HTML: {OUTPUT_DIR}/{table_name}.html")
########################################################################################

########################################################################################
# 7c. GENERATE ALL TABLES AT ONCE

print(f"\nGenerating All Tables:")

all_tables = table_gen.generate_all_tables(submission_names=all_submissions)

print(f"Generated {len(all_tables)} tables:")
for name in all_tables.keys():
    print(f"{name} (shape: {all_tables[name].shape})")
########################################################################################