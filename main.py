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
    FigureGenerator,
)
from cordex_bench_eval.config import load_config_file

########################################################################################
# 1. CONFIGURATION

CONFIG_PATH = "/gpfs/projects/meteo/WORK/gonzabad/cordex-bench-eval/configs/config.json"
METRICS_PATH = "/gpfs/projects/meteo/WORK/gonzabad/cordex-bench-eval/configs/metrics.json"
TABLES_CONFIG_PATH = "/gpfs/projects/meteo/WORK/gonzabad/cordex-bench-eval/configs/tables.json"
FIGURES_CONFIG_PATH = "/gpfs/projects/meteo/WORK/gonzabad/cordex-bench-eval/configs/figures.json"

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

# ########################################################################################
# # 4. LOAD A SPECIFIC SUBMISSION

# # Select which submission/experiment to evaluate
# SUBMISSION_NAME = "DeepESD-v1_IFCA_m1"
# TRAINING_EXPERIMENT = "ESD_pseudo_reality"
# EVALUATION_EXPERIMENT = "Perfect_cross_validation"
# USE_OROGRAPHY = True
# DOMAIN = "SA_Domain"
# VARIABLE = "pr"

# # Create data loader
# loader = SubmissionDataLoader(catalog)

# # Explore available data
# print(f"\n=== Available Data for {SUBMISSION_NAME} ===")
# print(f"Domains: {loader.available_domains(SUBMISSION_NAME)}")
# print(f"Experiments in {DOMAIN}: {loader.available_experiments(SUBMISSION_NAME, DOMAIN)}")

# experiment_name = f"{TRAINING_EXPERIMENT}_{'OROG' if USE_OROGRAPHY else 'NO_OROG'}"
# print(f"Periods in {experiment_name}: {loader.available_periods(SUBMISSION_NAME, DOMAIN, experiment_name)}")

# # Load the prediction dataset
# prediction_ds = loader.load_dataset(submission_name=SUBMISSION_NAME,
#                                     domain=DOMAIN,
#                                     training_experiment=TRAINING_EXPERIMENT,
#                                     evaluation_experiment=EVALUATION_EXPERIMENT,
#                                     variable=VARIABLE,
#                                     use_orography=USE_OROGRAPHY)
# print(f"Prediction Dataset: {prediction_ds}")
# ########################################################################################

# ########################################################################################
# # 5. LOAD GROUND TRUTH DATA

# gt_loader = GroundTruthDataLoader(config.groundtruth_path)

# print("Available Ground Truth Domains:")
# print(f"Domains: {gt_loader.available_domains()}")

# # Load training target
# gt_train = gt_loader.load_train_target(domain=DOMAIN,
#                                        training_experiment=TRAINING_EXPERIMENT,
#                                        variable=VARIABLE)
# print(f"Ground Truth (Train): {gt_train}")

# # Load test target
# gt_test = gt_loader.load_test_target(domain=DOMAIN,
#                                      training_experiment=TRAINING_EXPERIMENT,
#                                      evaluation_experiment=EVALUATION_EXPERIMENT,
#                                      variable=VARIABLE)
# print(f"Ground Truth (Test): {gt_test}")
# ########################################################################################

# ########################################################################################
# # 6. RUN EVALUATION

# evaluator = Evaluator(config=config, catalog=catalog, metrics_registry_path=METRICS_PATH,
#                       ml_benchmark_path=config.mlbenchmark_path)

# # Compute default metrics for the variable (defined in metrics.json)
# print("Running Evaluation:")
# results = evaluator.evaluate(submission_name=SUBMISSION_NAME,
#                              domain=DOMAIN,
#                              training_experiment=TRAINING_EXPERIMENT,
#                              evaluation_experiment=EVALUATION_EXPERIMENT,
#                              variable=VARIABLE,
#                              use_orography=USE_OROGRAPHY)

# print(f"Computed metrics: {list(results.keys())}")

# # List saved metric files
# metric_files = evaluator.available_metric_files(submission_name=SUBMISSION_NAME,
#                                                 domain=DOMAIN,
#                                                 training_experiment=TRAINING_EXPERIMENT,
#                                                 evaluation_experiment=EVALUATION_EXPERIMENT,
#                                                 variable=VARIABLE,
#                                                 use_orography=USE_OROGRAPHY)
# print(f"Saved metric files: {metric_files}")

# # Load a specific metric
# sdii_ratio_ds = evaluator.load_metric(submission_name=SUBMISSION_NAME,
#                                       domain=DOMAIN,
#                                       training_experiment=TRAINING_EXPERIMENT,
#                                       evaluation_experiment=EVALUATION_EXPERIMENT,
#                                       variable=VARIABLE,
#                                       metric_name="sdii_ratio",
#                                       use_orography=USE_OROGRAPHY)
# print(f"sdii_ratio Result: {sdii_ratio_ds}")
# ########################################################################################

# ########################################################################################
# # 7. GENERATE COMPARISON TABLES

# # Initialize table generator
# table_gen = TableGenerator(config=config,
#                            catalog=catalog,
#                            table_config_path=TABLES_CONFIG_PATH)

# # Get all available submissions
# all_submissions = [entry.name for entry in catalog.all_entries()]
# print(f"\nAvailable submissions for tables: {all_submissions}")
# ########################################################################################

# ########################################################################################
# # 7a. GENERATE A SINGLE TABLE

# print("\n--- Generating Single Table ---")
# table_name = "esd_perfect_extrapolation"

# df_single = table_gen.generate_table(table_name=table_name,
#                                      submission_names=all_submissions,
#                                      domains=["NZ_Domain", "SA_Domain", "ALPS_Domain"],
#                                      include_average=True)

# print(f"\nTable: {table_name}")
# print(f"Shape: {df_single.shape}")
# print("\nTable Preview:")
# print(df_single)
# ########################################################################################

# ########################################################################################
# # 7b. EXPORT SINGLE TABLE TO MULTIPLE FORMATS

# OUTPUT_DIR = "/gpfs/projects/meteo/WORK/gonzabad/cordex-bench-eval/tables_output"

# # Export to CSV
# table_gen.export_table(df=df_single,
#                        output_path=f"{OUTPUT_DIR}/{table_name}.csv",
#                        format="csv")
# print(f"Saved CSV: {OUTPUT_DIR}/{table_name}.csv")

# # Export to HTML
# table_gen.export_table(df=df_single,
#                        output_path=f"{OUTPUT_DIR}/{table_name}.html",
#                        format="html")
# print(f"Saved HTML: {OUTPUT_DIR}/{table_name}.html")
# ########################################################################################

# ########################################################################################
# # 7c. GENERATE ALL TABLES AT ONCE

# print(f"\nGenerating All Tables:")

# all_tables = table_gen.generate_all_tables(submission_names=all_submissions)

# print(f"Generated {len(all_tables)} tables:")
# for name in all_tables.keys():
#     print(f"{name} (shape: {all_tables[name].shape})")
# ########################################################################################

########################################################################################
# 8. GENERATE DAILY FIGURES

FIGURES_OUTPUT_DIR = "/gpfs/projects/meteo/WORK/gonzabad/cordex-bench-eval/figures_output"

print("Generating Daily Figure")

figure_gen = FigureGenerator(config=config,
                             catalog=catalog,
                             figure_config_path=FIGURES_CONFIG_PATH)

figure_name = "sa_pr_daily_snapshot"
figure_path = figure_gen.generate_figure(figure_name=figure_name,
                                         submission_names=["DeepESD-v1_IFCA_m1"],
                                         output_dir=FIGURES_OUTPUT_DIR,
                                         file_format="pdf")

print(f"Saved figure: {figure_path}")
########################################################################################
