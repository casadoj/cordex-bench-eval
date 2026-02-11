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
    DiagnosticFigureGenerator,
)

from cordex_bench_eval.config import load_config_file

########################################################################################
# SET THE CONFIGURATION

CONFIG_PATH = "/lustre/gmeteo/WORK/abadj/cordex-bench-eval/configs/config.json"
METRICS_PATH = "/lustre/gmeteo/WORK/abadj/cordex-bench-eval/configs/metrics.json"
TABLES_CONFIG_PATH = "/lustre/gmeteo/WORK/abadj/cordex-bench-eval/configs/tables.json"
FIGURES_CONFIG_PATH = "/lustre/gmeteo/WORK/abadj/cordex-bench-eval/configs/figures.json"
DIAGNOSTIC_FIGURES_CONFIG_PATH = "/lustre/gmeteo/WORK/abadj/cordex-bench-eval/configs/diagnostic_figures.json"

config = load_config_file(CONFIG_PATH)
########################################################################################

########################################################################################
# INGEST NEW SUBMISSIONS

# handler = SubmissionHandler(config)

# archive_path = "/lustre/gmeteo/WORK/abadj/cordex-bench-eval/data/incoming/FlowMatching-v1-orog.zip"
# ingested = handler.ingest(archive_path=archive_path)
# print(f"Ingested {len(ingested)} new submission(s)")
########################################################################################

########################################################################################
# BROWSE THE SUBMISSION CATALOG

catalog = SubmissionCatalog(config.catalog_path)

# List all available submissions
print("Available Submissions:")
for entry in catalog.all_entries():
    print(f"{entry.name} ({entry.status})")
########################################################################################

########################################################################################
# LOAD A SPECIFIC SUBMISSION

# Select the specific submission to load
SUBMISSION_NAME = "FlowMatching-v1-orog"
TRAINING_EXPERIMENT = "Emulator_hist_future"
EVALUATION_EXPERIMENT = "Perfect_cross_validation"
DOMAIN = "NZ_Domain"
VARIABLE = "pr"

# Create data loader
loader = SubmissionDataLoader(catalog)

# Explore available data
print(f"Available data for {SUBMISSION_NAME}:")
print(f"Domains: {loader.available_domains(SUBMISSION_NAME)}")
print(f"Experiments in {DOMAIN}: {loader.available_experiments(SUBMISSION_NAME, DOMAIN)}")
print(f"Periods in {TRAINING_EXPERIMENT}: {loader.available_periods(SUBMISSION_NAME, DOMAIN, TRAINING_EXPERIMENT)}")

# Load the prediction dataset
prediction_ds = loader.load_dataset(submission_name=SUBMISSION_NAME,
                                    domain=DOMAIN,
                                    training_experiment=TRAINING_EXPERIMENT,
                                    evaluation_experiment=EVALUATION_EXPERIMENT,
                                    variable=VARIABLE)
print(f"Prediction dataset: {prediction_ds}")
########################################################################################

########################################################################################
# LOAD GROUND TRUTH DATA

gt_loader = GroundTruthDataLoader(config.groundtruth_path)

print(f"Ground truth domains available: {gt_loader.available_domains()}")

# Load training target
gt_train = gt_loader.load_train_target(domain=DOMAIN,
                                       training_experiment=TRAINING_EXPERIMENT,
                                       variable=VARIABLE)
print(f"Ground truth (train): {gt_train}")

# Load test target
gt_test = gt_loader.load_test_target(domain=DOMAIN,
                                     training_experiment=TRAINING_EXPERIMENT,
                                     evaluation_experiment=EVALUATION_EXPERIMENT,
                                     variable=VARIABLE)
print(f"Ground truth (test): {gt_test}")
########################################################################################

########################################################################################
# # RUN EVALUATION

# # Set the evaluator
# evaluator = Evaluator(config=config, catalog=catalog, metrics_registry_path=METRICS_PATH,
#                       ml_benchmark_path=config.mlbenchmark_path)

# # Compute default metrics for the variable (defined in metrics.json)
# results = evaluator.evaluate(submission_name=SUBMISSION_NAME,
#                              domain=DOMAIN,
#                              training_experiment=TRAINING_EXPERIMENT,
#                              evaluation_experiment=EVALUATION_EXPERIMENT,
#                              variable=VARIABLE)

# print(f"Computed metrics: {list(results.keys())}")

# # List saved metric files
# metric_files = evaluator.available_metric_files(submission_name=SUBMISSION_NAME,
#                                                 domain=DOMAIN,
#                                                 training_experiment=TRAINING_EXPERIMENT,
#                                                 evaluation_experiment=EVALUATION_EXPERIMENT,
#                                                 variable=VARIABLE)
# print(f"Saved metric files for {SUBMISSION_NAME}: {metric_files}")

# # Load a specific metric
# metric_to_load = "sdii_ratio"
# metric_ds = evaluator.load_metric(submission_name=SUBMISSION_NAME,
#                                   domain=DOMAIN,
#                                   training_experiment=TRAINING_EXPERIMENT,
#                                   evaluation_experiment=EVALUATION_EXPERIMENT,
#                                   variable=VARIABLE,
#                                   metric_name=metric_to_load)
# print(f"{metric_to_load} result: {metric_ds}")
########################################################################################

########################################################################################
# # GENERATE COMPARISON TABLES

# # Initialize table generator
# table_gen = TableGenerator(config=config,
#                            catalog=catalog,
#                            table_config_path=TABLES_CONFIG_PATH)

# # Get all available submissions
# all_submissions = [entry.name for entry in catalog.all_entries()]

# # Tables defined in config/tables.json
# table_name = "pr_emulator_perfect_cross_validation"

# df_single = table_gen.generate_table(table_name=table_name,
#                                      submission_names=all_submissions,
#                                      domains=["ALPS_Domain"], # More domains can be added here
#                                      include_average=False)

# print(f"Table: {table_name}")
# print(f"Shape: {df_single.shape}")
# print("Table Preview:")
# print(df_single)
########################################################################################

########################################################################################
# # EXPORT A TABLE TO MULTIPLE FORMATS

# OUTPUT_DIR = "/lustre/gmeteo/WORK/abadj/cordex-bench-eval"

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
########################################################################################

########################################################################################
# GENERATE FIGURES OF DAILY MAPS

FIGURES_OUTPUT_DIR = "/lustre/gmeteo/WORK/abadj/cordex-bench-eval"

figure_gen = FigureGenerator(config=config,
                             catalog=catalog,
                             figure_config_path=FIGURES_CONFIG_PATH)

# Get all available submissions
all_submissions = [entry.name for entry in catalog.all_entries()]

# Figures defined in config/figures.json
figure_name = "nz_pr_daily_snapshot"
figure_path = figure_gen.generate_figure(figure_name=figure_name,
                                         submission_names=all_submissions,
                                         output_dir=FIGURES_OUTPUT_DIR)

print(f"Saved figure: {figure_path}")
########################################################################################

########################################################################################
# # GENERATE DIAGNOSTIC SPATIAL MAPS

# DIAGNOSTIC_FIGURES_OUTPUT_DIR = "/lustre/gmeteo/WORK/abadj/cordex-bench-eval/"

# # Initialize evaluator (needed to load metrics)
# evaluator = Evaluator(config=config, catalog=catalog, metrics_registry_path=METRICS_PATH,
#                       ml_benchmark_path=config.mlbenchmark_path)

# diagnostic_fig_gen = DiagnosticFigureGenerator(config=config,
#                                                catalog=catalog,
#                                                evaluator=evaluator,
#                                                figure_config_path=DIAGNOSTIC_FIGURES_CONFIG_PATH)

# # Diagnostic figures defined in config/diagnostic_figures.json
# diagnostic_figure_name = "alps_pr_rmse_map"
# diagnostic_path = diagnostic_fig_gen.generate_figure(figure_name=diagnostic_figure_name,
#                                                      submission_names=all_submissions,
#                                                      output_dir=DIAGNOSTIC_FIGURES_OUTPUT_DIR)

# print(f"Saved diagnostic figure: {diagnostic_path}")
########################################################################################