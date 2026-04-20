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
    SubmissionValidator,
    SubmissionCatalog,
    SubmissionDataLoader,
    GroundTruthDataLoader,
    Evaluator,
    TableGenerator,
    FigureGenerator,
    DiagnosticFigureGenerator,
    PsdFigureGenerator,
)

from cordex_bench_eval.config import load_config_file

########################################################################################
# SET THE CONFIGURATION

CONFIG_PATH = "/lustre/gmeteo/WORK/abadj/cordex-bench-eval/configs/config.json"
METRICS_PATH = "/lustre/gmeteo/WORK/abadj/cordex-bench-eval/configs/metrics.json"
TABLES_CONFIG_PATH = "/lustre/gmeteo/WORK/abadj/cordex-bench-eval/configs/tables.json"
FIGURES_CONFIG_PATH = "/lustre/gmeteo/WORK/abadj/cordex-bench-eval/configs/figures.json"
DIAGNOSTIC_FIGURES_CONFIG_PATH = "/lustre/gmeteo/WORK/abadj/cordex-bench-eval/configs/diagnostic_figures.json"
PSD_FIGURES_CONFIG_PATH = "/lustre/gmeteo/WORK/abadj/cordex-bench-eval/configs/psd_figures.json"

config = load_config_file(CONFIG_PATH)
catalog = SubmissionCatalog(config.catalog_path)
########################################################################################

########################################################################################
# VALIDATE STAGED SUBMISSIONS

# Validate one submission by name:
report = SubmissionValidator(config=config, catalog=catalog).validate_submission("GNN4CD-orog")

print("Submission validation results:")
status = "VALID" if report.is_valid else "INVALID"
print(f"{report.submission_name}: {status} "
      f"(checked_files={report.checked_files}, missing_files={report.missing_files}, "
      f"errors={report.error_count}, warnings={report.warning_count})")

if report.missing_file_paths:
    print("Missing files:")
    for missing_path in report.missing_file_paths:
        print(f"  - {missing_path}")

for issue in report.issues:
    if issue.code in {"MISSING_REQUIRED_FILE", "EXPECTED_FILE_NOT_A_FILE"}:
        continue
    print(f"  - [{issue.severity}] {issue.code}: {issue.message}")
########################################################################################

########################################################################################
# INGEST NEW SUBMISSIONS

# handler = SubmissionHandler(config)

# archive_path = "/lustre/gmeteo/WORK/abadj/cordex-bench-eval/data/incoming/FlowMatching-v1-orog.zip"
# ingested = handler.ingest(archive_path=archive_path)
# print(f"Ingested {len(ingested)} new submission(s)")
########################################################################################

########################################################################################
# # BROWSE THE SUBMISSION CATALOG

# # List all available submissions
# print("Available Submissions:")
# for entry in catalog.all_entries():
#     print(f"{entry.name} ({entry.status})")
########################################################################################

########################################################################################
# # LOAD A SPECIFIC SUBMISSION

# # Select the specific submission to load
# SUBMISSION_NAME = "CNRM-UNeT"
# TRAINING_EXPERIMENT = "Emulator_hist_future"
# EVALUATION_EXPERIMENT = "Perfect_cross_validation"
# DOMAIN = "ALPS_Domain"
# VARIABLE = "tasmax"

# # Create data loader
# loader = SubmissionDataLoader(catalog)

# # Explore available data
# print(f"Available data for {SUBMISSION_NAME}:")
# print(f"Domains: {loader.available_domains(SUBMISSION_NAME)}")
# print(f"Experiments in {DOMAIN}: {loader.available_experiments(SUBMISSION_NAME, DOMAIN)}")
# print(f"Periods in {TRAINING_EXPERIMENT}: {loader.available_periods(SUBMISSION_NAME, DOMAIN, TRAINING_EXPERIMENT)}")

# # Load the prediction dataset
# prediction_ds = loader.load_dataset(submission_name=SUBMISSION_NAME,
#                                     domain=DOMAIN,
#                                     training_experiment=TRAINING_EXPERIMENT,
#                                     evaluation_experiment=EVALUATION_EXPERIMENT,
#                                     variable=VARIABLE)
# print(f"Prediction dataset: {prediction_ds}")
# print(prediction_ds.tasmax.mean())
########################################################################################

########################################################################################
# # LOAD GROUND TRUTH DATA

# gt_loader = GroundTruthDataLoader(config.groundtruth_path)

# print(f"Ground truth domains available: {gt_loader.available_domains()}")

# # Load training target
# gt_train = gt_loader.load_train_target(domain=DOMAIN,
#                                        training_experiment=TRAINING_EXPERIMENT,
#                                        variable=VARIABLE)
# print(f"Ground truth (train): {gt_train}")

# # Load test target
# gt_test = gt_loader.load_test_target(domain=DOMAIN,
#                                      training_experiment=TRAINING_EXPERIMENT,
#                                      evaluation_experiment=EVALUATION_EXPERIMENT,
#                                      variable=VARIABLE)
# print(f"Ground truth (test): {gt_test}")
########################################################################################

########################################################################################
# # RUN EVALUATION

# # Evaluation settings
# SUBMISSION_NAMES = ["FlowMatching-v1-orog", "FlowMatching-v1", "UiBCorrDiff-orog", "ANN", "ANN-orog", "RCMGEM-mv-orog"]
# TRAINING_EXPERIMENTS = ["Emulator_hist_future"]
# EVALUATION_EXPERIMENTS = ["Perfect_cross_validation"]
# DOMAINS = ["ALPS_Domain", "NZ_Domain"]
# VARIABLE = "pr"

# # Set the evaluator
# evaluator = Evaluator(config=config, catalog=catalog, metrics_registry_path=METRICS_PATH,
#                       ml_benchmark_path=config.mlbenchmark_path)

# evaluation_report = evaluator.evaluate_multiple(submission_names=SUBMISSION_NAMES,
#                                                 domain=DOMAINS,
#                                                 training_experiment=TRAINING_EXPERIMENTS,
#                                                 evaluation_experiment=EVALUATION_EXPERIMENTS,
#                                                 variable=VARIABLE,
#                                                 recompute_existing=False)
# print(f"Evaluation report by context: {evaluation_report}")

# # List saved metric files
# metric_files_by_submission = {
#     f"{name}|{domain_name}|{training_name}|{evaluation_name}|{VARIABLE}": evaluator.available_metric_files(
#         submission_name=name,
#         domain=domain_name,
#         training_experiment=training_name,
#         evaluation_experiment=evaluation_name,
#         variable=VARIABLE,
#     )
#     for name in SUBMISSION_NAMES
#     for domain_name in DOMAINS
#     for training_name in TRAINING_EXPERIMENTS
#     for evaluation_name in EVALUATION_EXPERIMENTS
# }
# print(f"Saved metric files by submission: {metric_files_by_submission}")

# # Load a specific metric
# metric_submission = SUBMISSION_NAMES[0]
# metric_domain = DOMAINS[0]
# metric_training = TRAINING_EXPERIMENTS[0]
# metric_evaluation = EVALUATION_EXPERIMENTS[0]
# metric_to_load = "sdii_ratio"
# metric_ds = evaluator.load_metric(submission_name=metric_submission,
#                                   domain=metric_domain,
#                                   training_experiment=metric_training,
#                                   evaluation_experiment=metric_evaluation,
#                                   variable=VARIABLE,
#                                   metric_name=metric_to_load)
# print(f"{metric_to_load} result for {metric_submission} ({metric_domain}, {VARIABLE}): {metric_ds}")
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
#                                      domains=["ALPS_Domain", "NZ_Domain", "SA_Domain"], # More domains can be added here
#                                      include_average=True)

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
# # GENERATE FIGURES OF DAILY MAPS

# FIGURES_OUTPUT_DIR = "/lustre/gmeteo/WORK/abadj/cordex-bench-eval"

# figure_gen = FigureGenerator(config=config,
#                              catalog=catalog,
#                              figure_config_path=FIGURES_CONFIG_PATH)

# # Get all available submissions
# # all_submissions = [entry.name for entry in catalog.all_entries()]
# all_submissions = ["FlowMatching-v1", 
#                    "FlowMatching-v1-orog", "UiBCorrDiff-orog",
#                    "ANN", "ANN-orog",
#                    "Prithvi-UNet", "Prithvi-UNet-orog",
#                    "RCMGEM-mv-orog"]

# # Figures defined in config/figures.json
# figure_name = 'nz_pr_daily_snapshot'
# figure_path = figure_gen.generate_figure(figure_name=figure_name,
#                                          submission_names=all_submissions,
#                                          output_dir=FIGURES_OUTPUT_DIR)

# print(f"Saved figure: {figure_path}")
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

# # Get all available submissions
# all_submissions = [entry.name for entry in catalog.all_entries()]

# # Diagnostic figures defined in config/diagnostic_figures.json
# diagnostic_figure_name = "alps_pr_rmse_map"
# diagnostic_path = diagnostic_fig_gen.generate_figure(figure_name=diagnostic_figure_name,
#                                                      submission_names=all_submissions,
#                                                      output_dir=DIAGNOSTIC_FIGURES_OUTPUT_DIR)

# print(f"Saved diagnostic figure: {diagnostic_path}")
########################################################################################

########################################################################################
# # GENERATE PSD COMPARISON FIGURES

# PSD_FIGURES_OUTPUT_DIR = "/lustre/gmeteo/WORK/abadj/cordex-bench-eval/"

# # Initialize evaluator (needed to load metrics)
# evaluator = Evaluator(config=config, catalog=catalog, metrics_registry_path=METRICS_PATH,
#                       ml_benchmark_path=config.mlbenchmark_path)

# psd_fig_gen = PsdFigureGenerator(config=config,
#                                  catalog=catalog,
#                                  evaluator=evaluator,
#                                  figure_config_path=PSD_FIGURES_CONFIG_PATH)

# # Get all available submissions
# all_submissions = [entry.name for entry in catalog.all_entries()]

# # PSD figures defined in config/psd_figures.json
# psd_figure_name = "pr_emulator_perfect_cv_psd"
# psd_path = psd_fig_gen.generate_figure(figure_name=psd_figure_name,
#                                        submission_names=all_submissions,
#                                        output_dir=PSD_FIGURES_OUTPUT_DIR)

# print(f"Saved PSD figure: {psd_path}")
########################################################################################