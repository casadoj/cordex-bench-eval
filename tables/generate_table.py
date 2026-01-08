"""
Generate a table of metrics for a given submission and domain.
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

# Set the configuration paths
CONFIG_PATH = "/gpfs/projects/meteo/WORK/gonzabad/cordex-bench-eval/configs/config.json"
METRICS_PATH = "/gpfs/projects/meteo/WORK/gonzabad/cordex-bench-eval/configs/metrics.json"
TABLES_CONFIG_PATH = "/gpfs/projects/meteo/WORK/gonzabad/cordex-bench-eval/configs/tables.json"
OUTPUT_DIR = "/gpfs/projects/meteo/WORK/gonzabad/cordex-bench-eval/tables/tables_html"

# Load configuration
config = load_config_file(CONFIG_PATH)

# Load catalog
catalog = SubmissionCatalog(config.catalog_path)

# Select which submission/experiment to evaluate
SUBMISSION_NAME = "DeepESD-v1_IFCA_m1"
TRAINING_EXPERIMENT = "Emulator_hist_future"
EVALUATION_EXPERIMENT = "Perfect_cross_validation"
USE_OROGRAPHY = True
DOMAINS = ["NZ_Domain", "SA_Domain", "ALPS_Domain"]
VARIABLE = "tasmax"

# Run the evaluation
evaluator = Evaluator(config=config, catalog=catalog, metrics_registry_path=METRICS_PATH,
                      ml_benchmark_path=config.mlbenchmark_path)

# Compute the required diagnostic files for the table
for domain in DOMAINS:
    results = evaluator.evaluate(submission_name=SUBMISSION_NAME,
                                 domain=domain,
                                 training_experiment=TRAINING_EXPERIMENT,
                                 evaluation_experiment=EVALUATION_EXPERIMENT,
                                 variable=VARIABLE,
                                 use_orography=USE_OROGRAPHY)

# Generate table
table_gen = TableGenerator(config=config,
                           catalog=catalog,
                           table_config_path=TABLES_CONFIG_PATH)

table_name = "tasmax_perfect_cross_validation"
df_single = table_gen.generate_table(table_name=table_name,
                                     submission_names=[SUBMISSION_NAME],
                                     domains=DOMAINS,
                                     include_average=True)

print(df_single)

# Export the table to HTML
table_gen.export_table(df=df_single,
                       output_path=f"{OUTPUT_DIR}/{table_name}.html",
                       format="html")
                       
print(f"Saved HTML: {OUTPUT_DIR}/{table_name}.html")

# Export the table to CSV
table_gen.export_table(df=df_single,
                       output_path=f"{OUTPUT_DIR}/{table_name}.csv",
                       format="csv")
                       
print(f"Saved CSV: {OUTPUT_DIR}/{table_name}.csv")