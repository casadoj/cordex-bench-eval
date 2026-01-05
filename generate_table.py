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

# Configuration
CONFIG_PATH = "/gpfs/projects/meteo/WORK/gonzabad/cordex-bench-eval/configs/config.json"
METRICS_PATH = "/gpfs/projects/meteo/WORK/gonzabad/cordex-bench-eval/configs/metrics.json"
TABLES_CONFIG_PATH = "/gpfs/projects/meteo/WORK/gonzabad/cordex-bench-eval/configs/tables.json"
config = load_config_file(CONFIG_PATH)

# Catalog
catalog = SubmissionCatalog(config.catalog_path)

# Select which submission/experiment to evaluate
SUBMISSION_NAME = "DeepESD-v1_IFCA_m1"
TRAINING_EXPERIMENT = "ESD_pseudo_reality"
EVALUATION_EXPERIMENT = "Perfect_cross_validation"
USE_OROGRAPHY = True
DOMAINS = ["NZ_Domain", "SA_Domain", "ALPS_Domain"]
VARIABLE = "pr"

# Run the evaluation
evaluator = Evaluator(config=config, catalog=catalog, metrics_registry_path=METRICS_PATH,
                      ml_benchmark_path=config.mlbenchmark_path)

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

table_name = "pr_perfect_cross_validation"
df_single = table_gen.generate_table(table_name=table_name,
                                     submission_names=[SUBMISSION_NAME],
                                     domains=DOMAINS,
                                     include_average=True)
print(df_single)