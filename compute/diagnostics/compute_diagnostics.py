"""
Compute Diagnostics for All Models
"""

from cordex_bench_eval import (
    SubmissionCatalog,
    Evaluator,
)

from cordex_bench_eval.config import load_config_file

########################################################################################
# CONFIGURATION

CONFIG_PATH = "/lustre/gmeteo/WORK/abadj/cordex-bench-eval/configs/config.json"
METRICS_PATH = "/lustre/gmeteo/WORK/abadj/cordex-bench-eval/configs/metrics.json"

# Evaluation settings
VARIABLE = ["pr", "tasmax"]
DOMAINS = ["ALPS_Domain", "NZ_Domain", "SA_Domain"]
TRAINING_EXPERIMENTS = ["ESD_pseudo_reality", "Emulator_hist_future"]
# Evaluation experiments for each training experiment based on the benchmark specification
EVALUATION_EXPERIMENTS = [
    # For ESD_pseudo_reality
    "Perfect_cross_validation",
    "Imperfect_cross_validation",
    "Perfect_extrapolation",
    "Imperfect_extrapolation",
    "Perfect_extrapolation_GCM_transferability",
    # For Emulator_hist_future
    "Perfect_cross_validation",
    "Imperfect_cross_validation",
    "Perfect_interpolation",
    "Imperfect_interpolation",
    "Perfect_interpolation_GCM_transferability",
    "Imperfect_interpolation_GCM_transferability",
]

# Set to True to recompute metrics even if they already exist
RECOMPUTE_EXISTING = False

# Model to compute diagnostics for
model_to_evaluate = "Prithvi-UNet"

########################################################################################
# LOAD CONFIGURATION AND CATALOG

config = load_config_file(CONFIG_PATH)
catalog = SubmissionCatalog(config.catalog_path)

# COMPUTE DIAGNOSTICS

evaluator = Evaluator(config=config,
                      catalog=catalog,
                      metrics_registry_path=METRICS_PATH,
                      ml_benchmark_path=config.mlbenchmark_path)

for variable in VARIABLE:
    evaluation_report = evaluator.evaluate_multiple(submission_names=[model_to_evaluate],
                                                    domain=DOMAINS,
                                                    training_experiment=TRAINING_EXPERIMENTS,
                                                    evaluation_experiment=EVALUATION_EXPERIMENTS,
                                                    variable=variable,
                                                    recompute_existing=RECOMPUTE_EXISTING)
    print(f"[{variable}] {evaluation_report}")
