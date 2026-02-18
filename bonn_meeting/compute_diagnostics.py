"""
Compute diagnostics (evaluation metrics) for Bonn meeting submissions.

Run this before generate_all.py. Diagnostics are saved to the project's
diagnostic_path (config.diagnostic_path). Tables and figures then load
these pre-computed metrics.

Edit the settings below to choose which experiments, domains, and variables
to run. Only valid (training_experiment, evaluation_experiment) pairs from
cordex_bench_eval.mappings.EVAL_SETTINGS are used.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cordex_bench_eval import SubmissionCatalog, Evaluator
from cordex_bench_eval.config import load_config_file
from cordex_bench_eval.mappings import EVAL_SETTINGS

# Paths
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
CONFIG_PATH = PROJECT_ROOT / "configs/config.json"
METRICS_PATH = PROJECT_ROOT / "configs/metrics.json"

# Edit these to control what gets computed
SUBMISSION_NAMES = ["RCMGEM-mv-orog"] # ["FlowMatching-v1", "FlowMatching-v1-orog", "UiBCorrDiff-orog", "RCMGEM-mv-orog"]
DOMAINS = ["ALPS_Domain", "NZ_Domain", "SA_Domain"]
TRAINING_EXPERIMENTS = ["ESD_pseudo_reality", "Emulator_hist_future"]
EVALUATION_EXPERIMENTS = ["Perfect_extrapolation", "Perfect_cross_validation",
                          "Imperfect_interpolation_GCM_transferability"]
VARIABLES = ["pr", "tasmax"]
RECOMPUTE_EXISTING = False


def _valid_evaluations_for(training_exp: str) -> list[str]:
    """Return evaluation experiments in EVALUATION_EXPERIMENTS that exist for this training experiment."""
    if training_exp not in EVAL_SETTINGS:
        return []
    valid = set(EVAL_SETTINGS[training_exp])
    return [e for e in EVALUATION_EXPERIMENTS if e in valid]


def main():
    config = load_config_file(str(CONFIG_PATH))
    catalog = SubmissionCatalog(config.catalog_path)
    evaluator = Evaluator(config=config,
                          catalog=catalog,
                          metrics_registry_path=str(METRICS_PATH),
                          ml_benchmark_path=config.mlbenchmark_path)

    for variable in VARIABLES:
        for training_exp in TRAINING_EXPERIMENTS:
            eval_exps = _valid_evaluations_for(training_exp)
            if not eval_exps:
                skip_msg = ", ".join(sorted(EVAL_SETTINGS.get(training_exp, [])))
                print(f"\n[skip] {training_exp}: none of {EVALUATION_EXPERIMENTS} valid (available: {skip_msg})")
                continue
            print(f"\n[variable={variable}] training={training_exp}, domains={DOMAINS}, evaluation={eval_exps}")
            evaluator.evaluate_multiple(submission_names=SUBMISSION_NAMES,
                                        domain=DOMAINS,
                                        training_experiment=training_exp,
                                        evaluation_experiment=eval_exps,
                                        variable=variable,
                                        recompute_existing=RECOMPUTE_EXISTING)

    print("\nDone. Diagnostics saved to:", config.diagnostic_path)


if __name__ == "__main__":
    main()
