"""
Generate PSD comparison figures for CORDEX benchmark submissions.

This script generates log-log power spectral density comparison plots
across domains for selected submissions and evaluation experiments.
"""

import sys
from pathlib import Path

# Add repo root for imports (same pattern as tables/generate_tables.py)
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cordex_bench_eval import SubmissionCatalog, PsdFigureGenerator, Evaluator
from cordex_bench_eval.config import load_config_file

# Paths
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.json"
METRICS_PATH = PROJECT_ROOT / "configs" / "metrics.json"
PSD_CONFIG_PATH = ROOT / "configs" / "psd_figures.json"
OUTPUT_DIR = ROOT / "figs"

# Submissions to include (set to None to use all entries from the catalog)
SUBMISSION_NAMES = ['DeepESD-IFCAv1', 'DeepESD-IFCAv1-orog',
                    'DeepESDcrps_IFCAv2', 'DeepESDcrps_IFCAv2-orog',
                    'ViT-IFCAv1', 'ViT-IFCAv1-orog',
                    'FlowMatching-v1', 'FlowMatching-v1-orog',
                    'UiBCorrDiff-orog', 'RCMGEM-mv-orog',
                    'CorrDiff-TW1_NTNU-CWA', 'CorrDiff-TW1_NTNU-CWA-orog',
                    'Prithvi-UNet', 'Prithvi-UNet-orog',
                    'XGBoost_IDL', 'DeepESD_IDL',
                    'deepsensor_bas', 'deepsensor_orog_bas',
                    'ResGAN', 'ResGAN-orog',
                    'CNRM-UNeT',
                    'Rossby-UNet', 'Rossby-UNet-orog',
                    'EnScale', 'EnScale-orog', 'EnScale-linex',
                    'spaGAN', 'spaGAN-orog',
                    'DetUNet', 'DetUNet-orog',
                    'ParamUNET']

# Figure names from configs/psd_figures.json -- edit this list to choose what to build
FIGURE_NAMES = ["pr_emulator_hist_future_perfect_interpolation",
                "tasmax_emulator_hist_future_perfect_interpolation"]

# Output format: "png", "pdf", "svg", etc.
FORMAT = "pdf"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config_file(str(CONFIG_PATH))
    catalog = SubmissionCatalog(config.catalog_path)

    # Evaluator is required for PsdFigureGenerator to load metric files
    evaluator = Evaluator(config=config,
                          catalog=catalog,
                          metrics_registry_path=str(METRICS_PATH),
                          ml_benchmark_path=config.mlbenchmark_path)

    psd_fig_gen = PsdFigureGenerator(config=config,
                                     catalog=catalog,
                                     evaluator=evaluator,
                                     figure_config_path=str(PSD_CONFIG_PATH))

    print(f"Generating {len(FIGURE_NAMES)} PSD figure(s) -> {OUTPUT_DIR}/")
    for name in FIGURE_NAMES:
        fig_path = psd_fig_gen.generate_figure(figure_name=name,
                                               submission_names=SUBMISSION_NAMES,
                                               output_dir=OUTPUT_DIR,
                                               file_format=FORMAT)
        print(f"  {name} → {fig_path}")

    print("Done.")


if __name__ == "__main__":
    main()
