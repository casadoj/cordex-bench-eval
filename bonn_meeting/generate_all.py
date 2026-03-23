"""
Generate all figures and tables for submissions.

Output structure:
  bonn_meeting/
    figures/           → daily snapshot maps
    diagnostic_figures/ → spatial metric maps
    tables/            → CSV and HTML comparison tables
"""

import sys
from pathlib import Path

# Add parent for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cordex_bench_eval import (
    SubmissionCatalog,
    Evaluator,
    TableGenerator,
    FigureGenerator,
    DiagnosticFigureGenerator,
)
from cordex_bench_eval.config import load_config_file

# Paths
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
CONFIG_PATH = PROJECT_ROOT / "configs/config.json"
METRICS_PATH = PROJECT_ROOT / "configs/metrics.json"
# Bonn meeting local configs (edit these to customize outputs)
TABLES_CONFIG_PATH = ROOT / "configs/tables.json"
FIGURES_CONFIG_PATH = ROOT / "configs/figures.json"
DIAGNOSTIC_FIGURES_CONFIG_PATH = ROOT / "configs/diagnostic_figures.json"

# Output dirs
FIGURES_DIR = ROOT / "figures"
DIAGNOSTIC_FIGURES_DIR = ROOT / "diagnostic_figures"
TABLES_DIR = ROOT / "tables"

# Domains to include
DOMAINS = ["ALPS_Domain", "NZ_Domain", "SA_Domain"]


def main():
    for d in [FIGURES_DIR, DIAGNOSTIC_FIGURES_DIR, TABLES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    config = load_config_file(str(CONFIG_PATH))
    catalog = SubmissionCatalog(config.catalog_path)

    # Override submissions if needed (e.g. subset for meeting)
    submission_names = ['DeepESD-IFCAv1', 'DeepESD-IFCAv1-orog',
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
                        'spaGAN', 'spaGAN_orog',
                        'DetUNet', 'DetUNet_orog',
                        'ParamUNET',
                        'DeepESDcrps_IFCAv2', 'DeepESDcrps_IFCAv2-orog']

    # print("Generating tables...")
    # table_gen = TableGenerator(config=config,
    #                            catalog=catalog,
    #                            table_config_path=str(TABLES_CONFIG_PATH))
    # table_names = ["pr_emulator_hist_future_perfect_cross_validation",
    #                "tasmax_emulator_hist_future_perfect_cross_validation",
    #                "pr_emulator_hist_future_imperfect_interpolation",
    #                "tasmax_emulator_hist_future_imperfect_interpolation",
    #                "pr_emulator_hist_future_imperfect_interpolation_GCM_transferability",
    #                "tasmax_emulator_hist_future_imperfect_interpolation_GCM_transferability",
    #                "pr_emulator_hist_future_perfect_interpolation_GCM_transferability",
    #                "tasmax_emulator_hist_future_perfect_interpolation_GCM_transferability"]
    # for name in table_names:
    #     df = table_gen.generate_table(table_name=name,
    #                                   submission_names=submission_names,
    #                                   domains=DOMAINS,
    #                                   include_average=True)
    #     for fmt in ("csv", "html"):
    #         out = TABLES_DIR / f"{name}.{fmt}"
    #         table_gen.export_table(df=df, output_path=str(out), format=fmt)
    #     print(f"  {name} → {TABLES_DIR}/")

    print("Generating figures (daily snapshots)...")
    figure_gen = FigureGenerator(config=config,
                                 catalog=catalog,
                                 figure_config_path=str(FIGURES_CONFIG_PATH))
    figure_names = ["sa_pr_emulator_hist_future_perfect_cross_validation_1",
                    "alps_pr_emulator_hist_future_perfect_cross_validation_1",
                    "nz_pr_emulator_hist_future_perfect_cross_validation_1"]

    for name in figure_names:
        figure_gen.generate_figure(figure_name=name,
                                   submission_names=submission_names,
                                   output_dir=str(FIGURES_DIR))
        print(f"  {name}")

    # print("Generating diagnostic figures (spatial maps)...")
    # evaluator = Evaluator(config=config,
    #                       catalog=catalog,
    #                       metrics_registry_path=str(METRICS_PATH),
    #                       ml_benchmark_path=config.mlbenchmark_path)
    # diag_gen = DiagnosticFigureGenerator(config=config,
    #                                      catalog=catalog,
    #                                      evaluator=evaluator,
    #                                      figure_config_path=str(DIAGNOSTIC_FIGURES_CONFIG_PATH))
    # diag_names = ["alps_emulator_hist_future_perfect_cross_validation_txx_map",
    #               "nz_emulator_hist_future_perfect_cross_validation_txx_map",
    #             #   "sa_emulator_hist_future_perfect_cross_validation_txx_map",
    #               "alps_emulator_hist_future_imperfect_interpolation_gcm_transferability_txx_map",
    #               "nz_emulator_hist_future_imperfect_interpolation_gcm_transferability_txx_map",
    #             #   "sa_emulator_hist_future_imperfect_interpolation_gcm_transferability_txx_map",
    #               "alps_esd_pseudo_reality_perfect_extrapolation_txx_map",
    #               "nz_esd_pseudo_reality_perfect_extrapolation_txx_map"]
    #             #   "sa_esd_pseudo_reality_perfect_extrapolation_txx_map"]
    # for name in diag_names:
    #     diag_gen.generate_figure(figure_name=name,
    #                              submission_names=submission_names,
    #                              output_dir=str(DIAGNOSTIC_FIGURES_DIR))
    #     print(f"  {name}")

    # print("Done. Outputs in:")
    # print(f"  Tables:        {TABLES_DIR}")
    # print(f"  Figures:       {FIGURES_DIR}")
    # print(f"  Diagnostics:   {DIAGNOSTIC_FIGURES_DIR}")


if __name__ == "__main__":
    main()
