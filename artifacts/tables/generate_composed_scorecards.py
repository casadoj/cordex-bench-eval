import sys
from pathlib import Path

# Add repo root for imports (same pattern as generate_tables.py)
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cordex_bench_eval import SubmissionCatalog, ScorecardGenerator
from cordex_bench_eval.config import load_config_file

# Paths
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.json"
TABLES_CONFIG_PATH = ROOT / "configs" / "tables.json"
TABLES_DIR = ROOT / "output"

# Domains to include
DOMAINS = ["ALPS_Domain", "NZ_Domain", "SA_Domain"]

# Submissions to include (set to None to use all entries from the catalog)

submission_names = ['DeepESD-IFCAv1', 'DeepESD-IFCAv1-orog',
                    'ANN', 'ANN-orog',
                    'Prithvi-UNet', 'Prithvi-UNet-orog',
                    'XGBoost_IDL',
                    'deepsensor_bas', 'deepsensor_orog_bas',
                    'CNRM-UNeT',
                    'Rossby-UNet', 'Rossby-UNet-orog',
                    'DetUNet', 'DetUNet-orog',
                    'DetUNet-v2', 'DetUNet-v2-orog',
                    'GNN4CD-orog',
                    # 'DeepESDcrps_IFCAv2', 'DeepESDcrps_IFCAv2-orog',
                    'DeepESD_IDL',
                    'ParamUNET',
                    'EnScale', 'EnScale-orog', 'EnScale-linex',
                    'ViT-IFCAv1', 'ViT-IFCAv1-orog',
                    'FlowMatching-v1', 'FlowMatching-v1-orog',
                    'CorrDiff-TW1_NTNU-CWA', 'CorrDiff-TW1_NTNU-CWA-orog',
                    'UiBCorrDiff-orog',
                    'RCMGEM-mv-orog',
                    'RCMFlow-orog',
                    'ParamDiffusion-orog',
                    'ResGAN', 'ResGAN-orog',
                    'ResGAN-v2', 'ResGAN-v2-orog',
                    'spaGAN', 'spaGAN-orog']

###########################################################
# Include AV (average) column per metric
INCLUDE_AVERAGE = True

# Include Overall Rank column (aggregate rank across all metrics/domains)
INCLUDE_OVERALL_RANK = True

# Colormap for cell coloring
CMAP = "Greys_r"

# Normalization strategy for cell coloring.
# None for plain min-max, or one of: "rank", "percentile", "robust_zscore".
NORMALIZATION = "rank"

# Show only the AV (average across domains) column per metric.
AVERAGE_ONLY = False

# Number of discrete colour bins in the colorbar.
N_COLOR_LEVELS = 20

# Extra horizontal gap between metric groups (and before Overall), in cell-width units.
# 0 reproduces the old tight layout; ~0.15–0.25 gives visible white separation between blocks.
METRIC_COLUMN_GAP = 0.2

# If True, colorbar runs worse → better (left → right). If False, better is on the left.
COLORBAR_WORSE_ON_LEFT = True

# Border around cells for the best model(s) per metric column
HIGHLIGHT_BEST = True
# Matplotlib colour for that border
BEST_CELL_BORDER_COLOR = "yellow"
# Border stroke width in points
BEST_CELL_BORDER_LINEWIDTH = 1.5

# Colour for numeric values inside each cell
CELL_TEXT_COLOR = "deeppink"

# If True, the lower triangle (second experiment) shows the absolute
# difference with the upper triangle (first experiment) instead of the raw value.
# Computed as: |lower_value - upper_value|, except the Overall column where
# the lower triangle still shows experiment 2's overall rank.
LOWER_TRIANGLE_DIFF = False
###########################################################

# Pairs of table names to render as dual-experiment scorecards.
# Each entry is a (name1, name2) tuple: name1 → upper-left triangle,
# name2 → lower-right triangle.
TABLE_PAIRS = [("tasmax_esd_pseudo_reality_perfect_cross_validation",
                "tasmax_esd_pseudo_reality_perfect_extrapolation"),
               ("pr_esd_pseudo_reality_perfect_cross_validation",
                "pr_esd_pseudo_reality_perfect_extrapolation"),
               ("tasmax_emulator_hist_future_perfect_cross_validation",
                "tasmax_emulator_hist_future_perfect_interpolation"),
               ("pr_emulator_hist_future_perfect_cross_validation",
                "pr_emulator_hist_future_perfect_interpolation")]


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config_file(str(CONFIG_PATH))
    catalog = SubmissionCatalog(config.catalog_path)

    scorecard_gen = ScorecardGenerator(config=config,
                                       catalog=catalog,
                                       table_config_path=str(TABLES_CONFIG_PATH))

    table_config = scorecard_gen.table_generator.table_config

    print(f"Generating {len(TABLE_PAIRS)} composed scorecard(s) → {TABLES_DIR}/")
    for name1, name2 in TABLE_PAIRS:
        dfs = scorecard_gen.generate_scorecard(table_name=[name1, name2],
                                               submission_names=submission_names,
                                               domains=DOMAINS,
                                               include_average=INCLUDE_AVERAGE,
                                               include_overall_rank=INCLUDE_OVERALL_RANK)
        desc1 = table_config.get("tables", {}).get(name1, {}).get("description", name1)
        desc2 = table_config.get("tables", {}).get(name2, {}).get("description", name2)
        description = f"{desc1}  |  {desc2}"
        out = TABLES_DIR / f"{name1}_vs_{name2}_scorecard.pdf"
        scorecard_gen.export_scorecard_pdf(df=dfs, output_path=str(out),
                                           description=description,
                                           cmap=CMAP,
                                           normalization=NORMALIZATION,
                                           average_only=AVERAGE_ONLY,
                                           n_color_levels=N_COLOR_LEVELS,
                                           highlight_best=HIGHLIGHT_BEST,
                                           cell_text_color=CELL_TEXT_COLOR,
                                           best_cell_border_color=BEST_CELL_BORDER_COLOR,
                                           best_cell_border_linewidth=BEST_CELL_BORDER_LINEWIDTH,
                                           colorbar_worse_on_left=COLORBAR_WORSE_ON_LEFT,
                                           metric_column_gap=METRIC_COLUMN_GAP,
                                           lower_triangle_diff=LOWER_TRIANGLE_DIFF)
        print(f"  {name1}  |  {name2}")


if __name__ == "__main__":
    main()
