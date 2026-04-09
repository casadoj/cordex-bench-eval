import sys
from pathlib import Path

# Add repo root for imports (same pattern as bonn_meeting/generate_all.py)
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cordex_bench_eval import SubmissionCatalog, TableGenerator
from cordex_bench_eval.config import load_config_file

# Paths
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.json"
TABLES_CONFIG_PATH = ROOT / "configs" / "tables.json"
TABLES_DIR = ROOT / "output"

# Domains to include
DOMAINS = ["ALPS_Domain", "NZ_Domain", "SA_Domain"]

# Submissions to include (set to None to use all entries from the catalog)
submission_names = ['DeepESD-IFCAv1', 'DeepESD-IFCAv1-orog',
                    'DeepESDcrps_IFCAv2', 'DeepESDcrps_IFCAv2-orog',
                    'ViT-IFCAv1', 'ViT-IFCAv1-orog',
                    'FlowMatching-v1', 'FlowMatching-v1-orog',
                    'ANN', 'ANN-orog',
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
                    'ParamUNET',
                    'ResGAN-v2', 'ResGAN-v2-orog',
                    'DetUNet-v2', 'DetUNet-v2-orog']

# Export formats: any of "csv", "html", "latex"
FORMATS = ("html",)

# Include AV (average) row per metric
INCLUDE_AVERAGE = True

# Table names from configs/tables.json — edit this list to choose what to build
TABLE_NAMES = ["tasmax_emulator_hist_future_imperfect_interpolation_gcm_transferability",
               "pr_emulator_hist_future_imperfect_interpolation_gcm_transferability"]


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config_file(str(CONFIG_PATH))
    catalog = SubmissionCatalog(config.catalog_path)

    table_gen = TableGenerator(config=config,
                               catalog=catalog,
                               table_config_path=str(TABLES_CONFIG_PATH))

    print(f"Generating {len(TABLE_NAMES)} table(s) → {TABLES_DIR}/")
    for name in TABLE_NAMES:
        df = table_gen.generate_table(table_name=name,
                                      submission_names=submission_names,
                                      domains=DOMAINS,
                                      include_average=INCLUDE_AVERAGE)
        for fmt in FORMATS:
            out = TABLES_DIR / f"{name}.{fmt}"
            table_gen.export_table(df=df, output_path=str(out), format=fmt)
        print(f"  {name}")

if __name__ == "__main__":
    main()
