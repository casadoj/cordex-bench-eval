import sys
import json
from pathlib import Path
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cordex_bench_eval import SubmissionCatalog, TableGenerator
from cordex_bench_eval.config import load_config_file

### Configuration ###

TABLE_NAME = "tasmax_esd_pseudo_reality_perfect_extrapolation"

X_METRIC = "Mean_bias"
Y_METRIC = "TXX_bias"

DOMAINS = ["ALPS_Domain", "NZ_Domain", "SA_Domain"]

SUBMISSION_NAMES = ['DeepESD-IFCAv1', 'DeepESD-IFCAv1-orog',
                    'ViT-IFCAv1', 'ViT-IFCAv1-orog',
                    'FlowMatching-v1', 'FlowMatching-v1-orog',
                    # 'ANN', 'ANN-orog',
                    'CorrDiff-TW1_NTNU-CWA', 'CorrDiff-TW1_NTNU-CWA-orog',
                    'Prithvi-UNet', 'Prithvi-UNet-orog',
                    'ResGAN', 'ResGAN-orog',
                    'Rossby-UNet', 'Rossby-UNet-orog',
                    'EnScale', 'EnScale-orog',
                    'spaGAN', 'spaGAN-orog',
                    'DetUNet', 'DetUNet-orog',
                    'ResGAN-v2', 'ResGAN-v2-orog',
                    'DetUNet-v2', 'DetUNet-v2-orog']
#####################

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.json"
TABLES_CONFIG_PATH = PROJECT_ROOT / "artifacts" / "tables" / "configs" / "tables.json"
FIGS_DIR = ROOT / "figs"

DOMAIN_SHORT = {"ALPS_Domain": "ALPS",
                "NZ_Domain": "NZ",
                "SA_Domain": "SA"}


def _base_name(name, uses_orog):
    """Compute the model family base name for grouping."""
    if not uses_orog:
        return name
    if name.endswith("-orog"):
        return name[:-5]
    if name.endswith("_orog"):
        return name[:-5]
    if "_orog_" in name:
        return name.replace("_orog_", "_", 1)
    return name


def main():
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config_file(str(CONFIG_PATH))
    catalog = SubmissionCatalog(config.catalog_path)

    orog_flags = {n: catalog.uses_orography(n) for n in SUBMISSION_NAMES}
    base_names = {n: _base_name(n, orog_flags[n]) for n in SUBMISSION_NAMES}

    families = OrderedDict()
    for name in SUBMISSION_NAMES:
        families.setdefault(base_names[name], []).append(name)

    _t20 = plt.cm.tab20(np.arange(20))
    _t20b = plt.cm.tab20b(np.arange(20))
    _all_colors = np.vstack([_t20, _t20b])
    family_color = {base: _all_colors[i % len(_all_colors)]
                    for i, base in enumerate(families)}

    table_gen = TableGenerator(config=config,
                               catalog=catalog,
                               table_config_path=str(TABLES_CONFIG_PATH))

    df = table_gen.generate_table(table_name=TABLE_NAME,
                                  submission_names=SUBMISSION_NAMES,
                                  domains=DOMAINS,
                                  include_average=False)

    domain_labels = [DOMAIN_SHORT.get(d, d) for d in DOMAINS]

    fig, axes = plt.subplots(len(DOMAINS), 1, figsize=(5, 3.5 * len(DOMAINS)),
                             sharex=False, sharey=False)
    if len(DOMAINS) == 1:
        axes = [axes]

    for ax, dlabel in zip(axes, domain_labels):
        for name in SUBMISSION_NAMES:
            base = base_names[name]
            x = df.loc[(X_METRIC, dlabel), name]
            y = df.loc[(Y_METRIC, dlabel), name]
            if np.isnan(x) or np.isnan(y):
                continue
            marker = "o" if orog_flags[name] else "x"
            size = 50 if orog_flags[name] else 60
            ax.scatter(x, y, color=family_color[base],
                       marker=marker, s=size, zorder=3)

        ax.set_xlabel(X_METRIC)
        ax.set_ylabel(Y_METRIC)
        ax.set_title(dlabel)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")

    model_handles = [
        mlines.Line2D([], [], color=family_color[base], marker="s",
                       linestyle="None", markersize=7, label=base)
        for base in families
    ]
    orog_handle = mlines.Line2D([], [], color="black", marker="o",
                                 linestyle="None", markersize=7, label="orog")
    no_orog_handle = mlines.Line2D([], [], color="black", marker="x",
                                    linestyle="None", markersize=7, label="no orog")

    all_handles = model_handles + [orog_handle, no_orog_handle]
    fig.legend(handles=all_handles, loc="lower center",
               ncol=min(len(all_handles), 4),
               bbox_to_anchor=(0.5, -0.02), frameon=True)

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    out_path = FIGS_DIR / f"{TABLE_NAME}_orog.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
