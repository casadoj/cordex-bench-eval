"""
Script to generate spatial diagnostic figures for pr across three domains.
"""

from cordex_bench_eval import (
    SubmissionCatalog,
    Evaluator,
    FigureGenerator,
)
from cordex_bench_eval.config import load_config_file

# Set the configuration paths
CONFIG_PATH = "/lustre/gmeteo/WORK/abadj/cordex-bench-eval/configs/config.json"
METRICS_PATH = "/lustre/gmeteo/WORK/abadj/cordex-bench-eval/configs/metrics.json"
FIGURES_CONFIG_PATH = "/lustre/gmeteo/WORK/abadj/cordex-bench-eval/configs/figures.json"
OUTPUT_DIR = "/lustre/gmeteo/WORK/abadj/cordex-bench-eval/figs/figs_pdf"

def main():

    # Load configuration and catalogues
    config = load_config_file(CONFIG_PATH)
    catalog = SubmissionCatalog(config.catalog_path)
    
    # Set generator
    fig_gen = FigureGenerator(config=config,
                              catalog=catalog,
                              figure_config_path=FIGURES_CONFIG_PATH)
    
    # Select submissions to include
    all_submissions = [entry.name for entry in catalog.all_entries()]
    
    # Set figures to generate
    pr_figures = ["alps_pr_daily_snapshot",
                  "sa_pr_daily_snapshot",
                  "nz_pr_daily_snapshot"]
    
    for figure_name in pr_figures:  
        path = fig_gen.generate_figure(figure_name=figure_name,
                                       submission_names=all_submissions,
                                       target_date="1985-04-04",
                                       output_dir=OUTPUT_DIR)
        print(f"Saved: {path}")

if __name__ == "__main__":
    main()

