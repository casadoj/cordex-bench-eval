from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt

from final_scorecard import (
    plot_scorecard_all_regions,
    plot_scorecard_combined_rank,
    plot_scorecard_regions_avg,
)


def _save_pdf_png(base_path: Path, render: Callable[[str], None]) -> None:
    for ext in (".pdf", ".png"):
        render(str(base_path.with_suffix(ext)))
        plt.close("all")


def main() -> None:
    out = Path(__file__).resolve().parent
    out.mkdir(exist_ok=True)

    for var in ("pr", "tasmax"):
        _save_pdf_png(
            out / f"scorecard_{var}_all_regions",
            lambda save_path, v=var: plot_scorecard_all_regions(variable=v, save_path=save_path),
        )
        _save_pdf_png(
            out / f"scorecard_{var}_regions_avg",
            lambda save_path, v=var: plot_scorecard_regions_avg(variable=v, save_path=save_path),
        )

    _save_pdf_png(
        out / "scorecard_combined_overall_rank",
        lambda save_path: plot_scorecard_combined_rank(save_path=save_path),
    )


if __name__ == "__main__":
    main()
