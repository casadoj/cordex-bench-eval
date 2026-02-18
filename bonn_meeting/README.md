# Bonn Meeting Outputs

Generate figures and tables for meeting submissions.

## Structure

| Folder              | Content                          |
|---------------------|----------------------------------|
| `configs/`          | Local tables, figures, diagnostic configs |
| `figures/`          | Daily snapshot maps (pr)         |
| `diagnostic_figures/` | Spatial metric maps (RMSE, TXX, etc.) |
| `tables/`           | Comparison tables (CSV, HTML)    |

## Usage

1. Compute diagnostics (metrics; run first, can be slow):
   ```bash
   python bonn_meeting/compute_diagnostics.py
   ```
2. Generate tables and figures:
   ```bash
   python bonn_meeting/generate_all.py
   ```

## Configurable

- **configs/tables.json** – table definitions and metrics
- **configs/figures.json** – daily snapshot figures
- **configs/diagnostic_figures.json** – spatial metric maps
- **generate_all.py** – `submission_names`, `DOMAINS`, and which table/figure names to generate
