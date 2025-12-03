from cordex_bench_eval import SubmissionHandler, SubmissionCatalog, load_config_file

config = load_config_file("/gpfs/projects/meteo/WORK/gonzabad/cordex-bench-eval/configs/config.json")
handler = SubmissionHandler(config)
ingested = handler.ingest()

catalog = SubmissionCatalog(config.catalog_path)
print(catalog.all_entries())