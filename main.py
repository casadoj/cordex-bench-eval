from cordex_bench_eval import SubmissionHandler, SubmissionCatalog, load_config_file, SubmissionDataLoader

# Load config file
config = load_config_file("/gpfs/projects/meteo/WORK/gonzabad/cordex-bench-eval/configs/config.json")

# Create submission handler and ingest submissions
# THis is onyl required for the first time to ingest the submissions
handler = SubmissionHandler(config)
ingested = handler.ingest()

# Create submission catalog and print all entries
catalog = SubmissionCatalog(config.catalog_path)
print(catalog.all_entries())

# Create submission data loader
loader = SubmissionDataLoader(catalog)

# Entry to load
entry_name = "DeepESD-v1_IFCA_m1"

# List available domains, experiments, variables, and predictions
print(loader.available_domains(entry_name))
print(loader.available_experiments(entry_name, "ALPS_Domain"))
print(loader.available_variables(entry_name, "ALPS_Domain", "Emulator_Hist_Future_NO_OROG"))
print(loader.available_predictions(entry_name, "ALPS_Domain", "Emulator_Hist_Future_NO_OROG", "pr"))

# Load dataset
ds = loader.load_dataset(submission_name=entry_name,
                         domain="NZ_Domain",
                         experiment="Emulator_Hist_Future_NO_OROG",
                         variable="pr",
                         prediction="Extrapolation_perfect")
print(ds)