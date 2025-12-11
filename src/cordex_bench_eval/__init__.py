"""
CORDEX Benchmark Evaluation Utilities

This package provides tools for evaluating deep learning model predictions
against ground truth data in the CORDEX-ML benchmark.

Key Components
--------------
SubmissionHandler
    Ingests submission archives (.zip) from an incoming directory, extracts
    them to staging, and registers them in the catalog.

SubmissionCatalog
    JSON-backed catalog that tracks all registered submissions with metadata.

SubmissionDataLoader
    Loads prediction NetCDF files from staged submissions.

GroundTruthDataLoader
    Loads ground truth data for training and testing.

Evaluator
    Computes configurable metrics on predictions vs ground truth.

Example Usage
-------------
>>> from cordex_bench_eval import SubmissionCatalog, SubmissionDataLoader, Evaluator
>>> from cordex_bench_eval.config import load_config_file
>>>
>>> # Load configuration
>>> config = load_config_file("configs/config.json")
>>>
>>> # Load a prediction
>>> catalog = SubmissionCatalog(config.catalog_path)
>>> loader = SubmissionDataLoader(catalog)
>>> ds = loader.load_dataset(
...     submission_name="DeepESD-v1_IFCA_m1",
...     domain="ALPS_Domain",
...     experiment="Emulator_Hist_Future",
...     use_orography=True,
...     variable="pr",
...     prediction="Extrapolation_perfect",
... )
>>>
>>> # Run evaluation
>>> evaluator = Evaluator(config=config, catalog=catalog, ...)
>>> results = evaluator.evaluate(...)
"""

from .catalog import CatalogEntry, SubmissionCatalog
from .config import SubmissionWorkflowConfig, load_config_dict, load_config_file
from .data_loader import GroundTruthDataLoader, SubmissionDataLoader
from .evaluator import Evaluator
from .submission_handler import SubmissionHandler

__all__ = [
    # Configuration
    "SubmissionWorkflowConfig",
    "load_config_file",
    "load_config_dict",
    # Catalog
    "SubmissionCatalog",
    "CatalogEntry",
    # Ingestion
    "SubmissionHandler",
    # Data loading
    "SubmissionDataLoader",
    "GroundTruthDataLoader",
    # Evaluation
    "Evaluator",
]
