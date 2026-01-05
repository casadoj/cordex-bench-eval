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

TableGenerator
    Generates comparison tables from computed metrics.

FigureGenerator
    Creates daily figures that compare predictions against ground truth.
"""

from .catalog import CatalogEntry, SubmissionCatalog
from .config import SubmissionWorkflowConfig, load_config_dict, load_config_file
from .data_loader import GroundTruthDataLoader, SubmissionDataLoader
from .evaluator import Evaluator
from .figure_generator import FigureGenerator
from .submission_handler import SubmissionHandler
from .table_generator import TableGenerator

__all__ = [# Configuration
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
           # Figures
           "FigureGenerator",
           # Table generation
           "TableGenerator"]
