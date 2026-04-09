"""
CORDEX Benchmark Evaluation Utilities

This package provides tools for evaluating deep learning model predictions
against ground truth data in the CORDEX-ML benchmark.

Key Components
--------------
SubmissionHandler
    Ingests submission archives (.zip) from an incoming directory, extracts
    them to staging, and registers them in the catalog.

SubmissionValidator
    Validates staged submissions for required files, naming, and NetCDF
    structure needed by metrics and figures.

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

DiagnosticFigureGenerator
    Creates spatial maps of computed diagnostics for multiple submissions.

PsdFigureGenerator
    Creates log-log PSD comparison plots across domains for multiple submissions.
"""

from .catalog import CatalogEntry, SubmissionCatalog
from .config import SubmissionWorkflowConfig, load_config_dict, load_config_file
from .data_loader import GroundTruthDataLoader, SubmissionDataLoader
from .diagnostic_figure_generator import DiagnosticFigureGenerator
from .evaluator import Evaluator
from .figure_generator import FigureGenerator
from .psd_figure_generator import PsdFigureGenerator
from .scorecard_generator import ScorecardGenerator
from .submission_handler import SubmissionHandler
from .submission_validator import SubmissionValidator
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
           "SubmissionValidator",
           # Data loading
           "SubmissionDataLoader",
           "GroundTruthDataLoader",
           # Evaluation
           "Evaluator",
           # Figures
           "FigureGenerator",
           "DiagnosticFigureGenerator",
           "PsdFigureGenerator",
           # Table generation
           "TableGenerator",
           "ScorecardGenerator"]
