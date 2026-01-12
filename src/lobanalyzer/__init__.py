"""
LOB Dataset Analyzer - Statistical analysis for limit order book data.

A library for analyzing LOB feature data exported from the feature-extractor-MBO-LOB
Rust pipeline. Provides streaming analysis, data validation, and signal-label
correlation analysis.

Modules:
    constants: Feature indices, label encoding, schema version
    streaming: Memory-efficient day-by-day data iteration
    analysis: Statistical analysis functions
    validation: Data quality validation utilities

Quick Start:
    >>> from lobanalyzer import FEATURE_COUNT, SCHEMA_VERSION
    >>> from lobanalyzer.streaming import iter_days, DayData
    >>> from lobanalyzer.analysis import compute_data_quality
    >>> 
    >>> # Iterate over data one day at a time
    >>> for day in iter_days(Path("data/exports/nvda_balanced"), "train"):
    ...     print(f"Day {day.date}: {day.n_samples} samples")
    >>> 
    >>> # Validate data quality
    >>> quality = compute_data_quality(features)
    >>> print(f"Clean: {quality.is_clean}")

Version:
    Schema 2.2 - 98 features (40 LOB + 8 derived + 36 MBO + 14 signals)
"""

__version__ = "0.2.0"

# Re-export key constants for convenient access
from lobanalyzer.constants import (
    # Feature counts
    FEATURE_COUNT,
    LOB_FEATURE_COUNT,
    DERIVED_FEATURE_COUNT,
    MBO_FEATURE_COUNT,
    SIGNAL_FEATURE_COUNT,
    # Schema version
    SCHEMA_VERSION,
    # Feature indices
    FeatureIndex,
    SignalIndex,
    # Label encoding
    LABEL_DOWN,
    LABEL_STABLE,
    LABEL_UP,
    LABEL_NAMES,
    NUM_CLASSES,
)

__all__ = [
    # Version
    "__version__",
    # Feature counts
    "FEATURE_COUNT",
    "LOB_FEATURE_COUNT",
    "DERIVED_FEATURE_COUNT",
    "MBO_FEATURE_COUNT",
    "SIGNAL_FEATURE_COUNT",
    # Schema version
    "SCHEMA_VERSION",
    # Feature indices
    "FeatureIndex",
    "SignalIndex",
    # Label encoding
    "LABEL_DOWN",
    "LABEL_STABLE",
    "LABEL_UP",
    "LABEL_NAMES",
    "NUM_CLASSES",
]
