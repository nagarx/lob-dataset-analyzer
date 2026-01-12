"""
Data validation utilities for LOB datasets.

Re-exports validation functions from the analysis module for convenience.
This module provides a focused interface for data quality validation.

Usage:
    >>> from lobanalyzer.validation import (
    ...     validate_file_structure,
    ...     compute_data_quality,
    ...     compute_shape_validation,
    ... )
    >>> 
    >>> # Validate file structure
    >>> inventories = validate_file_structure(Path("data/exports/nvda_balanced"))
    >>> 
    >>> # Check data quality
    >>> quality = compute_data_quality(features)
    >>> assert quality.is_clean, f"NaN columns: {quality.columns_with_nan}"
"""

from lobanalyzer.analysis.data_overview import (
    # File validation
    FileInventory,
    discover_files,
    validate_file_structure,
    # Shape validation
    ShapeValidation,
    compute_shape_validation,
    # Data quality
    DataQuality,
    compute_data_quality,
    # Categorical validation
    CategoricalValidation,
    validate_categorical_feature,
    compute_all_categorical_validations,
)

__all__ = [
    "FileInventory",
    "discover_files",
    "validate_file_structure",
    "ShapeValidation",
    "compute_shape_validation",
    "DataQuality",
    "compute_data_quality",
    "CategoricalValidation",
    "validate_categorical_feature",
    "compute_all_categorical_validations",
]
