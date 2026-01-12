"""
Memory-efficient streaming iterators for LOB datasets.

This module provides generators for iterating over datasets one day at a time,
keeping memory usage bounded regardless of dataset size.

Design Principles:
    - Process one day at a time (never load entire dataset into memory)
    - Explicit memory management with gc.collect() after each day
    - Support both export formats (aligned and legacy)
    - Support memory-mapped files for even lower memory usage

Functions:
    iter_days: Iterate over raw day data
    iter_days_aligned: Iterate over aligned day data (for correlation analysis)
    count_days: Count days without loading data
    get_dates: Get sorted list of dates without loading data

Memory Budget:
    Target: < 4GB for any dataset size
    One day of data: ~100K samples × 98 features × 4 bytes = ~40MB (float32)
"""

import gc
import warnings
import numpy as np
from pathlib import Path
from typing import Generator, List, Optional

from .day_data import (
    DayData,
    AlignedDayData,
    align_features_for_day,
    WINDOW_SIZE,
    STRIDE,
)


def _detect_format(split_dir: Path) -> str:
    """
    Detect export format: 'aligned' (*_sequences.npy) or 'legacy' (*_features.npy).
    
    Args:
        split_dir: Path to split directory (train/, val/, or test/)
    
    Returns:
        'aligned' or 'legacy'
    
    Raises:
        ValueError: If no data files found
    """
    if list(split_dir.glob('*_sequences.npy')):
        return 'aligned'
    elif list(split_dir.glob('*_features.npy')):
        return 'legacy'
    else:
        raise ValueError(f"No data files found in {split_dir}")


def iter_days(
    data_dir: Path,
    split: str,
    dtype: np.dtype = np.float32,
    mmap_mode: Optional[str] = None,
) -> Generator[DayData, None, None]:
    """
    Iterate over days in a split, yielding one day at a time.
    
    This is the primary memory-efficient data access pattern.
    Each day is loaded, processed, then freed before the next.
    
    Automatically handles both export formats:
        - ALIGNED: *_sequences.npy [N_seq, 100, 98] - extracts last timestep
        - LEGACY: *_features.npy [N_samples, 98]
    
    Args:
        data_dir: Path to dataset root (e.g., data/exports/nvda_balanced)
        split: One of 'train', 'val', 'test'
        dtype: Data type for features (default: float32 for memory efficiency)
        mmap_mode: If 'r', use memory-mapped files (read-only, even more efficient)
    
    Yields:
        DayData for each day in chronological order
    
    Raises:
        FileNotFoundError: If split directory doesn't exist
    
    Example:
        >>> for day in iter_days(Path("data/exports/nvda_balanced"), "train"):
        ...     labels = day.get_labels(0)  # First horizon
        ...     print(f"{day.date}: {day.n_samples} samples")
        ...     # Memory freed after each iteration
    
    Memory Usage:
        ~40MB per day (float32, 100K samples × 98 features)
    """
    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    export_format = _detect_format(split_dir)
    
    if export_format == 'aligned':
        data_files = sorted(split_dir.glob('*_sequences.npy'))
        suffix = '_sequences'
    else:
        data_files = sorted(split_dir.glob('*_features.npy'))
        suffix = '_features'
    
    for data_file in data_files:
        date = data_file.stem.replace(suffix, '')
        label_file = data_file.parent / f"{date}_labels.npy"
        
        if not label_file.exists():
            warnings.warn(f"Label file not found for {date}, skipping")
            continue
        
        # Load with specified dtype and mmap mode
        if mmap_mode:
            raw_features = np.load(data_file, mmap_mode=mmap_mode)
            labels = np.load(label_file, mmap_mode=mmap_mode)
        else:
            raw_features = np.load(data_file)
            labels = np.load(label_file)
        
        # Handle 3D sequences: extract last timestep
        if len(raw_features.shape) == 3:
            # [N_seq, window_size, n_features] -> [N_seq, n_features]
            features = raw_features[:, -1, :].astype(dtype, copy=False)
        else:
            features = raw_features.astype(dtype, copy=False)
        
        # Detect multi-horizon labels
        is_multi_horizon = labels.ndim == 2
        num_horizons = labels.shape[1] if is_multi_horizon else 1
        
        yield DayData(
            date=date,
            features=features,
            labels=labels,
            n_samples=features.shape[0],
            n_labels=labels.shape[0],
            is_multi_horizon=is_multi_horizon,
            num_horizons=num_horizons,
        )
        
        # Explicit cleanup (important for memory)
        if not mmap_mode:
            del features, labels, raw_features
            gc.collect()


def iter_days_aligned(
    data_dir: Path,
    split: str,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    dtype: np.dtype = np.float32,
) -> Generator[AlignedDayData, None, None]:
    """
    Iterate over days, yielding ALIGNED feature-label pairs.
    
    This is the memory-efficient equivalent of load_split_aligned().
    Use this for any analysis that needs signal-label correlation.
    
    CRITICAL: This function performs correct per-day alignment, avoiding the
    day-boundary drift that occurs with global alignment on concatenated data.
    
    Args:
        data_dir: Path to dataset root (e.g., data/exports/nvda_balanced)
        split: One of 'train', 'val', 'test'
        window_size: Samples per sequence window (default: 100)
        stride: Samples between sequence starts (default: 10)
        dtype: Data type for features (default: float32)
    
    Yields:
        AlignedDayData for each day in chronological order
    
    Raises:
        FileNotFoundError: If split directory doesn't exist
    
    Example:
        >>> for day in iter_days_aligned(Path("data/exports/nvda_balanced"), "train"):
        ...     # day.features[i] corresponds to day.labels[i]
        ...     labels = day.get_labels(0)
        ...     corr = np.corrcoef(day.features[:, 84], labels)[0, 1]
        ...     print(f"{day.date}: OFI correlation = {corr:.4f}")
    
    Memory Usage:
        ~40MB per day (float32) - same as iter_days() but with aligned features
    """
    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    export_format = _detect_format(split_dir)
    
    if export_format == 'aligned':
        data_files = sorted(split_dir.glob('*_sequences.npy'))
        suffix = '_sequences'
    else:
        data_files = sorted(split_dir.glob('*_features.npy'))
        suffix = '_features'
    
    for data_file in data_files:
        date = data_file.stem.replace(suffix, '')
        label_file = data_file.parent / f"{date}_labels.npy"
        
        if not label_file.exists():
            warnings.warn(f"Label file not found for {date}, skipping")
            continue
        
        # Load raw data
        raw_data = np.load(data_file)
        labels = np.load(label_file)
        n_labels = labels.shape[0]
        
        # Detect multi-horizon labels
        is_multi_horizon = labels.ndim == 2
        num_horizons = labels.shape[1] if is_multi_horizon else 1
        
        if export_format == 'aligned':
            # ALIGNED format: 3D sequences [N_seq, 100, 98] - extract last timestep
            if len(raw_data.shape) == 3:
                aligned_features = raw_data[:, -1, :].astype(dtype, copy=False)
            else:
                aligned_features = raw_data.astype(dtype, copy=False)
        else:
            # LEGACY format: Need to align
            features = raw_data.astype(dtype, copy=False)
            aligned_features = align_features_for_day(
                features, n_labels, window_size, stride
            )
            del features
        
        yield AlignedDayData(
            date=date,
            features=aligned_features,
            labels=labels,
            n_pairs=n_labels,
            is_multi_horizon=is_multi_horizon,
            num_horizons=num_horizons,
        )
        
        # Explicit cleanup
        del raw_data, aligned_features, labels
        gc.collect()


def count_days(data_dir: Path, split: str) -> int:
    """
    Count days in a split without loading data.
    
    Args:
        data_dir: Path to dataset root
        split: One of 'train', 'val', 'test'
    
    Returns:
        Number of days in the split
    
    Example:
        >>> n_train = count_days(Path("data/exports/nvda_balanced"), "train")
        >>> print(f"Training days: {n_train}")
    """
    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        return 0
    
    # Try new format first, then legacy
    seq_files = list(split_dir.glob('*_sequences.npy'))
    if seq_files:
        return len(seq_files)
    return len(list(split_dir.glob('*_features.npy')))


def get_dates(data_dir: Path, split: str) -> List[str]:
    """
    Get sorted list of dates in a split without loading data.
    
    Args:
        data_dir: Path to dataset root
        split: One of 'train', 'val', 'test'
    
    Returns:
        List of date strings in chronological order
    
    Example:
        >>> dates = get_dates(Path("data/exports/nvda_balanced"), "train")
        >>> print(f"First day: {dates[0]}, Last day: {dates[-1]}")
    """
    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        return []
    
    # Try new format first
    seq_files = sorted(split_dir.glob('*_sequences.npy'))
    if seq_files:
        return [f.stem.replace('_sequences', '') for f in seq_files]
    
    # Fall back to legacy
    feat_files = sorted(split_dir.glob('*_features.npy'))
    return [f.stem.replace('_features', '') for f in feat_files]
