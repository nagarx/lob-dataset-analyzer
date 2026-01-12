"""
High-level streaming convenience functions for dataset analysis.

These functions provide simple, memory-efficient interfaces for common
analysis tasks by wrapping the lower-level streaming iterators.

Design Principles:
    - O(n_features) memory: constant regardless of dataset size
    - Automatic cleanup: gc.collect() after each day
    - Multi-horizon support: use get_labels(0) for backward compatibility

Usage:
    >>> from lobanalyzer.streaming import compute_streaming_overview
    >>> 
    >>> # Get complete dataset overview
    >>> overview = compute_streaming_overview(Path("data/exports/nvda"), symbol="NVDA")
    >>> print(f"Total: {overview['total_samples']:,} samples across {overview['total_days']} days")
    >>> 
    >>> # Analyze label distribution and autocorrelation
    >>> labels = compute_streaming_label_analysis(Path("data/exports/nvda"), split="train")
    >>> print(f"Label ACF(1) = {labels['autocorrelation']['lag_1']:.4f}")
"""

import gc
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

from lobanalyzer.streaming.iterators import iter_days
from lobanalyzer.streaming.day_data import WINDOW_SIZE, STRIDE
from lobanalyzer.analysis.streaming_stats import (
    RunningStats,
    StreamingColumnStats,
    StreamingLabelCounter,
    StreamingDataQuality,
)


def compute_streaming_overview(
    data_dir: Path,
    symbol: str = "UNKNOWN",
    dtype: np.dtype = np.float32,
) -> Dict[str, Any]:
    """
    Compute dataset overview with streaming (memory-efficient).
    
    Memory usage: O(n_features) - constant regardless of dataset size.
    
    Args:
        data_dir: Path to dataset root
        symbol: Symbol name
        dtype: Data type for loading
    
    Returns:
        Dict with overview statistics:
            - symbol: Symbol name
            - data_dir: Path to data
            - date_range: (start_date, end_date) tuple
            - total_days: Total days across all splits
            - train_days, val_days, test_days: Days per split
            - total_samples: Total feature rows
            - total_labels: Total labels
            - feature_count: Number of features (98)
            - data_quality: Dict with NaN/Inf counts
            - label_distribution: Dict with Up/Down/Stable counts
            - signal_stats: Dict with per-column statistics
    
    Example:
        >>> overview = compute_streaming_overview(Path("data/nvda"), "NVDA")
        >>> print(f"Total samples: {overview['total_samples']:,}")
        >>> print(f"Clean data: {overview['data_quality']['is_clean']}")
    """
    data_dir = Path(data_dir)
    
    # Initialize streaming counters
    n_features = 98  # Known from schema
    column_stats = StreamingColumnStats(n_columns=n_features)
    label_counter = StreamingLabelCounter()
    data_quality = StreamingDataQuality()
    
    all_dates = []
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    total_samples = 0
    
    # Process each split
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        
        day_count = 0
        for day in iter_days(data_dir, split, dtype=dtype):
            # Update streaming statistics
            column_stats.update_batch(day.features)
            # Use first horizon for multi-horizon data (backward compatible)
            label_counter.update(day.get_labels(0))
            data_quality.update(day.features)
            
            all_dates.append(day.date)
            total_samples += day.n_samples
            day_count += 1
            
            # Memory freed automatically after each day
        
        split_counts[split] = day_count
    
    # Build result
    sorted_dates = sorted(all_dates)
    
    return {
        'symbol': symbol,
        'data_dir': str(data_dir),
        'date_range': (sorted_dates[0], sorted_dates[-1]) if sorted_dates else (None, None),
        'total_days': len(all_dates),
        'train_days': split_counts['train'],
        'val_days': split_counts['val'],
        'test_days': split_counts['test'],
        'total_samples': total_samples,
        'total_labels': label_counter.total,
        'feature_count': n_features,
        'data_quality': {
            'total_values': data_quality.total_values,
            'finite_count': data_quality.finite_count,
            'nan_count': data_quality.nan_count,
            'inf_count': data_quality.inf_count,
            'is_clean': data_quality.is_clean,
            'columns_with_nan': sorted(data_quality.columns_with_nan),
            'columns_with_inf': sorted(data_quality.columns_with_inf),
        },
        'label_distribution': {
            'total': label_counter.total,
            'down_count': label_counter.down_count,
            'stable_count': label_counter.stable_count,
            'up_count': label_counter.up_count,
            'down_pct': label_counter.down_pct,
            'stable_pct': label_counter.stable_pct,
            'up_pct': label_counter.up_pct,
        },
        'signal_stats': column_stats.get_summary(),
    }


def _compute_acf(labels: np.ndarray, max_lag: int = 100) -> np.ndarray:
    """
    Compute autocorrelation function efficiently.
    
    Args:
        labels: 1D label array
        max_lag: Maximum lag to compute
    
    Returns:
        ACF values from lag 0 to max_lag
    """
    n = len(labels)
    labels_float = labels.astype(np.float32)
    mean = labels_float.mean()
    var = labels_float.var()
    
    if var == 0:
        return np.ones(min(max_lag + 1, n))
    
    acf = np.zeros(min(max_lag + 1, n))
    acf[0] = 1.0
    
    # Use numpy vectorization for efficiency
    for lag in range(1, len(acf)):
        cov = np.mean((labels_float[:-lag] - mean) * (labels_float[lag:] - mean))
        acf[lag] = cov / var
    
    return acf


def compute_streaming_label_analysis(
    data_dir: Path,
    split: str = 'train',
    dtype: np.dtype = np.float32,
    max_samples_for_acf: int = 100000,
) -> Dict[str, Any]:
    """
    Compute label analysis with streaming.
    
    Some analyses (like autocorrelation) require all labels in memory,
    but labels are small (1 byte per sample typically).
    
    Args:
        data_dir: Path to dataset root
        split: Which split to analyze
        dtype: Data type for loading features (labels are always int8)
        max_samples_for_acf: Maximum samples to use for ACF computation
    
    Returns:
        Dict with label analysis results:
            - split: Split name
            - date_range: (start_date, end_date)
            - n_days: Number of days
            - distribution: Dict with Up/Down/Stable counts and percentages
            - autocorrelation: Dict with lag-1/5/10 ACF and first 20 values
            - transition_matrix: Dict with label transition probabilities
            - day_stats: List of per-day statistics
    
    Example:
        >>> labels = compute_streaming_label_analysis(Path("data/nvda"), "train")
        >>> print(f"ACF(1) = {labels['autocorrelation']['lag_1']:.4f}")
        >>> print(f"Stable pct = {labels['distribution']['stable_pct']:.1f}%")
    """
    data_dir = Path(data_dir)
    
    # Labels are small enough to collect
    all_labels = []
    dates = []
    
    # Streaming counters for per-day stats
    day_stats = []
    transition_counts = np.zeros((3, 3), dtype=np.int64)
    label_map = {-1: 0, 0: 1, 1: 2}
    
    for day in iter_days(data_dir, split, dtype=dtype):
        # Use first horizon for multi-horizon data (backward compatible)
        day_labels = day.get_labels(0)
        all_labels.append(day_labels)
        dates.append(day.date)
        
        # Per-day distribution
        day_stats.append({
            'date': day.date,
            'n_labels': len(day_labels),
            'up_pct': float(100 * (day_labels == 1).mean()),
            'down_pct': float(100 * (day_labels == -1).mean()),
            'stable_pct': float(100 * (day_labels == 0).mean()),
        })
        
        # Update transition counts
        for i in range(len(day_labels) - 1):
            from_idx = label_map.get(day_labels[i], 1)
            to_idx = label_map.get(day_labels[i + 1], 1)
            transition_counts[from_idx, to_idx] += 1
    
    # Concatenate labels (small memory footprint)
    labels = np.concatenate(all_labels)
    del all_labels
    gc.collect()
    
    # Compute distribution
    total = len(labels)
    distribution = {
        'total': total,
        'down_count': int((labels == -1).sum()),
        'stable_count': int((labels == 0).sum()),
        'up_count': int((labels == 1).sum()),
    }
    distribution['down_pct'] = 100 * distribution['down_count'] / total
    distribution['stable_pct'] = 100 * distribution['stable_count'] / total
    distribution['up_pct'] = 100 * distribution['up_count'] / total
    
    # Autocorrelation (subsample if needed)
    if len(labels) > max_samples_for_acf:
        step = len(labels) // max_samples_for_acf
        labels_for_acf = labels[::step][:max_samples_for_acf]
    else:
        labels_for_acf = labels
    
    acf = _compute_acf(labels_for_acf, max_lag=100)
    
    # Transition probabilities
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    transition_probs = np.divide(
        transition_counts, row_sums, 
        where=row_sums > 0, 
        out=np.zeros_like(transition_counts, dtype=float)
    )
    
    return {
        'split': split,
        'date_range': (dates[0], dates[-1]) if dates else (None, None),
        'n_days': len(dates),
        'distribution': distribution,
        'autocorrelation': {
            'lag_1': float(acf[1]) if len(acf) > 1 else 0.0,
            'lag_5': float(acf[5]) if len(acf) > 5 else 0.0,
            'lag_10': float(acf[10]) if len(acf) > 10 else 0.0,
            'acf_values': acf[:20].tolist(),  # First 20 lags
        },
        'transition_matrix': {
            'labels': [-1, 0, 1],
            'probabilities': transition_probs.tolist(),
        },
        'day_stats': day_stats,
    }


def compute_streaming_signal_stats(
    data_dir: Path,
    split: str = 'train',
    signal_indices: Optional[List[int]] = None,
    dtype: np.dtype = np.float32,
) -> Dict[str, Any]:
    """
    Compute signal statistics with streaming.
    
    Memory: O(n_signals) - constant regardless of dataset size.
    
    Args:
        data_dir: Path to dataset root
        split: Which split to analyze
        signal_indices: Which signals to analyze (default: 84-91 core signals)
        dtype: Data type for loading
    
    Returns:
        Dict mapping signal name to statistics:
            - index: Feature index
            - n: Number of samples
            - mean: Signal mean
            - std: Signal standard deviation
            - min: Minimum value
            - max: Maximum value
    
    Example:
        >>> stats = compute_streaming_signal_stats(Path("data/nvda"), "train")
        >>> print(f"OFI mean: {stats['true_ofi']['mean']:.4f}")
    """
    if signal_indices is None:
        signal_indices = list(range(84, 92))  # Core signals
    
    signal_names = {
        84: 'true_ofi',
        85: 'depth_norm_ofi',
        86: 'executed_pressure',
        87: 'signed_mp_delta_bps',
        88: 'trade_asymmetry',
        89: 'cancel_asymmetry',
        90: 'fragility_score',
        91: 'depth_asymmetry',
    }
    
    # Initialize streaming stats for each signal
    signal_stats = {idx: RunningStats() for idx in signal_indices}
    
    # Process each day
    for day in iter_days(data_dir, split, dtype=dtype):
        for idx in signal_indices:
            col = day.features[:, idx]
            finite_mask = np.isfinite(col)
            if finite_mask.any():
                for val in col[finite_mask]:
                    signal_stats[idx].update(float(val))
    
    # Build results
    results = {}
    for idx, stats in signal_stats.items():
        name = signal_names.get(idx, f'signal_{idx}')
        results[name] = {
            'index': idx,
            'n': stats.n,
            'mean': stats.mean,
            'std': stats.std,
            'min': stats.min_val if stats.n > 0 else None,
            'max': stats.max_val if stats.n > 0 else None,
        }
    
    return results


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def estimate_memory_usage(
    data_dir: Path,
    dtype: np.dtype = np.float32,
) -> Dict[str, Any]:
    """
    Estimate memory usage without loading data.
    
    Handles both export formats:
    - NEW aligned: *_sequences.npy [N_seq, 100, 98]
    - LEGACY: *_features.npy [N_samples, 98]
    
    Args:
        data_dir: Path to dataset root
        dtype: Data type that would be used
    
    Returns:
        Dict with memory estimates per split:
            - samples: Number of samples
            - bytes: Memory in bytes
            - mb: Memory in megabytes
            - gb: Memory in gigabytes
    
    Example:
        >>> mem = estimate_memory_usage(Path("data/nvda"))
        >>> print(f"Total: {mem['total']['gb']:.2f} GB")
    """
    data_dir = Path(data_dir)
    bytes_per_element = np.dtype(dtype).itemsize
    n_features = 98
    
    estimates = {}
    total_samples = 0
    
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        
        # Count samples from file sizes - try new format first
        split_samples = 0
        seq_files = list(split_dir.glob('*_sequences.npy'))
        
        if seq_files:
            # NEW format: shape is [N_seq, window_size, n_features]
            for seq_file in seq_files:
                with open(seq_file, 'rb') as f:
                    np.lib.format.read_magic(f)
                    shape, _, _ = np.lib.format.read_array_header_1_0(f)
                    split_samples += shape[0]  # N_seq (number of sequences)
        else:
            # LEGACY format: shape is [N_samples, n_features]
            for feat_file in split_dir.glob('*_features.npy'):
                with open(feat_file, 'rb') as f:
                    np.lib.format.read_magic(f)
                    shape, _, _ = np.lib.format.read_array_header_1_0(f)
                    split_samples += shape[0]
        
        bytes_needed = split_samples * n_features * bytes_per_element
        estimates[split] = {
            'samples': split_samples,
            'bytes': bytes_needed,
            'mb': bytes_needed / (1024 * 1024),
            'gb': bytes_needed / (1024 * 1024 * 1024),
        }
        total_samples += split_samples
    
    total_bytes = total_samples * n_features * bytes_per_element
    estimates['total'] = {
        'samples': total_samples,
        'bytes': total_bytes,
        'mb': total_bytes / (1024 * 1024),
        'gb': total_bytes / (1024 * 1024 * 1024),
    }
    
    return estimates


def get_memory_efficient_config() -> Dict[str, Any]:
    """
    Get recommended configuration for memory-efficient processing.
    
    Returns:
        Dict with configuration recommendations:
            - dtype: Recommended data type ('float32')
            - mmap_mode: Memory-mapped mode ('r')
            - max_days_in_memory: Days to keep in memory (1)
            - gc_after_each_day: Whether to gc.collect() (True)
            - subsample_for_expensive_ops: Subsample for O(n²) ops (True)
            - max_samples_for_acf: Max samples for autocorrelation (100000)
            - max_samples_for_correlation: Max for correlation (500000)
    
    Example:
        >>> config = get_memory_efficient_config()
        >>> print(f"Recommended dtype: {config['dtype']}")
    """
    return {
        'dtype': 'float32',  # vs float64
        'mmap_mode': 'r',    # Memory-mapped read
        'max_days_in_memory': 1,  # Process one day at a time
        'gc_after_each_day': True,
        'subsample_for_expensive_ops': True,
        'max_samples_for_acf': 100000,
        'max_samples_for_correlation': 500000,
    }
