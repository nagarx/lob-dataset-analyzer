"""
Day data containers for memory-efficient analysis.

This module provides containers for holding one day's worth of LOB data,
designed for minimal memory footprint and streaming iteration.

Design Principles:
    - Process one day at a time to keep memory bounded
    - Support both single-horizon and multi-horizon labels
    - Clean separation between raw and aligned data

Classes:
    DayData: Raw day data (features may not align with labels)
    AlignedDayData: Features aligned 1:1 with labels (for correlation analysis)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


# Alignment constants (must match Rust export configuration)
WINDOW_SIZE = 100  # Samples per sequence window
STRIDE = 10        # Samples between sequence starts


@dataclass
class DayData:
    """
    Container for a single day's raw data (minimal memory footprint).
    
    Supports both single-horizon and multi-horizon labels:
        - Single-horizon: labels shape is (M,)
        - Multi-horizon: labels shape is (M, n_horizons)
    
    Attributes:
        date: Date string in YYYY-MM-DD format
        features: (N, 98) feature array for the day
        labels: (M,) or (M, n_horizons) label array
        n_samples: Number of feature samples (N)
        n_labels: Number of labels (M)
        is_multi_horizon: True if labels have multiple horizons
        num_horizons: Number of prediction horizons
    
    Note:
        For aligned format (*_sequences.npy), N == M.
        For legacy format, N > M (features not yet aligned).
    """
    date: str
    features: np.ndarray  # (N, 98)
    labels: np.ndarray    # (M,) or (M, n_horizons)
    n_samples: int
    n_labels: int
    is_multi_horizon: bool = False
    num_horizons: int = 1
    
    @property
    def memory_bytes(self) -> int:
        """Approximate memory usage in bytes."""
        return self.features.nbytes + self.labels.nbytes
    
    @property
    def memory_mb(self) -> float:
        """Memory usage in megabytes."""
        return self.memory_bytes / (1024 * 1024)
    
    def get_labels(self, horizon_idx: Optional[int] = 0) -> np.ndarray:
        """
        Get labels for a specific horizon.
        
        Args:
            horizon_idx: Which horizon to return (0-based).
                         None returns all labels (full array).
                         Default: 0 (first/only horizon)
        
        Returns:
            Label array: (M,) for single horizon, (M, n_horizons) if horizon_idx=None
        
        Raises:
            ValueError: If horizon_idx is out of range
        
        Example:
            >>> labels = day.get_labels(0)  # First horizon
            >>> all_labels = day.get_labels(None)  # All horizons
        """
        if horizon_idx is None:
            return self.labels
        
        if not self.is_multi_horizon:
            if horizon_idx != 0:
                raise ValueError(
                    f"Single-horizon data only has horizon_idx=0, got {horizon_idx}"
                )
            return self.labels
        
        if horizon_idx < 0 or horizon_idx >= self.num_horizons:
            raise ValueError(
                f"horizon_idx {horizon_idx} out of range [0, {self.num_horizons})"
            )
        
        return self.labels[:, horizon_idx]


@dataclass
class AlignedDayData:
    """
    Container for a single day's ALIGNED data.
    
    Features are aligned 1:1 with labels - this is the correct data
    structure for signal-label correlation analysis.
    
    For aligned format (*_sequences.npy), the last timestep of each
    sequence is extracted to align with the corresponding label.
    
    Attributes:
        date: Date string in YYYY-MM-DD format
        features: (N_labels, 98) features aligned with labels
        labels: (N_labels,) or (N_labels, n_horizons) label array
        n_pairs: Number of aligned feature-label pairs
        is_multi_horizon: True if labels have multiple horizons
        num_horizons: Number of prediction horizons
    
    Contract:
        len(features) == len(labels) (for single-horizon)
        len(features) == labels.shape[0] (for multi-horizon)
    """
    date: str
    features: np.ndarray  # (N_labels, 98) - aligned with labels
    labels: np.ndarray    # (N_labels,) or (N_labels, n_horizons)
    n_pairs: int          # Number of aligned feature-label pairs
    is_multi_horizon: bool = False
    num_horizons: int = 1
    
    @property
    def memory_bytes(self) -> int:
        """Approximate memory usage in bytes."""
        return self.features.nbytes + self.labels.nbytes
    
    @property
    def memory_mb(self) -> float:
        """Memory usage in megabytes."""
        return self.memory_bytes / (1024 * 1024)
    
    def get_labels(self, horizon_idx: Optional[int] = 0) -> np.ndarray:
        """
        Get labels for a specific horizon.
        
        Args:
            horizon_idx: Which horizon to return (0-based).
                         None returns all labels (full array).
                         Default: 0 (first/only horizon)
        
        Returns:
            Label array: (N,) for single horizon, (N, n_horizons) if horizon_idx=None
        
        Raises:
            ValueError: If horizon_idx is out of range
        """
        if horizon_idx is None:
            return self.labels
        
        if not self.is_multi_horizon:
            if horizon_idx != 0:
                raise ValueError(
                    f"Single-horizon data only has horizon_idx=0, got {horizon_idx}"
                )
            return self.labels
        
        if horizon_idx < 0 or horizon_idx >= self.num_horizons:
            raise ValueError(
                f"horizon_idx {horizon_idx} out of range [0, {self.num_horizons})"
            )
        
        return self.labels[:, horizon_idx]


def align_features_for_day(
    features: np.ndarray,
    n_labels: int,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> np.ndarray:
    """
    Align features with labels for a SINGLE day (legacy format only).
    
    Each label corresponds to the END of a sequence window.
    We extract the feature vector at the end of each window for alignment.
    
    Formula:
        For label[i], the corresponding feature is at:
            feat_idx = i * stride + window_size - 1
        
        This is the LAST feature in the sequence window:
            [i * stride, i * stride + window_size)
    
    Args:
        features: (N_samples, N_features) array from a single day
        n_labels: Number of labels for this day
        window_size: Samples per sequence window (default: 100)
        stride: Samples between sequence starts (default: 10)
    
    Returns:
        aligned_features: (n_labels, N_features) array
    
    Example (single day with 1000 samples, window=100, stride=10):
        - label[0] → feature[99]   (end of window [0, 100))
        - label[1] → feature[109]  (end of window [10, 110))
        - label[2] → feature[119]  (end of window [20, 120))
        - ...
        - label[90] → feature[999] (end of window [900, 1000))
    
    Note:
        This function is only needed for LEGACY format (*_features.npy).
        ALIGNED format (*_sequences.npy) is already 1:1 aligned.
    """
    n_features = features.shape[1]
    aligned = np.zeros((n_labels, n_features), dtype=features.dtype)
    
    for i in range(n_labels):
        # Feature index at end of sequence window
        feat_idx = i * stride + window_size - 1
        
        # Boundary check (should only happen if n_labels doesn't match formula)
        if feat_idx >= features.shape[0]:
            feat_idx = features.shape[0] - 1
        
        aligned[i] = features[feat_idx]
    
    return aligned
