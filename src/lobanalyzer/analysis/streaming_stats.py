"""
Streaming (online) statistics algorithms.

Provides memory-efficient incremental statistics computation:
    - RunningStats: Welford's algorithm for mean/variance
    - StreamingColumnStats: Multi-column statistics
    - StreamingLabelCounter: Label distribution counter
    - StreamingDataQuality: Data quality checker

Reference:
    Welford, B. P. (1962). "Note on a method for calculating 
    corrected sums of squares and products"

Memory Budget:
    - RunningStats: O(1) per signal
    - StreamingColumnStats: O(n_columns)
    - Total for 98 features: < 10KB
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Any


@dataclass
class RunningStats:
    """
    Welford's online algorithm for computing mean and variance.
    
    Numerically stable, single-pass, constant memory.
    
    Formula:
        mean_n = mean_{n-1} + (x_n - mean_{n-1}) / n
        M2_n = M2_{n-1} + (x_n - mean_{n-1}) * (x_n - mean_n)
        variance = M2 / n
    
    Attributes:
        n: Number of samples seen
        mean: Running mean
        M2: Sum of squared deviations (for variance)
        min_val: Minimum value seen
        max_val: Maximum value seen
    """
    n: int = 0
    mean: float = 0.0
    M2: float = 0.0  # Sum of squared deviations
    min_val: float = float('inf')
    max_val: float = float('-inf')
    
    def update(self, x: float) -> None:
        """
        Update with a single value.
        
        Args:
            x: New observation
        """
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        self.min_val = min(self.min_val, x)
        self.max_val = max(self.max_val, x)
    
    def update_batch(self, values: np.ndarray) -> None:
        """
        Update with a batch of values.
        
        Args:
            values: Array of observations
        """
        for x in values.flat:
            self.update(x)
    
    @property
    def variance(self) -> float:
        """Population variance."""
        return self.M2 / self.n if self.n > 0 else 0.0
    
    @property
    def std(self) -> float:
        """Population standard deviation."""
        return np.sqrt(self.variance)
    
    @classmethod
    def merge(cls, a: 'RunningStats', b: 'RunningStats') -> 'RunningStats':
        """
        Merge two RunningStats (for parallel computation).
        
        Args:
            a: First RunningStats
            b: Second RunningStats
        
        Returns:
            Combined RunningStats
        """
        if a.n == 0:
            return b
        if b.n == 0:
            return a
        
        combined = cls()
        combined.n = a.n + b.n
        delta = b.mean - a.mean
        combined.mean = (a.n * a.mean + b.n * b.mean) / combined.n
        combined.M2 = a.M2 + b.M2 + delta * delta * a.n * b.n / combined.n
        combined.min_val = min(a.min_val, b.min_val)
        combined.max_val = max(a.max_val, b.max_val)
        return combined


@dataclass
class StreamingColumnStats:
    """
    Streaming statistics for multiple columns.
    
    Memory: O(n_columns) - constant regardless of data size.
    
    Attributes:
        n_columns: Number of columns to track
        stats: List of RunningStats, one per column
    """
    n_columns: int
    stats: List[RunningStats] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        if not self.stats:
            self.stats = [RunningStats() for _ in range(self.n_columns)]
    
    def update(self, row: np.ndarray) -> None:
        """
        Update with a single row.
        
        Args:
            row: 1D array of values
        """
        for i, val in enumerate(row):
            if np.isfinite(val):
                self.stats[i].update(float(val))
    
    def update_batch(self, data: np.ndarray) -> None:
        """
        Update with a batch of rows (2D array).
        
        Args:
            data: 2D array (n_samples, n_columns)
        """
        for col_idx in range(min(data.shape[1], self.n_columns)):
            col = data[:, col_idx]
            finite_mask = np.isfinite(col)
            if finite_mask.any():
                for val in col[finite_mask]:
                    self.stats[col_idx].update(float(val))
    
    def get_summary(self) -> Dict[int, Dict[str, float]]:
        """
        Get summary for all columns.
        
        Returns:
            Dict mapping column_idx -> statistics dict
        """
        return {
            i: {
                'n': s.n,
                'mean': s.mean,
                'std': s.std,
                'min': s.min_val if s.n > 0 else None,
                'max': s.max_val if s.n > 0 else None,
            }
            for i, s in enumerate(self.stats)
        }


@dataclass
class StreamingLabelCounter:
    """
    Streaming label distribution counter.
    
    Memory: O(1) - constant regardless of data size.
    
    Attributes:
        down_count: Count of Down labels (-1)
        stable_count: Count of Stable labels (0)
        up_count: Count of Up labels (1)
        total: Total label count
    """
    down_count: int = 0
    stable_count: int = 0
    up_count: int = 0
    total: int = 0
    
    def update(self, labels: np.ndarray) -> None:
        """
        Update with a batch of labels.
        
        Args:
            labels: Array of labels {-1, 0, 1}
        """
        self.down_count += int((labels == -1).sum())
        self.stable_count += int((labels == 0).sum())
        self.up_count += int((labels == 1).sum())
        self.total += len(labels)
    
    @property
    def down_pct(self) -> float:
        """Percentage of Down labels."""
        return 100 * self.down_count / self.total if self.total > 0 else 0
    
    @property
    def stable_pct(self) -> float:
        """Percentage of Stable labels."""
        return 100 * self.stable_count / self.total if self.total > 0 else 0
    
    @property
    def up_pct(self) -> float:
        """Percentage of Up labels."""
        return 100 * self.up_count / self.total if self.total > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            'total': self.total,
            'down_count': self.down_count,
            'stable_count': self.stable_count,
            'up_count': self.up_count,
            'down_pct': self.down_pct,
            'stable_pct': self.stable_pct,
            'up_pct': self.up_pct,
        }


@dataclass
class StreamingDataQuality:
    """
    Streaming data quality checker.
    
    Memory: O(n_columns) - for tracking which columns have issues.
    
    Attributes:
        total_values: Total values seen
        nan_count: Count of NaN values
        inf_count: Count of Inf values
        columns_with_nan: Set of column indices with NaN
        columns_with_inf: Set of column indices with Inf
    """
    total_values: int = 0
    nan_count: int = 0
    inf_count: int = 0
    columns_with_nan: Set[int] = field(default_factory=set)
    columns_with_inf: Set[int] = field(default_factory=set)
    
    def update(self, features: np.ndarray) -> None:
        """
        Update with a batch of features.
        
        Args:
            features: 2D feature array (n_samples, n_features)
        """
        self.total_values += features.size
        
        nan_mask = np.isnan(features)
        inf_mask = np.isinf(features)
        
        self.nan_count += int(nan_mask.sum())
        self.inf_count += int(inf_mask.sum())
        
        # Track columns with issues
        nan_cols = np.where(nan_mask.any(axis=0))[0]
        inf_cols = np.where(inf_mask.any(axis=0))[0]
        
        self.columns_with_nan.update(nan_cols.tolist())
        self.columns_with_inf.update(inf_cols.tolist())
    
    @property
    def is_clean(self) -> bool:
        """True if no NaN or Inf values detected."""
        return self.nan_count == 0 and self.inf_count == 0
    
    @property
    def finite_count(self) -> int:
        """Count of finite values."""
        return self.total_values - self.nan_count - self.inf_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            'total_values': self.total_values,
            'finite_count': self.finite_count,
            'nan_count': self.nan_count,
            'inf_count': self.inf_count,
            'is_clean': self.is_clean,
            'columns_with_nan': sorted(self.columns_with_nan),
            'columns_with_inf': sorted(self.columns_with_inf),
        }
