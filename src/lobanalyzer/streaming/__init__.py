"""
Streaming data loading for memory-efficient analysis.

Provides day-by-day iteration over LOB datasets to enable
analysis of large datasets without loading everything into memory.

Design Principles:
    - Memory-efficient: Process one day at a time (~40MB per day)
    - Schema-aware: Validates feature counts and label shapes
    - Multi-horizon support: Handles both single and multi-horizon labels
    - Format-agnostic: Handles both aligned and legacy export formats

Usage:
    >>> from lobanalyzer.streaming import iter_days, iter_days_aligned, DayData
    >>> 
    >>> # Simple iteration over raw data
    >>> for day in iter_days(Path("data/exports/nvda_balanced"), "train"):
    ...     print(f"Day {day.date}: {day.n_samples} samples")
    >>> 
    >>> # Aligned iteration for correlation analysis
    >>> for day in iter_days_aligned(Path("data/exports/nvda_balanced"), "train"):
    ...     labels = day.get_labels(0)
    ...     corr = np.corrcoef(day.features[:, 84], labels)[0, 1]
    ...     print(f"{day.date}: OFI r = {corr:.4f}")
    >>> 
    >>> # High-level convenience functions
    >>> from lobanalyzer.streaming import compute_streaming_overview
    >>> overview = compute_streaming_overview(Path("data/exports/nvda"), "NVDA")
    >>> print(f"Total: {overview['total_samples']:,} samples")

Memory Budget:
    - Target: < 4GB for any dataset size
    - Per-day: ~40MB (float32, 100K samples × 98 features)
"""

from lobanalyzer.streaming.day_data import (
    DayData,
    AlignedDayData,
    align_features_for_day,
    WINDOW_SIZE,
    STRIDE,
)

from lobanalyzer.streaming.iterators import (
    iter_days,
    iter_days_aligned,
    count_days,
    get_dates,
)

from lobanalyzer.streaming.convenience import (
    compute_streaming_overview,
    compute_streaming_label_analysis,
    compute_streaming_signal_stats,
    estimate_memory_usage,
    get_memory_efficient_config,
)

__all__ = [
    # Data containers
    "DayData",
    "AlignedDayData",
    # Alignment utility
    "align_features_for_day",
    # Constants
    "WINDOW_SIZE",
    "STRIDE",
    # Iterators
    "iter_days",
    "iter_days_aligned",
    "count_days",
    "get_dates",
    # Convenience functions
    "compute_streaming_overview",
    "compute_streaming_label_analysis",
    "compute_streaming_signal_stats",
    "estimate_memory_usage",
    "get_memory_efficient_config",
]
