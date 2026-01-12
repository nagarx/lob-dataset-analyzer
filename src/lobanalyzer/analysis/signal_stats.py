"""
Signal distribution statistics.

Analyzes each signal's statistical properties:
- Mean, std, min, max
- Skewness, kurtosis
- Outlier frequency
- Normality tests
- Stationarity tests (Augmented Dickey-Fuller)
- Rolling statistics over time

References:
    - Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for
      autoregressive time series with a unit root.
    - D'Agostino, R. B., & Pearson, E. S. (1973). Tests for departure from normality.
"""

import numpy as np
from scipy.stats import skew, kurtosis, normaltest
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import warnings

from lobanalyzer.constants import get_signal_info, CORE_SIGNAL_INDICES


@dataclass
class StationarityResult:
    """
    Augmented Dickey-Fuller stationarity test result.
    
    Attributes:
        signal_name: Name of the signal
        signal_index: Index in feature array
        adf_statistic: ADF test statistic (more negative = stronger rejection of null)
        p_value: p-value for the test (< 0.05 suggests stationarity)
        critical_values: Critical values at 1%, 5%, 10% significance levels
        is_stationary: True if p < 0.05 (reject null of unit root)
        interpretation: Human-readable interpretation
    """
    signal_name: str
    signal_index: int
    adf_statistic: float
    p_value: float
    critical_values: Dict[str, float]
    is_stationary: bool  # p < 0.05
    interpretation: str


@dataclass
class RollingStatsResult:
    """
    Rolling statistics analysis for detecting non-stationarity.
    
    Detects mean drift and variance changes over time, which can indicate
    regime changes or non-stationarity that the ADF test might miss.
    
    Attributes:
        signal_name: Name of the signal
        signal_index: Index in feature array
        window_size: Size of each rolling window
        n_windows: Number of windows analyzed
        mean_drift: Change in mean from first to last window
        std_drift: Change in std from first to last window
        max_mean: Maximum window mean
        min_mean: Minimum window mean
        mean_range: max_mean - min_mean
        is_mean_stable: True if mean_range < 0.5 * overall_std
        is_std_stable: True if |std_drift| < 0.3 * overall_std
    """
    signal_name: str
    signal_index: int
    window_size: int
    n_windows: int
    mean_drift: float  # Change in mean from first to last window
    std_drift: float  # Change in std from first to last window
    max_mean: float
    min_mean: float
    mean_range: float  # max_mean - min_mean
    is_mean_stable: bool  # Mean doesn't drift too much
    is_std_stable: bool  # Std doesn't drift too much


@dataclass
class DistributionStats:
    """
    Distribution statistics for a single signal.
    
    Attributes:
        signal_index: Index in feature array
        signal_name: Human-readable name
        mean: Sample mean
        std: Sample standard deviation
        min: Minimum value
        max: Maximum value
        median: Sample median
        skewness: Third standardized moment (0 = symmetric)
        kurtosis: Excess kurtosis (0 = normal, > 0 = heavy tails)
        pct_outliers: Percentage of values with |z-score| > 3
        p_normal: p-value from D'Agostino-Pearson normality test
    """
    signal_index: int
    signal_name: str
    mean: float
    std: float
    min: float
    max: float
    median: float
    skewness: float
    kurtosis: float
    pct_outliers: float
    p_normal: float


def compute_distribution_stats(
    features: np.ndarray,
    signal_indices: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Compute distribution statistics for each signal.
    
    Args:
        features: (N, F) feature array where F >= max(signal_indices)
        signal_indices: Which signal indices to analyze (default: CORE_SIGNAL_INDICES)
    
    Returns:
        DataFrame with columns:
            - index, name, mean, std, min, max, median
            - skewness, kurtosis, pct_outliers, p_normal
    
    Example:
        >>> features = np.random.randn(10000, 98)
        >>> df_stats = compute_distribution_stats(features)
        >>> print(df_stats[['name', 'mean', 'std', 'skewness']])
    """
    if signal_indices is None:
        signal_indices = CORE_SIGNAL_INDICES
    
    signal_info = get_signal_info()
    results = []
    
    for idx in signal_indices:
        info = signal_info.get(idx, {'name': f'signal_{idx}'})
        signal = features[:, idx]
        
        # Basic stats
        mean = float(np.mean(signal))
        std = float(np.std(signal))
        min_val = float(np.min(signal))
        max_val = float(np.max(signal))
        median = float(np.median(signal))
        
        # Higher moments
        signal_skew = float(skew(signal))
        signal_kurt = float(kurtosis(signal))  # Excess kurtosis
        
        # Outlier frequency (|z| > 3)
        if std > 1e-10:
            z_scores = np.abs((signal - mean) / std)
            pct_outliers = 100.0 * float(np.mean(z_scores > 3))
        else:
            pct_outliers = 0.0
        
        # Normality test (subsample for speed)
        subsample = signal[::100][:5000]
        if len(subsample) >= 20:
            try:
                _, p_normal = normaltest(subsample)
                p_normal = float(p_normal)
            except Exception:
                p_normal = np.nan
        else:
            p_normal = np.nan
        
        results.append({
            'index': idx,
            'name': info.get('name', f'signal_{idx}'),
            'mean': mean,
            'std': std,
            'min': min_val,
            'max': max_val,
            'median': median,
            'skewness': signal_skew,
            'kurtosis': signal_kurt,
            'pct_outliers': pct_outliers,
            'p_normal': p_normal,
        })
    
    return pd.DataFrame(results)


def compute_stationarity_test(
    signal: np.ndarray,
    max_samples: int = 100000,
) -> Tuple[float, float, Dict[str, float], bool]:
    """
    Perform Augmented Dickey-Fuller test for stationarity.
    
    The ADF test null hypothesis is that the series has a unit root (non-stationary).
    If p < 0.05, we reject the null and conclude the series is stationary.
    
    Formula (Dickey & Fuller, 1979):
        ΔX_t = α + βt + γX_{t-1} + Σδ_i ΔX_{t-i} + ε_t
        H0: γ = 0 (unit root, non-stationary)
        H1: γ < 0 (stationary)
    
    Args:
        signal: 1D signal array
        max_samples: Maximum samples to use (ADF is O(n²))
    
    Returns:
        adf_stat: ADF test statistic
        p_value: p-value (< 0.05 suggests stationarity)
        critical_values: Dict of critical values at 1%, 5%, 10%
        is_stationary: True if p < 0.05
    
    Note:
        Returns (0.0, 1.0, {}, False) if statsmodels is not installed or test fails.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        warnings.warn("statsmodels not installed, skipping ADF test")
        return 0.0, 1.0, {}, False
    
    # Subsample for efficiency
    if len(signal) > max_samples:
        step = len(signal) // max_samples
        signal = signal[::step][:max_samples]
    
    try:
        result = adfuller(signal, autolag='AIC')
        adf_stat = float(result[0])
        p_value = float(result[1])
        critical_values = {k: float(v) for k, v in result[4].items()}
        is_stationary = p_value < 0.05
    except Exception as e:
        warnings.warn(f"ADF test failed: {e}")
        return 0.0, 1.0, {}, False
    
    return adf_stat, p_value, critical_values, is_stationary


def compute_all_stationarity_tests(
    features: np.ndarray,
    signal_indices: Optional[List[int]] = None,
) -> List[StationarityResult]:
    """
    Perform stationarity tests for all signals.
    
    Args:
        features: (N, F) feature array
        signal_indices: Which signals to test (default: CORE_SIGNAL_INDICES)
    
    Returns:
        List of StationarityResult for each signal
    
    Example:
        >>> features = np.random.randn(10000, 98)
        >>> results = compute_all_stationarity_tests(features)
        >>> for r in results:
        ...     print(f"{r.signal_name}: p={r.p_value:.4f}, stationary={r.is_stationary}")
    """
    if signal_indices is None:
        signal_indices = CORE_SIGNAL_INDICES
    
    signal_info = get_signal_info()
    results = []
    
    for idx in signal_indices:
        info = signal_info.get(idx, {'name': f'signal_{idx}'})
        signal = features[:, idx]
        
        adf_stat, p_value, critical_values, is_stationary = compute_stationarity_test(signal)
        
        if is_stationary:
            interpretation = "Stationary: No unit root, safe for modeling"
        elif p_value < 0.1:
            interpretation = "Marginally stationary: Weak evidence against unit root"
        else:
            interpretation = "Non-stationary: Consider differencing or detrending"
        
        results.append(StationarityResult(
            signal_name=info.get('name', f'signal_{idx}'),
            signal_index=idx,
            adf_statistic=adf_stat,
            p_value=p_value,
            critical_values=critical_values,
            is_stationary=is_stationary,
            interpretation=interpretation,
        ))
    
    return results


def compute_rolling_stats(
    signal: np.ndarray,
    window_size: int = 10000,
    n_windows: int = 10,
) -> Tuple[float, float, float, float, float, bool, bool]:
    """
    Compute rolling statistics to detect non-stationarity.
    
    Analyzes how the signal's mean and std change over time, which can
    reveal regime changes or drift that the ADF test might miss.
    
    Args:
        signal: 1D signal array
        window_size: Size of each rolling window
        n_windows: Number of windows to analyze
    
    Returns:
        Tuple of (mean_drift, std_drift, max_mean, min_mean, mean_range, 
                  is_mean_stable, is_std_stable)
    
    Stability Thresholds:
        - is_mean_stable: mean_range < 0.5 × overall_std
        - is_std_stable: |std_drift| < 0.3 × overall_std
    """
    n = len(signal)
    if n < window_size * 2:
        # Not enough data for meaningful analysis
        return 0.0, 0.0, 0.0, 0.0, 0.0, True, True
    
    step = max(1, (n - window_size) // n_windows)
    
    means = []
    stds = []
    
    for i in range(0, n - window_size, step):
        window = signal[i:i + window_size]
        means.append(float(np.mean(window)))
        stds.append(float(np.std(window)))
    
    if len(means) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0, True, True
    
    means = np.array(means)
    stds = np.array(stds)
    
    # Drift = change from first to last window
    mean_drift = float(means[-1] - means[0])
    std_drift = float(stds[-1] - stds[0])
    
    max_mean = float(np.max(means))
    min_mean = float(np.min(means))
    mean_range = max_mean - min_mean
    
    # Stability checks (thresholds based on normalized data)
    overall_std = float(np.std(signal))
    is_mean_stable = mean_range < 0.5 * overall_std
    is_std_stable = abs(std_drift) < 0.3 * overall_std
    
    return mean_drift, std_drift, max_mean, min_mean, mean_range, is_mean_stable, is_std_stable


def compute_all_rolling_stats(
    features: np.ndarray,
    signal_indices: Optional[List[int]] = None,
    window_size: int = 10000,
    n_windows: int = 10,
) -> List[RollingStatsResult]:
    """
    Compute rolling statistics for all signals.
    
    Args:
        features: (N, F) feature array
        signal_indices: Which signals to analyze (default: CORE_SIGNAL_INDICES)
        window_size: Size of each rolling window
        n_windows: Number of windows
    
    Returns:
        List of RollingStatsResult for each signal
    """
    if signal_indices is None:
        signal_indices = CORE_SIGNAL_INDICES
    
    signal_info = get_signal_info()
    results = []
    
    for idx in signal_indices:
        info = signal_info.get(idx, {'name': f'signal_{idx}'})
        signal = features[:, idx]
        
        (mean_drift, std_drift, max_mean, min_mean, mean_range,
         is_mean_stable, is_std_stable) = compute_rolling_stats(signal, window_size, n_windows)
        
        results.append(RollingStatsResult(
            signal_name=info.get('name', f'signal_{idx}'),
            signal_index=idx,
            window_size=window_size,
            n_windows=n_windows,
            mean_drift=mean_drift,
            std_drift=std_drift,
            max_mean=max_mean,
            min_mean=min_mean,
            mean_range=mean_range,
            is_mean_stable=is_mean_stable,
            is_std_stable=is_std_stable,
        ))
    
    return results


def print_distribution_summary(df_stats: pd.DataFrame) -> None:
    """
    Print formatted distribution statistics.
    
    Args:
        df_stats: DataFrame from compute_distribution_stats
    """
    print("=" * 100)
    print("SIGNAL DISTRIBUTION STATISTICS")
    print("=" * 100)
    
    # Display key columns
    display_cols = ['name', 'mean', 'std', 'skewness', 'kurtosis', 'pct_outliers']
    df_display = df_stats[display_cols].copy()
    
    # Format for readability
    for col in ['mean', 'std', 'skewness', 'kurtosis']:
        df_display[col] = df_display[col].apply(lambda x: f"{x:+.4f}")
    df_display['pct_outliers'] = df_display['pct_outliers'].apply(lambda x: f"{x:.2f}%")
    
    print(df_display.to_string(index=False))
    
    # Key insights
    print("\n" + "=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100)
    
    # Highly skewed signals
    skewed = df_stats[df_stats['skewness'].abs() > 1.0]
    if len(skewed) > 0:
        print("\nHighly Skewed Signals (|skewness| > 1.0):")
        for _, row in skewed.iterrows():
            direction = "right" if row['skewness'] > 0 else "left"
            print(f"  • {row['name']}: skewness = {row['skewness']:+.2f} ({direction}-tailed)")
    
    # Heavy-tailed signals
    heavy_tailed = df_stats[df_stats['kurtosis'] > 3.0]
    if len(heavy_tailed) > 0:
        print("\nHeavy-Tailed Signals (excess kurtosis > 3.0):")
        for _, row in heavy_tailed.iterrows():
            print(f"  • {row['name']}: kurtosis = {row['kurtosis']:.2f} (normal = 0)")
    
    # High outlier signals
    high_outliers = df_stats[df_stats['pct_outliers'] > 1.0]
    if len(high_outliers) > 0:
        print("\nHigh Outlier Frequency (> 1% outside 3σ):")
        for _, row in high_outliers.iterrows():
            print(f"  • {row['name']}: {row['pct_outliers']:.2f}% outliers")
    
    # Non-normal signals
    non_normal = df_stats[df_stats['p_normal'] < 0.01]
    if len(non_normal) > 0:
        print("\nNon-Normal Signals (p < 0.01):")
        print(f"  • All {len(non_normal)} tested signals are non-normal (expected after Z-scoring)")
    
    print("\n" + "=" * 100)


def print_stationarity_summary(
    stationarity_results: List[StationarityResult],
    rolling_results: Optional[List[RollingStatsResult]] = None,
) -> None:
    """
    Print formatted stationarity analysis.
    
    Args:
        stationarity_results: List of StationarityResult from compute_all_stationarity_tests
        rolling_results: Optional list of RollingStatsResult from compute_all_rolling_stats
    """
    print("\n" + "=" * 80)
    print("STATIONARITY ANALYSIS")
    print("=" * 80)
    
    print("\n1. AUGMENTED DICKEY-FULLER TEST")
    print("-" * 60)
    print(f"{'Signal':<25} {'ADF Stat':>10} {'p-value':>10} {'Stationary':>12}")
    
    for s in stationarity_results:
        status = "✅ Yes" if s.is_stationary else "❌ No"
        print(f"{s.signal_name:<25} {s.adf_statistic:>10.4f} {s.p_value:>10.4e} {status:>12}")
    
    if rolling_results:
        print("\n2. ROLLING STATISTICS (Mean/Std Stability)")
        print("-" * 60)
        print(f"{'Signal':<25} {'Mean Drift':>11} {'Mean Range':>11} {'Mean OK':>9} {'Std OK':>8}")
        
        for r in rolling_results:
            mean_ok = "✅" if r.is_mean_stable else "❌"
            std_ok = "✅" if r.is_std_stable else "❌"
            print(f"{r.signal_name:<25} {r.mean_drift:>+11.4f} {r.mean_range:>11.4f} {mean_ok:>9} {std_ok:>8}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    non_stationary = [s.signal_name for s in stationarity_results if not s.is_stationary]
    if non_stationary:
        print(f"\n  ⚠️ Non-stationary signals (may need differencing):")
        for name in non_stationary:
            print(f"    • {name}")
    else:
        print("\n  ✅ All signals are stationary")
    
    if rolling_results:
        unstable_mean = [r.signal_name for r in rolling_results if not r.is_mean_stable]
        if unstable_mean:
            print(f"\n  ⚠️ Signals with unstable mean over time:")
            for name in unstable_mean:
                print(f"    • {name}")
    
    print("\n" + "=" * 80)
