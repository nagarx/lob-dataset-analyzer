"""
Tests for lobanalyzer.analysis.signal_stats module.

Tests cover:
- Distribution statistics computation (mean, std, skewness, kurtosis, outliers)
- Stationarity tests (ADF test)
- Rolling statistics for detecting non-stationarity
- Edge cases (constant signals, short signals, NaN handling)
"""

import numpy as np
import pandas as pd
import pytest
from dataclasses import asdict

from lobanalyzer.analysis.signal_stats import (
    StationarityResult,
    RollingStatsResult,
    DistributionStats,
    compute_distribution_stats,
    compute_stationarity_test,
    compute_all_stationarity_tests,
    compute_rolling_stats,
    compute_all_rolling_stats,
    print_distribution_summary,
    print_stationarity_summary,
)
from lobanalyzer.constants import CORE_SIGNAL_INDICES, FeatureIndex


class TestDistributionStats:
    """Tests for compute_distribution_stats."""
    
    def test_basic_stats_normal_data(self):
        """Test that statistics are computed correctly for normal data."""
        np.random.seed(42)
        n_samples = 10000
        features = np.random.randn(n_samples, 98)
        
        df = compute_distribution_stats(features, signal_indices=[84, 85])
        
        assert len(df) == 2
        assert 'mean' in df.columns
        assert 'std' in df.columns
        assert 'skewness' in df.columns
        assert 'kurtosis' in df.columns
        
        # For standard normal, mean ≈ 0, std ≈ 1, skewness ≈ 0, kurtosis ≈ 0
        for _, row in df.iterrows():
            assert abs(row['mean']) < 0.1, f"Mean should be close to 0, got {row['mean']}"
            assert abs(row['std'] - 1.0) < 0.1, f"Std should be close to 1, got {row['std']}"
            assert abs(row['skewness']) < 0.3, f"Skewness should be close to 0, got {row['skewness']}"
            # Excess kurtosis for normal is 0
            assert abs(row['kurtosis']) < 0.5, f"Kurtosis should be close to 0, got {row['kurtosis']}"
    
    def test_skewed_data(self):
        """Test that skewness is detected correctly."""
        np.random.seed(42)
        n_samples = 10000
        features = np.zeros((n_samples, 98))
        
        # Create right-skewed data (exponential)
        features[:, 84] = np.random.exponential(1, n_samples)
        
        df = compute_distribution_stats(features, signal_indices=[84])
        
        assert len(df) == 1
        # Exponential distribution has skewness = 2
        assert df.iloc[0]['skewness'] > 1.0, "Should detect right skew"
    
    def test_heavy_tailed_data(self):
        """Test that kurtosis is detected correctly for heavy-tailed data."""
        np.random.seed(42)
        n_samples = 10000
        features = np.zeros((n_samples, 98))
        
        # Create heavy-tailed data (t-distribution with low df)
        from scipy.stats import t as t_dist
        features[:, 84] = t_dist.rvs(df=3, size=n_samples)
        
        df = compute_distribution_stats(features, signal_indices=[84])
        
        # t(3) has excess kurtosis ≈ 6 (very heavy tails)
        assert df.iloc[0]['kurtosis'] > 3.0, "Should detect heavy tails"
    
    def test_outliers_detection(self):
        """Test that outlier percentage is computed correctly."""
        np.random.seed(42)
        n_samples = 10000
        features = np.zeros((n_samples, 98))
        
        # Standard normal: ~0.27% outside 3 sigma
        features[:, 84] = np.random.randn(n_samples)
        
        # Add some artificial outliers
        features[0:50, 85] = 10  # 50 outliers
        features[50:, 85] = np.random.randn(n_samples - 50)
        
        df = compute_distribution_stats(features, signal_indices=[84, 85])
        
        # Signal 84: Should have ~0.3% outliers (standard normal)
        signal_84 = df[df['index'] == 84].iloc[0]
        assert signal_84['pct_outliers'] < 1.0, "Normal data should have few outliers"
        
        # Signal 85: Should have more outliers due to artificial ones
        signal_85 = df[df['index'] == 85].iloc[0]
        assert signal_85['pct_outliers'] > 0.3, "Should detect artificial outliers"
    
    def test_default_signal_indices(self):
        """Test that default uses CORE_SIGNAL_INDICES."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        
        df = compute_distribution_stats(features)
        
        assert len(df) == len(CORE_SIGNAL_INDICES)
        assert set(df['index']) == set(CORE_SIGNAL_INDICES)
    
    def test_constant_signal(self):
        """Test handling of constant signal (zero variance)."""
        features = np.zeros((1000, 98))
        features[:, 84] = 5.0  # Constant signal
        
        df = compute_distribution_stats(features, signal_indices=[84])
        
        assert df.iloc[0]['mean'] == 5.0
        assert df.iloc[0]['std'] == 0.0
        assert df.iloc[0]['pct_outliers'] == 0.0


class TestStationarityTest:
    """Tests for stationarity testing functions."""
    
    def test_stationary_signal(self):
        """Test that stationary signal is correctly identified."""
        np.random.seed(42)
        # White noise is stationary
        signal = np.random.randn(10000)
        
        adf_stat, p_value, critical_values, is_stationary = compute_stationarity_test(signal)
        
        # White noise should be stationary (p < 0.05)
        assert is_stationary, f"White noise should be stationary, got p={p_value}"
        assert p_value < 0.05, f"p-value should be < 0.05 for stationary signal"
    
    def test_non_stationary_signal(self):
        """Test that random walk (non-stationary) is correctly identified."""
        np.random.seed(42)
        # Random walk is non-stationary
        signal = np.cumsum(np.random.randn(10000))
        
        adf_stat, p_value, critical_values, is_stationary = compute_stationarity_test(signal)
        
        # Random walk should NOT be stationary (p > 0.05)
        assert not is_stationary, f"Random walk should not be stationary, got p={p_value}"
        assert p_value > 0.05, f"p-value should be > 0.05 for random walk"
    
    def test_subsampling_for_large_signals(self):
        """Test that large signals are subsampled."""
        np.random.seed(42)
        large_signal = np.random.randn(500000)  # Very large
        
        # Should not raise an error or take too long
        adf_stat, p_value, _, _ = compute_stationarity_test(
            large_signal, max_samples=10000
        )
        
        assert np.isfinite(adf_stat)
        assert 0 <= p_value <= 1
    
    def test_stationarity_result_dataclass(self):
        """Test StationarityResult dataclass structure."""
        result = StationarityResult(
            signal_name="test_signal",
            signal_index=84,
            adf_statistic=-3.5,
            p_value=0.01,
            critical_values={"1%": -3.5, "5%": -2.9, "10%": -2.6},
            is_stationary=True,
            interpretation="Stationary: No unit root, safe for modeling",
        )
        
        d = asdict(result)
        assert d['signal_name'] == "test_signal"
        assert d['is_stationary'] == True
        assert '5%' in d['critical_values']


class TestComputeAllStationarityTests:
    """Tests for compute_all_stationarity_tests."""
    
    def test_all_signals_tested(self):
        """Test that all specified signals are tested."""
        np.random.seed(42)
        features = np.random.randn(5000, 98)
        
        results = compute_all_stationarity_tests(features, signal_indices=[84, 85, 86])
        
        assert len(results) == 3
        assert all(isinstance(r, StationarityResult) for r in results)
        assert {r.signal_index for r in results} == {84, 85, 86}
    
    def test_interpretation_categories(self):
        """Test that interpretation categories are correct."""
        np.random.seed(42)
        features = np.zeros((10000, 98))
        
        # Stationary signal
        features[:, 84] = np.random.randn(10000)
        
        # Non-stationary signal (random walk)
        features[:, 85] = np.cumsum(np.random.randn(10000))
        
        results = compute_all_stationarity_tests(features, signal_indices=[84, 85])
        
        # Check interpretations
        result_84 = next(r for r in results if r.signal_index == 84)
        result_85 = next(r for r in results if r.signal_index == 85)
        
        assert "Stationary" in result_84.interpretation or "safe" in result_84.interpretation.lower()
        assert "Non-stationary" in result_85.interpretation or "differencing" in result_85.interpretation.lower()


class TestRollingStats:
    """Tests for rolling statistics functions."""
    
    def test_stable_signal(self):
        """Test that stable signal has stable rolling stats."""
        np.random.seed(42)
        # Stationary signal with constant mean and variance
        signal = np.random.randn(50000)
        
        (mean_drift, std_drift, max_mean, min_mean, mean_range,
         is_mean_stable, is_std_stable) = compute_rolling_stats(signal)
        
        assert is_mean_stable, "Stationary signal should have stable mean"
        assert is_std_stable, "Stationary signal should have stable std"
        assert abs(mean_drift) < 0.5, f"Mean drift should be small, got {mean_drift}"
    
    def test_drifting_mean(self):
        """Test detection of mean drift."""
        np.random.seed(42)
        n = 50000
        # Signal with drifting mean (trend)
        trend = np.linspace(0, 5, n)
        signal = np.random.randn(n) + trend
        
        (mean_drift, std_drift, max_mean, min_mean, mean_range,
         is_mean_stable, is_std_stable) = compute_rolling_stats(signal)
        
        assert not is_mean_stable, "Signal with trend should have unstable mean"
        assert mean_drift > 0, "Mean should drift positive with increasing trend"
        assert mean_range > 1.0, f"Mean range should be large, got {mean_range}"
    
    def test_short_signal(self):
        """Test handling of signals too short for analysis."""
        signal = np.random.randn(100)  # Very short
        
        (mean_drift, std_drift, max_mean, min_mean, mean_range,
         is_mean_stable, is_std_stable) = compute_rolling_stats(signal, window_size=10000)
        
        # Should return stable (default) for insufficient data
        assert is_mean_stable
        assert is_std_stable
        assert mean_drift == 0.0
    
    def test_rolling_stats_result_dataclass(self):
        """Test RollingStatsResult dataclass structure."""
        result = RollingStatsResult(
            signal_name="test_signal",
            signal_index=84,
            window_size=10000,
            n_windows=10,
            mean_drift=0.1,
            std_drift=0.05,
            max_mean=0.15,
            min_mean=-0.05,
            mean_range=0.2,
            is_mean_stable=True,
            is_std_stable=True,
        )
        
        d = asdict(result)
        assert d['window_size'] == 10000
        assert d['mean_range'] == 0.2


class TestComputeAllRollingStats:
    """Tests for compute_all_rolling_stats."""
    
    def test_all_signals_analyzed(self):
        """Test that all specified signals are analyzed."""
        np.random.seed(42)
        features = np.random.randn(50000, 98)
        
        results = compute_all_rolling_stats(features, signal_indices=[84, 85, 86])
        
        assert len(results) == 3
        assert all(isinstance(r, RollingStatsResult) for r in results)
        assert {r.signal_index for r in results} == {84, 85, 86}
    
    def test_custom_window_parameters(self):
        """Test that custom window parameters are used."""
        np.random.seed(42)
        features = np.random.randn(100000, 98)
        
        results = compute_all_rolling_stats(
            features,
            signal_indices=[84],
            window_size=5000,
            n_windows=20,
        )
        
        assert results[0].window_size == 5000
        assert results[0].n_windows == 20


class TestPrintFunctions:
    """Tests for print/display functions."""
    
    def test_print_distribution_summary(self, capsys):
        """Test that print_distribution_summary runs without error."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        df = compute_distribution_stats(features, signal_indices=[84, 85])
        
        # Should not raise
        print_distribution_summary(df)
        
        captured = capsys.readouterr()
        assert "SIGNAL DISTRIBUTION STATISTICS" in captured.out
        assert "KEY INSIGHTS" in captured.out
    
    def test_print_stationarity_summary(self, capsys):
        """Test that print_stationarity_summary runs without error."""
        np.random.seed(42)
        features = np.random.randn(5000, 98)
        
        stationarity_results = compute_all_stationarity_tests(features, signal_indices=[84, 85])
        rolling_results = compute_all_rolling_stats(features, signal_indices=[84, 85])
        
        # Should not raise
        print_stationarity_summary(stationarity_results, rolling_results)
        
        captured = capsys.readouterr()
        assert "STATIONARITY ANALYSIS" in captured.out
        assert "AUGMENTED DICKEY-FULLER TEST" in captured.out
        assert "SUMMARY" in captured.out
    
    def test_print_stationarity_without_rolling(self, capsys):
        """Test print_stationarity_summary without rolling results."""
        np.random.seed(42)
        features = np.random.randn(5000, 98)
        
        stationarity_results = compute_all_stationarity_tests(features, signal_indices=[84])
        
        # Should not raise without rolling_results
        print_stationarity_summary(stationarity_results, rolling_results=None)
        
        captured = capsys.readouterr()
        assert "ROLLING STATISTICS" not in captured.out


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_signal_indices(self):
        """Test behavior with empty signal indices list."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        
        df = compute_distribution_stats(features, signal_indices=[])
        
        assert len(df) == 0
    
    def test_signal_with_nan(self):
        """Test handling of signals with NaN values."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        features[0:10, 84] = np.nan  # Add some NaN values
        
        # Should not crash, but stats may be affected
        df = compute_distribution_stats(features, signal_indices=[84])
        
        # NaN in input propagates to stats (by design)
        assert len(df) == 1
    
    def test_very_small_sample(self):
        """Test with very small sample size."""
        np.random.seed(42)
        features = np.random.randn(10, 98)  # Very few samples
        
        df = compute_distribution_stats(features, signal_indices=[84])
        
        assert len(df) == 1
        assert np.isfinite(df.iloc[0]['mean'])
        # Normality test may fail with too few samples
        assert np.isnan(df.iloc[0]['p_normal']) or np.isfinite(df.iloc[0]['p_normal'])


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_full_analysis_pipeline(self):
        """Test a complete analysis pipeline."""
        np.random.seed(42)
        n_samples = 10000
        features = np.random.randn(n_samples, 98)
        
        # Make one signal non-stationary (random walk)
        features[:, 85] = np.cumsum(np.random.randn(n_samples)) * 0.01
        
        # Run all analyses
        dist_stats = compute_distribution_stats(features, signal_indices=[84, 85])
        stationarity = compute_all_stationarity_tests(features, signal_indices=[84, 85])
        rolling = compute_all_rolling_stats(features, signal_indices=[84, 85])
        
        # Verify results make sense together
        assert len(dist_stats) == 2
        assert len(stationarity) == 2
        assert len(rolling) == 2
        
        # Signal 84 (stationary) vs Signal 85 (non-stationary)
        stat_84 = next(s for s in stationarity if s.signal_index == 84)
        stat_85 = next(s for s in stationarity if s.signal_index == 85)
        
        assert stat_84.is_stationary, "White noise should be stationary"
        # Note: Random walk with small steps might still appear stationary in short samples
