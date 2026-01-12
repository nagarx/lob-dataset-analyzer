"""
Tests for lobanalyzer.analysis.predictive_power module.

Tests cover:
- Correlation computation (Pearson, Spearman)
- AUC computation for Up/Down classification
- Mutual information estimation
- Binned probability analysis
- Predictive grouping
"""

import numpy as np
import pandas as pd
import pytest

from lobanalyzer.analysis.predictive_power import (
    compute_signal_metrics,
    compute_all_signal_metrics,
    compute_binned_probabilities,
    identify_predictive_groups,
    print_predictive_summary,
)
from lobanalyzer.constants import LABEL_DOWN, LABEL_STABLE, LABEL_UP, CORE_SIGNAL_INDICES


class TestComputeSignalMetrics:
    """Tests for compute_signal_metrics."""
    
    def test_basic_metrics_computed(self):
        """Test that all basic metrics are computed."""
        np.random.seed(42)
        signal = np.random.randn(1000)
        labels = np.random.choice([-1, 0, 1], 1000)
        
        metrics = compute_signal_metrics(signal, labels)
        
        assert 'n_samples' in metrics
        assert 'pearson_r' in metrics
        assert 'spearman_r' in metrics
        assert 'auc_up' in metrics
        assert 'auc_down' in metrics
        assert 'mutual_info' in metrics
        assert 'mean_up' in metrics
        assert metrics['n_samples'] == 1000
    
    def test_perfect_positive_correlation(self):
        """Test with perfectly correlated signal and labels."""
        np.random.seed(42)
        n = 1000
        labels = np.random.choice([-1, 0, 1], n)
        signal = labels.astype(float) + 0.01 * np.random.randn(n)  # Add tiny noise
        
        metrics = compute_signal_metrics(signal, labels, expected_sign='+')
        
        assert metrics['pearson_r'] > 0.95, f"Expected high positive r, got {metrics['pearson_r']}"
        assert metrics['sign_consistent'] == True
    
    def test_negative_correlation(self):
        """Test with negatively correlated signal and labels."""
        np.random.seed(42)
        n = 1000
        labels = np.random.choice([-1, 0, 1], n)
        signal = -labels.astype(float) + 0.01 * np.random.randn(n)  # Negative correlation
        
        metrics = compute_signal_metrics(signal, labels, expected_sign='+')
        
        assert metrics['pearson_r'] < -0.95, f"Expected negative r, got {metrics['pearson_r']}"
        assert metrics['sign_consistent'] == False  # Expected +, got -
    
    def test_auc_for_perfect_discriminator(self):
        """Test AUC when signal perfectly separates Up from others."""
        np.random.seed(42)
        n = 1000
        labels = np.random.choice([-1, 0, 1], n)
        
        # Signal = 1 when Up, 0 otherwise (perfect discriminator)
        signal = (labels == LABEL_UP).astype(float)
        signal += 0.01 * np.random.randn(n)  # Small noise
        
        metrics = compute_signal_metrics(signal, labels)
        
        assert metrics['auc_up'] > 0.95, f"Expected high AUC_up, got {metrics['auc_up']}"
    
    def test_random_signal_auc_near_half(self):
        """Test that random signal gives AUC near 0.5."""
        np.random.seed(42)
        signal = np.random.randn(1000)
        labels = np.random.choice([-1, 0, 1], 1000)
        
        metrics = compute_signal_metrics(signal, labels)
        
        # Random signal should have AUC close to 0.5
        assert 0.4 < metrics['auc_up'] < 0.6, f"Expected AUC near 0.5, got {metrics['auc_up']}"
        assert 0.4 < metrics['auc_down'] < 0.6, f"Expected AUC near 0.5, got {metrics['auc_down']}"
    
    def test_mutual_information_nonnegative(self):
        """Test that mutual information is non-negative."""
        np.random.seed(42)
        signal = np.random.randn(1000)
        labels = np.random.choice([-1, 0, 1], 1000)
        
        metrics = compute_signal_metrics(signal, labels)
        
        assert metrics['mutual_info'] >= 0, "MI should be non-negative"
        assert metrics['mi_bits'] >= 0, "MI in bits should be non-negative"
    
    def test_conditional_means(self):
        """Test that conditional means are computed correctly."""
        np.random.seed(42)
        n = 1000
        signal = np.zeros(n)
        labels = np.zeros(n)
        
        # Set specific patterns
        signal[:333] = 1.0  # Up
        signal[333:666] = 0.0  # Stable
        signal[666:] = -1.0  # Down
        
        labels[:333] = LABEL_UP
        labels[333:666] = LABEL_STABLE
        labels[666:] = LABEL_DOWN
        
        metrics = compute_signal_metrics(signal, labels)
        
        assert abs(metrics['mean_up'] - 1.0) < 0.01
        assert abs(metrics['mean_stable'] - 0.0) < 0.01
        assert abs(metrics['mean_down'] - (-1.0)) < 0.01
    
    def test_empty_signal_handling(self):
        """Test handling of empty signal."""
        signal = np.array([])
        labels = np.array([])
        
        metrics = compute_signal_metrics(signal, labels)
        
        assert metrics['n_samples'] == 0
        assert 'error' in metrics
    
    def test_nan_filtering(self):
        """Test that NaN values are filtered out."""
        np.random.seed(42)
        signal = np.random.randn(100)
        labels = np.random.choice([-1, 0, 1], 100)
        
        # Add some NaN values
        signal[0:10] = np.nan
        
        metrics = compute_signal_metrics(signal, labels)
        
        assert metrics['n_samples'] == 90  # 100 - 10 NaN


class TestComputeAllSignalMetrics:
    """Tests for compute_all_signal_metrics."""
    
    def test_all_signals_analyzed(self):
        """Test that all specified signals are analyzed."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        labels = np.random.choice([-1, 0, 1], 1000)
        
        df = compute_all_signal_metrics(features, labels, signal_indices=[84, 85, 86])
        
        assert len(df) == 3
        assert set(df['index']) == {84, 85, 86}
    
    def test_default_signal_indices(self):
        """Test that default uses CORE_SIGNAL_INDICES."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        labels = np.random.choice([-1, 0, 1], 1000)
        
        df = compute_all_signal_metrics(features, labels)
        
        assert len(df) == len(CORE_SIGNAL_INDICES)
    
    def test_dataframe_columns(self):
        """Test that DataFrame has expected columns."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        labels = np.random.choice([-1, 0, 1], 1000)
        
        df = compute_all_signal_metrics(features, labels, signal_indices=[84])
        
        expected_columns = ['index', 'name', 'expected_sign', 'n_samples', 
                          'pearson_r', 'pearson_p', 'spearman_r', 'spearman_p',
                          'auc_up', 'auc_down', 'mutual_info', 'mi_bits',
                          'sign_consistent', 'mean_up', 'mean_stable', 'mean_down']
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"


class TestComputeBinnedProbabilities:
    """Tests for compute_binned_probabilities."""
    
    def test_correct_number_of_bins(self):
        """Test that correct number of bins are created."""
        np.random.seed(42)
        signal = np.random.randn(1000)
        labels = np.random.choice([-1, 0, 1], 1000)
        
        df = compute_binned_probabilities(signal, labels, n_bins=10)
        
        # May have fewer bins if duplicates are dropped
        assert len(df) <= 10
        assert len(df) >= 5  # Should have at least some bins
    
    def test_probabilities_sum_to_one(self):
        """Test that label probabilities sum to 1 in each bin."""
        np.random.seed(42)
        signal = np.random.randn(1000)
        labels = np.random.choice([-1, 0, 1], 1000)
        
        df = compute_binned_probabilities(signal, labels, n_bins=10)
        
        for _, row in df.iterrows():
            prob_sum = row['p_up'] + row['p_down'] + row['p_stable']
            assert abs(prob_sum - 1.0) < 0.01, f"Probabilities sum to {prob_sum}, not 1"
    
    def test_signal_mean_increases_with_bin(self):
        """Test that signal_mean generally increases with bin number."""
        np.random.seed(42)
        signal = np.random.randn(1000)
        labels = np.random.choice([-1, 0, 1], 1000)
        
        df = compute_binned_probabilities(signal, labels, n_bins=10)
        
        # First bin mean should be less than last bin mean
        if len(df) >= 2:
            assert df.iloc[0]['signal_mean'] < df.iloc[-1]['signal_mean']
    
    def test_monotonic_relationship_detected(self):
        """Test that monotonic signal-label relationship is detected in bins."""
        np.random.seed(42)
        n = 1000
        # Create signal where high values predict Up
        signal = np.random.randn(n)
        
        # Labels based on signal: high signal -> Up, low -> Down
        labels = np.zeros(n, dtype=int)
        labels[signal > 1] = LABEL_UP
        labels[signal < -1] = LABEL_DOWN
        
        df = compute_binned_probabilities(signal, labels, n_bins=10)
        
        # In highest bin, p_up should be higher than in lowest bin
        if len(df) >= 2:
            assert df.iloc[-1]['p_up'] > df.iloc[0]['p_up']
    
    def test_empty_signal_handling(self):
        """Test handling of empty signal."""
        signal = np.array([])
        labels = np.array([])
        
        df = compute_binned_probabilities(signal, labels)
        
        assert len(df) == 0


class TestIdentifyPredictiveGroups:
    """Tests for identify_predictive_groups."""
    
    def test_primary_group_identified(self):
        """Test that strong predictors are placed in primary group."""
        df = pd.DataFrame({
            'name': ['signal_a', 'signal_b', 'signal_c'],
            'pearson_r': [0.1, 0.02, 0.005],
            'sign_consistent': [True, True, True],
        })
        
        groups = identify_predictive_groups(df)
        
        # signal_a has |r| >= 0.05, should be primary
        assert len(groups['primary']) == 1
        assert groups['primary'][0][0] == 'signal_a'
    
    def test_contrarian_group_identified(self):
        """Test that contrarian signals are grouped separately."""
        df = pd.DataFrame({
            'name': ['signal_a', 'signal_b'],
            'pearson_r': [0.1, 0.08],
            'sign_consistent': [True, False],
        })
        
        groups = identify_predictive_groups(df)
        
        # signal_b has sign_consistent=False
        assert len(groups['contrarian']) == 1
        assert groups['contrarian'][0][0] == 'signal_b'
    
    def test_low_priority_identified(self):
        """Test that weak signals are in low_priority group."""
        df = pd.DataFrame({
            'name': ['signal_a'],
            'pearson_r': [0.005],  # |r| < 0.01
            'sign_consistent': [True],
        })
        
        groups = identify_predictive_groups(df)
        
        assert len(groups['low_priority']) == 1
    
    def test_redundancy_detection(self):
        """Test that redundant signals are identified."""
        df = pd.DataFrame({
            'name': ['signal_a', 'signal_b'],
            'pearson_r': [0.1, 0.05],
            'sign_consistent': [True, True],
        })
        
        # Correlation matrix showing high correlation between a and b
        corr = np.array([[1.0, 0.8], [0.8, 1.0]])
        names = ['signal_a', 'signal_b']
        
        groups = identify_predictive_groups(df, corr, names)
        
        # signal_b should be redundant (lower |r| but correlated with signal_a)
        redundant_names = [name for name, r in groups['redundant']]
        assert 'signal_b' in redundant_names


class TestPrintPredictiveSummary:
    """Tests for print_predictive_summary."""
    
    def test_print_runs_without_error(self, capsys):
        """Test that print_predictive_summary runs without error."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        labels = np.random.choice([-1, 0, 1], 1000)
        
        df = compute_all_signal_metrics(features, labels, signal_indices=[84, 85])
        
        print_predictive_summary(df)
        
        captured = capsys.readouterr()
        assert "SIGNAL PREDICTIVE POWER ANALYSIS" in captured.out
        assert "SIGNAL RANKING" in captured.out
        assert "RECOMMENDATIONS" in captured.out
    
    def test_print_with_correlation_matrix(self, capsys):
        """Test print with correlation matrix for redundancy detection."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        labels = np.random.choice([-1, 0, 1], 1000)
        
        df = compute_all_signal_metrics(features, labels, signal_indices=[84, 85])
        
        from lobanalyzer.analysis.signal_correlations import compute_signal_correlation_matrix
        corr, names = compute_signal_correlation_matrix(features, signal_indices=[84, 85])
        
        print_predictive_summary(df, corr, names)
        
        captured = capsys.readouterr()
        assert "REDUNDANT PAIRS" in captured.out


class TestIntegration:
    """Integration tests."""
    
    def test_full_predictive_analysis_pipeline(self):
        """Test a complete predictive analysis pipeline."""
        np.random.seed(42)
        n = 2000
        features = np.zeros((n, 98))
        
        # Create a signal with known predictive power
        signal = np.random.randn(n)
        features[:, 84] = signal
        
        # Create labels correlated with signal
        labels = np.zeros(n, dtype=int)
        labels[signal > 0.5] = LABEL_UP
        labels[signal < -0.5] = LABEL_DOWN
        
        # Run analysis
        df = compute_all_signal_metrics(features, labels, signal_indices=[84])
        df_bins = compute_binned_probabilities(features[:, 84], labels)
        groups = identify_predictive_groups(df)
        
        # Verify
        assert len(df) == 1
        assert df.iloc[0]['pearson_r'] > 0.5, "Should have positive correlation"
        assert len(df_bins) > 0
        assert len(groups['primary']) == 1
    
    def test_analysis_with_mixed_predictive_power(self):
        """Test with signals of varying predictive power."""
        np.random.seed(42)
        n = 2000
        features = np.zeros((n, 98))
        
        # Create base signal for labels
        base = np.random.randn(n)
        labels = np.zeros(n, dtype=int)
        labels[base > 0.5] = LABEL_UP
        labels[base < -0.5] = LABEL_DOWN
        
        # Signal 84: Strong predictor (correlated with labels)
        features[:, 84] = base + 0.1 * np.random.randn(n)
        
        # Signal 85: Weak predictor (mostly noise)
        features[:, 85] = 0.2 * base + np.random.randn(n)
        
        # Signal 86: Random (no predictive power)
        features[:, 86] = np.random.randn(n)
        
        df = compute_all_signal_metrics(features, labels, signal_indices=[84, 85, 86])
        
        # Sort by absolute correlation
        df_sorted = df.sort_values('pearson_r', key=abs, ascending=False)
        
        # Signal 84 should be best, 86 should be worst
        assert df_sorted.iloc[0]['index'] == 84
        assert df_sorted.iloc[-1]['index'] == 86
