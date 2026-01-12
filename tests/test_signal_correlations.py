"""
Tests for lobanalyzer.analysis.signal_correlations module.

Tests cover:
- Correlation matrix computation
- Redundant pair detection
- PCA analysis
- VIF computation
- Signal clustering
- Edge cases and error handling
"""

import numpy as np
import pandas as pd
import pytest
from dataclasses import asdict

from lobanalyzer.analysis.signal_correlations import (
    PCAResult,
    VIFResult,
    SignalCluster,
    compute_signal_correlation_matrix,
    find_redundant_pairs,
    compute_pca_analysis,
    compute_vif,
    cluster_signals,
    print_correlation_summary,
    print_advanced_correlation_summary,
)
from lobanalyzer.constants import CORE_SIGNAL_INDICES


class TestCorrelationMatrix:
    """Tests for compute_signal_correlation_matrix."""
    
    def test_correlation_matrix_shape(self):
        """Test that correlation matrix has correct shape."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        
        corr, names = compute_signal_correlation_matrix(features, signal_indices=[84, 85, 86])
        
        assert corr.shape == (3, 3)
        assert len(names) == 3
    
    def test_correlation_matrix_diagonal_is_one(self):
        """Test that diagonal of correlation matrix is 1."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        
        corr, _ = compute_signal_correlation_matrix(features, signal_indices=[84, 85, 86])
        
        np.testing.assert_array_almost_equal(np.diag(corr), np.ones(3), decimal=5)
    
    def test_correlation_matrix_symmetric(self):
        """Test that correlation matrix is symmetric."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        
        corr, _ = compute_signal_correlation_matrix(features)
        
        np.testing.assert_array_almost_equal(corr, corr.T, decimal=10)
    
    def test_high_correlation_detected(self):
        """Test that high correlation between signals is detected."""
        np.random.seed(42)
        features = np.zeros((1000, 98))
        
        # Create two highly correlated signals
        signal = np.random.randn(1000)
        features[:, 84] = signal
        features[:, 85] = signal + 0.1 * np.random.randn(1000)  # Nearly identical
        features[:, 86] = np.random.randn(1000)  # Independent
        
        corr, names = compute_signal_correlation_matrix(features, signal_indices=[84, 85, 86])
        
        # Correlation between 84 and 85 should be very high
        assert corr[0, 1] > 0.9, f"Expected high correlation, got {corr[0, 1]}"
        
        # Correlation with 86 should be low
        assert abs(corr[0, 2]) < 0.3, f"Expected low correlation, got {corr[0, 2]}"
    
    def test_default_signal_indices(self):
        """Test that default uses CORE_SIGNAL_INDICES."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        
        corr, names = compute_signal_correlation_matrix(features)
        
        assert len(names) == len(CORE_SIGNAL_INDICES)


class TestRedundantPairs:
    """Tests for find_redundant_pairs."""
    
    def test_find_redundant_pairs_basic(self):
        """Test that redundant pairs are found correctly."""
        # Create a correlation matrix with known correlations
        corr = np.array([
            [1.0, 0.8, 0.1],
            [0.8, 1.0, 0.2],
            [0.1, 0.2, 1.0],
        ])
        names = ['signal_a', 'signal_b', 'signal_c']
        
        pairs = find_redundant_pairs(corr, names, threshold=0.5)
        
        assert len(pairs) == 1
        assert pairs[0]['signal_1'] == 'signal_a'
        assert pairs[0]['signal_2'] == 'signal_b'
        assert pairs[0]['correlation'] == 0.8
    
    def test_no_redundant_pairs(self):
        """Test behavior when no pairs exceed threshold."""
        corr = np.eye(3)  # Identity matrix (no correlations)
        names = ['a', 'b', 'c']
        
        pairs = find_redundant_pairs(corr, names, threshold=0.5)
        
        assert len(pairs) == 0
    
    def test_sorted_by_correlation(self):
        """Test that pairs are sorted by absolute correlation."""
        corr = np.array([
            [1.0, 0.6, -0.9],
            [0.6, 1.0, 0.7],
            [-0.9, 0.7, 1.0],
        ])
        names = ['a', 'b', 'c']
        
        pairs = find_redundant_pairs(corr, names, threshold=0.5)
        
        # Highest absolute correlation should be first
        assert abs(pairs[0]['correlation']) >= abs(pairs[1]['correlation'])
    
    def test_negative_correlation_included(self):
        """Test that negative correlations above threshold are included."""
        corr = np.array([
            [1.0, -0.8],
            [-0.8, 1.0],
        ])
        names = ['a', 'b']
        
        pairs = find_redundant_pairs(corr, names, threshold=0.5)
        
        assert len(pairs) == 1
        assert pairs[0]['correlation'] == -0.8


class TestPCAAnalysis:
    """Tests for compute_pca_analysis."""
    
    def test_pca_variance_sums_to_one(self):
        """Test that explained variance ratios sum to approximately 1."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        
        result = compute_pca_analysis(features, signal_indices=[84, 85, 86])
        
        assert abs(sum(result.explained_variance_ratio) - 1.0) < 0.01
    
    def test_pca_cumulative_variance_monotonic(self):
        """Test that cumulative variance is monotonically increasing."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        
        result = compute_pca_analysis(features, signal_indices=[84, 85, 86, 87])
        
        for i in range(len(result.cumulative_variance) - 1):
            assert result.cumulative_variance[i] <= result.cumulative_variance[i + 1]
    
    def test_pca_components_threshold(self):
        """Test that n_components_90 and n_components_95 are computed correctly."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        
        result = compute_pca_analysis(features)
        
        # Components for 90% should be <= components for 95%
        assert result.n_components_90 <= result.n_components_95
        # Both should be <= total components
        assert result.n_components_90 <= result.n_components
        assert result.n_components_95 <= result.n_components
    
    def test_pca_dominant_signals(self):
        """Test that dominant signals per component are identified."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        
        result = compute_pca_analysis(features, signal_indices=[84, 85, 86])
        
        assert len(result.dominant_signal_per_component) == 3
        # All dominant signals should be from our input signals
        for signal in result.dominant_signal_per_component:
            assert signal in result.signal_names
    
    def test_pca_insufficient_samples_raises(self):
        """Test that insufficient samples raises ValueError."""
        features = np.random.randn(50, 98)  # Too few samples
        
        with pytest.raises(ValueError, match="Insufficient"):
            compute_pca_analysis(features)
    
    def test_pca_result_dataclass(self):
        """Test PCAResult dataclass structure."""
        result = PCAResult(
            n_components=3,
            explained_variance_ratio=[0.5, 0.3, 0.2],
            cumulative_variance=[0.5, 0.8, 1.0],
            n_components_95=3,
            n_components_90=2,
            component_loadings=[[0.7, 0.5, 0.5], [0.5, 0.7, 0.5], [0.5, 0.5, 0.7]],
            signal_names=['a', 'b', 'c'],
            dominant_signal_per_component=['a', 'b', 'c'],
        )
        
        d = asdict(result)
        assert d['n_components'] == 3
        assert len(d['explained_variance_ratio']) == 3


class TestVIF:
    """Tests for compute_vif."""
    
    def test_vif_independent_signals(self):
        """Test that independent signals have VIF close to 1."""
        np.random.seed(42)
        n = 1000
        features = np.zeros((n, 98))
        
        # Create completely independent signals
        for i in [84, 85, 86]:
            features[:, i] = np.random.randn(n)
        
        results = compute_vif(features, signal_indices=[84, 85, 86])
        
        for r in results:
            # VIF should be close to 1 for independent signals
            assert r.vif < 2.0, f"Expected VIF close to 1, got {r.vif} for {r.signal_name}"
            assert not r.is_problematic
            assert not r.is_severe
    
    def test_vif_collinear_signals(self):
        """Test that collinear signals have high VIF."""
        np.random.seed(42)
        n = 1000
        features = np.zeros((n, 98))
        
        # Create collinear signals
        base = np.random.randn(n)
        features[:, 84] = base
        features[:, 85] = base + 0.01 * np.random.randn(n)  # Nearly identical
        features[:, 86] = np.random.randn(n)  # Independent
        
        results = compute_vif(features, signal_indices=[84, 85, 86])
        
        # Signal 84 and 85 should have high VIF
        vif_84 = next(r for r in results if r.signal_index == 84)
        vif_85 = next(r for r in results if r.signal_index == 85)
        
        assert vif_84.vif > 10, f"Expected high VIF, got {vif_84.vif}"
        assert vif_85.vif > 10, f"Expected high VIF, got {vif_85.vif}"
        assert vif_84.is_severe
        assert vif_85.is_severe
    
    def test_vif_insufficient_samples_raises(self):
        """Test that insufficient samples raises ValueError."""
        features = np.random.randn(50, 98)  # Too few samples
        
        with pytest.raises(ValueError, match="Insufficient"):
            compute_vif(features)
    
    def test_vif_result_dataclass(self):
        """Test VIFResult dataclass structure."""
        result = VIFResult(
            signal_name="test_signal",
            signal_index=84,
            vif=6.5,
            is_problematic=True,
            is_severe=False,
        )
        
        d = asdict(result)
        assert d['vif'] == 6.5
        assert d['is_problematic'] == True


class TestClusterSignals:
    """Tests for cluster_signals."""
    
    def test_independent_signals_separate_clusters(self):
        """Test that independent signals are in separate clusters."""
        corr = np.eye(3)  # No correlations
        names = ['a', 'b', 'c']
        indices = [84, 85, 86]
        
        clusters = cluster_signals(corr, names, indices, threshold=0.5)
        
        # Each signal should be in its own cluster
        assert len(clusters) == 3
        for c in clusters:
            assert len(c.signals) == 1
    
    def test_correlated_signals_same_cluster(self):
        """Test that correlated signals are grouped together."""
        corr = np.array([
            [1.0, 0.8, 0.1],
            [0.8, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ])
        names = ['a', 'b', 'c']
        indices = [84, 85, 86]
        
        clusters = cluster_signals(corr, names, indices, threshold=0.5)
        
        # a and b should be in same cluster, c separate
        assert len(clusters) == 2
        
        # Find cluster with multiple signals
        multi_cluster = [c for c in clusters if len(c.signals) > 1][0]
        assert set(multi_cluster.signals) == {'a', 'b'}
    
    def test_cluster_mean_correlation(self):
        """Test that mean within-cluster correlation is computed correctly."""
        corr = np.array([
            [1.0, 0.8, 0.7],
            [0.8, 1.0, 0.75],
            [0.7, 0.75, 1.0],
        ])
        names = ['a', 'b', 'c']
        indices = [84, 85, 86]
        
        clusters = cluster_signals(corr, names, indices, threshold=0.5)
        
        # All signals should be in one cluster
        assert len(clusters) == 1
        
        # Mean correlation should be (0.8 + 0.7 + 0.75) / 3 = 0.75
        expected_mean = (0.8 + 0.7 + 0.75) / 3
        assert abs(clusters[0].mean_within_correlation - expected_mean) < 0.01
    
    def test_signal_cluster_dataclass(self):
        """Test SignalCluster dataclass structure."""
        cluster = SignalCluster(
            cluster_id=0,
            signals=['a', 'b'],
            signal_indices=[84, 85],
            mean_within_correlation=0.75,
        )
        
        d = asdict(cluster)
        assert d['cluster_id'] == 0
        assert len(d['signals']) == 2


class TestPrintFunctions:
    """Tests for print/display functions."""
    
    def test_print_correlation_summary(self, capsys):
        """Test that print_correlation_summary runs without error."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        corr, names = compute_signal_correlation_matrix(features, signal_indices=[84, 85])
        
        print_correlation_summary(corr, names)
        
        captured = capsys.readouterr()
        assert "SIGNAL CORRELATION ANALYSIS" in captured.out
        assert "RECOMMENDATIONS" in captured.out
    
    def test_print_advanced_correlation_summary_pca(self, capsys):
        """Test print_advanced_correlation_summary with PCA."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        pca_result = compute_pca_analysis(features, signal_indices=[84, 85, 86])
        
        print_advanced_correlation_summary(pca_result=pca_result)
        
        captured = capsys.readouterr()
        assert "PCA ANALYSIS" in captured.out
        assert "Variance Explained" in captured.out
    
    def test_print_advanced_correlation_summary_vif(self, capsys):
        """Test print_advanced_correlation_summary with VIF."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        vif_results = compute_vif(features, signal_indices=[84, 85])
        
        print_advanced_correlation_summary(vif_results=vif_results)
        
        captured = capsys.readouterr()
        assert "VIF ANALYSIS" in captured.out
    
    def test_print_advanced_correlation_summary_clusters(self, capsys):
        """Test print_advanced_correlation_summary with clusters."""
        corr = np.eye(3)
        names = ['a', 'b', 'c']
        clusters = cluster_signals(corr, names, [84, 85, 86])
        
        print_advanced_correlation_summary(clusters=clusters)
        
        captured = capsys.readouterr()
        assert "SIGNAL CLUSTERING" in captured.out


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_single_signal(self):
        """Test behavior with a single signal."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        
        corr, names = compute_signal_correlation_matrix(features, signal_indices=[84])
        
        # np.corrcoef with single signal returns scalar 1.0
        # This is expected behavior - single signal correlation with itself is 1
        assert len(names) == 1
        # corr might be a scalar or 2D depending on numpy version
        if hasattr(corr, 'shape') and corr.shape:
            assert corr.shape == (1, 1) or corr.shape == ()
        else:
            assert corr == 1.0 or (hasattr(corr, 'item') and corr.item() == 1.0)
    
    def test_two_signals(self):
        """Test behavior with two signals."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        
        corr, names = compute_signal_correlation_matrix(features, signal_indices=[84, 85])
        
        assert corr.shape == (2, 2)
        assert len(names) == 2
    
    def test_nan_values_filtered(self):
        """Test that NaN values are filtered in PCA and VIF."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        features[0:10, 84] = np.nan  # Add some NaN values
        
        # Should still work (fewer valid samples)
        result = compute_pca_analysis(features, signal_indices=[84, 85, 86])
        assert result.n_components == 3
        
        vif = compute_vif(features, signal_indices=[84, 85, 86])
        assert len(vif) == 3


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_full_correlation_analysis_pipeline(self):
        """Test a complete correlation analysis pipeline."""
        np.random.seed(42)
        n = 2000
        features = np.zeros((n, 98))
        
        # Create signals with known structure:
        # - 84, 85: highly correlated
        # - 86: independent
        # - 87: moderately correlated with 84
        base = np.random.randn(n)
        features[:, 84] = base
        features[:, 85] = base + 0.05 * np.random.randn(n)
        features[:, 86] = np.random.randn(n)
        features[:, 87] = 0.6 * base + 0.8 * np.random.randn(n)
        
        # Run all analyses
        corr, names = compute_signal_correlation_matrix(features, signal_indices=[84, 85, 86, 87])
        pairs = find_redundant_pairs(corr, names, threshold=0.5)
        pca = compute_pca_analysis(features, signal_indices=[84, 85, 86, 87])
        vif = compute_vif(features, signal_indices=[84, 85, 86, 87])
        clusters = cluster_signals(corr, names, [84, 85, 86, 87], threshold=0.5)
        
        # Verify results
        assert len(pairs) >= 1, "Should find at least one redundant pair (84-85)"
        assert pca.n_components_90 <= 4, "Should need <= 4 components for 90% variance"
        
        # 84 and 85 should have high VIF
        vif_84 = next(v for v in vif if v.signal_index == 84)
        assert vif_84.vif > 5, f"Expected high VIF for signal 84, got {vif_84.vif}"
