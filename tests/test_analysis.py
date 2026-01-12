"""
Tests for the analysis module.

Validates data quality checks, label analysis, and streaming statistics.
"""

import numpy as np
import pytest


class TestDataQuality:
    """Test data quality functions."""
    
    def test_clean_data(self) -> None:
        """Data quality reports clean for valid data."""
        from lobanalyzer.analysis import compute_data_quality
        
        features = np.random.randn(100, 98).astype(np.float32)
        quality = compute_data_quality(features)
        
        assert quality.is_clean is True
        assert quality.nan_count == 0
        assert quality.inf_count == 0
        assert quality.finite_count == 100 * 98
        assert len(quality.columns_with_nan) == 0
        assert len(quality.columns_with_inf) == 0

    def test_data_with_nan(self) -> None:
        """Data quality detects NaN values."""
        from lobanalyzer.analysis import compute_data_quality
        
        features = np.random.randn(100, 98).astype(np.float32)
        features[0, 0] = np.nan
        features[50, 10] = np.nan
        
        quality = compute_data_quality(features)
        
        assert quality.is_clean is False
        assert quality.nan_count == 2
        assert 0 in quality.columns_with_nan
        assert 10 in quality.columns_with_nan

    def test_data_with_inf(self) -> None:
        """Data quality detects Inf values."""
        from lobanalyzer.analysis import compute_data_quality
        
        features = np.random.randn(100, 98).astype(np.float32)
        features[0, 5] = np.inf
        features[50, 5] = -np.inf
        
        quality = compute_data_quality(features)
        
        assert quality.is_clean is False
        assert quality.inf_count == 2
        assert 5 in quality.columns_with_inf


class TestLabelDistribution:
    """Test label distribution functions."""
    
    def test_balanced_labels(self) -> None:
        """Balanced labels have low imbalance ratio."""
        from lobanalyzer.analysis import compute_label_distribution
        
        # Equal distribution
        labels = np.array([-1] * 100 + [0] * 100 + [1] * 100)
        dist = compute_label_distribution(labels)
        
        assert dist.total == 300
        assert dist.down_count == 100
        assert dist.stable_count == 100
        assert dist.up_count == 100
        assert dist.imbalance_ratio == 1.0
        assert dist.is_balanced is True

    def test_imbalanced_labels(self) -> None:
        """Imbalanced labels have high imbalance ratio."""
        from lobanalyzer.analysis import compute_label_distribution
        
        # 80% stable
        labels = np.array([-1] * 50 + [0] * 400 + [1] * 50)
        dist = compute_label_distribution(labels)
        
        assert dist.total == 500
        assert dist.stable_count == 400
        assert dist.imbalance_ratio == 8.0  # 400/50
        assert dist.is_balanced is False

    def test_majority_minority_class(self) -> None:
        """Correctly identifies majority and minority classes."""
        from lobanalyzer.analysis.label_analysis import compute_label_distribution
        
        labels = np.array([-1] * 10 + [0] * 50 + [1] * 40)
        dist = compute_label_distribution(labels)
        
        assert dist.majority_class == "Stable"
        assert dist.minority_class == "Down"


class TestAutocorrelation:
    """Test autocorrelation analysis."""
    
    def test_constant_labels(self) -> None:
        """Handles constant labels gracefully."""
        from lobanalyzer.analysis import compute_autocorrelation
        
        labels = np.zeros(100)
        acf = compute_autocorrelation(labels, max_lag=10)
        
        assert acf.lag_1_acf == 1.0  # Constant series has ACF=1

    def test_random_labels(self) -> None:
        """Random labels have low autocorrelation."""
        from lobanalyzer.analysis import compute_autocorrelation
        
        np.random.seed(42)
        labels = np.random.choice([-1, 0, 1], size=10000)
        acf = compute_autocorrelation(labels, max_lag=10)
        
        # Random labels should have ACF near 0
        assert abs(acf.lag_1_acf) < 0.1
        assert "random" in acf.interpretation.lower() or "no significant" in acf.interpretation.lower()


class TestTransitionMatrix:
    """Test transition matrix computation."""
    
    def test_deterministic_sequence(self) -> None:
        """Transition matrix for deterministic sequence."""
        from lobanalyzer.analysis import compute_transition_matrix
        
        # Sequence: -1 -> 0 -> 1 -> -1 -> 0 -> 1 -> ...
        labels = np.array([-1, 0, 1] * 100)
        tm = compute_transition_matrix(labels)
        
        assert tm.labels == [-1, 0, 1]
        # -1 always goes to 0
        assert tm.probabilities[0][1] == 1.0
        # 0 always goes to 1
        assert tm.probabilities[1][2] == 1.0

    def test_self_transitions(self) -> None:
        """Transition matrix captures self-transitions."""
        from lobanalyzer.analysis import compute_transition_matrix
        
        # Labels that cluster
        labels = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
        tm = compute_transition_matrix(labels)
        
        # High self-transition probability
        assert tm.probabilities[0][0] == 2/3  # -1 -> -1


class TestRunningStats:
    """Test Welford's online statistics."""
    
    def test_mean_variance(self) -> None:
        """Running stats match numpy mean/variance."""
        from lobanalyzer.analysis import RunningStats
        
        data = np.random.randn(1000)
        
        stats = RunningStats()
        for x in data:
            stats.update(x)
        
        assert np.isclose(stats.mean, data.mean(), atol=1e-10)
        assert np.isclose(stats.variance, data.var(), atol=1e-10)
        assert np.isclose(stats.min_val, data.min(), atol=1e-10)
        assert np.isclose(stats.max_val, data.max(), atol=1e-10)

    def test_merge(self) -> None:
        """Merging stats preserves correct mean/variance."""
        from lobanalyzer.analysis import RunningStats
        
        data1 = np.random.randn(500)
        data2 = np.random.randn(500)
        
        stats1 = RunningStats()
        for x in data1:
            stats1.update(x)
        
        stats2 = RunningStats()
        for x in data2:
            stats2.update(x)
        
        merged = RunningStats.merge(stats1, stats2)
        all_data = np.concatenate([data1, data2])
        
        assert np.isclose(merged.mean, all_data.mean(), atol=1e-10)
        assert np.isclose(merged.variance, all_data.var(), atol=1e-10)

    def test_empty_merge(self) -> None:
        """Merging with empty stats works correctly."""
        from lobanalyzer.analysis import RunningStats
        
        stats1 = RunningStats()
        stats1.update(1.0)
        stats1.update(2.0)
        
        stats2 = RunningStats()  # Empty
        
        merged = RunningStats.merge(stats1, stats2)
        assert merged.n == 2
        assert merged.mean == 1.5


class TestStreamingLabelCounter:
    """Test streaming label counter."""
    
    def test_counting(self) -> None:
        """Label counter counts correctly."""
        from lobanalyzer.analysis import StreamingLabelCounter
        
        counter = StreamingLabelCounter()
        
        labels1 = np.array([-1, 0, 1, 0, 1])
        labels2 = np.array([-1, -1, 0])
        
        counter.update(labels1)
        counter.update(labels2)
        
        assert counter.total == 8
        assert counter.down_count == 3
        assert counter.stable_count == 3
        assert counter.up_count == 2

    def test_percentages(self) -> None:
        """Label counter percentages are correct."""
        from lobanalyzer.analysis import StreamingLabelCounter
        
        counter = StreamingLabelCounter()
        counter.update(np.array([-1] * 25 + [0] * 50 + [1] * 25))
        
        assert counter.down_pct == 25.0
        assert counter.stable_pct == 50.0
        assert counter.up_pct == 25.0


class TestStreamingDataQuality:
    """Test streaming data quality checker."""
    
    def test_clean_data(self) -> None:
        """Quality checker reports clean for valid data."""
        from lobanalyzer.analysis import StreamingDataQuality
        
        quality = StreamingDataQuality()
        quality.update(np.random.randn(100, 98))
        quality.update(np.random.randn(100, 98))
        
        assert quality.is_clean is True
        assert quality.total_values == 200 * 98
        assert quality.nan_count == 0

    def test_nan_detection(self) -> None:
        """Quality checker detects NaN values."""
        from lobanalyzer.analysis import StreamingDataQuality
        
        quality = StreamingDataQuality()
        
        features = np.random.randn(100, 98)
        features[0, 0] = np.nan
        quality.update(features)
        
        assert quality.is_clean is False
        assert quality.nan_count == 1
        assert 0 in quality.columns_with_nan


class TestPackageImports:
    """Test that analysis module exports work correctly."""
    
    def test_import_from_analysis(self) -> None:
        """Can import key exports from analysis module."""
        from lobanalyzer.analysis import (
            # Data overview
            compute_data_quality,
            compute_label_distribution,
            DataQuality,
            LabelDistribution,
            # Label analysis
            compute_autocorrelation,
            compute_transition_matrix,
            AutocorrelationResult,
            TransitionMatrix,
            # Streaming stats
            RunningStats,
            StreamingLabelCounter,
            StreamingDataQuality,
        )
        
        # Basic functionality check
        features = np.random.randn(10, 98).astype(np.float32)
        quality = compute_data_quality(features)
        assert isinstance(quality, DataQuality)

    def test_import_from_validation(self) -> None:
        """Can import from validation module."""
        from lobanalyzer.validation import (
            compute_data_quality,
            compute_shape_validation,
            DataQuality,
            ShapeValidation,
        )
        
        features = np.random.randn(10, 98).astype(np.float32)
        quality = compute_data_quality(features)
        assert isinstance(quality, DataQuality)
