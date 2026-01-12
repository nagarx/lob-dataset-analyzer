"""
Tests for streaming convenience functions.

These test the high-level streaming analysis functions that wrap
the lower-level iterators and provide easy-to-use interfaces.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from typing import Dict, Any

from lobanalyzer.streaming.convenience import (
    compute_streaming_overview,
    compute_streaming_label_analysis,
    compute_streaming_signal_stats,
    estimate_memory_usage,
    get_memory_efficient_config,
    _compute_acf,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_dataset(tmp_path: Path) -> Path:
    """Create a minimal dataset for testing."""
    # Create train/val/test splits
    split_idx = 0
    for split in ['train', 'val', 'test']:
        split_dir = tmp_path / split
        split_dir.mkdir()
        
        # Create 2 days per split with unique dates
        dates = [f"2025-01-{10 + split_idx * 2 + i:02d}" for i in range(2)]
        for date in dates:
            # Sequences: (N_seq, window_size, n_features)
            n_seq = 100 if split == 'train' else 50
            sequences = np.random.randn(n_seq, 100, 98).astype(np.float32)
            
            # Labels: (N_seq,) with values -1, 0, 1
            labels = np.random.choice([-1, 0, 1], size=n_seq, p=[0.2, 0.5, 0.3]).astype(np.int8)
            
            np.save(split_dir / f"{date}_sequences.npy", sequences)
            np.save(split_dir / f"{date}_labels.npy", labels)
        
        split_idx += 1
    
    return tmp_path


@pytest.fixture
def sample_dataset_with_nan(tmp_path: Path) -> Path:
    """Create a dataset with some NaN/Inf values for data quality testing."""
    split_dir = tmp_path / 'train'
    split_dir.mkdir()
    
    # Create sequences with some bad values
    # NOTE: iter_days extracts the LAST timestep, so put bad values at index 99 (not 0)
    n_seq = 100
    sequences = np.random.randn(n_seq, 100, 98).astype(np.float32)
    sequences[0, 99, 0] = np.nan  # Last timestep, feature 0
    sequences[1, 99, 1] = np.inf  # Last timestep, feature 1
    sequences[2, 99, 2] = -np.inf  # Last timestep, feature 2
    
    labels = np.random.choice([-1, 0, 1], size=n_seq).astype(np.int8)
    
    np.save(split_dir / "2025-01-01_sequences.npy", sequences)
    np.save(split_dir / "2025-01-01_labels.npy", labels)
    
    return tmp_path


# =============================================================================
# Test _compute_acf
# =============================================================================

class TestComputeACF:
    """Tests for the ACF computation helper."""
    
    def test_constant_labels_acf_is_one(self):
        """Constant labels should have ACF = 1 for all lags."""
        labels = np.ones(1000)
        acf = _compute_acf(labels, max_lag=20)
        assert all(a == 1.0 for a in acf)
    
    def test_random_labels_acf_near_zero(self):
        """Random labels should have ACF near 0 for non-zero lags."""
        np.random.seed(42)
        labels = np.random.choice([-1, 0, 1], size=10000)
        acf = _compute_acf(labels, max_lag=20)
        
        assert acf[0] == 1.0  # Lag 0 always 1
        # Non-zero lags should be small for random data
        assert all(abs(a) < 0.1 for a in acf[1:])
    
    def test_acf_length(self):
        """ACF should have correct length."""
        labels = np.random.choice([-1, 0, 1], size=1000)
        acf = _compute_acf(labels, max_lag=50)
        assert len(acf) == 51  # 0 to 50 inclusive
    
    def test_acf_short_array(self):
        """ACF should handle arrays shorter than max_lag."""
        labels = np.random.choice([-1, 0, 1], size=20)
        acf = _compute_acf(labels, max_lag=100)
        assert len(acf) == 20  # Limited by array length


# =============================================================================
# Test compute_streaming_overview
# =============================================================================

class TestComputeStreamingOverview:
    """Tests for compute_streaming_overview."""
    
    def test_returns_expected_keys(self, sample_dataset: Path):
        """Overview should return all expected keys."""
        result = compute_streaming_overview(sample_dataset, symbol="TEST")
        
        expected_keys = [
            'symbol', 'data_dir', 'date_range', 'total_days',
            'train_days', 'val_days', 'test_days', 'total_samples',
            'total_labels', 'feature_count', 'data_quality',
            'label_distribution', 'signal_stats'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_correct_split_counts(self, sample_dataset: Path):
        """Should count days per split correctly."""
        result = compute_streaming_overview(sample_dataset)
        
        assert result['train_days'] == 2
        assert result['val_days'] == 2
        assert result['test_days'] == 2
        assert result['total_days'] == 6
    
    def test_symbol_is_recorded(self, sample_dataset: Path):
        """Symbol parameter should be recorded."""
        result = compute_streaming_overview(sample_dataset, symbol="NVDA")
        assert result['symbol'] == "NVDA"
    
    def test_data_quality_clean_data(self, sample_dataset: Path):
        """Clean data should be detected."""
        result = compute_streaming_overview(sample_dataset)
        assert result['data_quality']['is_clean'] == True
        assert result['data_quality']['nan_count'] == 0
        assert result['data_quality']['inf_count'] == 0
    
    def test_data_quality_with_bad_values(self, sample_dataset_with_nan: Path):
        """NaN/Inf values should be detected."""
        result = compute_streaming_overview(sample_dataset_with_nan)
        assert result['data_quality']['is_clean'] == False
        assert result['data_quality']['nan_count'] > 0 or result['data_quality']['inf_count'] > 0


# =============================================================================
# Test compute_streaming_label_analysis
# =============================================================================

class TestComputeStreamingLabelAnalysis:
    """Tests for compute_streaming_label_analysis."""
    
    def test_returns_expected_keys(self, sample_dataset: Path):
        """Label analysis should return all expected keys."""
        result = compute_streaming_label_analysis(sample_dataset, split='train')
        
        expected_keys = [
            'split', 'date_range', 'n_days', 'distribution',
            'autocorrelation', 'transition_matrix', 'day_stats'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_distribution_sums_to_total(self, sample_dataset: Path):
        """Label counts should sum to total."""
        result = compute_streaming_label_analysis(sample_dataset, split='train')
        dist = result['distribution']
        
        total = dist['down_count'] + dist['stable_count'] + dist['up_count']
        assert total == dist['total']
    
    def test_percentages_sum_to_100(self, sample_dataset: Path):
        """Label percentages should sum to ~100."""
        result = compute_streaming_label_analysis(sample_dataset, split='train')
        dist = result['distribution']
        
        total_pct = dist['down_pct'] + dist['stable_pct'] + dist['up_pct']
        assert abs(total_pct - 100) < 0.01
    
    def test_autocorrelation_structure(self, sample_dataset: Path):
        """Autocorrelation should have expected structure."""
        result = compute_streaming_label_analysis(sample_dataset, split='train')
        acf = result['autocorrelation']
        
        assert 'lag_1' in acf
        assert 'lag_5' in acf
        assert 'lag_10' in acf
        assert 'acf_values' in acf
        assert len(acf['acf_values']) == 20
    
    def test_transition_matrix_structure(self, sample_dataset: Path):
        """Transition matrix should be 3x3."""
        result = compute_streaming_label_analysis(sample_dataset, split='train')
        tm = result['transition_matrix']
        
        assert tm['labels'] == [-1, 0, 1]
        assert len(tm['probabilities']) == 3
        assert all(len(row) == 3 for row in tm['probabilities'])
    
    def test_transition_rows_sum_to_one(self, sample_dataset: Path):
        """Transition matrix rows should sum to ~1."""
        result = compute_streaming_label_analysis(sample_dataset, split='train')
        probs = result['transition_matrix']['probabilities']
        
        for row in probs:
            if sum(row) > 0:  # Skip empty rows
                assert abs(sum(row) - 1.0) < 0.01
    
    def test_day_stats_count(self, sample_dataset: Path):
        """Should have one entry per day."""
        result = compute_streaming_label_analysis(sample_dataset, split='train')
        assert len(result['day_stats']) == 2  # 2 days in train


# =============================================================================
# Test compute_streaming_signal_stats
# =============================================================================

class TestComputeStreamingSignalStats:
    """Tests for compute_streaming_signal_stats."""
    
    def test_returns_all_core_signals(self, sample_dataset: Path):
        """Should return stats for all core signals."""
        result = compute_streaming_signal_stats(sample_dataset, split='train')
        
        expected_signals = [
            'true_ofi', 'depth_norm_ofi', 'executed_pressure',
            'signed_mp_delta_bps', 'trade_asymmetry', 'cancel_asymmetry',
            'fragility_score', 'depth_asymmetry'
        ]
        for sig in expected_signals:
            assert sig in result, f"Missing signal: {sig}"
    
    def test_signal_stats_structure(self, sample_dataset: Path):
        """Each signal should have expected stats."""
        result = compute_streaming_signal_stats(sample_dataset, split='train')
        
        for name, stats in result.items():
            assert 'index' in stats
            assert 'n' in stats
            assert 'mean' in stats
            assert 'std' in stats
            assert 'min' in stats
            assert 'max' in stats
    
    def test_custom_signal_indices(self, sample_dataset: Path):
        """Should work with custom signal indices."""
        result = compute_streaming_signal_stats(
            sample_dataset, 
            split='train',
            signal_indices=[84, 85]  # Only first 2
        )
        
        assert 'true_ofi' in result
        assert 'depth_norm_ofi' in result
        assert len(result) == 2


# =============================================================================
# Test estimate_memory_usage
# =============================================================================

class TestEstimateMemoryUsage:
    """Tests for estimate_memory_usage."""
    
    def test_returns_expected_keys(self, sample_dataset: Path):
        """Should return estimates for all splits."""
        result = estimate_memory_usage(sample_dataset)
        
        assert 'train' in result
        assert 'val' in result
        assert 'test' in result
        assert 'total' in result
    
    def test_estimate_structure(self, sample_dataset: Path):
        """Each estimate should have samples, bytes, mb, gb."""
        result = estimate_memory_usage(sample_dataset)
        
        for split, estimate in result.items():
            assert 'samples' in estimate
            assert 'bytes' in estimate
            assert 'mb' in estimate
            assert 'gb' in estimate
    
    def test_total_is_sum_of_splits(self, sample_dataset: Path):
        """Total samples should equal sum of splits."""
        result = estimate_memory_usage(sample_dataset)
        
        split_total = (
            result['train']['samples'] + 
            result['val']['samples'] + 
            result['test']['samples']
        )
        assert result['total']['samples'] == split_total


# =============================================================================
# Test get_memory_efficient_config
# =============================================================================

class TestGetMemoryEfficientConfig:
    """Tests for get_memory_efficient_config."""
    
    def test_returns_expected_keys(self):
        """Config should have all expected keys."""
        config = get_memory_efficient_config()
        
        expected_keys = [
            'dtype', 'mmap_mode', 'max_days_in_memory',
            'gc_after_each_day', 'subsample_for_expensive_ops',
            'max_samples_for_acf', 'max_samples_for_correlation'
        ]
        for key in expected_keys:
            assert key in config, f"Missing key: {key}"
    
    def test_reasonable_defaults(self):
        """Defaults should be reasonable."""
        config = get_memory_efficient_config()
        
        assert config['dtype'] == 'float32'  # Not float64
        assert config['max_days_in_memory'] == 1  # Memory efficient
        assert config['gc_after_each_day'] == True
