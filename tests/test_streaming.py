"""
Tests for the streaming module.

Validates day data containers, alignment, and iterators.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import shutil


class TestDayData:
    """Test DayData container."""
    
    def test_single_horizon_labels(self) -> None:
        """DayData handles single-horizon labels correctly."""
        from lobanalyzer.streaming import DayData
        
        features = np.random.randn(100, 98).astype(np.float32)
        labels = np.random.choice([-1, 0, 1], size=100)
        
        day = DayData(
            date="2025-01-01",
            features=features,
            labels=labels,
            n_samples=100,
            n_labels=100,
            is_multi_horizon=False,
            num_horizons=1,
        )
        
        assert day.date == "2025-01-01"
        assert day.n_samples == 100
        assert day.n_labels == 100
        assert day.is_multi_horizon is False
        assert day.num_horizons == 1
        assert day.get_labels(0).shape == (100,)
        assert np.array_equal(day.get_labels(0), labels)

    def test_multi_horizon_labels(self) -> None:
        """DayData handles multi-horizon labels correctly."""
        from lobanalyzer.streaming import DayData
        
        features = np.random.randn(100, 98).astype(np.float32)
        labels = np.random.choice([-1, 0, 1], size=(100, 5))
        
        day = DayData(
            date="2025-01-01",
            features=features,
            labels=labels,
            n_samples=100,
            n_labels=100,
            is_multi_horizon=True,
            num_horizons=5,
        )
        
        assert day.is_multi_horizon is True
        assert day.num_horizons == 5
        assert day.get_labels(0).shape == (100,)
        assert day.get_labels(2).shape == (100,)
        assert day.get_labels(None).shape == (100, 5)

    def test_get_labels_invalid_horizon(self) -> None:
        """get_labels raises error for invalid horizon index."""
        from lobanalyzer.streaming import DayData
        
        features = np.random.randn(100, 98).astype(np.float32)
        labels = np.random.choice([-1, 0, 1], size=100)
        
        day = DayData(
            date="2025-01-01",
            features=features,
            labels=labels,
            n_samples=100,
            n_labels=100,
            is_multi_horizon=False,
            num_horizons=1,
        )
        
        with pytest.raises(ValueError, match="Single-horizon"):
            day.get_labels(1)

    def test_memory_properties(self) -> None:
        """Memory properties work correctly."""
        from lobanalyzer.streaming import DayData
        
        features = np.zeros((100, 98), dtype=np.float32)
        labels = np.zeros(100, dtype=np.int8)
        
        day = DayData(
            date="2025-01-01",
            features=features,
            labels=labels,
            n_samples=100,
            n_labels=100,
        )
        
        expected_bytes = 100 * 98 * 4 + 100 * 1  # float32 + int8
        assert day.memory_bytes == expected_bytes
        assert day.memory_mb == expected_bytes / (1024 * 1024)


class TestAlignedDayData:
    """Test AlignedDayData container."""
    
    def test_single_horizon(self) -> None:
        """AlignedDayData handles single-horizon correctly."""
        from lobanalyzer.streaming import AlignedDayData
        
        features = np.random.randn(100, 98).astype(np.float32)
        labels = np.random.choice([-1, 0, 1], size=100)
        
        day = AlignedDayData(
            date="2025-01-01",
            features=features,
            labels=labels,
            n_pairs=100,
            is_multi_horizon=False,
            num_horizons=1,
        )
        
        assert day.n_pairs == 100
        assert len(day.features) == len(day.labels)
        assert day.get_labels(0).shape == (100,)

    def test_multi_horizon(self) -> None:
        """AlignedDayData handles multi-horizon correctly."""
        from lobanalyzer.streaming import AlignedDayData
        
        features = np.random.randn(100, 98).astype(np.float32)
        labels = np.random.choice([-1, 0, 1], size=(100, 3))
        
        day = AlignedDayData(
            date="2025-01-01",
            features=features,
            labels=labels,
            n_pairs=100,
            is_multi_horizon=True,
            num_horizons=3,
        )
        
        assert day.is_multi_horizon is True
        assert day.num_horizons == 3
        assert day.get_labels(0).shape == (100,)
        assert day.get_labels(None).shape == (100, 3)


class TestAlignmentFunction:
    """Test align_features_for_day function."""
    
    def test_alignment_formula(self) -> None:
        """Alignment uses correct formula: feat_idx = i * stride + window_size - 1."""
        from lobanalyzer.streaming import align_features_for_day
        
        # Create features with known values
        n_samples = 1000
        n_features = 98
        features = np.arange(n_samples * n_features).reshape(n_samples, n_features).astype(np.float32)
        
        # With window=100, stride=10, we expect:
        # label[0] -> feature[99]
        # label[1] -> feature[109]
        # etc.
        n_labels = 91  # (1000 - 100) // 10 + 1
        
        aligned = align_features_for_day(features, n_labels, window_size=100, stride=10)
        
        assert aligned.shape == (n_labels, n_features)
        
        # Check first few alignments
        assert np.array_equal(aligned[0], features[99])
        assert np.array_equal(aligned[1], features[109])
        assert np.array_equal(aligned[2], features[119])

    def test_alignment_boundary(self) -> None:
        """Alignment handles boundary cases correctly."""
        from lobanalyzer.streaming import align_features_for_day
        
        features = np.arange(500 * 98).reshape(500, 98).astype(np.float32)
        n_labels = 50
        
        aligned = align_features_for_day(features, n_labels, window_size=100, stride=10)
        
        # Last valid label should use feature at (49 * 10 + 100 - 1) = 589
        # But we only have 500 samples, so it should use 499
        assert aligned.shape == (50, 98)


class TestConstants:
    """Test streaming constants."""
    
    def test_constants_values(self) -> None:
        """Constants have expected values."""
        from lobanalyzer.streaming import WINDOW_SIZE, STRIDE
        
        assert WINDOW_SIZE == 100
        assert STRIDE == 10


class TestIterators:
    """Test iterator functions with mock data."""
    
    @pytest.fixture
    def mock_dataset_dir(self) -> Path:
        """Create a temporary mock dataset."""
        tmp_dir = tempfile.mkdtemp()
        data_dir = Path(tmp_dir)
        
        # Create train split with aligned format
        train_dir = data_dir / "train"
        train_dir.mkdir(parents=True)
        
        # Create 3 days of mock data
        for day in ["2025-01-01", "2025-01-02", "2025-01-03"]:
            # Aligned format: [N_seq, window_size, n_features]
            sequences = np.random.randn(100, 100, 98).astype(np.float32)
            labels = np.random.choice([-1, 0, 1], size=100)
            
            np.save(train_dir / f"{day}_sequences.npy", sequences)
            np.save(train_dir / f"{day}_labels.npy", labels)
        
        yield data_dir
        
        # Cleanup
        shutil.rmtree(tmp_dir)

    def test_iter_days(self, mock_dataset_dir: Path) -> None:
        """iter_days yields DayData for each day."""
        from lobanalyzer.streaming import iter_days
        
        days = list(iter_days(mock_dataset_dir, "train"))
        
        assert len(days) == 3
        assert days[0].date == "2025-01-01"
        assert days[1].date == "2025-01-02"
        assert days[2].date == "2025-01-03"
        
        for day in days:
            assert day.n_samples == 100
            assert day.n_labels == 100
            assert day.features.shape == (100, 98)

    def test_iter_days_aligned(self, mock_dataset_dir: Path) -> None:
        """iter_days_aligned yields AlignedDayData for each day."""
        from lobanalyzer.streaming import iter_days_aligned
        
        days = list(iter_days_aligned(mock_dataset_dir, "train"))
        
        assert len(days) == 3
        
        for day in days:
            assert day.n_pairs == 100
            assert len(day.features) == len(day.labels)

    def test_count_days(self, mock_dataset_dir: Path) -> None:
        """count_days returns correct count."""
        from lobanalyzer.streaming import count_days
        
        assert count_days(mock_dataset_dir, "train") == 3
        assert count_days(mock_dataset_dir, "val") == 0

    def test_get_dates(self, mock_dataset_dir: Path) -> None:
        """get_dates returns sorted list of dates."""
        from lobanalyzer.streaming import get_dates
        
        dates = get_dates(mock_dataset_dir, "train")
        
        assert dates == ["2025-01-01", "2025-01-02", "2025-01-03"]

    def test_missing_split_directory(self, mock_dataset_dir: Path) -> None:
        """iter_days raises FileNotFoundError for missing split."""
        from lobanalyzer.streaming import iter_days
        
        with pytest.raises(FileNotFoundError):
            list(iter_days(mock_dataset_dir, "nonexistent"))


class TestMultiHorizonSupport:
    """Test multi-horizon label support."""
    
    @pytest.fixture
    def mock_multi_horizon_dir(self) -> Path:
        """Create a temporary dataset with multi-horizon labels."""
        tmp_dir = tempfile.mkdtemp()
        data_dir = Path(tmp_dir)
        
        train_dir = data_dir / "train"
        train_dir.mkdir(parents=True)
        
        # Create data with multi-horizon labels
        sequences = np.random.randn(100, 100, 98).astype(np.float32)
        labels = np.random.choice([-1, 0, 1], size=(100, 5))  # 5 horizons
        
        np.save(train_dir / "2025-01-01_sequences.npy", sequences)
        np.save(train_dir / "2025-01-01_labels.npy", labels)
        
        yield data_dir
        
        shutil.rmtree(tmp_dir)

    def test_iter_days_multi_horizon(self, mock_multi_horizon_dir: Path) -> None:
        """iter_days detects multi-horizon labels."""
        from lobanalyzer.streaming import iter_days
        
        days = list(iter_days(mock_multi_horizon_dir, "train"))
        
        assert len(days) == 1
        day = days[0]
        
        assert day.is_multi_horizon is True
        assert day.num_horizons == 5
        assert day.get_labels(0).shape == (100,)
        assert day.get_labels(None).shape == (100, 5)

    def test_iter_days_aligned_multi_horizon(self, mock_multi_horizon_dir: Path) -> None:
        """iter_days_aligned handles multi-horizon labels."""
        from lobanalyzer.streaming import iter_days_aligned
        
        days = list(iter_days_aligned(mock_multi_horizon_dir, "train"))
        
        assert len(days) == 1
        day = days[0]
        
        assert day.is_multi_horizon is True
        assert day.num_horizons == 5


class TestPackageImports:
    """Test that streaming module exports work correctly."""
    
    def test_import_from_streaming(self) -> None:
        """Can import all exports from streaming module."""
        from lobanalyzer.streaming import (
            DayData,
            AlignedDayData,
            align_features_for_day,
            WINDOW_SIZE,
            STRIDE,
            iter_days,
            iter_days_aligned,
            count_days,
            get_dates,
        )
        
        assert WINDOW_SIZE == 100
        assert STRIDE == 10
