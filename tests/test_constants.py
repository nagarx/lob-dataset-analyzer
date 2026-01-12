"""
Tests for the constants module.

Validates that all feature indices, label encodings, and slices are correct
according to the Schema v2.1 data contract.

These tests document the exact behavior expected by the feature-extractor-MBO-LOB
Rust pipeline export format.
"""

import pytest


class TestFeatureCounts:
    """Test feature count constants."""

    def test_total_feature_count(self) -> None:
        """Total feature count must be exactly 98."""
        from lobanalyzer.constants import FEATURE_COUNT
        assert FEATURE_COUNT == 98, f"Expected 98 features, got {FEATURE_COUNT}"

    def test_feature_count_breakdown(self) -> None:
        """Feature counts must sum to 98."""
        from lobanalyzer.constants import (
            LOB_FEATURE_COUNT,
            DERIVED_FEATURE_COUNT,
            MBO_FEATURE_COUNT,
            SIGNAL_FEATURE_COUNT,
            FEATURE_COUNT,
        )
        expected_sum = LOB_FEATURE_COUNT + DERIVED_FEATURE_COUNT + MBO_FEATURE_COUNT + SIGNAL_FEATURE_COUNT
        assert expected_sum == 98, f"Feature sum: {expected_sum}"
        assert FEATURE_COUNT == expected_sum

    def test_individual_feature_counts(self) -> None:
        """Individual feature group counts."""
        from lobanalyzer.constants import (
            LOB_FEATURE_COUNT,
            DERIVED_FEATURE_COUNT,
            MBO_FEATURE_COUNT,
            SIGNAL_FEATURE_COUNT,
        )
        assert LOB_FEATURE_COUNT == 40, "Raw LOB should have 40 features (10 levels × 4)"
        assert DERIVED_FEATURE_COUNT == 8, "Derived should have 8 features"
        assert MBO_FEATURE_COUNT == 36, "MBO should have 36 features"
        assert SIGNAL_FEATURE_COUNT == 14, "Signals should have 14 features"


class TestSchemaVersion:
    """Test schema version constant."""

    def test_schema_version_value(self) -> None:
        """Schema version must be 2.1."""
        from lobanalyzer.constants import SCHEMA_VERSION
        assert SCHEMA_VERSION == 2.1, f"Expected 2.1, got {SCHEMA_VERSION}"

    def test_schema_version_type(self) -> None:
        """Schema version must be a float."""
        from lobanalyzer.constants import SCHEMA_VERSION
        assert isinstance(SCHEMA_VERSION, float)


class TestLabelEncoding:
    """Test label encoding constants."""

    def test_original_label_values(self) -> None:
        """Original labels are {-1, 0, 1}."""
        from lobanalyzer.constants import LABEL_DOWN, LABEL_STABLE, LABEL_UP
        assert LABEL_DOWN == -1, "Down label should be -1"
        assert LABEL_STABLE == 0, "Stable label should be 0"
        assert LABEL_UP == 1, "Up label should be 1"

    def test_shifted_label_values(self) -> None:
        """Shifted labels are {0, 1, 2} for PyTorch."""
        from lobanalyzer.constants import (
            SHIFTED_LABEL_DOWN,
            SHIFTED_LABEL_STABLE,
            SHIFTED_LABEL_UP,
        )
        assert SHIFTED_LABEL_DOWN == 0, "Shifted Down should be 0"
        assert SHIFTED_LABEL_STABLE == 1, "Shifted Stable should be 1"
        assert SHIFTED_LABEL_UP == 2, "Shifted Up should be 2"

    def test_num_classes(self) -> None:
        """Number of classes is 3."""
        from lobanalyzer.constants import NUM_CLASSES
        assert NUM_CLASSES == 3

    def test_label_names(self) -> None:
        """Label names mapping is correct."""
        from lobanalyzer.constants import LABEL_NAMES, SHIFTED_LABEL_NAMES
        
        assert LABEL_NAMES[-1] == "Down"
        assert LABEL_NAMES[0] == "Stable"
        assert LABEL_NAMES[1] == "Up"
        
        assert SHIFTED_LABEL_NAMES[0] == "Down"
        assert SHIFTED_LABEL_NAMES[1] == "Stable"
        assert SHIFTED_LABEL_NAMES[2] == "Up"

    def test_get_label_name_original(self) -> None:
        """get_label_name works for original encoding."""
        from lobanalyzer.constants import get_label_name
        
        assert get_label_name(-1, shifted=False) == "Down"
        assert get_label_name(0, shifted=False) == "Stable"
        assert get_label_name(1, shifted=False) == "Up"

    def test_get_label_name_shifted(self) -> None:
        """get_label_name works for shifted encoding."""
        from lobanalyzer.constants import get_label_name
        
        assert get_label_name(0, shifted=True) == "Down"
        assert get_label_name(1, shifted=True) == "Stable"
        assert get_label_name(2, shifted=True) == "Up"


class TestFeatureIndexEnum:
    """Test FeatureIndex enum values."""

    def test_lob_price_indices(self) -> None:
        """LOB price indices are in expected positions."""
        from lobanalyzer.constants import FeatureIndex
        
        # Ask prices: 0-9
        assert FeatureIndex.ASK_PRICE_L0 == 0
        assert FeatureIndex.ASK_PRICE_L9 == 9
        
        # Bid prices: 20-29
        assert FeatureIndex.BID_PRICE_L0 == 20
        assert FeatureIndex.BID_PRICE_L9 == 29

    def test_lob_size_indices(self) -> None:
        """LOB size indices are in expected positions."""
        from lobanalyzer.constants import FeatureIndex
        
        # Ask sizes: 10-19
        assert FeatureIndex.ASK_SIZE_L0 == 10
        assert FeatureIndex.ASK_SIZE_L9 == 19
        
        # Bid sizes: 30-39
        assert FeatureIndex.BID_SIZE_L0 == 30
        assert FeatureIndex.BID_SIZE_L9 == 39

    def test_derived_feature_indices(self) -> None:
        """Derived feature indices are in 40-47."""
        from lobanalyzer.constants import FeatureIndex
        
        assert FeatureIndex.MID_PRICE == 40
        assert FeatureIndex.SPREAD == 41
        assert FeatureIndex.SPREAD_BPS == 42
        assert FeatureIndex.TOTAL_BID_VOLUME == 43
        assert FeatureIndex.TOTAL_ASK_VOLUME == 44
        assert FeatureIndex.VOLUME_IMBALANCE == 45
        assert FeatureIndex.WEIGHTED_MID_PRICE == 46
        assert FeatureIndex.PRICE_IMPACT == 47

    def test_signal_indices(self) -> None:
        """Trading signal indices are in 84-97."""
        from lobanalyzer.constants import FeatureIndex
        
        assert FeatureIndex.TRUE_OFI == 84
        assert FeatureIndex.DEPTH_NORM_OFI == 85
        assert FeatureIndex.EXECUTED_PRESSURE == 86
        assert FeatureIndex.SIGNED_MP_DELTA_BPS == 87
        assert FeatureIndex.TRADE_ASYMMETRY == 88
        assert FeatureIndex.CANCEL_ASYMMETRY == 89
        assert FeatureIndex.FRAGILITY_SCORE == 90
        assert FeatureIndex.DEPTH_ASYMMETRY == 91
        
        # Safety gates
        assert FeatureIndex.BOOK_VALID == 92
        assert FeatureIndex.TIME_REGIME == 93
        assert FeatureIndex.MBO_READY == 94
        assert FeatureIndex.DT_SECONDS == 95
        assert FeatureIndex.INVALIDITY_DELTA == 96
        assert FeatureIndex.SCHEMA_VERSION_FEATURE == 97


class TestSignalIndex:
    """Test SignalIndex convenience enum."""

    def test_signal_index_values(self) -> None:
        """SignalIndex values match FeatureIndex."""
        from lobanalyzer.constants import FeatureIndex, SignalIndex
        
        assert SignalIndex.TRUE_OFI == FeatureIndex.TRUE_OFI
        assert SignalIndex.DEPTH_NORM_OFI == FeatureIndex.DEPTH_NORM_OFI
        assert SignalIndex.BOOK_VALID == FeatureIndex.BOOK_VALID
        assert SignalIndex.SCHEMA_VERSION == FeatureIndex.SCHEMA_VERSION_FEATURE

    def test_signal_index_range(self) -> None:
        """All SignalIndex values are in 84-97."""
        from lobanalyzer.constants import SignalIndex
        
        for signal in SignalIndex:
            assert 84 <= signal <= 97, f"{signal.name} = {signal} is out of range [84, 97]"


class TestFeatureSlices:
    """Test feature group slices."""

    def test_lob_slices(self) -> None:
        """LOB slices cover indices 0-39."""
        from lobanalyzer.constants import (
            LOB_ASK_PRICES,
            LOB_ASK_SIZES,
            LOB_BID_PRICES,
            LOB_BID_SIZES,
            LOB_ALL,
        )
        
        # Check slice boundaries
        assert LOB_ASK_PRICES == slice(0, 10)
        assert LOB_ASK_SIZES == slice(10, 20)
        assert LOB_BID_PRICES == slice(20, 30)
        assert LOB_BID_SIZES == slice(30, 40)
        assert LOB_ALL == slice(0, 40)

    def test_category_slices(self) -> None:
        """Category slices cover correct ranges."""
        from lobanalyzer.constants import DERIVED_ALL, MBO_ALL, SIGNALS_ALL
        
        assert DERIVED_ALL == slice(40, 48)
        assert MBO_ALL == slice(48, 84)
        assert SIGNALS_ALL == slice(84, 98)


class TestFeatureGroups:
    """Test feature group tuples."""

    def test_safety_gates(self) -> None:
        """SAFETY_GATES contains required gate features."""
        from lobanalyzer.constants import SAFETY_GATES, FeatureIndex
        
        assert FeatureIndex.BOOK_VALID in SAFETY_GATES
        assert FeatureIndex.MBO_READY in SAFETY_GATES
        assert FeatureIndex.INVALIDITY_DELTA in SAFETY_GATES
        assert len(SAFETY_GATES) == 3

    def test_primary_signals(self) -> None:
        """PRIMARY_SIGNALS contains most important signals."""
        from lobanalyzer.constants import PRIMARY_SIGNALS, FeatureIndex
        
        assert FeatureIndex.TRUE_OFI in PRIMARY_SIGNALS
        assert FeatureIndex.EXECUTED_PRESSURE in PRIMARY_SIGNALS
        assert FeatureIndex.BOOK_VALID in PRIMARY_SIGNALS

    def test_asymmetry_signals(self) -> None:
        """ASYMMETRY_SIGNALS contains all asymmetry features."""
        from lobanalyzer.constants import ASYMMETRY_SIGNALS, FeatureIndex
        
        assert FeatureIndex.TRADE_ASYMMETRY in ASYMMETRY_SIGNALS
        assert FeatureIndex.CANCEL_ASYMMETRY in ASYMMETRY_SIGNALS
        assert FeatureIndex.DEPTH_ASYMMETRY in ASYMMETRY_SIGNALS
        assert len(ASYMMETRY_SIGNALS) == 3


class TestSignConventions:
    """Test sign convention documentation."""

    def test_unsigned_features(self) -> None:
        """UNSIGNED_FEATURES contains only PRICE_IMPACT."""
        from lobanalyzer.constants import UNSIGNED_FEATURES, FeatureIndex
        
        assert len(UNSIGNED_FEATURES) == 1
        assert FeatureIndex.PRICE_IMPACT in UNSIGNED_FEATURES


class TestAnalysisConstants:
    """Test analysis-specific constants."""

    def test_signal_names(self) -> None:
        """SIGNAL_NAMES maps core signals to names."""
        from lobanalyzer.constants import SIGNAL_NAMES, FeatureIndex
        
        assert SIGNAL_NAMES[FeatureIndex.TRUE_OFI] == "true_ofi"
        assert SIGNAL_NAMES[FeatureIndex.DEPTH_NORM_OFI] == "depth_norm_ofi"
        assert SIGNAL_NAMES[FeatureIndex.EXECUTED_PRESSURE] == "executed_pressure"
        assert SIGNAL_NAMES[FeatureIndex.SIGNED_MP_DELTA_BPS] == "signed_mp_delta_bps"
        assert SIGNAL_NAMES[FeatureIndex.TRADE_ASYMMETRY] == "trade_asymmetry"
        assert SIGNAL_NAMES[FeatureIndex.CANCEL_ASYMMETRY] == "cancel_asymmetry"
        assert SIGNAL_NAMES[FeatureIndex.FRAGILITY_SCORE] == "fragility_score"
        assert SIGNAL_NAMES[FeatureIndex.DEPTH_ASYMMETRY] == "depth_asymmetry"
        # Safety gates are NOT in SIGNAL_NAMES
        assert FeatureIndex.BOOK_VALID not in SIGNAL_NAMES

    def test_core_signal_indices(self) -> None:
        """CORE_SIGNAL_INDICES contains indices 84-91."""
        from lobanalyzer.constants import CORE_SIGNAL_INDICES, FeatureIndex
        
        expected = [
            FeatureIndex.TRUE_OFI,
            FeatureIndex.DEPTH_NORM_OFI,
            FeatureIndex.EXECUTED_PRESSURE,
            FeatureIndex.SIGNED_MP_DELTA_BPS,
            FeatureIndex.TRADE_ASYMMETRY,
            FeatureIndex.CANCEL_ASYMMETRY,
            FeatureIndex.FRAGILITY_SCORE,
            FeatureIndex.DEPTH_ASYMMETRY,
        ]
        assert CORE_SIGNAL_INDICES == expected
        assert len(CORE_SIGNAL_INDICES) == 8


class TestPackageImports:
    """Test that the package structure works correctly."""

    def test_import_from_lobanalyzer(self) -> None:
        """Can import core constants from lobanalyzer directly."""
        from lobanalyzer import (
            FeatureIndex,
            SignalIndex,
            FEATURE_COUNT,
            SCHEMA_VERSION,
            LABEL_DOWN,
            LABEL_STABLE,
            LABEL_UP,
            NUM_CLASSES,
            LABEL_NAMES,
        )
        
        assert FEATURE_COUNT == 98
        assert SCHEMA_VERSION == 2.1
        assert NUM_CLASSES == 3

    def test_import_from_constants_submodule(self) -> None:
        """Can import from lobanalyzer.constants."""
        from lobanalyzer.constants import (
            LOB_FEATURE_COUNT,
            DERIVED_FEATURE_COUNT,
            MBO_FEATURE_COUNT,
            SIGNAL_FEATURE_COUNT,
            LOB_ALL,
            DERIVED_ALL,
            MBO_ALL,
            SIGNALS_ALL,
        )
        
        assert LOB_FEATURE_COUNT == 40
        assert DERIVED_FEATURE_COUNT == 8
        assert MBO_FEATURE_COUNT == 36
        assert SIGNAL_FEATURE_COUNT == 14
