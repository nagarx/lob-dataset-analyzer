"""
Authoritative Feature Index Map (v2.1 — 98 Features).

This module defines the EXACT feature indices matching the Rust pipeline export.
This is a DATA CONTRACT - any changes must be synchronized with:
    - feature-extractor-MBO-LOB/src/features/mod.rs
    - lob-model-trainer/src/lobtrainer/constants/
    - plan/03-FEATURE-INDEX-MAP-v2.md

Source: plan/03-FEATURE-INDEX-MAP-v2.md

Feature Layout Overview:
    | Range   | Count | Category        |
    |---------|-------|-----------------|
    | 0-39    | 40    | Raw LOB         |
    | 40-47   | 8     | Derived         |
    | 48-83   | 36    | MBO             |
    | 84-97   | 14    | Trading Signals |
    | Total   | 98    |                 |

Sign Convention (RULE.md §9):
    - All directional signals follow: > 0 = BULLISH, < 0 = BEARISH
    - Exception: PRICE_IMPACT (47) is unsigned - cannot determine direction

Schema Version History:
    - v2.0: Initial signal layer implementation
    - v2.1: Fixed sign convention for net_trade_flow (56) and net_cancel_flow (55)
"""

from enum import IntEnum
from typing import Final, Dict, List

# =============================================================================
# Schema Version
# =============================================================================

SCHEMA_VERSION: Final[float] = 2.1
"""
Schema version for feature export format. Must match Rust pipeline.

Version History:
    - 2.0: Initial signal layer
    - 2.1: Fixed net_trade_flow (56) and net_cancel_flow (55) sign convention
"""

# =============================================================================
# Label Encoding
# =============================================================================

LABEL_DOWN: Final[int] = -1
"""Price moved down (bearish)."""

LABEL_STABLE: Final[int] = 0
"""Price stayed within threshold (neutral)."""

LABEL_UP: Final[int] = 1
"""Price moved up (bullish)."""

NUM_CLASSES: Final[int] = 3
"""Number of label classes for classification."""

LABEL_NAMES: Final[Dict[int, str]] = {
    LABEL_DOWN: "Down",
    LABEL_STABLE: "Stable",
    LABEL_UP: "Up",
}
"""Human-readable label names (original encoding: {-1, 0, 1})."""

# =============================================================================
# Shifted Label Encoding (for PyTorch CrossEntropyLoss)
# =============================================================================
# PyTorch CrossEntropyLoss requires labels in {0, 1, ..., num_classes-1}.
# We shift original labels {-1, 0, 1} to {0, 1, 2} during training.
#
# Mapping: original + 1 = shifted
#   -1 (Down)   -> 0
#    0 (Stable) -> 1
#   +1 (Up)     -> 2

SHIFTED_LABEL_DOWN: Final[int] = 0
"""Down label after shift (original: -1)."""

SHIFTED_LABEL_STABLE: Final[int] = 1
"""Stable label after shift (original: 0)."""

SHIFTED_LABEL_UP: Final[int] = 2
"""Up label after shift (original: +1)."""

SHIFTED_LABEL_NAMES: Final[Dict[int, str]] = {
    SHIFTED_LABEL_DOWN: "Down",
    SHIFTED_LABEL_STABLE: "Stable",
    SHIFTED_LABEL_UP: "Up",
}
"""Human-readable label names (shifted encoding: {0, 1, 2})."""


def get_label_name(label: int, shifted: bool = False) -> str:
    """
    Get human-readable name for a label value.
    
    Args:
        label: Label value (-1/0/1 for original, 0/1/2 for shifted)
        shifted: True if using shifted encoding (PyTorch), False for original
    
    Returns:
        Label name: "Down", "Stable", or "Up"
    
    Example:
        >>> get_label_name(-1, shifted=False)  # "Down"
        >>> get_label_name(0, shifted=True)    # "Down"
    """
    mapping = SHIFTED_LABEL_NAMES if shifted else LABEL_NAMES
    return mapping.get(label, str(label))

# =============================================================================
# Feature Counts
# =============================================================================

LOB_FEATURE_COUNT: Final[int] = 40
"""Raw LOB features: 10 levels × 4 values (ask_prices, ask_sizes, bid_prices, bid_sizes)."""

DERIVED_FEATURE_COUNT: Final[int] = 8
"""Derived features: mid_price, spread, spread_bps, volumes, microprice, etc."""

MBO_FEATURE_COUNT: Final[int] = 36
"""MBO features: order flow rates, queue stats, institutional detection, etc."""

SIGNAL_FEATURE_COUNT: Final[int] = 14
"""Trading signals: OFI, asymmetry, regime, safety gates, etc."""

FEATURE_COUNT: Final[int] = (
    LOB_FEATURE_COUNT + DERIVED_FEATURE_COUNT + MBO_FEATURE_COUNT + SIGNAL_FEATURE_COUNT
)
"""Total feature count: 98."""

assert FEATURE_COUNT == 98, f"Feature count mismatch: expected 98, got {FEATURE_COUNT}"


# =============================================================================
# Feature Index Enum
# =============================================================================


class FeatureIndex(IntEnum):
    """
    Complete feature index mapping for 98-feature export (Schema v2.1).
    
    Usage:
        >>> features[:, FeatureIndex.TRUE_OFI]  # Access OFI signal
        >>> features[:, FeatureIndex.MID_PRICE]  # Access mid price
    
    Sign Conventions (RULE.md §9):
        - All directional signals: > 0 = BULLISH, < 0 = BEARISH
        - Exception: PRICE_IMPACT (47) is unsigned - do not use for direction
    
    LOB Feature Layout (matches Rust pipeline output):
        - Indices 0-9:   Ask prices (level 0-9, best ask at index 0)
        - Indices 10-19: Ask sizes (volume at each ask level)
        - Indices 20-29: Bid prices (level 0-9, best bid at index 20)
        - Indices 30-39: Bid sizes (volume at each bid level)
    
    This layout groups by SIDE first (ask, then bid), then by TYPE (prices, then sizes).
    Consistent with FI-2010 convention where ask comes before bid.
    
    Spread Invariant: ASK_PRICE_L0 > BID_PRICE_L0 (when book is valid)
    """
    
    # =========================================================================
    # Raw LOB Features (40) — Indices 0-39
    # =========================================================================
    
    # Ask prices (levels 0-9)
    ASK_PRICE_L0 = 0
    ASK_PRICE_L1 = 1
    ASK_PRICE_L2 = 2
    ASK_PRICE_L3 = 3
    ASK_PRICE_L4 = 4
    ASK_PRICE_L5 = 5
    ASK_PRICE_L6 = 6
    ASK_PRICE_L7 = 7
    ASK_PRICE_L8 = 8
    ASK_PRICE_L9 = 9
    
    # Ask sizes (levels 0-9)
    ASK_SIZE_L0 = 10
    ASK_SIZE_L1 = 11
    ASK_SIZE_L2 = 12
    ASK_SIZE_L3 = 13
    ASK_SIZE_L4 = 14
    ASK_SIZE_L5 = 15
    ASK_SIZE_L6 = 16
    ASK_SIZE_L7 = 17
    ASK_SIZE_L8 = 18
    ASK_SIZE_L9 = 19
    
    # Bid prices (levels 0-9)
    BID_PRICE_L0 = 20
    BID_PRICE_L1 = 21
    BID_PRICE_L2 = 22
    BID_PRICE_L3 = 23
    BID_PRICE_L4 = 24
    BID_PRICE_L5 = 25
    BID_PRICE_L6 = 26
    BID_PRICE_L7 = 27
    BID_PRICE_L8 = 28
    BID_PRICE_L9 = 29
    
    # Bid sizes (levels 0-9)
    BID_SIZE_L0 = 30
    BID_SIZE_L1 = 31
    BID_SIZE_L2 = 32
    BID_SIZE_L3 = 33
    BID_SIZE_L4 = 34
    BID_SIZE_L5 = 35
    BID_SIZE_L6 = 36
    BID_SIZE_L7 = 37
    BID_SIZE_L8 = 38
    BID_SIZE_L9 = 39
    
    # =========================================================================
    # Derived Features (8) — Indices 40-47
    # =========================================================================
    
    MID_PRICE = 40
    """Mid price: (best_bid + best_ask) / 2."""
    
    SPREAD = 41
    """Spread in dollars: best_ask - best_bid."""
    
    SPREAD_BPS = 42
    """Spread in basis points: spread / mid_price × 10000."""
    
    TOTAL_BID_VOLUME = 43
    """Total bid volume across all levels."""
    
    TOTAL_ASK_VOLUME = 44
    """Total ask volume across all levels."""
    
    VOLUME_IMBALANCE = 45
    """Volume imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol). Range: [-1, 1]."""
    
    WEIGHTED_MID_PRICE = 46
    """Microprice (Stoikov): (bid×ask_vol + ask×bid_vol) / (bid_vol + ask_vol)."""
    
    PRICE_IMPACT = 47
    """⚠️ UNSIGNED: |mid - microprice|. Cannot determine direction."""
    
    # =========================================================================
    # MBO Features (36) — Indices 48-83
    # =========================================================================
    
    # Order flow rates (6)
    ADD_RATE_BID = 48
    ADD_RATE_ASK = 49
    CANCEL_RATE_BID = 50
    CANCEL_RATE_ASK = 51
    TRADE_RATE_BID = 52
    TRADE_RATE_ASK = 53
    
    # Net flows (3)
    NET_ORDER_FLOW = 54
    """Net order flow: (add_bid - add_ask) / total. > 0 = BULLISH."""
    
    NET_CANCEL_FLOW = 55
    """Net cancel flow: (cancel_ask - cancel_bid) / total. > 0 = BULLISH."""
    
    NET_TRADE_FLOW = 56
    """Net trade flow: (trade_ask - trade_bid) / total. > 0 = BULLISH."""
    
    # Conviction indicators (3)
    AGGRESSIVE_ORDER_RATIO = 57
    ORDER_FLOW_VOLATILITY = 58
    FLOW_REGIME_INDICATOR = 59
    
    # Size distribution (8)
    SIZE_MEAN_BID = 60
    SIZE_MEAN_ASK = 61
    SIZE_STD_BID = 62
    SIZE_STD_ASK = 63
    SIZE_MAX_BID = 64
    SIZE_MAX_ASK = 65
    SIZE_SKEWNESS = 66
    SIZE_CONCENTRATION = 67
    
    # Queue depth (6)
    QUEUE_SIZE_BID = 68
    QUEUE_SIZE_ASK = 69
    AVERAGE_QUEUE_POSITION = 70
    LEVEL_CONCENTRATION = 71
    DEPTH_TICKS_BID = 72
    DEPTH_TICKS_ASK = 73
    
    # Institutional detection (4)
    LARGE_ORDER_COUNT_BID = 74
    LARGE_ORDER_COUNT_ASK = 75
    INSTITUTIONAL_RATIO_BID = 76
    INSTITUTIONAL_RATIO_ASK = 77
    
    # Core MBO metrics (6)
    ORDER_LIFETIME_MEAN = 78
    ORDER_LIFETIME_STD = 79
    CANCEL_RATIO = 80
    MODIFY_RATIO = 81
    QUEUE_SIZE_AHEAD = 82
    ORDER_COUNT_ACTIVE = 83
    
    # =========================================================================
    # Trading Signals (14) — Indices 84-97
    # =========================================================================
    
    TRUE_OFI = 84
    """Cont et al. (2014) Order Flow Imbalance. > 0 = BUY pressure."""
    
    DEPTH_NORM_OFI = 85
    """Depth-normalized OFI: true_ofi / avg_depth. > 0 = BUY pressure."""
    
    EXECUTED_PRESSURE = 86
    """Trade confirmation: trade_rate_ask - trade_rate_bid. > 0 = net buying."""
    
    SIGNED_MP_DELTA_BPS = 87
    """Microprice deviation from mid in bps. > 0 = BUY pressure."""
    
    TRADE_ASYMMETRY = 88
    """Normalized executed pressure. Range: [-1, 1]. > 0 = BUY pressure."""
    
    CANCEL_ASYMMETRY = 89
    """Cancel imbalance. Range: [-1, 1]. > 0 = bullish (sellers pulling)."""
    
    FRAGILITY_SCORE = 90
    """Book fragility: level_concentration / ln(avg_depth)."""
    
    DEPTH_ASYMMETRY = 91
    """Depth imbalance. Range: [-1, 1]. > 0 = bullish (stronger support)."""
    
    BOOK_VALID = 92
    """Safety gate: 1.0 if book is valid, 0.0 otherwise. MUST check."""
    
    TIME_REGIME = 93
    """Market session: 0=OPEN, 1=EARLY, 2=MIDDAY, 3=CLOSE, 4=CLOSED."""
    
    MBO_READY = 94
    """Warmup gate: 1.0 if MBO features are ready. MUST check."""
    
    DT_SECONDS = 95
    """Sample duration in seconds (wall-clock time)."""
    
    INVALIDITY_DELTA = 96
    """Feed quality: Count of crossed/locked events. 0 = clean."""
    
    SCHEMA_VERSION_FEATURE = 97
    """Fixed value = 2.1. For forward compatibility checking."""


# =============================================================================
# Signal-Specific Index Class (Convenience)
# =============================================================================


class SignalIndex(IntEnum):
    """
    Convenience enum for just the 14 trading signals (indices 84-97).
    
    Usage:
        >>> signals = features[:, SignalIndex.TRUE_OFI:SignalIndex.SCHEMA_VERSION + 1]
    """
    
    TRUE_OFI = 84
    DEPTH_NORM_OFI = 85
    EXECUTED_PRESSURE = 86
    SIGNED_MP_DELTA_BPS = 87
    TRADE_ASYMMETRY = 88
    CANCEL_ASYMMETRY = 89
    FRAGILITY_SCORE = 90
    DEPTH_ASYMMETRY = 91
    BOOK_VALID = 92
    TIME_REGIME = 93
    MBO_READY = 94
    DT_SECONDS = 95
    INVALIDITY_DELTA = 96
    SCHEMA_VERSION = 97


# =============================================================================
# Feature Groups (for selective loading)
# =============================================================================

# LOB level slices
LOB_ASK_PRICES = slice(0, 10)
LOB_ASK_SIZES = slice(10, 20)
LOB_BID_PRICES = slice(20, 30)
LOB_BID_SIZES = slice(30, 40)
LOB_ALL = slice(0, 40)

# Category slices
DERIVED_ALL = slice(40, 48)
MBO_ALL = slice(48, 84)
SIGNALS_ALL = slice(84, 98)

# Commonly used feature groups
SAFETY_GATES = (FeatureIndex.BOOK_VALID, FeatureIndex.MBO_READY, FeatureIndex.INVALIDITY_DELTA)
"""Features that must be checked before using other signals."""

PRIMARY_SIGNALS = (FeatureIndex.TRUE_OFI, FeatureIndex.EXECUTED_PRESSURE, FeatureIndex.BOOK_VALID)
"""Most important signals per research (Cont et al. 2014)."""

ASYMMETRY_SIGNALS = (
    FeatureIndex.TRADE_ASYMMETRY,
    FeatureIndex.CANCEL_ASYMMETRY,
    FeatureIndex.DEPTH_ASYMMETRY,
)
"""Normalized asymmetry signals, all in range [-1, 1]."""


# =============================================================================
# Sign Convention Notes
# =============================================================================

UNSIGNED_FEATURES = frozenset({FeatureIndex.PRICE_IMPACT})
"""Unsigned features. Cannot be used for directional signals."""


# =============================================================================
# Analysis-Specific Constants
# =============================================================================

SIGNAL_NAMES: Final[Dict[int, str]] = {
    FeatureIndex.TRUE_OFI: "true_ofi",
    FeatureIndex.DEPTH_NORM_OFI: "depth_norm_ofi",
    FeatureIndex.EXECUTED_PRESSURE: "executed_pressure",
    FeatureIndex.SIGNED_MP_DELTA_BPS: "signed_mp_delta_bps",
    FeatureIndex.TRADE_ASYMMETRY: "trade_asymmetry",
    FeatureIndex.CANCEL_ASYMMETRY: "cancel_asymmetry",
    FeatureIndex.FRAGILITY_SCORE: "fragility_score",
    FeatureIndex.DEPTH_ASYMMETRY: "depth_asymmetry",
}
"""Human-readable signal names for analysis output (excludes safety gates)."""

CORE_SIGNAL_INDICES: Final[List[int]] = [
    FeatureIndex.TRUE_OFI,
    FeatureIndex.DEPTH_NORM_OFI,
    FeatureIndex.EXECUTED_PRESSURE,
    FeatureIndex.SIGNED_MP_DELTA_BPS,
    FeatureIndex.TRADE_ASYMMETRY,
    FeatureIndex.CANCEL_ASYMMETRY,
    FeatureIndex.FRAGILITY_SCORE,
    FeatureIndex.DEPTH_ASYMMETRY,
]
"""Core signal indices for correlation analysis (indices 84-91)."""


def get_signal_info() -> Dict[int, Dict]:
    """
    Return metadata about each signal feature (indices 84-97).
    
    Returns:
        Dict mapping signal_index -> {name, description, type, expected_sign}
        
    Expected sign interpretation:
        - '+': Positive signal → expect Up label (bullish)
        - '-': Positive signal → expect Down label (bearish)
        - '?': Direction unclear or not applicable
        - 'N/A': Categorical/binary, not directional
    """
    return {
        84: {
            'name': 'true_ofi',
            'description': 'Cont et al. Order Flow Imbalance',
            'type': 'continuous',
            'expected_sign': '+',  # Positive OFI → expect Up
        },
        85: {
            'name': 'depth_norm_ofi',
            'description': 'OFI normalized by average depth',
            'type': 'continuous',
            'expected_sign': '+',
        },
        86: {
            'name': 'executed_pressure',
            'description': 'Net executed trade imbalance',
            'type': 'continuous',
            'expected_sign': '+',
        },
        87: {
            'name': 'signed_mp_delta_bps',
            'description': 'Microprice deviation from mid (bps)',
            'type': 'continuous',
            'expected_sign': '+',
        },
        88: {
            'name': 'trade_asymmetry',
            'description': 'Trade count imbalance ratio',
            'type': 'continuous',
            'expected_sign': '+',
        },
        89: {
            'name': 'cancel_asymmetry',
            'description': 'Cancel imbalance ratio',
            'type': 'continuous',
            'expected_sign': '+',
        },
        90: {
            'name': 'fragility_score',
            'description': 'Book concentration / ln(depth)',
            'type': 'continuous',
            'expected_sign': '?',
        },
        91: {
            'name': 'depth_asymmetry',
            'description': 'Depth imbalance ratio',
            'type': 'continuous',
            'expected_sign': '+',
        },
        92: {
            'name': 'book_valid',
            'description': 'Book validity flag',
            'type': 'binary',
            'expected_sign': 'N/A',
        },
        93: {
            'name': 'time_regime',
            'description': 'Market session encoding (0=OPEN, 1=EARLY, 2=MIDDAY, 3=CLOSE, 4=CLOSED)',
            'type': 'categorical',
            'expected_sign': 'N/A',
        },
        94: {
            'name': 'mbo_ready',
            'description': 'MBO warmup complete flag',
            'type': 'binary',
            'expected_sign': 'N/A',
        },
        95: {
            'name': 'dt_seconds',
            'description': 'Time since last sample',
            'type': 'continuous',
            'expected_sign': '?',
        },
        96: {
            'name': 'invalidity_delta',
            'description': 'Feed quality indicator (0 = clean)',
            'type': 'continuous',
            'expected_sign': 'N/A',
        },
        97: {
            'name': 'schema_version',
            'description': 'Export schema version (fixed value)',
            'type': 'categorical',
            'expected_sign': 'N/A',
        },
    }
