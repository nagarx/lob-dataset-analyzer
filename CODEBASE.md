# LOB-Dataset-Analyzer: Codebase Technical Reference

> **Version**: 0.1.0  
> **Schema**: 2.2  
> **Last Updated**: January 2026  
> **Purpose**: Technical reference for LLM coders and maintainers

---

## Table of Contents

1. [Overview](#1-overview)
2. [Module Architecture](#2-module-architecture)
3. [Constants Module](#3-constants-module)
4. [Streaming Module](#4-streaming-module)
5. [Analysis Module](#5-analysis-module)
6. [Validation Module](#6-validation-module)
7. [Testing Patterns](#7-testing-patterns)
8. [Integration with Pipeline](#8-integration-with-pipeline)

---

## 1. Overview

### Purpose

Python library for statistical analysis of LOB (Limit Order Book) feature data exported from the `feature-extractor-MBO-LOB` Rust pipeline. Designed for:

- **Memory-efficient streaming analysis** of large datasets
- **Data validation** and quality checks
- **Signal-label correlation analysis** for feature engineering
- **Reusable analysis functions** across different symbols/datasets

### Core Dependencies

```toml
[dependencies]
numpy = ">=1.24"      # Numerical operations
scipy = ">=1.10"      # Statistical tests
```

### Data Contract (Schema v2.2)

| Property | Value |
|----------|-------|
| Features | 98 per timestep |
| Sequence Layout | `(N_seq, 100, 98)` - pre-aligned |
| Labels (original) | `{-1, 0, 1}` = Down, Stable, Up |
| Multi-Horizon | Optional `(N_seq, H)` for multiple horizons |

---

## 2. Module Architecture

```
src/lobanalyzer/
├── __init__.py                    # Public API exports, version
├── constants/
│   ├── __init__.py                # Module exports
│   └── feature_index.py           # Feature indices, label encoding, schema version
│
├── streaming/
│   ├── __init__.py                # Module exports
│   ├── day_data.py                # DayData, AlignedDayData containers
│   └── iterators.py               # iter_days, iter_days_aligned
│
├── analysis/
│   ├── __init__.py                # Module exports (comprehensive)
│   ├── data_overview.py           # Data quality, validation, summaries
│   ├── label_analysis.py          # Label distribution, autocorrelation
│   ├── streaming_stats.py         # Online algorithms (Welford, counters)
│   ├── signal_stats.py            # ADF stationarity, rolling stats, distribution
│   ├── signal_correlations.py     # PCA, VIF, correlation matrix, clustering
│   ├── predictive_power.py        # Signal-label metrics, AUC, mutual information
│   ├── temporal_dynamics.py       # Autocorrelation, lead-lag, predictive decay
│   ├── generalization.py          # Walk-forward validation, day statistics
│   └── intraday_seasonality.py    # Regime-stratified analysis
│
└── validation/
    └── __init__.py                # Re-exports from analysis.data_overview

tests/
├── __init__.py
├── test_constants.py              # Feature index, label encoding tests
├── test_streaming.py              # DayData, iterators tests
├── test_analysis.py               # Core analysis function tests
├── test_signal_stats.py           # Stationarity, rolling stats tests
├── test_signal_correlations.py    # PCA, VIF, correlation tests
└── test_predictive_power.py       # Predictive metrics tests
```

---

## 3. Constants Module

### Feature Index Mapping

```python
class FeatureIndex:
    """Zero-based indices into the 98-feature vector."""
    
    # Raw LOB (40 features: indices 0-39)
    ASK_PRICE_0 = 0      # Best ask price
    ASK_PRICE_9 = 9      # Level 10 ask price
    ASK_SIZE_0 = 10      # Best ask size
    ASK_SIZE_9 = 19      # Level 10 ask size
    BID_PRICE_0 = 20     # Best bid price
    BID_PRICE_9 = 29     # Level 10 bid price
    BID_SIZE_0 = 30      # Best bid size
    BID_SIZE_9 = 39      # Level 10 bid size
    
    # Derived Features (8 features: indices 40-47)
    MID_PRICE = 40
    SPREAD = 41
    SPREAD_BPS = 42
    TOTAL_BID_VOLUME = 43
    TOTAL_ASK_VOLUME = 44
    VOLUME_IMBALANCE = 45
    WEIGHTED_MID_PRICE = 46
    PRICE_IMPACT = 47
    
    # MBO Features (36 features: indices 48-83)
    # ... order flow, size distribution, queue metrics
    
    # Trading Signals (14 features: indices 84-97)
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
    SCHEMA_VERSION_FEATURE = 97
```

### Label Encoding

```python
# Original labels (from Rust pipeline)
LABEL_DOWN: Final[int] = -1     # Price moved down
LABEL_STABLE: Final[int] = 0    # Price within threshold
LABEL_UP: Final[int] = 1        # Price moved up
NUM_CLASSES: Final[int] = 3

LABEL_NAMES: Final[Dict[int, str]] = {
    LABEL_DOWN: "Down",
    LABEL_STABLE: "Stable",
    LABEL_UP: "Up",
}
```

---

## 4. Streaming Module

Memory-efficient day-by-day iteration over datasets.

### DayData Container

```python
@dataclass
class DayData:
    """Container for a single day's raw data."""
    date: str                  # "YYYY-MM-DD"
    features: np.ndarray       # (N, 98)
    labels: np.ndarray         # (M,) or (M, n_horizons)
    n_samples: int
    n_labels: int
    is_multi_horizon: bool
    num_horizons: int
    
    def get_labels(self, horizon_idx: Optional[int] = 0) -> np.ndarray:
        """Get labels for a specific horizon."""
```

### AlignedDayData Container

```python
@dataclass
class AlignedDayData:
    """Container for aligned feature-label pairs (1:1 correspondence)."""
    date: str
    features: np.ndarray       # (N_labels, 98) - aligned with labels
    labels: np.ndarray         # (N_labels,) or (N_labels, n_horizons)
    n_pairs: int
    is_multi_horizon: bool
    num_horizons: int
```

### Iterators

```python
def iter_days(
    data_dir: Path,
    split: str,
    dtype: np.dtype = np.float32,
    mmap_mode: Optional[str] = None,
) -> Generator[DayData, None, None]:
    """Iterate over days, yielding raw data."""

def iter_days_aligned(
    data_dir: Path,
    split: str,
    window_size: int = 100,
    stride: int = 10,
    dtype: np.dtype = np.float32,
) -> Generator[AlignedDayData, None, None]:
    """Iterate over days, yielding aligned feature-label pairs."""
```

### Constants

```python
WINDOW_SIZE = 100  # Samples per sequence window
STRIDE = 10        # Samples between sequence starts
```

---

## 5. Analysis Module

### Data Quality

```python
@dataclass
class DataQuality:
    total_values: int
    finite_count: int
    nan_count: int
    inf_count: int
    pct_finite: float
    pct_nan: float
    pct_inf: float
    columns_with_nan: List[int]
    columns_with_inf: List[int]
    
    @property
    def is_clean(self) -> bool:
        return self.nan_count == 0 and self.inf_count == 0

def compute_data_quality(features: np.ndarray) -> DataQuality:
    """Compute data quality metrics for a feature array."""
```

### Label Distribution

```python
@dataclass
class LabelDistribution:
    total: int
    down_count: int
    stable_count: int
    up_count: int
    down_pct: float
    stable_pct: float
    up_pct: float
    imbalance_ratio: float
    
    @property
    def is_balanced(self) -> bool:
        return self.imbalance_ratio < 1.5

def compute_label_distribution(labels: np.ndarray) -> LabelDistribution:
    """Compute label distribution statistics."""
```

### Streaming Statistics (Welford's Algorithm)

```python
@dataclass
class RunningStats:
    """Online mean/variance computation."""
    n: int = 0
    mean: float = 0.0
    M2: float = 0.0
    min_val: float = float('inf')
    max_val: float = float('-inf')
    
    def update(self, x: float) -> None:
        """Update with a single value."""
    
    @property
    def variance(self) -> float:
        return self.M2 / self.n if self.n > 0 else 0.0
    
    @classmethod
    def merge(cls, a: 'RunningStats', b: 'RunningStats') -> 'RunningStats':
        """Merge two RunningStats (for parallel computation)."""
```

### Autocorrelation Analysis

```python
@dataclass
class AutocorrelationResult:
    lags: List[int]
    acf_values: List[float]
    confidence_interval: float
    lag_1_acf: float
    lag_5_acf: float
    lag_10_acf: float
    interpretation: str

def compute_autocorrelation(
    labels: np.ndarray,
    max_lag: int = 100,
) -> AutocorrelationResult:
    """Compute autocorrelation function for labels."""
```

### Transition Matrix

```python
@dataclass
class TransitionMatrix:
    labels: List[int]
    counts: List[List[int]]
    probabilities: List[List[float]]
    stationary_probs: List[float]
    persistence_deviation: Dict[str, float]

def compute_transition_matrix(labels: np.ndarray) -> TransitionMatrix:
    """Compute Markov transition matrix for labels."""
```

### Signal Statistics (signal_stats.py)

```python
@dataclass
class StationarityResult:
    """ADF stationarity test result."""
    signal_name: str
    adf_statistic: float
    p_value: float
    is_stationary: bool

@dataclass
class RollingStatsResult:
    """Rolling statistics for non-stationarity detection."""
    signal_name: str
    mean_drift: float
    std_drift: float
    is_mean_stable: bool

def compute_all_stationarity_tests(
    features: np.ndarray,
    signal_indices: List[int] = None,
) -> List[StationarityResult]:
    """Perform ADF tests for all signals."""

def compute_all_rolling_stats(
    features: np.ndarray,
    signal_indices: List[int] = None,
) -> List[RollingStatsResult]:
    """Compute rolling statistics for non-stationarity detection."""
```

### Signal Correlations (signal_correlations.py)

```python
@dataclass
class PCAResult:
    """PCA analysis result."""
    explained_variance_ratio: List[float]
    n_components_95: int  # Components for 95% variance
    dominant_signal_per_component: List[str]

@dataclass
class VIFResult:
    """Variance Inflation Factor result."""
    signal_name: str
    vif: float
    is_problematic: bool  # VIF > 5

def compute_signal_correlation_matrix(
    features: np.ndarray,
    signal_indices: List[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Compute correlation matrix between signals."""

def compute_pca_analysis(
    features: np.ndarray,
    signal_indices: List[int] = None,
) -> PCAResult:
    """Perform PCA on signals."""

def compute_vif(
    features: np.ndarray,
    signal_indices: List[int] = None,
) -> List[VIFResult]:
    """Compute VIF for multicollinearity detection."""
```

### Predictive Power (predictive_power.py)

```python
def compute_signal_metrics(
    signal: np.ndarray,
    labels: np.ndarray,
    expected_sign: str = '?',
) -> Dict:
    """
    Compute comprehensive predictive metrics for a signal.
    
    Returns:
        pearson_r, spearman_r: Correlations
        auc_up, auc_down: AUC for Up/Down classification
        mutual_info: Mutual information
        mean_up, mean_stable, mean_down: Conditional means
    """

def compute_binned_probabilities(
    signal: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Bin signal and compute label probabilities per bin."""

def identify_predictive_groups(
    df_metrics: pd.DataFrame,
    corr_matrix: Optional[np.ndarray] = None,
) -> Dict[str, List[Tuple[str, float]]]:
    """Group signals by predictive power and redundancy."""
```

### Temporal Dynamics (temporal_dynamics.py)

```python
@dataclass
class SignalAutocorrelation:
    """Autocorrelation analysis for a signal."""
    signal_name: str
    half_life: int  # Lag where ACF < 0.5
    decay_rate: float

@dataclass
class LeadLagRelation:
    """Lead-lag relationship between signals."""
    leader: str
    follower: str
    optimal_lag: int
    max_correlation: float

@dataclass
class PredictiveDecay:
    """Signal-label correlation decay with lag."""
    signal_name: str
    half_life: int  # Lag where |corr| halves
    optimal_lag: int

def run_temporal_dynamics_analysis(
    features: np.ndarray,
    labels: np.ndarray,
    signal_indices: List[int] = None,
) -> TemporalDynamicsSummary:
    """Run complete temporal dynamics analysis."""
```

### Generalization (generalization.py)

```python
@dataclass
class DayStatistics:
    """Statistics for a single trading day."""
    date: str
    n_samples: int
    label_up_pct: float

@dataclass
class SignalDayStats:
    """Per-day statistics for a signal."""
    signal_name: str
    mean_of_correlations: float
    stability_score: float

@dataclass
class WalkForwardResult:
    """Walk-forward validation fold result."""
    train_days: List[str]
    test_day: str
    signal_correlations: Dict[str, float]

def run_generalization_analysis(
    data_dir: Path,
    split: str = 'train',
) -> GeneralizationSummary:
    """Run complete generalization analysis."""
```

### Intraday Seasonality (intraday_seasonality.py)

```python
# Regime encoding (from Rust pipeline)
REGIME_NAMES = {
    0: "OPEN",    # 9:30-9:45 ET
    1: "EARLY",   # 9:45-10:30 ET
    2: "MIDDAY",  # 10:30-15:30 ET
    3: "CLOSE",   # 15:30-16:00 ET
    4: "CLOSED",  # After hours
}

@dataclass
class SignalRegimeCorrelation:
    """Signal-label correlation for a regime."""
    signal_name: str
    regime: int
    correlation: float
    is_significant: bool

@dataclass
class SignalSeasonality:
    """Seasonality analysis for a signal."""
    signal_name: str
    correlation_range: float
    is_regime_dependent: bool

def run_intraday_seasonality_analysis(
    features: np.ndarray,
    labels: np.ndarray,
) -> IntradaySeasonalitySummary:
    """Run intraday seasonality analysis."""
```

---

## 6. Validation Module

Re-exports validation utilities from `analysis.data_overview`:

```python
from lobanalyzer.validation import (
    validate_file_structure,
    compute_data_quality,
    compute_shape_validation,
    compute_all_categorical_validations,
)
```

---

## 7. Testing Patterns

### Test Structure

```
tests/
├── test_constants.py          # 27 tests: Feature indices, labels, schema
├── test_streaming.py          # 17 tests: DayData, iterators, alignment
├── test_analysis.py           # 19 tests: Quality, distribution, stats
├── test_signal_stats.py       # 30 tests: ADF, rolling stats, distribution
├── test_signal_correlations.py # 27 tests: PCA, VIF, correlation, clustering
└── test_predictive_power.py   # 25 tests: Metrics, binning, grouping
```

### Total: 144+ tests

### Example Tests

```python
def test_clean_data() -> None:
    """Data quality reports clean for valid data."""
    from lobanalyzer.analysis import compute_data_quality
    
    features = np.random.randn(100, 98).astype(np.float32)
    quality = compute_data_quality(features)
    
    assert quality.is_clean is True
    assert quality.nan_count == 0

def test_welford_mean_variance() -> None:
    """Running stats match numpy mean/variance."""
    from lobanalyzer.analysis import RunningStats
    
    data = np.random.randn(1000)
    stats = RunningStats()
    for x in data:
        stats.update(x)
    
    assert np.isclose(stats.mean, data.mean(), atol=1e-10)
    assert np.isclose(stats.variance, data.var(), atol=1e-10)
```

---

## 8. Integration with Pipeline

### Data Source

Data is exported from `feature-extractor-MBO-LOB` Rust pipeline:

```
data/exports/nvda_balanced/
├── train/
│   ├── 2025-02-03_sequences.npy   # (N_seq, 100, 98)
│   ├── 2025-02-03_labels.npy      # (N_seq,) or (N_seq, H)
│   └── ...
├── val/
└── test/
```

### Usage Example

```python
from pathlib import Path
from lobanalyzer.streaming import iter_days_aligned
from lobanalyzer.analysis import compute_data_quality, compute_label_distribution

data_dir = Path("../data/exports/nvda_balanced")

# Stream through data one day at a time
total_samples = 0
for day in iter_days_aligned(data_dir, "train"):
    # Validate data quality
    quality = compute_data_quality(day.features)
    if not quality.is_clean:
        print(f"Warning: {day.date} has NaN/Inf values")
    
    # Analyze labels
    labels = day.get_labels(0)
    dist = compute_label_distribution(labels)
    print(f"{day.date}: {dist.stable_pct:.1f}% stable")
    
    total_samples += day.n_pairs

print(f"Total samples: {total_samples}")
```

---

## Version History

| Version | Schema | Changes |
|---------|--------|---------|
| 0.2.0 | 2.2 | Full analysis suite: signal_stats, signal_correlations, predictive_power, temporal_dynamics, generalization, intraday_seasonality |
| 0.1.0 | 2.2 | Initial release: constants, streaming, basic analysis modules |

---

*Last updated: January 12, 2026*
