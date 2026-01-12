# LOB Dataset Analyzer

Statistical analysis library for limit order book feature data.

**Version**: 0.2.0 | **Schema**: 2.2 | **Tests**: 144 passing

## Overview

This library provides comprehensive tools for analyzing LOB feature data exported from the `feature-extractor-MBO-LOB` Rust pipeline. It emphasizes:

- **Memory efficiency** - Stream through large datasets one day at a time
- **Schema compliance** - Constants and validation matching the 98-feature schema
- **Reusability** - Analysis functions work across any symbol/dataset
- **Comprehensive analysis** - Signal statistics, correlations, predictive power, temporal dynamics

## Quick Start

```bash
# Install
cd lob-dataset-analyzer
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## Usage

### Stream Through Data

```python
from pathlib import Path
from lobanalyzer.streaming import iter_days, iter_days_aligned

# Iterate one day at a time (~40MB memory per day)
for day in iter_days(Path("../data/exports/nvda_balanced"), "train"):
    print(f"{day.date}: {day.n_samples} samples")

# Use aligned iterator for correlation analysis
for day in iter_days_aligned(Path("../data/exports/nvda_balanced"), "train"):
    labels = day.get_labels(0)  # First horizon
    ofi = day.features[:, 84]   # TRUE_OFI
    corr = np.corrcoef(ofi, labels)[0, 1]
    print(f"{day.date}: OFI correlation = {corr:.4f}")
```

### Validate Data Quality

```python
from lobanalyzer.validation import compute_data_quality

quality = compute_data_quality(features)
if not quality.is_clean:
    print(f"NaN columns: {quality.columns_with_nan}")
    print(f"Inf columns: {quality.columns_with_inf}")
```

### Analyze Labels

```python
from lobanalyzer.analysis import (
    compute_label_distribution,
    compute_autocorrelation,
    compute_transition_matrix,
)

# Distribution
dist = compute_label_distribution(labels)
print(f"Balanced: {dist.is_balanced}")
print(f"Stable %: {dist.stable_pct:.1f}%")

# Autocorrelation
acf = compute_autocorrelation(labels)
print(f"Lag-1 ACF: {acf.lag_1_acf:.4f}")
print(f"Interpretation: {acf.interpretation}")

# Transition matrix
tm = compute_transition_matrix(labels)
print(f"P(Up → Up): {tm.probabilities[2][2]:.3f}")
```

### Use Feature Constants

```python
from lobanalyzer.constants import (
    FEATURE_COUNT,
    FeatureIndex,
    SignalIndex,
    LABEL_NAMES,
)

# Access features by semantic name
ofi = features[:, FeatureIndex.TRUE_OFI]
spread = features[:, FeatureIndex.SPREAD_BPS]

# Signal range
signals = features[:, SignalIndex.TRUE_OFI:SignalIndex.DEPTH_ASYMMETRY + 1]

# Check label names
print(LABEL_NAMES[-1])  # "Down"
print(LABEL_NAMES[0])   # "Stable"
print(LABEL_NAMES[1])   # "Up"
```

## Data Contract (Schema v2.2)

| Property | Value |
|----------|-------|
| Features | 98 per timestep |
| Layout | `(N_seq, 100, 98)` for aligned format |
| Labels | `{-1, 0, 1}` = Down, Stable, Up |

### Feature Categories

| Range | Count | Category |
|-------|-------|----------|
| 0-39 | 40 | Raw LOB (10 levels × 4 values) |
| 40-47 | 8 | Derived (spread, microprice, etc.) |
| 48-83 | 36 | MBO (order flow, queue stats) |
| 84-97 | 14 | Trading Signals |

## Module Structure

```
lobanalyzer/
├── constants/     # Feature indices, labels, schema version
├── streaming/     # Memory-efficient day iteration
├── analysis/      # Full statistical analysis suite
│   ├── data_overview.py        # Data quality, validation
│   ├── label_analysis.py       # Distribution, autocorrelation
│   ├── signal_stats.py         # ADF stationarity, rolling stats
│   ├── signal_correlations.py  # PCA, VIF, correlation matrix
│   ├── predictive_power.py     # Signal-label metrics, AUC, MI
│   ├── temporal_dynamics.py    # Lead-lag, predictive decay
│   ├── generalization.py       # Walk-forward validation
│   └── intraday_seasonality.py # Regime-stratified analysis
└── validation/    # Data quality validation
```

## Key Modules

### `lobanalyzer.streaming`

```python
from lobanalyzer.streaming import (
    DayData,           # Raw day data container
    AlignedDayData,    # Aligned feature-label container
    iter_days,         # Iterate over raw data
    iter_days_aligned, # Iterate with alignment
    count_days,        # Count days without loading
    get_dates,         # Get date list without loading
)
```

### `lobanalyzer.analysis`

```python
from lobanalyzer.analysis import (
    # Data quality
    compute_data_quality,
    DataQuality,
    # Label analysis
    compute_label_distribution,
    compute_autocorrelation,
    compute_transition_matrix,
    # Streaming stats
    RunningStats,           # Welford's algorithm
    StreamingLabelCounter,  # Online label counting
    StreamingDataQuality,   # Streaming quality check
    # Signal statistics
    compute_all_stationarity_tests,  # ADF tests
    compute_all_rolling_stats,       # Rolling mean/std
    # Signal correlations
    compute_signal_correlation_matrix,
    compute_pca_analysis,
    compute_vif,
    # Predictive power
    compute_signal_metrics,          # Pearson/Spearman/AUC/MI
    compute_all_signal_metrics,      # For all signals
    compute_binned_probabilities,    # Non-linear analysis
    # Temporal dynamics
    compute_signal_autocorrelations, # Signal persistence
    compute_lead_lag_relations,      # Lead-lag detection
    run_temporal_dynamics_analysis,  # Full analysis
    # Generalization
    run_generalization_analysis,     # Walk-forward validation
    # Intraday seasonality
    run_intraday_seasonality_analysis,  # Regime analysis
)
```

### `lobanalyzer.constants`

```python
from lobanalyzer.constants import (
    FEATURE_COUNT,     # 98
    SCHEMA_VERSION,    # 2.2
    FeatureIndex,      # Enum with all indices
    SignalIndex,       # Signal-specific indices
    LABEL_DOWN,        # -1
    LABEL_STABLE,      # 0
    LABEL_UP,          # 1
)
```

## Running Tests

```bash
# All 144 tests
pytest tests/ -v

# Specific module
pytest tests/test_streaming.py -v
pytest tests/test_predictive_power.py -v
pytest tests/test_signal_correlations.py -v

# With coverage
pytest tests/ --cov=lobanalyzer --cov-report=term-missing
```

## Memory Efficiency

The library is designed for large datasets:

- **Per-day memory**: ~40MB (100K samples × 98 features × float32)
- **Total memory**: O(1 day) regardless of dataset size
- **Streaming stats**: O(n_features) for statistics

Example for 200 days of data:
- ❌ Load all: 200 × 40MB = 8GB
- ✅ Stream: ~40MB constant

## Related Documentation

- `CODEBASE.md` - Technical reference for developers
- `../feature-extractor-MBO-LOB/docs/full-data-pipeline.md` - Data pipeline docs

---

*Last updated: January 12, 2026*
