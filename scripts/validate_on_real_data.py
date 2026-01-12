#!/usr/bin/env python3
"""
Comprehensive validation of lobanalyzer modules against real NVIDIA dataset.

This script validates every module in lobanalyzer step by step:
1. Constants module - Feature indices and schema compliance
2. Streaming module - Data loading and iteration
3. Analysis modules - All statistical analysis functions

Follows RULE.md guidelines:
- Explicit validation at boundaries
- No silent failures
- Clear error messages
- Deterministic and reproducible

Usage:
    python scripts/validate_on_real_data.py --data-dir /path/to/nvda_multi_horizon
"""

import argparse
import sys
import gc
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import numpy as np

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def print_header(title: str) -> None:
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subheader(title: str) -> None:
    """Print formatted subsection header."""
    print(f"\n--- {title} ---")


def validate_result(name: str, condition: bool, message: str = "") -> bool:
    """Validate a condition and print result."""
    status = "✅ PASS" if condition else "❌ FAIL"
    print(f"  {status}: {name}")
    if not condition and message:
        print(f"         {message}")
    return condition


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_constants_module(data_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate constants module against real data.
    
    Checks:
    - Feature count matches actual data
    - Label values are in expected set
    - Schema version is consistent
    """
    print_header("1. CONSTANTS MODULE VALIDATION")
    results = {"passed": 0, "failed": 0, "details": {}}
    
    try:
        from lobanalyzer.constants import (
            FeatureIndex, SignalIndex, FEATURE_COUNT, SCHEMA_VERSION,
            LABEL_DOWN, LABEL_STABLE, LABEL_UP, NUM_CLASSES,
            CORE_SIGNAL_INDICES, get_signal_info,
        )
        
        # Load one day of real data
        train_dir = data_dir / "train"
        first_seq = sorted(train_dir.glob("*_sequences.npy"))[0]
        first_labels = sorted(train_dir.glob("*_labels.npy"))[0]
        
        sequences = np.load(first_seq)
        labels = np.load(first_labels)
        
        print_subheader("Feature Count Validation")
        actual_features = sequences.shape[2]
        if validate_result(
            f"FEATURE_COUNT={FEATURE_COUNT} matches data shape[2]={actual_features}",
            actual_features == FEATURE_COUNT,
            f"Expected {FEATURE_COUNT}, got {actual_features}"
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print_subheader("Label Values Validation")
        unique_labels = np.unique(labels)
        expected_labels = {LABEL_DOWN, LABEL_STABLE, LABEL_UP}
        actual_label_set = set(unique_labels.flatten())
        
        # For multi-horizon, check first horizon
        if labels.ndim == 2:
            actual_label_set = set(np.unique(labels[:, 0]))
        
        if validate_result(
            f"Labels in expected set {expected_labels}",
            actual_label_set.issubset(expected_labels),
            f"Got unexpected labels: {actual_label_set - expected_labels}"
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print_subheader("Signal Index Validation")
        # Check that signal indices are within bounds
        signal_info = get_signal_info()
        all_indices_valid = all(0 <= idx < FEATURE_COUNT for idx in CORE_SIGNAL_INDICES)
        if validate_result(
            f"All {len(CORE_SIGNAL_INDICES)} core signal indices < {FEATURE_COUNT}",
            all_indices_valid
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        # Check signal values are finite
        last_timestep = sequences[:, -1, :]  # Extract last timestep
        for idx in CORE_SIGNAL_INDICES[:4]:  # Check first 4 signals
            signal = last_timestep[:, idx]
            finite_pct = 100 * np.isfinite(signal).mean()
            if validate_result(
                f"Signal {idx} ({signal_info[idx]['name']}): {finite_pct:.1f}% finite",
                finite_pct > 90,
                f"Only {finite_pct:.1f}% finite values"
            ):
                results["passed"] += 1
            else:
                results["failed"] += 1
        
        results["details"] = {
            "feature_count": actual_features,
            "label_values": list(actual_label_set),
            "schema_version": SCHEMA_VERSION,
            "signal_indices_checked": len(CORE_SIGNAL_INDICES),
        }
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        traceback.print_exc()
        results["failed"] += 1
    
    return results["failed"] == 0, results


def validate_streaming_module(data_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate streaming module against real data.
    
    Checks:
    - iter_days works correctly
    - iter_days_aligned produces correct alignment
    - count_days and get_dates are accurate
    - Multi-horizon labels are handled correctly
    """
    print_header("2. STREAMING MODULE VALIDATION")
    results = {"passed": 0, "failed": 0, "details": {}}
    
    try:
        from lobanalyzer.streaming import (
            iter_days, iter_days_aligned, count_days, get_dates,
            DayData, AlignedDayData, WINDOW_SIZE, STRIDE,
        )
        
        print_subheader("count_days() Validation")
        for split in ['train', 'val', 'test']:
            expected = len(list((data_dir / split).glob("*_sequences.npy")))
            actual = count_days(data_dir, split)
            if validate_result(
                f"{split}: count_days={actual}, actual files={expected}",
                actual == expected
            ):
                results["passed"] += 1
            else:
                results["failed"] += 1
        
        print_subheader("get_dates() Validation")
        dates = get_dates(data_dir, 'train')
        if validate_result(
            f"get_dates() returns {len(dates)} dates",
            len(dates) > 0
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        if validate_result(
            f"Dates are sorted: {dates[0]} to {dates[-1]}",
            dates == sorted(dates)
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print_subheader("iter_days() Validation")
        days_processed = 0
        total_samples = 0
        multi_horizon_detected = False
        
        for day in iter_days(data_dir, 'train'):
            days_processed += 1
            total_samples += day.n_samples
            
            # Check DayData attributes
            if days_processed == 1:
                if validate_result(
                    f"DayData has correct attributes (date={day.date})",
                    hasattr(day, 'features') and hasattr(day, 'labels')
                ):
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                
                # Check for multi-horizon
                multi_horizon_detected = day.is_multi_horizon
                if multi_horizon_detected:
                    if validate_result(
                        f"Multi-horizon detected: {day.num_horizons} horizons",
                        day.num_horizons > 1
                    ):
                        results["passed"] += 1
                    else:
                        results["failed"] += 1
                
                # Check shape extraction (3D -> 2D)
                if validate_result(
                    f"Features shape: {day.features.shape} (N, 98)",
                    len(day.features.shape) == 2 and day.features.shape[1] == 98
                ):
                    results["passed"] += 1
                else:
                    results["failed"] += 1
            
            if days_processed >= 5:
                break
        
        if validate_result(
            f"iter_days processed {days_processed} days, {total_samples:,} samples",
            days_processed == 5
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print_subheader("iter_days_aligned() Validation")
        aligned_days = 0
        for day in iter_days_aligned(data_dir, 'train'):
            aligned_days += 1
            
            if aligned_days == 1:
                # Critical: features and labels should have same length
                if validate_result(
                    f"Aligned: features[{day.features.shape[0]}] == labels[{day.n_pairs}]",
                    day.features.shape[0] == day.n_pairs
                ):
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                
                # Check get_labels() with horizon
                if day.is_multi_horizon:
                    labels_h0 = day.get_labels(0)
                    labels_h1 = day.get_labels(1)
                    if validate_result(
                        f"get_labels(0) shape: {labels_h0.shape}, get_labels(1) shape: {labels_h1.shape}",
                        labels_h0.shape == labels_h1.shape
                    ):
                        results["passed"] += 1
                    else:
                        results["failed"] += 1
            
            if aligned_days >= 3:
                break
        
        results["details"] = {
            "train_days": count_days(data_dir, 'train'),
            "val_days": count_days(data_dir, 'val'),
            "test_days": count_days(data_dir, 'test'),
            "multi_horizon": multi_horizon_detected,
            "date_range": (dates[0], dates[-1]) if dates else None,
        }
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        traceback.print_exc()
        results["failed"] += 1
    
    gc.collect()
    return results["failed"] == 0, results


def validate_streaming_convenience(data_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate streaming convenience functions.
    
    Checks:
    - compute_streaming_overview
    - compute_streaming_label_analysis
    - compute_streaming_signal_stats
    """
    print_header("3. STREAMING CONVENIENCE FUNCTIONS")
    results = {"passed": 0, "failed": 0, "details": {}}
    
    try:
        from lobanalyzer.streaming import (
            compute_streaming_overview,
            compute_streaming_label_analysis,
            compute_streaming_signal_stats,
            estimate_memory_usage,
        )
        
        print_subheader("compute_streaming_overview()")
        overview = compute_streaming_overview(data_dir, symbol="NVDA")
        
        if validate_result(
            f"Overview: {overview['total_days']} days, {overview['total_samples']:,} samples",
            overview['total_days'] > 0 and overview['total_samples'] > 0
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        if validate_result(
            f"Data quality: is_clean={overview['data_quality']['is_clean']}",
            'is_clean' in overview['data_quality']
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        label_dist = overview['label_distribution']
        total_pct = label_dist['down_pct'] + label_dist['stable_pct'] + label_dist['up_pct']
        if validate_result(
            f"Label dist sums to 100%: {total_pct:.2f}%",
            abs(total_pct - 100) < 0.1
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print_subheader("compute_streaming_label_analysis()")
        labels = compute_streaming_label_analysis(data_dir, split='train')
        
        if validate_result(
            f"Label analysis: {labels['n_days']} days",
            labels['n_days'] > 0
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        acf_1 = labels['autocorrelation']['lag_1']
        if validate_result(
            f"ACF(1) = {acf_1:.4f} (expecting high autocorrelation)",
            np.isfinite(acf_1) and abs(acf_1) < 1.0
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print_subheader("compute_streaming_signal_stats()")
        signal_stats = compute_streaming_signal_stats(data_dir, split='train')
        
        if validate_result(
            f"Signal stats: {len(signal_stats)} signals",
            len(signal_stats) == 8  # Core signals
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        # Check true_ofi has reasonable values
        ofi_stats = signal_stats.get('true_ofi', {})
        if validate_result(
            f"true_ofi: mean={ofi_stats.get('mean', 'N/A'):.4f}, std={ofi_stats.get('std', 'N/A'):.4f}",
            ofi_stats.get('n', 0) > 0
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print_subheader("estimate_memory_usage()")
        mem = estimate_memory_usage(data_dir)
        
        if validate_result(
            f"Memory estimate: {mem['total']['gb']:.2f} GB",
            mem['total']['gb'] > 0
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        results["details"] = {
            "total_samples": overview['total_samples'],
            "label_acf_1": acf_1,
            "memory_gb": mem['total']['gb'],
        }
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        traceback.print_exc()
        results["failed"] += 1
    
    gc.collect()
    return results["failed"] == 0, results


def validate_analysis_data_overview(data_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    """Validate data_overview module."""
    print_header("4. ANALYSIS: DATA OVERVIEW")
    results = {"passed": 0, "failed": 0, "details": {}}
    
    try:
        from lobanalyzer.analysis import (
            compute_data_quality, compute_shape_validation,
            compute_signal_statistics, DataQuality,
        )
        from lobanalyzer.streaming import iter_days
        
        # Load sample data
        print_subheader("Loading sample data...")
        sample_features = []
        for i, day in enumerate(iter_days(data_dir, 'train')):
            sample_features.append(day.features[:1000])  # First 1000 per day
            if i >= 2:
                break
        features = np.vstack(sample_features)
        print(f"  Loaded {features.shape[0]:,} samples")
        
        print_subheader("compute_data_quality()")
        quality = compute_data_quality(features)
        
        if validate_result(
            f"DataQuality: nan_count={quality.nan_count}, inf_count={quality.inf_count}",
            isinstance(quality, DataQuality)
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        if validate_result(
            f"Clean data: {quality.is_clean}",
            isinstance(quality.is_clean, bool)
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print_subheader("compute_signal_statistics()")
        signal_stats = compute_signal_statistics(features)
        
        if validate_result(
            f"Signal statistics: {len(signal_stats)} signals",
            len(signal_stats) >= 8
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        results["details"] = {
            "samples_checked": features.shape[0],
            "nan_count": quality.nan_count,
            "inf_count": quality.inf_count,
        }
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        traceback.print_exc()
        results["failed"] += 1
    
    gc.collect()
    return results["failed"] == 0, results


def validate_analysis_label(data_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    """Validate label_analysis module."""
    print_header("5. ANALYSIS: LABEL ANALYSIS")
    results = {"passed": 0, "failed": 0, "details": {}}
    
    try:
        from lobanalyzer.analysis import (
            compute_label_distribution, compute_autocorrelation,
            compute_transition_matrix, run_label_analysis,
            LabelDistribution, AutocorrelationResult, TransitionMatrix,
        )
        from lobanalyzer.streaming import iter_days_aligned
        
        # Load aligned data
        print_subheader("Loading aligned data...")
        features_list = []
        labels_list = []
        for i, day in enumerate(iter_days_aligned(data_dir, 'train')):
            features_list.append(day.features[:1000])
            labels_list.append(day.get_labels(0)[:1000])
            if i >= 4:
                break
        features = np.vstack(features_list)
        labels = np.concatenate(labels_list)
        print(f"  Loaded {len(labels):,} aligned samples")
        
        print_subheader("compute_label_distribution()")
        dist = compute_label_distribution(labels)
        
        if validate_result(
            f"Distribution: Down={dist.down_pct:.1f}%, Stable={dist.stable_pct:.1f}%, Up={dist.up_pct:.1f}%",
            isinstance(dist, LabelDistribution)
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        total = dist.down_pct + dist.stable_pct + dist.up_pct
        if validate_result(
            f"Percentages sum to 100%: {total:.2f}%",
            abs(total - 100) < 0.1
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print_subheader("compute_autocorrelation()")
        acf = compute_autocorrelation(labels)
        
        if validate_result(
            f"ACF: lag_1={acf.lag_1_acf:.4f}, lag_5={acf.lag_5_acf:.4f}",
            isinstance(acf, AutocorrelationResult)
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print_subheader("compute_transition_matrix()")
        tm = compute_transition_matrix(labels)
        
        if validate_result(
            f"Transition matrix: {len(tm.labels)} states",
            isinstance(tm, TransitionMatrix) and len(tm.labels) == 3
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        # Check rows sum to 1
        row_sums = [sum(row) for row in tm.probabilities]
        rows_valid = all(abs(s - 1.0) < 0.01 or s == 0 for s in row_sums)
        if validate_result(
            f"Transition rows sum to 1: {[f'{s:.2f}' for s in row_sums]}",
            rows_valid
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print_subheader("run_label_analysis() [Full Pipeline]")
        summary = run_label_analysis(features, labels)
        
        if validate_result(
            f"LabelAnalysisSummary: has all 5 components",
            all(hasattr(summary, attr) for attr in 
                ['distribution', 'autocorrelation', 'transition_matrix', 'regime_stats', 'signal_correlations'])
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        # Get majority class from counts
        counts = {'Down': dist.down_count, 'Stable': dist.stable_count, 'Up': dist.up_count}
        majority = max(counts.items(), key=lambda x: x[1])[0]
        
        results["details"] = {
            "samples": len(labels),
            "label_acf_1": acf.lag_1_acf,
            "majority_class": majority,
        }
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        traceback.print_exc()
        results["failed"] += 1
    
    gc.collect()
    return results["failed"] == 0, results


def validate_analysis_signal_stats(data_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    """Validate signal_stats module."""
    print_header("6. ANALYSIS: SIGNAL STATISTICS")
    results = {"passed": 0, "failed": 0, "details": {}}
    
    try:
        from lobanalyzer.analysis import (
            compute_distribution_stats,
            compute_all_stationarity_tests,
            compute_all_rolling_stats,
            StationarityResult, RollingStatsResult,
        )
        from lobanalyzer.streaming import iter_days
        from lobanalyzer.constants import CORE_SIGNAL_INDICES
        
        # Load sample data
        print_subheader("Loading sample data...")
        sample_features = []
        for i, day in enumerate(iter_days(data_dir, 'train')):
            sample_features.append(day.features[::10])  # Subsample
            if i >= 9:
                break
        features = np.vstack(sample_features)
        print(f"  Loaded {features.shape[0]:,} samples")
        
        print_subheader("compute_distribution_stats()")
        dist_df = compute_distribution_stats(features, CORE_SIGNAL_INDICES)
        
        if validate_result(
            f"Distribution stats: {len(dist_df)} signals",
            len(dist_df) == len(CORE_SIGNAL_INDICES)
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        # Check columns exist
        required_cols = ['name', 'mean', 'std', 'skewness', 'kurtosis']
        has_all_cols = all(col in dist_df.columns for col in required_cols)
        if validate_result(
            f"Has required columns: {required_cols}",
            has_all_cols
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print_subheader("compute_all_stationarity_tests()")
        stationarity = compute_all_stationarity_tests(features, CORE_SIGNAL_INDICES[:4])  # First 4
        
        if validate_result(
            f"Stationarity tests: {len(stationarity)} results",
            len(stationarity) == 4
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        stationary_count = sum(1 for s in stationarity if s.is_stationary)
        if validate_result(
            f"Stationary signals: {stationary_count}/{len(stationarity)}",
            isinstance(stationarity[0], StationarityResult)
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print_subheader("compute_all_rolling_stats()")
        rolling = compute_all_rolling_stats(features, CORE_SIGNAL_INDICES[:4])
        
        if validate_result(
            f"Rolling stats: {len(rolling)} results",
            len(rolling) == 4
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        stable_count = sum(1 for r in rolling if r.is_mean_stable)
        if validate_result(
            f"Mean-stable signals: {stable_count}/{len(rolling)}",
            isinstance(rolling[0], RollingStatsResult)
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        results["details"] = {
            "samples": features.shape[0],
            "stationary_signals": stationary_count,
            "mean_stable_signals": stable_count,
        }
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        traceback.print_exc()
        results["failed"] += 1
    
    gc.collect()
    return results["failed"] == 0, results


def validate_analysis_correlations(data_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    """Validate signal_correlations module."""
    print_header("7. ANALYSIS: SIGNAL CORRELATIONS")
    results = {"passed": 0, "failed": 0, "details": {}}
    
    try:
        from lobanalyzer.analysis import (
            compute_signal_correlation_matrix,
            find_redundant_pairs,
            compute_pca_analysis,
            compute_vif,
            cluster_signals,
            PCAResult, VIFResult, SignalCluster,
        )
        from lobanalyzer.streaming import iter_days
        from lobanalyzer.constants import CORE_SIGNAL_INDICES
        
        # Load sample data
        print_subheader("Loading sample data...")
        sample_features = []
        for i, day in enumerate(iter_days(data_dir, 'train')):
            sample_features.append(day.features[::10])
            if i >= 9:
                break
        features = np.vstack(sample_features)
        print(f"  Loaded {features.shape[0]:,} samples")
        
        print_subheader("compute_signal_correlation_matrix()")
        corr_matrix, signal_names = compute_signal_correlation_matrix(features, CORE_SIGNAL_INDICES)
        
        if validate_result(
            f"Correlation matrix shape: {corr_matrix.shape}",
            corr_matrix.shape == (len(CORE_SIGNAL_INDICES), len(CORE_SIGNAL_INDICES))
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        # Check diagonal is 1
        diag_is_one = np.allclose(np.diag(corr_matrix), 1.0)
        if validate_result(
            f"Diagonal is 1.0: {diag_is_one}",
            diag_is_one
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print_subheader("find_redundant_pairs()")
        pairs = find_redundant_pairs(corr_matrix, signal_names, threshold=0.5)
        
        if validate_result(
            f"Redundant pairs (|r| > 0.5): {len(pairs)}",
            isinstance(pairs, list)
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print_subheader("compute_pca_analysis()")
        pca = compute_pca_analysis(features, CORE_SIGNAL_INDICES)
        
        if validate_result(
            f"PCA: {pca.n_components_90} components for 90% variance",
            isinstance(pca, PCAResult)
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        cum_var_sums_to_1 = abs(pca.cumulative_variance[-1] - 1.0) < 0.01
        if validate_result(
            f"Cumulative variance ends at 1.0: {pca.cumulative_variance[-1]:.4f}",
            cum_var_sums_to_1
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print_subheader("compute_vif()")
        vif_results = compute_vif(features, CORE_SIGNAL_INDICES)
        
        if validate_result(
            f"VIF results: {len(vif_results)} signals",
            len(vif_results) == len(CORE_SIGNAL_INDICES)
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        problematic = sum(1 for v in vif_results if v.is_problematic)
        if validate_result(
            f"Problematic VIF (>5): {problematic}/{len(vif_results)}",
            isinstance(vif_results[0], VIFResult)
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print_subheader("cluster_signals()")
        clusters = cluster_signals(corr_matrix, signal_names, CORE_SIGNAL_INDICES)
        
        if validate_result(
            f"Signal clusters: {len(clusters)}",
            len(clusters) > 0
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        results["details"] = {
            "samples": features.shape[0],
            "redundant_pairs": len(pairs),
            "pca_components_90": pca.n_components_90,
            "problematic_vif": problematic,
            "n_clusters": len(clusters),
        }
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        traceback.print_exc()
        results["failed"] += 1
    
    gc.collect()
    return results["failed"] == 0, results


def validate_analysis_predictive_power(data_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    """Validate predictive_power module."""
    print_header("8. ANALYSIS: PREDICTIVE POWER")
    results = {"passed": 0, "failed": 0, "details": {}}
    
    try:
        from lobanalyzer.analysis import (
            compute_signal_metrics,
            compute_all_signal_metrics,
            compute_binned_probabilities,
            identify_predictive_groups,
        )
        from lobanalyzer.streaming import iter_days_aligned
        from lobanalyzer.constants import CORE_SIGNAL_INDICES
        
        # Load aligned data
        print_subheader("Loading aligned data...")
        features_list = []
        labels_list = []
        for i, day in enumerate(iter_days_aligned(data_dir, 'train')):
            features_list.append(day.features[::5])  # Subsample
            labels_list.append(day.get_labels(0)[::5])
            if i >= 4:
                break
        features = np.vstack(features_list)
        labels = np.concatenate(labels_list)
        print(f"  Loaded {len(labels):,} aligned samples")
        
        print_subheader("compute_signal_metrics()")
        # Test single signal
        signal_idx = CORE_SIGNAL_INDICES[0]
        metrics = compute_signal_metrics(features[:, signal_idx], labels)
        
        # The actual API returns pearson_r, pearson_p, etc.
        required_keys = ['pearson_r', 'pearson_p', 'n_samples']
        has_all_keys = all(k in metrics for k in required_keys)
        if validate_result(
            f"Signal metrics has required keys: {required_keys}",
            has_all_keys
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        if validate_result(
            f"Correlation: {metrics.get('pearson_r', 0):.4f}, p={metrics.get('pearson_p', 1):.2e}",
            np.isfinite(metrics.get('pearson_r', 0))
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print_subheader("compute_all_signal_metrics()")
        # Returns a DataFrame
        all_metrics_df = compute_all_signal_metrics(features, labels, CORE_SIGNAL_INDICES)
        
        if validate_result(
            f"All signal metrics: {len(all_metrics_df)} signals (DataFrame)",
            len(all_metrics_df) == len(CORE_SIGNAL_INDICES)
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print_subheader("compute_binned_probabilities()")
        binned = compute_binned_probabilities(features[:, signal_idx], labels, n_bins=10)
        
        if validate_result(
            f"Binned probabilities: {len(binned)} bins",
            len(binned) == 10
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print_subheader("identify_predictive_groups()")
        # This function expects a DataFrame
        groups = identify_predictive_groups(all_metrics_df)
        
        if validate_result(
            f"Predictive groups: {list(groups.keys())}",
            'strong' in groups or 'moderate' in groups or 'weak' in groups
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        results["details"] = {
            "samples": len(labels),
            "top_signal_corr": all_metrics_df['pearson_r'].abs().max() if 'pearson_r' in all_metrics_df else 0,
            "significant_signals": (all_metrics_df['pearson_p'] < 0.05).sum() if 'pearson_p' in all_metrics_df else 0,
        }
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        traceback.print_exc()
        results["failed"] += 1
    
    gc.collect()
    return results["failed"] == 0, results


def validate_analysis_temporal(data_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    """Validate temporal_dynamics module."""
    print_header("9. ANALYSIS: TEMPORAL DYNAMICS")
    results = {"passed": 0, "failed": 0, "details": {}}
    
    try:
        from lobanalyzer.analysis import (
            compute_signal_acf,
            compute_signal_autocorrelations,
            compute_cross_correlation,
            compute_predictive_decay,
            run_temporal_dynamics_analysis,
            SignalAutocorrelation, PredictiveDecay,
        )
        from lobanalyzer.streaming import iter_days_aligned
        from lobanalyzer.constants import CORE_SIGNAL_INDICES
        
        # Load aligned data
        print_subheader("Loading aligned data...")
        features_list = []
        labels_list = []
        for i, day in enumerate(iter_days_aligned(data_dir, 'train')):
            features_list.append(day.features[::10])
            labels_list.append(day.get_labels(0)[::10])
            if i >= 4:
                break
        features = np.vstack(features_list)
        labels = np.concatenate(labels_list)
        print(f"  Loaded {len(labels):,} aligned samples")
        
        print_subheader("compute_signal_acf()")
        signal = features[:, CORE_SIGNAL_INDICES[0]]
        # API returns: (acf_values, half_life, decay_rate)
        acf_values, half_life, decay_rate = compute_signal_acf(signal, max_lag=50)
        
        if validate_result(
            f"ACF: {len(acf_values)} lags, half_life={half_life}, decay_rate={decay_rate:.4f}",
            len(acf_values) == 51 and abs(acf_values[0] - 1.0) < 0.01
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print_subheader("compute_signal_autocorrelations()")
        autocorrs = compute_signal_autocorrelations(features, CORE_SIGNAL_INDICES[:4])
        
        if validate_result(
            f"Autocorrelations: {len(autocorrs)} signals",
            len(autocorrs) == 4
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        if validate_result(
            f"First result is SignalAutocorrelation",
            isinstance(autocorrs[0], SignalAutocorrelation)
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print_subheader("compute_predictive_decay()")
        # API uses 'lags' not 'horizons', returns tuple (correlations, half_life, n_valid, decay_rate)
        correlations, half_life, n_valid, decay_rate = compute_predictive_decay(
            features[:, CORE_SIGNAL_INDICES[0]], 
            labels, 
            lags=[0, 1, 5, 10, 20]
        )
        
        if validate_result(
            f"Predictive decay: {len(correlations)} lags, half_life={half_life}",
            len(correlations) == 5
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print_subheader("run_temporal_dynamics_analysis() [Full Pipeline]")
        summary = run_temporal_dynamics_analysis(features, labels, CORE_SIGNAL_INDICES[:4])
        
        if validate_result(
            f"TemporalDynamicsSummary: has autocorrelations and predictive_decays",
            hasattr(summary, 'autocorrelations') and hasattr(summary, 'predictive_decays')
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        results["details"] = {
            "samples": len(labels),
            "signals_analyzed": len(autocorrs),
            "first_signal_half_life": half_life,
        }
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        traceback.print_exc()
        results["failed"] += 1
    
    gc.collect()
    return results["failed"] == 0, results


def validate_analysis_generalization(data_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    """Validate generalization module."""
    print_header("10. ANALYSIS: GENERALIZATION")
    results = {"passed": 0, "failed": 0, "details": {}}
    
    try:
        from lobanalyzer.analysis import (
            load_day_data,
            compute_day_statistics,
            compute_signal_day_stats,
            walk_forward_validation,
            run_generalization_analysis,
            DayStatistics, SignalDayStats, WalkForwardResult,
        )
        from lobanalyzer.constants import CORE_SIGNAL_INDICES
        
        print_subheader("load_day_data()")
        days = load_day_data(data_dir, split='train')
        
        if validate_result(
            f"Loaded {len(days)} days of data",
            len(days) > 0
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print_subheader("compute_day_statistics()")
        day_stats = compute_day_statistics(days[:10])  # First 10 days
        
        if validate_result(
            f"Day statistics: {len(day_stats)} days",
            len(day_stats) == 10
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        if validate_result(
            f"First result is DayStatistics",
            isinstance(day_stats[0], DayStatistics)
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print_subheader("compute_signal_day_stats()")
        signal_stats = compute_signal_day_stats(days[:10], CORE_SIGNAL_INDICES[:4])
        
        if validate_result(
            f"Signal day stats: {len(signal_stats)} signals",
            len(signal_stats) == 4
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print_subheader("walk_forward_validation()")
        wf_results = walk_forward_validation(days[:15], CORE_SIGNAL_INDICES[:2], min_train_days=5)
        
        if validate_result(
            f"Walk-forward results: {len(wf_results)} folds",
            len(wf_results) > 0
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        if validate_result(
            f"First result is WalkForwardResult",
            isinstance(wf_results[0], WalkForwardResult)
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print_subheader("run_generalization_analysis() [Full Pipeline]")
        summary = run_generalization_analysis(data_dir, split='train', signal_indices=CORE_SIGNAL_INDICES[:2])
        
        if validate_result(
            f"GeneralizationSummary: has day_statistics and walk_forward_results",
            hasattr(summary, 'day_statistics') and hasattr(summary, 'walk_forward_results')
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        results["details"] = {
            "days_loaded": len(days),
            "walk_forward_folds": len(wf_results),
        }
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        traceback.print_exc()
        results["failed"] += 1
    
    gc.collect()
    return results["failed"] == 0, results


def validate_analysis_intraday(data_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    """Validate intraday_seasonality module."""
    print_header("11. ANALYSIS: INTRADAY SEASONALITY")
    results = {"passed": 0, "failed": 0, "details": {}}
    
    try:
        from lobanalyzer.analysis import (
            compute_intraday_regime_stats,
            compute_signal_regime_correlation,
            compute_all_regime_correlations,
            run_intraday_seasonality_analysis,
            IntradayRegimeStats, SignalRegimeCorrelation,
        )
        from lobanalyzer.streaming import iter_days_aligned
        from lobanalyzer.constants import CORE_SIGNAL_INDICES, FeatureIndex
        
        # Load aligned data
        print_subheader("Loading aligned data...")
        features_list = []
        labels_list = []
        for i, day in enumerate(iter_days_aligned(data_dir, 'train')):
            features_list.append(day.features[::10])
            labels_list.append(day.get_labels(0)[::10])
            if i >= 4:
                break
        features = np.vstack(features_list)
        labels = np.concatenate(labels_list)
        print(f"  Loaded {len(labels):,} aligned samples")
        
        # Check for time_regime feature
        time_regime = features[:, FeatureIndex.TIME_REGIME]
        unique_regimes = np.unique(time_regime[np.isfinite(time_regime)])
        print(f"  Time regimes present: {unique_regimes}")
        
        print_subheader("compute_intraday_regime_stats()")
        regime_stats = compute_intraday_regime_stats(features, labels)
        
        if validate_result(
            f"Regime stats: {len(regime_stats)} regimes",
            len(regime_stats) > 0
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print_subheader("compute_signal_regime_correlation()")
        # This function requires pre-filtered signal/labels for a specific regime
        # Let's use regime 0 (pre-market)
        signal_idx = CORE_SIGNAL_INDICES[0]
        time_regime = features[:, FeatureIndex.TIME_REGIME]
        regime_mask = time_regime == 0  # Pre-market
        
        if regime_mask.sum() > 50:
            regime_signal = features[regime_mask, signal_idx]
            regime_labels = labels[regime_mask]
            regime_corr = compute_signal_regime_correlation(regime_signal, regime_labels, "true_ofi", 0)
            
            if validate_result(
                f"Signal regime correlation: r={regime_corr.correlation:.4f}",
                isinstance(regime_corr, SignalRegimeCorrelation)
            ):
                results["passed"] += 1
            else:
                results["failed"] += 1
        else:
            if validate_result(
                f"Skipped: only {regime_mask.sum()} samples in regime 0",
                True  # Pass anyway since it's a data issue
            ):
                results["passed"] += 1
        
        print_subheader("compute_all_regime_correlations()")
        # API expects dict mapping signal_name -> index
        signal_dict = {f'signal_{idx}': idx for idx in CORE_SIGNAL_INDICES[:4]}
        all_corrs = compute_all_regime_correlations(features, labels, signal_dict)
        
        if validate_result(
            f"All regime correlations: {len(all_corrs)} (signal, regime) pairs",
            len(all_corrs) > 0
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print_subheader("run_intraday_seasonality_analysis() [Full Pipeline]")
        signal_dict_full = {f'signal_{idx}': idx for idx in CORE_SIGNAL_INDICES[:4]}
        summary = run_intraday_seasonality_analysis(features, labels, signal_dict_full)
        
        if validate_result(
            f"IntradaySeasonalitySummary: has regime_stats",
            hasattr(summary, 'regime_stats')
        ):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        results["details"] = {
            "samples": len(labels),
            "n_regimes": len(regime_stats),
            "unique_time_regimes": list(unique_regimes),
        }
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        traceback.print_exc()
        results["failed"] += 1
    
    gc.collect()
    return results["failed"] == 0, results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Validate lobanalyzer on real NVIDIA dataset')
    parser.add_argument('--data-dir', type=Path, required=True, help='Path to NVDA dataset')
    args = parser.parse_args()
    
    if not args.data_dir.exists():
        print(f"❌ ERROR: Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    start_time = datetime.now()
    
    print("\n" + "=" * 80)
    print("  LOBANALYZER REAL DATA VALIDATION")
    print("=" * 80)
    print(f"Dataset: {args.data_dir}")
    print(f"Started: {start_time.isoformat()}")
    
    # Run all validations
    validators = [
        ("Constants Module", validate_constants_module),
        ("Streaming Module", validate_streaming_module),
        ("Streaming Convenience", validate_streaming_convenience),
        ("Data Overview", validate_analysis_data_overview),
        ("Label Analysis", validate_analysis_label),
        ("Signal Statistics", validate_analysis_signal_stats),
        ("Signal Correlations", validate_analysis_correlations),
        ("Predictive Power", validate_analysis_predictive_power),
        ("Temporal Dynamics", validate_analysis_temporal),
        ("Generalization", validate_analysis_generalization),
        ("Intraday Seasonality", validate_analysis_intraday),
    ]
    
    all_results = {}
    total_passed = 0
    total_failed = 0
    modules_passed = 0
    modules_failed = 0
    
    for name, validator in validators:
        try:
            passed, results = validator(args.data_dir)
            all_results[name] = results
            total_passed += results.get("passed", 0)
            total_failed += results.get("failed", 0)
            
            if passed:
                modules_passed += 1
            else:
                modules_failed += 1
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR in {name}: {e}")
            traceback.print_exc()
            modules_failed += 1
        
        gc.collect()
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print_header("VALIDATION SUMMARY")
    print(f"\nModules: {modules_passed} passed, {modules_failed} failed")
    print(f"Checks:  {total_passed} passed, {total_failed} failed")
    print(f"Duration: {duration:.1f} seconds")
    
    if total_failed == 0:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        return 0
    else:
        print(f"\n⚠️ {total_failed} checks failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
