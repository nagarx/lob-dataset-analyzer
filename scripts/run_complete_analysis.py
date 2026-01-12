#!/usr/bin/env python3
"""
Complete Dataset Analysis for LOB Training Decisions.

This script generates comprehensive analysis outputs to inform training strategy:

OUTPUT FILES:
├── dataset_overview.json         # Dataset summary, quality, label distribution
├── signal_analysis.json          # Signal statistics, stationarity, correlations
├── predictive_power.json         # Signal-label correlations, feature importance
├── temporal_dynamics.json        # Autocorrelation, decay rates, lookback recommendations
├── generalization_analysis.json  # Walk-forward results, stability scores
├── intraday_seasonality.json     # Regime-specific behavior
├── training_recommendations.json # Actionable recommendations for training
└── analysis_summary.txt          # Human-readable executive summary

Usage:
    python scripts/run_complete_analysis.py \
        --data-dir /path/to/nvda_multi_horizon \
        --output-dir ./analysis_output \
        --symbol NVDA

Author: HFT Pipeline Team
"""

import argparse
import json
import sys
import gc
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from typing import Dict, Any, List, Optional
import numpy as np

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def convert_for_json(obj):
    """Recursively convert numpy types and dataclasses for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if hasattr(obj, 'to_dict'):
        return convert_for_json(obj.to_dict())
    if hasattr(obj, '__dataclass_fields__'):
        return convert_for_json(asdict(obj))
    if isinstance(obj, dict):
        return {str(k) if isinstance(k, (np.integer, np.int64, np.int32)) else k: convert_for_json(v) 
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_for_json(item) for item in obj]
    return obj


def save_json(data: Dict, path: Path) -> None:
    """Save dict as JSON with numpy type handling."""
    clean_data = convert_for_json(data)
    
    with open(path, 'w') as f:
        json.dump(clean_data, f, indent=2)
    print(f"  ✓ Saved: {path.name}")


def print_header(title: str) -> None:
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_dataset_overview(
    data_dir: Path,
    symbol: str,
) -> Dict[str, Any]:
    """
    Generate comprehensive dataset overview.
    
    This answers:
    - How much data do we have?
    - Is the data clean?
    - What's the label distribution?
    - What's the memory footprint?
    """
    from lobanalyzer.streaming import (
        compute_streaming_overview,
        compute_streaming_label_analysis,
        estimate_memory_usage,
        count_days,
        get_dates,
    )
    
    print_header("1. DATASET OVERVIEW")
    
    # Get overview
    overview = compute_streaming_overview(data_dir, symbol=symbol)
    
    # Get label analysis for all splits
    label_analyses = {}
    for split in ['train', 'val', 'test']:
        try:
            label_analyses[split] = compute_streaming_label_analysis(data_dir, split=split)
        except FileNotFoundError:
            pass
    
    # Memory estimate
    memory = estimate_memory_usage(data_dir)
    
    # Date ranges
    date_ranges = {}
    for split in ['train', 'val', 'test']:
        try:
            dates = get_dates(data_dir, split)
            if dates:
                date_ranges[split] = {
                    'start': dates[0],
                    'end': dates[-1],
                    'n_days': len(dates),
                }
        except FileNotFoundError:
            pass
    
    result = {
        'metadata': {
            'symbol': symbol,
            'data_dir': str(data_dir),
            'analysis_timestamp': datetime.now().isoformat(),
        },
        'overview': overview,
        'date_ranges': date_ranges,
        'label_analysis_by_split': label_analyses,
        'memory_usage': memory,
    }
    
    # Print summary
    print(f"\n  Symbol: {symbol}")
    print(f"  Total Days: {overview['total_days']}")
    print(f"  Total Samples: {overview['total_samples']:,}")
    print(f"  Memory: {memory['total']['gb']:.2f} GB")
    print(f"  Data Quality: {'✓ Clean' if overview['data_quality']['is_clean'] else '⚠ Issues detected'}")
    
    ld = overview['label_distribution']
    print(f"\n  Label Distribution (overall):")
    print(f"    Down:   {ld['down_pct']:5.1f}%  ({ld['down_count']:,})")
    print(f"    Stable: {ld['stable_pct']:5.1f}%  ({ld['stable_count']:,})")
    print(f"    Up:     {ld['up_pct']:5.1f}%  ({ld['up_count']:,})")
    
    return result


def analyze_signals(
    data_dir: Path,
    max_samples: int = 50000,
) -> Dict[str, Any]:
    """
    Analyze signal statistics and relationships.
    
    This answers:
    - What are the statistical properties of each signal?
    - Are signals stationary (safe for ML)?
    - Which signals are redundant (highly correlated)?
    - How many independent factors exist (PCA)?
    """
    from lobanalyzer.streaming import iter_days
    from lobanalyzer.analysis import (
        compute_distribution_stats,
        compute_all_stationarity_tests,
        compute_all_rolling_stats,
        compute_signal_correlation_matrix,
        find_redundant_pairs,
        compute_pca_analysis,
        compute_vif,
        cluster_signals,
    )
    from lobanalyzer.constants import CORE_SIGNAL_INDICES, FEATURE_COUNT
    
    print_header("2. SIGNAL ANALYSIS")
    
    # Load sample data
    print("\n  Loading sample data...")
    sample_features = []
    samples_loaded = 0
    for day in iter_days(data_dir, 'train'):
        sample_features.append(day.features)
        samples_loaded += day.n_samples
        if samples_loaded >= max_samples:
            break
    
    features = np.vstack(sample_features)[:max_samples]
    print(f"  Loaded {features.shape[0]:,} samples")
    
    # Distribution statistics
    print("\n  Computing distribution statistics...")
    dist_stats = compute_distribution_stats(features, CORE_SIGNAL_INDICES)
    
    # Stationarity tests
    print("  Running stationarity tests...")
    stationarity = compute_all_stationarity_tests(features, CORE_SIGNAL_INDICES)
    
    # Rolling statistics
    print("  Computing rolling statistics...")
    rolling_stats = compute_all_rolling_stats(features, CORE_SIGNAL_INDICES)
    
    # Correlation analysis
    print("  Computing correlation matrix...")
    corr_matrix, signal_names = compute_signal_correlation_matrix(features, CORE_SIGNAL_INDICES)
    redundant_pairs = find_redundant_pairs(corr_matrix, signal_names, threshold=0.7)
    
    # PCA
    print("  Running PCA analysis...")
    pca_result = compute_pca_analysis(features, CORE_SIGNAL_INDICES)
    
    # VIF
    print("  Computing VIF...")
    vif_results = compute_vif(features, CORE_SIGNAL_INDICES)
    
    # Clustering
    print("  Clustering signals...")
    clusters = cluster_signals(corr_matrix, signal_names, CORE_SIGNAL_INDICES)
    
    result = {
        'samples_analyzed': features.shape[0],
        'feature_count': FEATURE_COUNT,
        'distribution_statistics': dist_stats.to_dict('records'),
        'stationarity': [asdict(s) for s in stationarity],
        'rolling_statistics': [asdict(r) for r in rolling_stats],
        'correlation_matrix': corr_matrix.tolist(),
        'signal_names': signal_names,
        'redundant_pairs': redundant_pairs,
        'pca': asdict(pca_result),
        'vif': [asdict(v) for v in vif_results],
        'signal_clusters': [asdict(c) for c in clusters],
    }
    
    # Summary
    stationary_count = sum(1 for s in stationarity if s.is_stationary)
    stable_count = sum(1 for r in rolling_stats if r.is_mean_stable)
    problematic_vif = sum(1 for v in vif_results if v.is_problematic)
    
    print(f"\n  Results:")
    print(f"    Stationary signals: {stationary_count}/{len(stationarity)} (ADF test)")
    print(f"    Mean-stable signals: {stable_count}/{len(rolling_stats)}")
    print(f"    Redundant pairs (|r|>0.7): {len(redundant_pairs)}")
    print(f"    PCA components for 90% var: {pca_result.n_components_90}")
    print(f"    Problematic VIF (>5): {problematic_vif}/{len(vif_results)}")
    print(f"    Signal clusters: {len(clusters)}")
    
    del features
    gc.collect()
    
    return result


def analyze_predictive_power(
    data_dir: Path,
    max_samples: int = 30000,
) -> Dict[str, Any]:
    """
    Analyze predictive power of signals.
    
    This answers:
    - Which signals are most predictive?
    - What is the correlation between signals and labels?
    - Which signals should be prioritized for the model?
    """
    from lobanalyzer.streaming import iter_days_aligned
    from lobanalyzer.analysis import (
        compute_all_signal_metrics,
        compute_binned_probabilities,
        identify_predictive_groups,
    )
    from lobanalyzer.constants import CORE_SIGNAL_INDICES
    
    print_header("3. PREDICTIVE POWER ANALYSIS")
    
    # Load aligned data
    print("\n  Loading aligned data...")
    features_list = []
    labels_list = []
    samples_loaded = 0
    
    for day in iter_days_aligned(data_dir, 'train'):
        features_list.append(day.features)
        labels_list.append(day.get_labels(0))  # First horizon
        samples_loaded += day.n_pairs
        if samples_loaded >= max_samples:
            break
    
    features = np.vstack(features_list)[:max_samples]
    labels = np.concatenate(labels_list)[:max_samples]
    print(f"  Loaded {len(labels):,} aligned samples")
    
    # Compute metrics for all signals
    print("\n  Computing predictive metrics...")
    metrics_df = compute_all_signal_metrics(features, labels, CORE_SIGNAL_INDICES)
    
    # Identify predictive groups
    groups = identify_predictive_groups(metrics_df)
    
    # Binned probabilities for top signals
    print("  Computing binned probabilities...")
    binned_probs = {}
    top_signals = metrics_df.nlargest(4, 'pearson_r')
    for _, row in top_signals.iterrows():
        signal_idx = int(row['index'])
        signal_name = row['name']
        binned = compute_binned_probabilities(features[:, signal_idx], labels, n_bins=10)
        binned_probs[signal_name] = binned
    
    result = {
        'samples_analyzed': len(labels),
        'signal_metrics': metrics_df.to_dict('records'),
        'predictive_groups': groups,
        'binned_probabilities': binned_probs,
        'feature_ranking': metrics_df.sort_values('pearson_r', key=abs, ascending=False)[
            ['name', 'pearson_r', 'pearson_p', 'spearman_r', 'auc_up', 'auc_down']
        ].to_dict('records'),
    }
    
    # Summary
    print(f"\n  Top Predictive Signals (by |correlation|):")
    top_5 = metrics_df.reindex(metrics_df['pearson_r'].abs().sort_values(ascending=False).index).head(5)
    for _, row in top_5.iterrows():
        print(f"    {row['name']:25s}: r={row['pearson_r']:+.4f}, p={row['pearson_p']:.2e}")
    
    print(f"\n  Predictive Groups:")
    for group, signals in groups.items():
        if signals:
            # Signals may be strings or tuples, handle both
            names = [s[0] if isinstance(s, tuple) else str(s) for s in signals[:3]]
            print(f"    {group}: {', '.join(names)}{'...' if len(signals) > 3 else ''}")
    
    del features, labels
    gc.collect()
    
    return result


def analyze_temporal_dynamics(
    data_dir: Path,
    max_samples: int = 20000,
) -> Dict[str, Any]:
    """
    Analyze temporal dynamics of signals.
    
    This answers:
    - How quickly does signal information decay?
    - What lookback window should I use for sequence models?
    - Is a sequence model justified (vs tabular)?
    """
    from lobanalyzer.streaming import iter_days_aligned
    from lobanalyzer.analysis import (
        run_temporal_dynamics_analysis,
    )
    from lobanalyzer.constants import CORE_SIGNAL_INDICES
    
    print_header("4. TEMPORAL DYNAMICS ANALYSIS")
    
    # Load aligned data
    print("\n  Loading aligned data...")
    features_list = []
    labels_list = []
    samples_loaded = 0
    
    for day in iter_days_aligned(data_dir, 'train'):
        features_list.append(day.features)
        labels_list.append(day.get_labels(0))
        samples_loaded += day.n_pairs
        if samples_loaded >= max_samples:
            break
    
    features = np.vstack(features_list)[:max_samples]
    labels = np.concatenate(labels_list)[:max_samples]
    print(f"  Loaded {len(labels):,} aligned samples")
    
    # Run full temporal analysis
    print("\n  Running temporal dynamics analysis...")
    summary = run_temporal_dynamics_analysis(features, labels, CORE_SIGNAL_INDICES)
    
    result = summary.to_dict()
    result['samples_analyzed'] = len(labels)
    
    # Summary
    print(f"\n  Results:")
    print(f"    Optimal lookback window: {summary.optimal_lookback} samples")
    print(f"    Sequence model justified: {summary.sequence_model_justified}")
    print(f"    Justification: {summary.justification}")
    
    print(f"\n  Signal Autocorrelation Half-lives:")
    for acf in summary.autocorrelations[:4]:
        print(f"    {acf.signal_name:25s}: {acf.half_life:3d} samples")
    
    del features, labels
    gc.collect()
    
    return result


def analyze_generalization(
    data_dir: Path,
) -> Dict[str, Any]:
    """
    Analyze generalization potential.
    
    This answers:
    - How stable are signal-label relationships across days?
    - Will my model generalize to new data?
    - Which signals are most reliable day-to-day?
    """
    from lobanalyzer.analysis import run_generalization_analysis
    from lobanalyzer.constants import CORE_SIGNAL_INDICES
    
    print_header("5. GENERALIZATION ANALYSIS")
    
    print(f"\n  Running walk-forward analysis...")
    
    summary = run_generalization_analysis(
        data_dir,
        split='train',
        signal_indices=CORE_SIGNAL_INDICES[:4],  # Top 4 signals
    )
    
    result = summary.to_dict()
    
    # Summary
    print(f"\n  Results:")
    print(f"    Overall stability score: {summary.overall_stability_score:.3f}")
    print(f"    Walk-forward avg accuracy: {summary.walk_forward_avg_accuracy:.1%}")
    print(f"    Assessment: {summary.generalization_assessment}")
    
    print(f"\n  Most stable signals: {', '.join(summary.most_stable_signals[:3])}")
    print(f"  Least stable signals: {', '.join(summary.least_stable_signals[:3])}")
    
    print(f"\n  Recommendations:")
    for rec in summary.recommendations[:3]:
        print(f"    • {rec}")
    
    gc.collect()
    
    return result


def analyze_intraday_seasonality(
    data_dir: Path,
    max_samples: int = 20000,
) -> Dict[str, Any]:
    """
    Analyze intraday seasonality.
    
    This answers:
    - Do signals behave differently at market open vs close?
    - Should I stratify training by time-of-day?
    - Are there regime-specific patterns?
    """
    from lobanalyzer.streaming import iter_days_aligned
    from lobanalyzer.analysis import run_intraday_seasonality_analysis
    from lobanalyzer.constants import CORE_SIGNAL_INDICES, get_signal_info
    
    print_header("6. INTRADAY SEASONALITY ANALYSIS")
    
    # Load aligned data
    print("\n  Loading aligned data...")
    features_list = []
    labels_list = []
    samples_loaded = 0
    
    for day in iter_days_aligned(data_dir, 'train'):
        features_list.append(day.features)
        labels_list.append(day.get_labels(0))
        samples_loaded += day.n_pairs
        if samples_loaded >= max_samples:
            break
    
    features = np.vstack(features_list)[:max_samples]
    labels = np.concatenate(labels_list)[:max_samples]
    print(f"  Loaded {len(labels):,} aligned samples")
    
    # Create signal dict
    signal_info = get_signal_info()
    signal_dict = {
        signal_info[idx]['name']: idx 
        for idx in CORE_SIGNAL_INDICES 
        if idx in signal_info
    }
    
    # Run analysis
    print("\n  Running seasonality analysis...")
    summary = run_intraday_seasonality_analysis(features, labels, signal_dict)
    
    # Convert to dict (may or may not have to_dict method)
    if hasattr(summary, 'to_dict'):
        result = summary.to_dict()
    else:
        result = asdict(summary)
    result['samples_analyzed'] = len(labels)
    
    # Summary
    print(f"\n  Regime Statistics:")
    for regime_stat in summary.regime_stats:
        regime_id = getattr(regime_stat, 'regime_id', getattr(regime_stat, 'regime', '?'))
        n_samples = getattr(regime_stat, 'n_samples', 0)
        up_pct = getattr(regime_stat, 'up_pct', 0)
        down_pct = getattr(regime_stat, 'down_pct', 0)
        print(f"    Regime {regime_id}: {n_samples:,} samples, Up={up_pct:.1f}%, Down={down_pct:.1f}%")
    
    print(f"\n  Recommendations:")
    for rec in summary.recommendations[:3]:
        print(f"    • {rec}")
    
    del features, labels
    gc.collect()
    
    return result


def generate_training_recommendations(
    overview: Dict,
    signals: Dict,
    predictive: Dict,
    temporal: Dict,
    generalization: Dict,
    seasonality: Dict,
) -> Dict[str, Any]:
    """
    Generate actionable training recommendations.
    """
    print_header("7. TRAINING RECOMMENDATIONS")
    
    recommendations = {
        'data_recommendations': [],
        'feature_recommendations': [],
        'model_recommendations': [],
        'training_recommendations': [],
        'priority_actions': [],
    }
    
    # Data recommendations
    label_dist = overview['overview']['label_distribution']
    if label_dist['stable_pct'] > 80:
        recommendations['data_recommendations'].append(
            f"High class imbalance: {label_dist['stable_pct']:.1f}% Stable. "
            "Consider focal loss, class weights, or oversampling minority classes."
        )
    
    if overview['overview']['data_quality']['is_clean']:
        recommendations['data_recommendations'].append(
            "Data quality is good - no NaN/Inf values detected."
        )
    else:
        recommendations['data_recommendations'].append(
            "⚠ Data quality issues detected. Review data_quality details in overview."
        )
    
    # Feature recommendations
    if 'pca' in signals:
        n_components = signals['pca']['n_components_90']
        n_signals = len(signals.get('signal_names', []))
        if n_components < n_signals * 0.6:
            recommendations['feature_recommendations'].append(
                f"High redundancy: Only {n_components} components needed for 90% variance "
                f"(from {n_signals} signals). Consider feature selection or PCA."
            )
    
    redundant = signals.get('redundant_pairs', [])
    if len(redundant) > 3:
        recommendations['feature_recommendations'].append(
            f"Found {len(redundant)} highly correlated signal pairs. "
            "Consider removing redundant features to reduce multicollinearity."
        )
    
    # Predictive power recommendations
    if 'predictive_groups' in predictive:
        groups = predictive['predictive_groups']
        if groups.get('primary'):
            # Signals may be tuples (name, correlation) or strings
            names = [s[0] if isinstance(s, (tuple, list)) else str(s) for s in groups['primary'][:3]]
            recommendations['feature_recommendations'].append(
                f"Primary predictive signals: {', '.join(names)}. "
                "Prioritize these in feature engineering."
            )
        if groups.get('low_priority'):
            names = [s[0] if isinstance(s, (tuple, list)) else str(s) for s in groups['low_priority'][:3]]
            recommendations['feature_recommendations'].append(
                f"Low-priority signals: {', '.join(names)}. "
                "Consider removing if model complexity is a concern."
            )
    
    # Model recommendations
    if temporal.get('sequence_model_justified'):
        lookback = temporal.get('optimal_lookback', 100)
        recommendations['model_recommendations'].append(
            f"Sequence model recommended. Use lookback window of ~{lookback} samples."
        )
    else:
        recommendations['model_recommendations'].append(
            "Tabular model may be sufficient - limited temporal structure detected."
        )
    
    # Training recommendations
    gen_score = generalization.get('overall_stability_score', 0)
    if gen_score < 0.5:
        recommendations['training_recommendations'].append(
            f"⚠ Low stability score ({gen_score:.2f}). Use aggressive regularization, "
            "cross-validation, and consider data augmentation."
        )
    elif gen_score > 0.8:
        recommendations['training_recommendations'].append(
            f"Good stability score ({gen_score:.2f}). Standard training should generalize well."
        )
    
    wf_acc = generalization.get('walk_forward_avg_accuracy', 0)
    if wf_acc > 0.4:  # Better than random for 3-class
        recommendations['training_recommendations'].append(
            f"Walk-forward accuracy: {wf_acc:.1%}. Simple baseline shows signal has predictive value."
        )
    
    # Seasonality recommendations
    if seasonality.get('recommendations'):
        recommendations['training_recommendations'].extend(
            seasonality['recommendations'][:2]
        )
    
    # Priority actions
    if label_dist['stable_pct'] > 80:
        recommendations['priority_actions'].append(
            "HIGH: Address class imbalance before training (focal loss recommended)"
        )
    
    if len(redundant) > 5:
        recommendations['priority_actions'].append(
            "MEDIUM: Reduce feature redundancy via selection or PCA"
        )
    
    if gen_score < 0.5:
        recommendations['priority_actions'].append(
            "HIGH: Use regularization and early stopping to prevent overfitting"
        )
    
    # Print summary
    print("\n  Priority Actions:")
    for action in recommendations['priority_actions'][:5]:
        print(f"    • {action}")
    
    print("\n  Model Recommendations:")
    for rec in recommendations['model_recommendations'][:3]:
        print(f"    • {rec}")
    
    return recommendations


def generate_summary_report(
    output_dir: Path,
    symbol: str,
    overview: Dict,
    signals: Dict,
    predictive: Dict,
    temporal: Dict,
    generalization: Dict,
    seasonality: Dict,
    recommendations: Dict,
) -> str:
    """Generate human-readable summary report."""
    
    lines = [
        "=" * 80,
        f"  LOB DATASET ANALYSIS REPORT: {symbol}",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "-" * 80,
        "DATASET OVERVIEW",
        "-" * 80,
        f"Total Days:    {overview['overview']['total_days']}",
        f"Total Samples: {overview['overview']['total_samples']:,}",
        f"Memory:        {overview['memory_usage']['total']['gb']:.2f} GB",
        f"Data Quality:  {'Clean' if overview['overview']['data_quality']['is_clean'] else 'Issues detected'}",
        "",
        "Label Distribution:",
    ]
    
    ld = overview['overview']['label_distribution']
    lines.extend([
        f"  Down:   {ld['down_pct']:5.1f}%  ({ld['down_count']:,} samples)",
        f"  Stable: {ld['stable_pct']:5.1f}%  ({ld['stable_count']:,} samples)",
        f"  Up:     {ld['up_pct']:5.1f}%  ({ld['up_count']:,} samples)",
        "",
        "-" * 80,
        "SIGNAL ANALYSIS",
        "-" * 80,
    ])
    
    stationary_count = sum(1 for s in signals['stationarity'] if s['is_stationary'])
    lines.extend([
        f"Stationary signals:         {stationary_count}/{len(signals['stationarity'])}",
        f"PCA components (90% var):   {signals['pca']['n_components_90']}",
        f"Redundant pairs (|r|>0.7):  {len(signals['redundant_pairs'])}",
        f"Signal clusters:            {len(signals['signal_clusters'])}",
        "",
        "-" * 80,
        "PREDICTIVE POWER (Top 5 Signals)",
        "-" * 80,
    ])
    
    for i, sig in enumerate(predictive['feature_ranking'][:5], 1):
        lines.append(f"  {i}. {sig['name']:25s} r={sig['pearson_r']:+.4f}")
    
    lines.extend([
        "",
        "-" * 80,
        "TEMPORAL DYNAMICS",
        "-" * 80,
        f"Optimal lookback:           {temporal['optimal_lookback']} samples",
        f"Sequence model justified:   {temporal['sequence_model_justified']}",
        f"Justification:              {temporal['justification']}",
        "",
        "-" * 80,
        "GENERALIZATION POTENTIAL",
        "-" * 80,
        f"Stability score:            {generalization['overall_stability_score']:.3f}",
        f"Walk-forward accuracy:      {generalization['walk_forward_avg_accuracy']:.1%}",
        f"Assessment:                 {generalization['generalization_assessment']}",
        "",
        "-" * 80,
        "TRAINING RECOMMENDATIONS",
        "-" * 80,
    ])
    
    for action in recommendations['priority_actions']:
        lines.append(f"  • {action}")
    
    lines.extend([
        "",
        "=" * 80,
        "  END OF REPORT",
        "=" * 80,
    ])
    
    report = "\n".join(lines)
    
    report_path = output_dir / "analysis_summary.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n  ✓ Saved: analysis_summary.txt")
    
    return report


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive LOB dataset analysis for training decisions'
    )
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Path to dataset (e.g., nvda_multi_horizon)')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output directory (default: data_dir/analysis_output)')
    parser.add_argument('--symbol', type=str, default='NVDA',
                        help='Symbol name for reports')
    parser.add_argument('--max-samples', type=int, default=50000,
                        help='Max samples for per-split analyses')
    
    args = parser.parse_args()
    
    if not args.data_dir.exists():
        print(f"❌ ERROR: Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    # Setup output directory
    output_dir = args.output_dir or args.data_dir / 'analysis_output'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("  LOB DATASET ANALYZER - COMPLETE ANALYSIS")
    print("=" * 70)
    print(f"Dataset:    {args.data_dir}")
    print(f"Output:     {output_dir}")
    print(f"Symbol:     {args.symbol}")
    print(f"Max samples: {args.max_samples:,}")
    
    start_time = datetime.now()
    
    # Run all analyses
    overview = analyze_dataset_overview(args.data_dir, args.symbol)
    save_json(overview, output_dir / 'dataset_overview.json')
    
    signals = analyze_signals(args.data_dir, max_samples=args.max_samples)
    save_json(signals, output_dir / 'signal_analysis.json')
    
    predictive = analyze_predictive_power(args.data_dir, max_samples=args.max_samples)
    save_json(predictive, output_dir / 'predictive_power.json')
    
    temporal = analyze_temporal_dynamics(args.data_dir, max_samples=args.max_samples)
    save_json(temporal, output_dir / 'temporal_dynamics.json')
    
    generalization = analyze_generalization(args.data_dir)
    save_json(generalization, output_dir / 'generalization_analysis.json')
    
    seasonality = analyze_intraday_seasonality(args.data_dir, max_samples=args.max_samples)
    save_json(seasonality, output_dir / 'intraday_seasonality.json')
    
    # Generate recommendations
    recommendations = generate_training_recommendations(
        overview, signals, predictive, temporal, generalization, seasonality
    )
    save_json(recommendations, output_dir / 'training_recommendations.json')
    
    # Generate human-readable report
    report = generate_summary_report(
        output_dir, args.symbol,
        overview, signals, predictive, temporal, generalization, seasonality,
        recommendations
    )
    
    # Final summary
    duration = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n  Duration: {duration:.1f} seconds")
    print(f"\n  Output files saved to: {output_dir}")
    print(f"\n  Files generated:")
    for f in sorted(output_dir.glob('*.json')) + sorted(output_dir.glob('*.txt')):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:35s} ({size_kb:6.1f} KB)")
    
    print(f"\n  📊 Read analysis_summary.txt for executive summary")
    print(f"  📁 Check JSON files for detailed analysis data")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
