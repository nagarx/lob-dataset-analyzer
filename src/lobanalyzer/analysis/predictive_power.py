"""
Signal predictive power analysis.

Determines which signals predict price movement using:
- Pearson/Spearman correlation
- AUC (Up vs Not-Up, Down vs Not-Down)
- Mutual Information
- Binned probability analysis

References:
    - Pearson, K. (1895). Notes on regression and inheritance.
    - Spearman, C. (1904). The proof and measurement of association between two things.
    - Shannon, C. E. (1948). A mathematical theory of communication.
    - Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information.
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from typing import Dict, List, Optional, Tuple
import pandas as pd

from lobanalyzer.constants import (
    LABEL_DOWN, LABEL_STABLE, LABEL_UP,
    get_signal_info, CORE_SIGNAL_INDICES,
)


def compute_signal_metrics(
    signal: np.ndarray,
    labels: np.ndarray,
    expected_sign: str = '?',
) -> Dict:
    """
    Compute comprehensive predictive metrics for a single signal.
    
    This function computes multiple measures of predictive power:
    - Pearson correlation: Linear relationship strength
    - Spearman correlation: Monotonic relationship (rank-based)
    - AUC: Discriminative power for classification
    - Mutual Information: General dependence (captures non-linear)
    
    Args:
        signal: (N,) array of signal values
        labels: (N,) array of labels {-1, 0, 1}
        expected_sign: '+' (bullish signal), '-' (bearish), or '?' (unknown)
    
    Returns:
        dict with:
            - n_samples: number of valid samples
            - pearson_r, pearson_p: Pearson correlation and p-value
            - spearman_r, spearman_p: Spearman correlation and p-value
            - auc_up: AUC for Up vs Not-Up
            - auc_down: AUC for Down vs Not-Down
            - mutual_info, mi_bits: Mutual information (nats and bits)
            - sign_consistent: whether sign matches expected
            - mean_up, mean_stable, mean_down: conditional means
    
    Example:
        >>> signal = np.random.randn(1000)
        >>> labels = np.random.choice([-1, 0, 1], 1000)
        >>> metrics = compute_signal_metrics(signal, labels, expected_sign='+')
        >>> print(f"Correlation: {metrics['pearson_r']:.4f}")
    """
    # Remove any NaN/Inf values
    valid_mask = np.isfinite(signal) & np.isfinite(labels)
    signal = signal[valid_mask]
    labels_clean = labels[valid_mask]
    
    n = len(signal)
    if n == 0:
        return {'n_samples': 0, 'error': 'No valid samples'}
    
    # 1. Pearson correlation (linear relationship)
    # Formula: r = cov(X, Y) / (std(X) × std(Y))
    pearson_r, pearson_p = pearsonr(signal, labels_clean)
    
    # 2. Spearman correlation (rank-based, captures monotonic relationships)
    # Formula: ρ = Pearson correlation of ranks
    spearman_r, spearman_p = spearmanr(signal, labels_clean)
    
    # 3. AUC for Up vs Not-Up (how well signal separates Up from others)
    y_up = (labels_clean == LABEL_UP).astype(int)
    if 0 < y_up.sum() < len(y_up):
        auc_up = roc_auc_score(y_up, signal)
    else:
        auc_up = 0.5  # No discrimination possible
    
    # 4. AUC for Down vs Not-Down (use NEGATIVE signal for consistency)
    # Higher signal should predict Up, lower should predict Down
    y_down = (labels_clean == LABEL_DOWN).astype(int)
    if 0 < y_down.sum() < len(y_down):
        auc_down = roc_auc_score(y_down, -signal)
    else:
        auc_down = 0.5
    
    # 5. Mutual Information (captures non-linear dependence)
    # Formula: MI(X; Y) = H(Y) - H(Y|X) = ∑∑ p(x,y) log(p(x,y) / (p(x)p(y)))
    # Shift labels from {-1, 0, 1} to {0, 1, 2} for sklearn
    labels_shifted = labels_clean.astype(int) + 1
    mi = mutual_info_classif(
        signal.reshape(-1, 1),
        labels_shifted,
        discrete_features=False,
        random_state=42,
    )[0]
    mi_bits = mi / np.log(2)  # Convert nats to bits
    
    # 6. Sign consistency check
    if expected_sign == '+':
        sign_consistent = pearson_r > 0
    elif expected_sign == '-':
        sign_consistent = pearson_r < 0
    else:
        sign_consistent = None  # Unknown expected sign
    
    # 7. Conditional means (mean signal value for each label)
    mean_up = float(signal[labels_clean == LABEL_UP].mean()) if (labels_clean == LABEL_UP).any() else np.nan
    mean_stable = float(signal[labels_clean == LABEL_STABLE].mean()) if (labels_clean == LABEL_STABLE).any() else np.nan
    mean_down = float(signal[labels_clean == LABEL_DOWN].mean()) if (labels_clean == LABEL_DOWN).any() else np.nan
    
    return {
        'n_samples': n,
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'auc_up': float(auc_up),
        'auc_down': float(auc_down),
        'mutual_info': float(mi),
        'mi_bits': float(mi_bits),
        'sign_consistent': sign_consistent,
        'mean_up': mean_up,
        'mean_stable': mean_stable,
        'mean_down': mean_down,
    }


def compute_all_signal_metrics(
    aligned_features: np.ndarray,
    labels: np.ndarray,
    signal_indices: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Compute predictive metrics for all signals.
    
    Args:
        aligned_features: (N_labels, F) aligned feature array (features[i] corresponds to labels[i])
        labels: (N_labels,) label array with values in {-1, 0, 1}
        signal_indices: Which signals to analyze (default: CORE_SIGNAL_INDICES)
    
    Returns:
        DataFrame with metrics for each signal, including:
            - index, name, expected_sign
            - All metrics from compute_signal_metrics
    
    Example:
        >>> df = compute_all_signal_metrics(aligned_features, labels)
        >>> print(df.sort_values('pearson_r', key=abs, ascending=False).head())
    """
    if signal_indices is None:
        signal_indices = CORE_SIGNAL_INDICES
    
    signal_info = get_signal_info()
    results = []
    
    for idx in signal_indices:
        info = signal_info.get(idx, {'name': f'signal_{idx}', 'expected_sign': '?'})
        signal = aligned_features[:, idx]
        
        metrics = compute_signal_metrics(
            signal, labels, info.get('expected_sign', '?')
        )
        
        results.append({
            'index': idx,
            'name': info.get('name', f'signal_{idx}'),
            'expected_sign': info.get('expected_sign', '?'),
            **metrics,
        })
    
    return pd.DataFrame(results)


def compute_binned_probabilities(
    signal: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Bin signal into quantiles and compute label probabilities per bin.
    
    This reveals non-linear relationships that correlation might miss.
    For example, a signal might only be predictive in extreme values.
    
    Args:
        signal: (N,) signal values
        labels: (N,) label values {-1, 0, 1}
        n_bins: Number of bins (default 10 = deciles)
    
    Returns:
        DataFrame with columns:
            - bin: bin number (0 = lowest signal values)
            - signal_mean, signal_min, signal_max
            - p_up, p_down, p_stable: probability of each label
            - n_samples: samples in this bin
    
    Example:
        >>> df_bins = compute_binned_probabilities(features[:, 84], labels)
        >>> # Check if p_up increases with signal value (expected for bullish signal)
        >>> print(df_bins[['bin', 'signal_mean', 'p_up']])
    """
    # Handle edge cases
    valid_mask = np.isfinite(signal)
    signal = signal[valid_mask]
    labels_clean = labels[valid_mask]
    
    if len(signal) == 0:
        return pd.DataFrame()
    
    # Create bins using quantiles (equal sample count per bin)
    try:
        bins = pd.qcut(signal, q=n_bins, labels=False, duplicates='drop')
    except ValueError:
        # Fall back to equal-width bins if quantiles fail (e.g., many ties)
        bins = pd.cut(signal, bins=n_bins, labels=False)
    
    results = []
    for b in range(int(np.nanmax(bins)) + 1):
        mask = bins == b
        if mask.sum() == 0:
            continue
        
        bin_labels = labels_clean[mask]
        bin_signal = signal[mask]
        
        results.append({
            'bin': int(b),
            'signal_mean': float(bin_signal.mean()),
            'signal_min': float(bin_signal.min()),
            'signal_max': float(bin_signal.max()),
            'p_up': float((bin_labels == LABEL_UP).mean()),
            'p_down': float((bin_labels == LABEL_DOWN).mean()),
            'p_stable': float((bin_labels == LABEL_STABLE).mean()),
            'n_samples': int(len(bin_labels)),
        })
    
    return pd.DataFrame(results)


def identify_predictive_groups(
    df_metrics: pd.DataFrame,
    corr_matrix: Optional[np.ndarray] = None,
    signal_names: Optional[List[str]] = None,
    primary_threshold: float = 0.05,
    low_threshold: float = 0.01,
    redundancy_threshold: float = 0.5,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Group signals by predictive power and redundancy.
    
    Args:
        df_metrics: DataFrame from compute_all_signal_metrics
        corr_matrix: Optional correlation matrix between signals
        signal_names: Signal names (same order as corr_matrix)
        primary_threshold: |r| threshold for primary features (default: 0.05)
        low_threshold: |r| threshold for low priority (default: 0.01)
        redundancy_threshold: Correlation threshold for redundancy (default: 0.5)
    
    Returns:
        Dict with groups:
            - 'primary': Strong predictors with consistent sign
            - 'contrarian': Opposite sign from expected
            - 'moderate': Weak but non-trivial predictors
            - 'redundant': Correlated with better predictor
            - 'low_priority': Very weak predictors
    
    Example:
        >>> groups = identify_predictive_groups(df_metrics, corr_matrix, signal_names)
        >>> print(f"Primary features: {[name for name, r in groups['primary']]}")
    """
    # Sort by absolute correlation
    df_sorted = df_metrics.sort_values('pearson_r', key=abs, ascending=False)
    
    # Build redundancy set
    redundant_with_better = set()
    if corr_matrix is not None and signal_names is not None:
        for i in range(len(signal_names)):
            for j in range(i + 1, len(signal_names)):
                r = corr_matrix[i, j]
                if abs(r) > redundancy_threshold:
                    # Find which signal has higher predictive power
                    r1 = df_metrics[df_metrics['name'] == signal_names[i]]['pearson_r'].abs().values
                    r2 = df_metrics[df_metrics['name'] == signal_names[j]]['pearson_r'].abs().values
                    if len(r1) > 0 and len(r2) > 0:
                        if r1[0] > r2[0]:
                            redundant_with_better.add(signal_names[j])
                        else:
                            redundant_with_better.add(signal_names[i])
    
    # Categorize signals
    groups = {
        'primary': [],
        'contrarian': [],
        'moderate': [],
        'redundant': [],
        'low_priority': [],
    }
    
    for _, row in df_sorted.iterrows():
        name = row['name']
        r = row['pearson_r']
        sign_ok = row['sign_consistent']
        
        if name in redundant_with_better:
            groups['redundant'].append((name, r))
        elif abs(r) < low_threshold:
            groups['low_priority'].append((name, r))
        elif sign_ok == False:
            groups['contrarian'].append((name, r))
        elif abs(r) >= primary_threshold:
            groups['primary'].append((name, r))
        else:
            groups['moderate'].append((name, r))
    
    return groups


def print_predictive_summary(
    df_metrics: pd.DataFrame,
    corr_matrix: Optional[np.ndarray] = None,
    signal_names: Optional[List[str]] = None,
) -> None:
    """
    Print formatted predictive power summary.
    
    Args:
        df_metrics: DataFrame from compute_all_signal_metrics
        corr_matrix: Optional correlation matrix for redundancy info
        signal_names: Signal names for correlation matrix
    """
    print("=" * 80)
    print("SIGNAL PREDICTIVE POWER ANALYSIS")
    print("=" * 80)
    
    # Sort by absolute Pearson correlation
    df_sorted = df_metrics.sort_values('pearson_r', key=abs, ascending=False)
    
    # Display ranking
    print("\n1. SIGNAL RANKING (by |Pearson r|):\n")
    print("   Rank | Signal                    | r       | AUC_up | AUC_down | Sign OK")
    print("   " + "-" * 70)
    
    for rank, (_, row) in enumerate(df_sorted.iterrows(), 1):
        sign_ok = '✓' if row['sign_consistent'] == True else '✗' if row['sign_consistent'] == False else '?'
        print(f"   #{rank:2d}  | {row['name']:25s} | {row['pearson_r']:+.4f} | {row['auc_up']:.4f} | {row['auc_down']:.4f}  | {sign_ok}")
    
    # Key findings
    print("\n" + "=" * 80)
    print("2. KEY FINDINGS")
    print("=" * 80)
    
    # Best predictor
    best = df_sorted.iloc[0]
    print(f"\n  • BEST PREDICTOR: {best['name']} (r = {best['pearson_r']:+.4f})")
    
    # Contrarian signals
    contrarian = df_metrics[df_metrics['sign_consistent'] == False]
    if len(contrarian) > 0:
        print(f"\n  • CONTRARIAN SIGNALS (opposite of expected sign):")
        for _, row in contrarian.iterrows():
            print(f"    - {row['name']}: expected {row['expected_sign']}, got r = {row['pearson_r']:+.4f}")
    
    # Redundant pairs (if provided)
    redundant_pairs = []
    if corr_matrix is not None and signal_names is not None:
        print(f"\n  • REDUNDANT PAIRS (|r| > 0.5):")
        for i in range(len(signal_names)):
            for j in range(i + 1, len(signal_names)):
                r = corr_matrix[i, j]
                if abs(r) > 0.5:
                    print(f"    - {signal_names[i]} ↔ {signal_names[j]}: r = {r:+.3f}")
                    redundant_pairs.append((signal_names[i], signal_names[j], r))
    
    # Get grouped signals
    groups = identify_predictive_groups(df_metrics, corr_matrix, signal_names)
    
    # Print recommendations
    print("\n" + "=" * 80)
    print("3. RECOMMENDATIONS (DATA-DRIVEN)")
    print("=" * 80)
    
    if groups['primary']:
        print(f"\n  GROUP A - PRIMARY FEATURES (|r| ≥ 0.05, consistent sign):")
        for name, r in groups['primary']:
            print(f"    • {name}: r = {r:+.4f}")
    
    if groups['contrarian']:
        print(f"\n  GROUP B - CONTRARIAN (opposite sign, use carefully):")
        for name, r in groups['contrarian']:
            print(f"    • {name}: r = {r:+.4f} (inverted sign)")
    
    if groups['moderate']:
        print(f"\n  GROUP C - MODERATE VALUE (|r| < 0.05 but > 0.01):")
        for name, r in groups['moderate']:
            print(f"    • {name}: r = {r:+.4f}")
    
    if groups['redundant']:
        print(f"\n  GROUP D - REDUNDANT (correlated with better predictor):")
        for name, r in groups['redundant']:
            print(f"    • {name}: r = {r:+.4f}")
    
    if groups['low_priority']:
        print(f"\n  GROUP E - LOW PRIORITY (|r| < 0.01):")
        for name, r in groups['low_priority']:
            print(f"    • {name}: r = {r:+.4f}")
    
    # Model recommendation
    print("\n  MODEL RECOMMENDATION:")
    n_primary = len(groups['primary'])
    n_contrarian = len(groups['contrarian'])
    if n_primary >= 2:
        print(f"    → Use {n_primary} primary features as base predictors")
    if n_contrarian > 0:
        print(f"    → Consider {n_contrarian} contrarian feature(s) as separate input(s)")
    if len(groups['redundant']) > 0:
        print(f"    → Avoid {len(groups['redundant'])} redundant feature(s) to reduce multicollinearity")
    
    print("\n" + "=" * 80)
    print("✅ SIGNAL PREDICTIVE POWER ANALYSIS COMPLETE")
    print("=" * 80)
