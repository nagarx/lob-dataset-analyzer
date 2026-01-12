"""
Statistical analysis modules for LOB datasets.

Provides comprehensive analysis functions for understanding signals, labels,
and their relationships in limit order book data.

Modules:
    - data_overview: Data validation, quality checks, dataset summaries
    - label_analysis: Label distribution, autocorrelation, transitions
    - streaming_stats: Memory-efficient online statistics algorithms

Usage:
    >>> from lobanalyzer.analysis import (
    ...     compute_data_quality,
    ...     compute_label_distribution,
    ...     RunningStats,
    ... )
    >>> 
    >>> # Check data quality
    >>> quality = compute_data_quality(features)
    >>> print(f"Clean: {quality.is_clean}")
    >>> 
    >>> # Analyze labels
    >>> dist = compute_label_distribution(labels)
    >>> print(f"Balanced: {dist.is_balanced}")

Memory Efficiency:
    For large datasets (>100 days), use streaming analysis functions
    which compute statistics incrementally without loading all data.
"""

from lobanalyzer.analysis.data_overview import (
    # File discovery
    FileInventory,
    discover_files,
    validate_file_structure,
    # Shape validation
    ShapeValidation,
    compute_shape_validation,
    # Data quality
    DataQuality,
    compute_data_quality,
    # Label distribution
    LabelDistribution,
    compute_label_distribution,
    # Categorical validation
    CategoricalValidation,
    validate_categorical_feature,
    compute_all_categorical_validations,
    # Signal statistics
    SignalStatistics,
    compute_signal_statistics,
    # Dataset summary
    DatasetSummary,
    print_data_overview,
)

from lobanalyzer.analysis.label_analysis import (
    # Distribution
    LabelDistribution as LabelDistributionAnalysis,
    compute_label_distribution as compute_label_dist_analysis,
    # Autocorrelation
    AutocorrelationResult,
    compute_autocorrelation,
    # Transition matrix
    TransitionMatrix,
    compute_transition_matrix,
    # Regime statistics
    RegimeStats,
    compute_regime_stats,
    # Signal correlations
    SignalCorrelation,
    compute_signal_label_correlations,
    # Summary
    LabelAnalysisSummary,
    run_label_analysis,
    print_label_analysis,
    # Constants
    REGIME_NAMES,
)

from lobanalyzer.analysis.streaming_stats import (
    RunningStats,
    StreamingColumnStats,
    StreamingLabelCounter,
    StreamingDataQuality,
)

from lobanalyzer.analysis.signal_stats import (
    # Dataclasses
    StationarityResult,
    RollingStatsResult,
    DistributionStats,
    # Distribution statistics
    compute_distribution_stats,
    # Stationarity tests
    compute_stationarity_test,
    compute_all_stationarity_tests,
    # Rolling statistics
    compute_rolling_stats,
    compute_all_rolling_stats,
    # Display functions
    print_distribution_summary,
    print_stationarity_summary,
)

from lobanalyzer.analysis.signal_correlations import (
    # Dataclasses
    PCAResult,
    VIFResult,
    SignalCluster,
    # Correlation matrix
    compute_signal_correlation_matrix,
    find_redundant_pairs,
    # PCA analysis
    compute_pca_analysis,
    # VIF analysis
    compute_vif,
    # Clustering
    cluster_signals,
    # Display functions
    print_correlation_summary,
    print_advanced_correlation_summary,
)

from lobanalyzer.analysis.predictive_power import (
    # Core metrics
    compute_signal_metrics,
    compute_all_signal_metrics,
    # Binned analysis
    compute_binned_probabilities,
    # Grouping
    identify_predictive_groups,
    # Display
    print_predictive_summary,
)

from lobanalyzer.analysis.temporal_dynamics import (
    # Dataclasses
    SignalAutocorrelation,
    LeadLagRelation,
    PredictiveDecay,
    LevelVsChangeAnalysis,
    TemporalDynamicsSummary,
    # Autocorrelation (renamed to avoid conflict with label_analysis.compute_autocorrelation)
    compute_autocorrelation as compute_signal_acf,
    compute_signal_autocorrelations,
    # Lead-lag
    compute_cross_correlation,
    compute_lead_lag_relations,
    # Predictive decay
    compute_predictive_decay,
    compute_all_predictive_decays,
    # Level vs change
    compute_level_vs_change,
    compute_all_level_vs_change,
    # Full analysis
    run_temporal_dynamics_analysis,
    print_temporal_dynamics,
)

from lobanalyzer.analysis.generalization import (
    # Dataclasses
    DayStatistics,
    SignalDayStats,
    WalkForwardResult,
    GeneralizationSummary,
    # Functions
    load_day_data,
    compute_day_statistics,
    compute_signal_day_stats,
    walk_forward_validation,
    run_generalization_analysis,
    print_generalization_analysis,
)

from lobanalyzer.analysis.intraday_seasonality import (
    # Dataclasses (rename to avoid conflict with label_analysis)
    RegimeStats as IntradayRegimeStats,
    SignalRegimeCorrelation,
    SignalSeasonality,
    IntradaySeasonalitySummary,
    # Functions
    compute_regime_stats as compute_intraday_regime_stats,
    compute_signal_regime_correlation,
    compute_all_regime_correlations,
    compute_signal_seasonality,
    compute_regime_importance,
    run_intraday_seasonality_analysis,
)

__all__ = [
    # Data overview
    "FileInventory",
    "discover_files",
    "validate_file_structure",
    "ShapeValidation",
    "compute_shape_validation",
    "DataQuality",
    "compute_data_quality",
    "LabelDistribution",
    "compute_label_distribution",
    "CategoricalValidation",
    "validate_categorical_feature",
    "compute_all_categorical_validations",
    "SignalStatistics",
    "compute_signal_statistics",
    "DatasetSummary",
    "print_data_overview",
    # Label analysis
    "LabelDistributionAnalysis",
    "compute_label_dist_analysis",
    "AutocorrelationResult",
    "compute_autocorrelation",
    "TransitionMatrix",
    "compute_transition_matrix",
    "RegimeStats",
    "compute_regime_stats",
    "SignalCorrelation",
    "compute_signal_label_correlations",
    "LabelAnalysisSummary",
    "run_label_analysis",
    "print_label_analysis",
    "REGIME_NAMES",
    # Streaming statistics
    "RunningStats",
    "StreamingColumnStats",
    "StreamingLabelCounter",
    "StreamingDataQuality",
    # Signal statistics
    "StationarityResult",
    "RollingStatsResult",
    "DistributionStats",
    "compute_distribution_stats",
    "compute_stationarity_test",
    "compute_all_stationarity_tests",
    "compute_rolling_stats",
    "compute_all_rolling_stats",
    "print_distribution_summary",
    "print_stationarity_summary",
    # Signal correlations
    "PCAResult",
    "VIFResult",
    "SignalCluster",
    "compute_signal_correlation_matrix",
    "find_redundant_pairs",
    "compute_pca_analysis",
    "compute_vif",
    "cluster_signals",
    "print_correlation_summary",
    "print_advanced_correlation_summary",
    # Predictive power
    "compute_signal_metrics",
    "compute_all_signal_metrics",
    "compute_binned_probabilities",
    "identify_predictive_groups",
    "print_predictive_summary",
    # Temporal dynamics
    "SignalAutocorrelation",
    "LeadLagRelation",
    "PredictiveDecay",
    "LevelVsChangeAnalysis",
    "TemporalDynamicsSummary",
    "compute_signal_acf",  # Signal autocorrelation (tuple return)
    "compute_signal_autocorrelations",
    "compute_cross_correlation",
    "compute_lead_lag_relations",
    "compute_predictive_decay",
    "compute_all_predictive_decays",
    "compute_level_vs_change",
    "compute_all_level_vs_change",
    "run_temporal_dynamics_analysis",
    "print_temporal_dynamics",
    # Generalization
    "DayStatistics",
    "SignalDayStats",
    "WalkForwardResult",
    "GeneralizationSummary",
    "load_day_data",
    "compute_day_statistics",
    "compute_signal_day_stats",
    "walk_forward_validation",
    "run_generalization_analysis",
    "print_generalization_analysis",
    # Intraday seasonality
    "IntradayRegimeStats",
    "SignalRegimeCorrelation",
    "SignalSeasonality",
    "IntradaySeasonalitySummary",
    "compute_intraday_regime_stats",
    "compute_signal_regime_correlation",
    "compute_all_regime_correlations",
    "compute_signal_seasonality",
    "compute_regime_importance",
    "run_intraday_seasonality_analysis",
]
