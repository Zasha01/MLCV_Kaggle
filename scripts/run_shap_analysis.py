#!/usr/bin/env python3
"""
SHAP (SHapley Additive exPlanations) Analysis for Student Score Prediction.

This script computes SHAP values for the best-performing model (CatBoost) and
generates interpretability visualizations including:
- Summary plots (global feature importance)
- Beeswarm plots (feature impact distribution)
- Dependence plots (feature interactions)
- Waterfall plots (individual predictions)

References:
- Lundberg & Lee (2017): "A Unified Approach to Interpreting Model Predictions"
- Lundberg et al. (2020): "From Local Explanations to Global Understanding"
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    FIGURES_DIR, RESULTS_DIR, SEED, CATBOOST_PARAMS,
    NUMERICAL_COLS, TARGET_COL
)
from src.data import load_data, prepare_datasets
from src.models import CatBoostModel
from src.visualization import setup_plotting


def train_model_for_shap(X_train, y_train, sample_size=50000):
    """
    Train CatBoost model on a sample for SHAP analysis.
    
    Using a sample because SHAP computation is expensive for large datasets.
    """
    print(f"\nTraining CatBoost on {sample_size} samples for SHAP analysis...")
    
    # Sample data for faster SHAP computation
    np.random.seed(SEED)
    if len(X_train) > sample_size:
        idx = np.random.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train.iloc[idx].reset_index(drop=True)
        y_sample = y_train.iloc[idx].reset_index(drop=True)
    else:
        X_sample = X_train.reset_index(drop=True)
        y_sample = y_train.reset_index(drop=True)
    
    # Split for validation
    split_idx = int(len(X_sample) * 0.8)
    X_tr, X_val = X_sample.iloc[:split_idx], X_sample.iloc[split_idx:]
    y_tr, y_val = y_sample.iloc[:split_idx], y_sample.iloc[split_idx:]
    
    # Train model
    model = CatBoostModel(params=CATBOOST_PARAMS.copy())
    model.fit(X_tr, y_tr, X_val, y_val)
    
    return model, X_sample, y_sample


def compute_shap_values(model, X_sample, background_size=1000):
    """
    Compute SHAP values using TreeExplainer.
    
    Args:
        model: Trained tree-based model
        X_sample: Data to explain
        background_size: Size of background dataset for interventional SHAP
    """
    print(f"\nComputing SHAP values for {len(X_sample)} samples...")
    print("This may take a few minutes...")
    
    # Create background dataset (smaller for efficiency)
    np.random.seed(SEED)
    bg_idx = np.random.choice(len(X_sample), min(background_size, len(X_sample)), replace=False)
    background = X_sample.iloc[bg_idx]
    
    # Create TreeExplainer
    explainer = shap.TreeExplainer(
        model.model,
        data=background,
        feature_perturbation='interventional'
    )
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_sample)
    
    # Create Explanation object for newer SHAP API
    explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_sample.values,
        feature_names=X_sample.columns.tolist()
    )
    
    return explanation, explainer


def plot_shap_summary(explanation, filename="shap_summary.png"):
    """Create SHAP summary plot (beeswarm)."""
    print(f"\nGenerating SHAP summary plot...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(
        explanation.values,
        explanation.data,
        feature_names=explanation.feature_names,
        show=False,
        max_display=20
    )
    plt.title("SHAP Summary Plot: Feature Impact on Exam Score Predictions", fontsize=14, pad=20)
    plt.tight_layout()
    
    filepath = FIGURES_DIR / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filepath}")


def plot_shap_bar(explanation, filename="shap_importance.png"):
    """Create SHAP bar plot (mean absolute SHAP values)."""
    print(f"\nGenerating SHAP importance bar plot...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(explanation, max_display=20, show=False)
    plt.title("Mean |SHAP Value|: Global Feature Importance", fontsize=14, pad=20)
    plt.tight_layout()
    
    filepath = FIGURES_DIR / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filepath}")


def plot_shap_dependence(explanation, X_sample, top_features, filename_prefix="shap_dependence"):
    """Create SHAP dependence plots for top features."""
    print(f"\nGenerating SHAP dependence plots for top features...")
    
    n_features = len(top_features)
    n_cols = 2
    n_rows = (n_features + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(top_features):
        if feature in X_sample.columns:
            feature_idx = X_sample.columns.tolist().index(feature)
            shap.dependence_plot(
                feature_idx,
                explanation.values,
                X_sample.values,
                feature_names=X_sample.columns.tolist(),
                ax=axes[i],
                show=False,
                interaction_index='auto'
            )
            axes[i].set_title(f"SHAP Dependence: {feature}", fontsize=11)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle("SHAP Dependence Plots: Feature Effects on Predictions", fontsize=14, y=1.02)
    plt.tight_layout()
    
    filepath = FIGURES_DIR / f"{filename_prefix}.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filepath}")


def plot_shap_waterfall(explanation, X_sample, y_sample, indices, filename="shap_waterfall.png"):
    """Create SHAP waterfall plots for individual predictions."""
    print(f"\nGenerating SHAP waterfall plots for individual samples...")
    
    n_samples = len(indices)
    fig, axes = plt.subplots(1, n_samples, figsize=(6 * n_samples, 8))
    
    if n_samples == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        plt.sca(axes[i])
        
        # Create single-sample explanation
        single_exp = shap.Explanation(
            values=explanation.values[idx],
            base_values=explanation.base_values if np.isscalar(explanation.base_values) else explanation.base_values[idx],
            data=explanation.data[idx],
            feature_names=explanation.feature_names
        )
        
        shap.plots.waterfall(single_exp, max_display=12, show=False)
        actual = y_sample.iloc[idx] if hasattr(y_sample, 'iloc') else y_sample[idx]
        axes[i].set_title(f"Sample {idx}\nActual: {actual:.1f}", fontsize=11)
    
    plt.suptitle("SHAP Waterfall: Individual Prediction Explanations", fontsize=14, y=1.02)
    plt.tight_layout()
    
    filepath = FIGURES_DIR / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filepath}")


def plot_shap_force(explanation, explainer, X_sample, idx, filename="shap_force.png"):
    """Create SHAP force plot for a single prediction."""
    print(f"\nGenerating SHAP force plot...")
    
    # Force plot for single prediction
    shap.force_plot(
        explainer.expected_value,
        explanation.values[idx],
        X_sample.iloc[idx],
        matplotlib=True,
        show=False
    )
    
    filepath = FIGURES_DIR / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filepath}")


def get_top_features(explanation, n=6):
    """Get top N features by mean absolute SHAP value."""
    mean_abs_shap = np.abs(explanation.values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': explanation.feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    return feature_importance['feature'].head(n).tolist()


def save_shap_statistics(explanation, filepath):
    """Save SHAP value statistics to CSV."""
    mean_shap = explanation.values.mean(axis=0)
    mean_abs_shap = np.abs(explanation.values).mean(axis=0)
    std_shap = explanation.values.std(axis=0)
    
    stats_df = pd.DataFrame({
        'feature': explanation.feature_names,
        'mean_shap': mean_shap,
        'mean_abs_shap': mean_abs_shap,
        'std_shap': std_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    stats_df.to_csv(filepath, index=False)
    print(f"Saved SHAP statistics: {filepath}")
    return stats_df


def main():
    """Run complete SHAP analysis pipeline."""
    setup_plotting()
    
    print("\n" + "="*70)
    print("SHAP ANALYSIS FOR STUDENT SCORE PREDICTION")
    print("="*70)
    
    # Load and prepare data
    print("\nLoading and preparing data...")
    train, test, _ = load_data()
    X_train, y_train, X_test, _, _ = prepare_datasets(train, test)
    
    print(f"Training set: {X_train.shape}")
    print(f"Features: {X_train.columns.tolist()[:10]}... ({len(X_train.columns)} total)")
    
    # Train model on sample
    sample_size = 30000  # Reduced for faster SHAP computation
    model, X_sample, y_sample = train_model_for_shap(X_train, y_train, sample_size)
    
    # Compute SHAP values
    explanation, explainer = compute_shap_values(model, X_sample, background_size=500)
    
    # Save SHAP statistics
    shap_stats = save_shap_statistics(explanation, RESULTS_DIR / "shap_statistics.csv")
    
    print("\n" + "-"*50)
    print("Top 10 Features by Mean |SHAP Value|:")
    print("-"*50)
    print(shap_stats.head(10).to_string(index=False))
    
    # Get top features for dependence plots
    top_features = get_top_features(explanation, n=6)
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING SHAP VISUALIZATIONS")
    print("="*70)
    
    # 1. Summary plot (beeswarm)
    plot_shap_summary(explanation, "shap_summary.png")
    
    # 2. Bar plot (importance)
    plot_shap_bar(explanation, "shap_importance.png")
    
    # 3. Dependence plots for top features
    plot_shap_dependence(explanation, X_sample, top_features, "shap_dependence")
    
    # 4. Waterfall plots for interesting samples
    # Select samples: low score, medium score, high score
    y_sorted_idx = y_sample.argsort()
    sample_indices = [
        y_sorted_idx.iloc[int(len(y_sorted_idx) * 0.1)],   # Low score
        y_sorted_idx.iloc[int(len(y_sorted_idx) * 0.5)],   # Medium score
        y_sorted_idx.iloc[int(len(y_sorted_idx) * 0.9)]    # High score
    ]
    plot_shap_waterfall(explanation, X_sample, y_sample, sample_indices, "shap_waterfall.png")
    
    # 5. Force plot for single prediction
    plot_shap_force(explanation, explainer, X_sample, sample_indices[2], "shap_force.png")
    
    # Copy figures to latex directory
    import shutil
    latex_figures = Path(__file__).parent.parent / "latex" / "figures"
    for fig_name in ["shap_summary.png", "shap_importance.png", "shap_dependence.png", "shap_waterfall.png"]:
        src = FIGURES_DIR / fig_name
        if src.exists():
            shutil.copy(src, latex_figures / fig_name)
            print(f"Copied {fig_name} to latex/figures/")
    
    print("\n" + "="*70)
    print("SHAP ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nFigures saved to: {FIGURES_DIR}")
    print(f"Statistics saved to: {RESULTS_DIR / 'shap_statistics.csv'}")
    print("\nKey Insights:")
    print(f"  - Most important feature: {shap_stats.iloc[0]['feature']}")
    print(f"  - Mean |SHAP|: {shap_stats.iloc[0]['mean_abs_shap']:.4f}")
    print(f"  - Base value (expected prediction): {explainer.expected_value:.2f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

