"""Plotting functions that save figures to disk."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from pathlib import Path
from ..config import PLOT_STYLE, PLOT_PALETTE, FIG_DPI


def setup_plotting():
    """Setup matplotlib/seaborn styling."""
    plt.style.use(PLOT_STYLE)
    sns.set_palette(PLOT_PALETTE)


def plot_target_distribution(y, save_path):
    """
    Plot target variable distribution.
    
    Args:
        y: Target variable series/array
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(y, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    axes[0].set_title('Target Distribution (Exam Score)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Exam Score')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(y.mean(), color='red', linestyle='--', 
                    label=f'Mean: {y.mean():.2f}', linewidth=2)
    axes[0].legend()
    
    # Boxplot
    axes[1].boxplot(y, vert=True)
    axes[1].set_title('Exam Score Box Plot', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Exam Score')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_correlation_matrix(df, cols, save_path):
    """
    Plot correlation matrix heatmap.
    
    Args:
        df: Dataframe containing the columns
        cols: List of column names to include
        save_path: Path to save the figure
    """
    correlation_data = df[cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_data, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_scatter_vs_target(df, cols, target_col, save_path, seed=42):
    """
    Plot scatter plots of features vs target with trend lines.
    
    Args:
        df: Dataframe
        cols: List of column names to plot
        target_col: Target column name
        save_path: Path to save the figure
        seed: Random seed for sampling
    """
    n_cols = len(cols)
    n_rows = (n_cols + 1) // 2
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5*n_rows))
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    # Sample for faster plotting
    sample_df = df.sample(n=min(10000, len(df)), random_state=seed)
    
    for idx, col in enumerate(cols):
        axes[idx].scatter(sample_df[col], sample_df[target_col], alpha=0.3, s=5)
        axes[idx].set_xlabel(col.replace('_', ' ').title())
        axes[idx].set_ylabel('Exam Score')
        axes[idx].set_title(f'{col.replace("_", " ").title()} vs Exam Score')
        
        # Add trend line
        z = np.polyfit(sample_df[col], sample_df[target_col], 1)
        p = np.poly1d(z)
        x_sorted = np.sort(sample_df[col])
        axes[idx].plot(x_sorted, p(x_sorted), "r--", alpha=0.8, linewidth=2)
    
    # Remove extra subplots
    for idx in range(len(cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_categorical_boxplots(df, cols, target_col, save_path, seed=42):
    """
    Plot boxplots for categorical features vs target.
    
    Args:
        df: Dataframe
        cols: List of categorical column names
        target_col: Target column name
        save_path: Path to save the figure
        seed: Random seed for sampling
    """
    n_cols = len(cols)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4*n_rows))
    axes = axes.flatten()
    
    # Sample for faster plotting
    sample_df = df.sample(n=min(50000, len(df)), random_state=seed)
    
    for idx, col in enumerate(cols):
        sample_df.boxplot(column=target_col, by=col, ax=axes[idx])
        axes[idx].set_title(f'Exam Score by {col.replace("_", " ").title()}')
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('Exam Score')
        plt.sca(axes[idx])
        plt.xticks(rotation=45)
    
    # Remove extra subplots
    for idx in range(len(cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle('Exam Score Distribution by Categorical Features', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_feature_importance(importance_df, save_path, top_n=20):
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        save_path: Path to save the figure
        top_n: Number of top features to show
    """
    top_features = importance_df.head(top_n).iloc[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance (Gain)', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_model_comparison(results_df, save_path):
    """
    Plot model performance comparison.
    
    Args:
        results_df: DataFrame with 'Model', 'CV_RMSE', and 'CV_Std' columns
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = results_df['Model'].values
    scores = results_df['CV_RMSE'].values
    stds = results_df.get('CV_Std', [0]*len(models)).values
    
    colors = ['#ff7f0e' if i == 0 else '#1f77b4' for i in range(len(models))]
    bars = ax.barh(range(len(models)), scores, xerr=stds, color=colors, alpha=0.7)
    
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel('Cross-Validation RMSE', fontsize=12)
    ax.set_title('Model Performance Comparison (Lower is Better)', 
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add value labels
    for i, (score, std) in enumerate(zip(scores, stds)):
        label = f'{score:.5f}' if std == 0 else f'{score:.5f} ± {std:.5f}'
        ax.text(score + 0.02, i, label, va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_residuals(y_true, y_pred, model_name, save_path):
    """
    Plot residual analysis: predicted vs actual and residual distribution.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        model_name: Name of the model (for title)
        save_path: Path to save the figure
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Predicted vs Actual
    axes[0].scatter(y_true, y_pred, alpha=0.3, s=1)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Exam Score')
    axes[0].set_ylabel('Predicted Exam Score')
    axes[0].set_title(f'Predicted vs Actual ({model_name})')
    axes[0].legend()
    
    # Residual distribution
    axes[1].hist(residuals, bins=50, alpha=0.7, edgecolor='black', color='coral')
    axes[1].axvline(0, color='red', linestyle='--', lw=2, label='Zero Error')
    axes[1].set_xlabel('Residual (Actual - Predicted)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Residual Distribution ({model_name})')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

