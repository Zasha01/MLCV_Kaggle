#!/usr/bin/env python3
"""Generate and save EDA plots."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import FIGURES_DIR, NUMERICAL_COLS, CATEGORICAL_COLS, TARGET_COL
from src.data import load_data, get_data_summary
from src.visualization import (
    plot_target_distribution,
    plot_correlation_matrix,
    plot_scatter_vs_target,
    plot_categorical_boxplots,
    setup_plotting
)


def main():
    """Run EDA and generate plots."""
    print("\n" + "="*70)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    # Setup plotting
    setup_plotting()
    
    # Load data
    train, test, _ = load_data()
    
    # Print summary
    get_data_summary(train)
    
    # Generate plots
    print("\nGenerating EDA plots...")
    
    # 1. Target distribution
    plot_target_distribution(
        train[TARGET_COL],
        FIGURES_DIR / "target_distribution.png"
    )
    
    # 2. Correlation matrix (numerical features + target)
    corr_cols = NUMERICAL_COLS + [TARGET_COL]
    plot_correlation_matrix(
        train,
        corr_cols,
        FIGURES_DIR / "correlation_matrix.png"
    )
    
    # 3. Scatter plots vs target
    plot_scatter_vs_target(
        train,
        NUMERICAL_COLS,
        TARGET_COL,
        FIGURES_DIR / "scatter_numerical_vs_target.png"
    )
    
    # 4. Categorical boxplots
    plot_categorical_boxplots(
        train,
        CATEGORICAL_COLS,
        TARGET_COL,
        FIGURES_DIR / "categorical_boxplots.png"
    )
    
    print("\n" + "="*70)
    print("EDA COMPLETE!")
    print(f"All plots saved to: {FIGURES_DIR}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

