#!/usr/bin/env python3
"""Train stacking ensemble with Ridge meta-learner."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import FIGURES_DIR, RESULTS_DIR, MODELS_DIR, SEED
from src.data import load_data, prepare_datasets
from src.models import (
    RidgeModel,
    LightGBMModel, XGBoostModel, CatBoostModel,
    StackingModel
)
from src.evaluation import save_results, calculate_rmse
from src.visualization import plot_residuals, setup_plotting


def main():
    """Run stacking pipeline with Ridge meta-learner."""
    # Setup plotting
    setup_plotting()
    
    # Load and prepare data
    print("\nLoading and preparing data...")
    train, test, _ = load_data()
    X_train, y_train, X_test, test_ids, train_ids = prepare_datasets(train, test)
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Define base models (using default params from config)
    base_model_configs = [
        (LightGBMModel, None),
        (XGBoostModel, None),
        (CatBoostModel, None),
    ]
    
    # Create stacking model with Ridge meta-learner
    stacking_model = StackingModel(
        base_model_configs=base_model_configs,
        meta_model_class=RidgeModel,
        meta_params=None,  # Use default Ridge params
        n_folds=5
    )
    
    # Train stacking model
    # This will:
    # 1. Generate OOF predictions for each base model
    # 2. Train base models on full training data
    # 3. Train Ridge meta-model on stacked OOF predictions
    stacking_model.fit(X_train, y_train)
    
    # Get predictions
    print("\nGenerating predictions...")
    
    # Training predictions (to calculate final OOF RMSE)
    train_predictions = stacking_model.predict(X_train)
    train_rmse = calculate_rmse(y_train, train_predictions)
    
    # Test predictions
    test_predictions = stacking_model.predict(X_test)
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Stacking Model (Ridge meta-learner)")
    print(f"Training RMSE: {train_rmse:.5f}")
    print(f"Base models: {', '.join(stacking_model.get_base_model_names())}")
    print(f"Meta-model: {stacking_model.get_meta_model_name()}")
    print(f"{'='*70}\n")
    
    # Save training predictions with residuals
    train_pred_df = pd.DataFrame({
        'id': train_ids,
        'actual': y_train.values,
        'predicted': train_predictions,
        'residual': y_train.values - train_predictions
    })
    save_results(train_pred_df, RESULTS_DIR / "stacking_train_predictions.csv", format='csv')
    
    # Save test predictions
    test_pred_df = pd.DataFrame({
        'id': test_ids,
        'predicted': test_predictions
    })
    save_results(test_pred_df, RESULTS_DIR / "stacking_test_predictions.csv", format='csv')
    
    # Create submission file
    submission_df = pd.DataFrame({
        'id': test_ids,
        'exam_score': test_predictions
    })
    save_results(submission_df, RESULTS_DIR / "stacking_submission.csv", format='csv')
    
    # Plot residuals
    plot_residuals(
        y_train.values, 
        train_predictions,
        "Stacking Model (Ridge Meta-Learner)",
        FIGURES_DIR / "stacking_residuals.png"
    )
    
    # Residual statistics
    residuals = y_train.values - train_predictions
    residual_stats = {
        'rmse': train_rmse,
        'mean_residual': float(np.mean(residuals)),
        'std_residual': float(np.std(residuals)),
        'min_residual': float(np.min(residuals)),
        'max_residual': float(np.max(residuals)),
        'mae': float(np.mean(np.abs(residuals)))
    }
    save_results(residual_stats, RESULTS_DIR / "stacking_residual_stats.json", format='json')
    
    print("\nResults saved:")
    print(f"  - Training predictions: {RESULTS_DIR / 'stacking_train_predictions.csv'}")
    print(f"  - Test predictions: {RESULTS_DIR / 'stacking_test_predictions.csv'}")
    print(f"  - Submission file: {RESULTS_DIR / 'stacking_submission.csv'}")
    print(f"  - Residual plot: {FIGURES_DIR / 'stacking_residuals.png'}")
    print(f"  - Residual stats: {RESULTS_DIR / 'stacking_residual_stats.json'}")
    
    print("\n" + "="*70)
    print("STACKING COMPLETE!")
    print("="*70 + "\n")
    
    # Optional: Compare with individual base model performance
    print("\nBase Model Performance (for comparison):")
    print("Note: These are averages across CV folds during stacking training")
    print("The stacking model combines these to achieve better performance")
    print("\nTo see detailed comparison, run scripts/run_training.py")


if __name__ == "__main__":
    main()
