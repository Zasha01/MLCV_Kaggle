#!/usr/bin/env python3
"""Train all models with cross-validation."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import FIGURES_DIR, RESULTS_DIR, SEED
from src.data import load_data, prepare_datasets
from src.models import (
    RidgeModel, RandomForestModel,
    LightGBMModel, XGBoostModel, CatBoostModel,
    EnsembleModel
)
from src.training import train_all_models
from src.visualization import (
    plot_model_comparison,
    plot_feature_importance,
    plot_residuals,
    setup_plotting
)
from src.evaluation import save_results, get_residual_stats


def main():
    """Run training pipeline."""
    # Setup plotting
    setup_plotting()
    
    # Load and prepare data
    print("\nLoading and preparing data...")
    train, test, _ = load_data()
    X_train, y_train, X_test, test_ids, train_ids = prepare_datasets(train, test)
    
    # Define models to train
    model_configs = [
        (RidgeModel, None),
        (RandomForestModel, None),
        (LightGBMModel, None),
        (XGBoostModel, None),
        (CatBoostModel, None),
    ]
    
    # Train all models
    all_results, summary_df = train_all_models(X_train, y_train, X_test, model_configs)
    
    # Save model comparison
    save_results(summary_df, RESULTS_DIR / "model_comparison.csv", format='csv')
    
    # Plot model comparison
    plot_model_comparison(summary_df, FIGURES_DIR / "model_comparison.png")
    
    # Get best single models for ensemble
    gbm_models = ['LightGBMModel', 'XGBoostModel', 'CatBoostModel']
    
    # Train ensemble (simple average)
    print("\n" + "="*70)
    print("ENSEMBLE - SIMPLE AVERAGE")
    print("="*70)
    ensemble_simple_pred = np.mean([
        all_results[name]['test_predictions'] for name in gbm_models
    ], axis=0)
    ensemble_simple_oof = np.mean([
        all_results[name]['oof_predictions'] for name in gbm_models
    ], axis=0)
    ensemble_simple_rmse = np.sqrt(np.mean((y_train - ensemble_simple_oof) ** 2))
    print(f"Simple Average Ensemble RMSE: {ensemble_simple_rmse:.5f}")
    
    # Train ensemble (optimized weights)
    print("\n" + "="*70)
    print("ENSEMBLE - OPTIMIZED WEIGHTS")
    print("="*70)
    
    # Create dummy model instances for ensemble
    from src.models.base import BaseModel
    
    class DummyModel(BaseModel):
        def __init__(self, name, predictions):
            super().__init__(name=name)
            self.predictions = predictions
            
        def fit(self, X, y, X_val=None, y_val=None):
            return self
        
        def predict(self, X):
            # Return predictions based on length
            if len(X) == len(self.predictions):
                return self.predictions
            # For train set, return a subset
            return self.predictions[:len(X)]
    
    dummy_models = [
        DummyModel(name, all_results[name]['oof_predictions'])
        for name in gbm_models
    ]
    
    ensemble_opt = EnsembleModel(dummy_models, mode='optimized')
    ensemble_opt.optimize_weights(X_train, y_train)
    
    print("Optimized weights:")
    weights = ensemble_opt.get_weights()
    for name, weight in weights.items():
        print(f"  {name}: {weight:.4f}")
    
    # Get optimized ensemble predictions
    ensemble_opt_oof = ensemble_opt.predict(X_train)
    ensemble_opt_rmse = np.sqrt(np.mean((y_train - ensemble_opt_oof) ** 2))
    print(f"\nOptimized Ensemble RMSE: {ensemble_opt_rmse:.5f}")
    
    # Apply optimized weights to test predictions
    test_dummy_models = [
        DummyModel(name, all_results[name]['test_predictions'])
        for name in gbm_models
    ]
    ensemble_opt_test = EnsembleModel(test_dummy_models, mode='optimized')
    ensemble_opt_test.weights = ensemble_opt.weights
    ensemble_opt_test_pred = ensemble_opt_test.predict(X_test)
    
    # Save ensemble weights
    save_results(weights, RESULTS_DIR / "ensemble_weights.json", format='json')
    
    # Update summary with ensemble results
    ensemble_rows = pd.DataFrame([
        {'Model': 'Ensemble (Simple)', 'CV_RMSE': ensemble_simple_rmse, 'CV_Std': 0.0},
        {'Model': 'Ensemble (Optimized)', 'CV_RMSE': ensemble_opt_rmse, 'CV_Std': 0.0}
    ])
    summary_df = pd.concat([summary_df, ensemble_rows], ignore_index=True).sort_values('CV_RMSE')
    
    # Save updated comparison
    save_results(summary_df, RESULTS_DIR / "model_comparison_with_ensemble.csv", format='csv')
    plot_model_comparison(summary_df, FIGURES_DIR / "model_comparison_with_ensemble.png")
    
    # Feature importance (from best GBM model)
    best_gbm = min(gbm_models, key=lambda x: all_results[x]['cv_mean'])
    feature_importance = all_results[best_gbm]['feature_importance']
    
    if feature_importance is not None:
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        save_results(importance_df, RESULTS_DIR / "feature_importance.csv", format='csv')
        plot_feature_importance(importance_df, FIGURES_DIR / "feature_importance.png")
    
    # Residual analysis for best model
    best_model_name = summary_df.iloc[0]['Model']
    if 'Ensemble' in best_model_name:
        best_oof = ensemble_opt_oof if 'Optimized' in best_model_name else ensemble_simple_oof
        best_test = ensemble_opt_test_pred if 'Optimized' in best_model_name else ensemble_simple_pred
    else:
        best_oof = all_results[best_model_name]['oof_predictions']
        best_test = all_results[best_model_name]['test_predictions']
    
    plot_residuals(y_train.values, best_oof, best_model_name, 
                   FIGURES_DIR / "residual_analysis.png")
    
    residual_stats = get_residual_stats(y_train.values, best_oof)
    save_results(residual_stats, RESULTS_DIR / "residual_stats.json", format='json')
    
    # Save OOF and test predictions for best models
    oof_df = pd.DataFrame({
        'id': train_ids,
        'actual': y_train.values,
        'best_model': best_oof
    })
    for name in gbm_models:
        oof_df[name] = all_results[name]['oof_predictions']
    
    save_results(oof_df, RESULTS_DIR / "oof_predictions.csv", format='csv')
    
    # Save test predictions
    test_pred_df = pd.DataFrame({
        'id': test_ids,
        'best_model': best_test
    })
    for name in gbm_models:
        test_pred_df[name] = all_results[name]['test_predictions']
    
    save_results(test_pred_df, RESULTS_DIR / "test_predictions.csv", format='csv')
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Plots saved to: {FIGURES_DIR}")
    print("="*70 + "\n")
    
    print(f"Best Model: {best_model_name}")
    print(f"Best CV RMSE: {summary_df.iloc[0]['CV_RMSE']:.5f}")


if __name__ == "__main__":
    main()


