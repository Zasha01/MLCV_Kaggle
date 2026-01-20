#!/usr/bin/env python3
"""
Train stacking ensemble with Ridge meta-learner.

This script uses SAVED OOF predictions from run_training.py instead of 
retraining base models. Much faster!

Usage:
    python scripts/run_stacking.py                    # Use saved predictions
    python scripts/run_stacking.py --retrain          # Retrain base models from scratch
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import FIGURES_DIR, RESULTS_DIR, SEED
from src.evaluation import save_results, calculate_rmse
from src.visualization import plot_residuals, setup_plotting


def load_saved_predictions():
    """Load OOF and test predictions from previous training run."""
    oof_path = RESULTS_DIR / "oof_predictions.csv"
    test_path = RESULTS_DIR / "test_predictions.csv"
    
    if not oof_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Saved predictions not found!\n"
            f"  Expected: {oof_path}\n"
            f"  Expected: {test_path}\n"
            f"Please run 'python scripts/run_training.py' first."
        )
    
    oof_df = pd.read_csv(oof_path)
    test_df = pd.read_csv(test_path)
    
    return oof_df, test_df


def train_stacking_from_saved(oof_df, test_df, meta_alpha=1.0):
    """
    Train stacking meta-learner using saved OOF predictions.
    
    Args:
        oof_df: DataFrame with columns [id, actual, LightGBMModel, XGBoostModel, CatBoostModel]
        test_df: DataFrame with columns [id, LightGBMModel, XGBoostModel, CatBoostModel]
        meta_alpha: Ridge regularization parameter
        
    Returns:
        tuple: (train_predictions, test_predictions, meta_model, scaler)
    """
    # Extract base model columns
    base_model_cols = ['LightGBMModel', 'XGBoostModel', 'CatBoostModel']
    
    # Prepare meta-features (OOF predictions as features)
    X_meta_train = oof_df[base_model_cols].values
    y_train = oof_df['actual'].values
    
    X_meta_test = test_df[base_model_cols].values
    
    print(f"\nMeta-feature shape (train): {X_meta_train.shape}")
    print(f"Meta-feature shape (test): {X_meta_test.shape}")
    
    # Standardize meta-features (important for Ridge)
    scaler = StandardScaler()
    X_meta_train_scaled = scaler.fit_transform(X_meta_train)
    X_meta_test_scaled = scaler.transform(X_meta_test)
    
    # Train Ridge meta-learner
    print(f"\nTraining Ridge meta-learner (alpha={meta_alpha})...")
    meta_model = Ridge(alpha=meta_alpha, random_state=SEED)
    meta_model.fit(X_meta_train_scaled, y_train)
    
    # Get coefficients (shows how much each base model contributes)
    print("\nMeta-learner coefficients:")
    for col, coef in zip(base_model_cols, meta_model.coef_):
        print(f"  {col}: {coef:.4f}")
    print(f"  Intercept: {meta_model.intercept_:.4f}")
    
    # Generate predictions
    train_predictions = meta_model.predict(X_meta_train_scaled)
    test_predictions = meta_model.predict(X_meta_test_scaled)
    
    return train_predictions, test_predictions, meta_model, scaler, base_model_cols


def main(retrain=False, meta_alpha=1.0):
    """Run stacking pipeline."""
    setup_plotting()
    
    print("\n" + "="*70)
    print("STACKING ENSEMBLE WITH RIDGE META-LEARNER")
    print("="*70)
    
    if retrain:
        print("\n[!] --retrain flag detected. This would retrain base models.")
        print("[!] Not implemented yet. Use saved predictions instead.")
        print("[!] Run 'python scripts/run_training.py' to retrain base models.\n")
        return
    
    # Load saved predictions
    print("\nLoading saved OOF and test predictions...")
    oof_df, test_df = load_saved_predictions()
    
    print(f"  OOF predictions: {len(oof_df)} samples")
    print(f"  Test predictions: {len(test_df)} samples")
    
    # Compare with simple blending baseline
    base_model_cols = ['LightGBMModel', 'XGBoostModel', 'CatBoostModel']
    y_train = oof_df['actual'].values
    
    # Simple average baseline
    simple_avg_oof = oof_df[base_model_cols].mean(axis=1).values
    simple_avg_rmse = calculate_rmse(y_train, simple_avg_oof)
    print(f"\nBaseline (simple average): RMSE = {simple_avg_rmse:.5f}")
    
    # Best single model
    for col in base_model_cols:
        rmse = calculate_rmse(y_train, oof_df[col].values)
        print(f"  {col}: RMSE = {rmse:.5f}")
    
    # Train stacking meta-learner
    train_preds, test_preds, meta_model, scaler, cols = train_stacking_from_saved(
        oof_df, test_df, meta_alpha=meta_alpha
    )
    
    # Calculate stacking RMSE
    stacking_rmse = calculate_rmse(y_train, train_preds)
    
    print(f"\n{'='*70}")
    print("STACKING RESULTS")
    print(f"{'='*70}")
    print(f"Stacking (Ridge meta-learner): RMSE = {stacking_rmse:.5f}")
    print(f"Simple Average baseline:       RMSE = {simple_avg_rmse:.5f}")
    print(f"Improvement over simple avg:   {(simple_avg_rmse - stacking_rmse):.5f} ({(simple_avg_rmse - stacking_rmse)/simple_avg_rmse*100:.2f}%)")
    print(f"{'='*70}")
    
    # Save results
    print("\nSaving results...")
    
    # Training predictions with residuals
    train_pred_df = pd.DataFrame({
        'id': oof_df['id'],
        'actual': y_train,
        'predicted': train_preds,
        'residual': y_train - train_preds
    })
    save_results(train_pred_df, RESULTS_DIR / "stacking_train_predictions.csv", format='csv')
    
    # Test predictions
    test_pred_df = pd.DataFrame({
        'id': test_df['id'],
        'predicted': test_preds
    })
    save_results(test_pred_df, RESULTS_DIR / "stacking_test_predictions.csv", format='csv')
    
    # Submission file
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'exam_score': np.clip(test_preds, y_train.min(), y_train.max())
    })
    save_results(submission_df, RESULTS_DIR / "stacking_submission.csv", format='csv')
    
    # Plot residuals
    plot_residuals(
        y_train, 
        train_preds,
        "Stacking (Ridge Meta-Learner)",
        FIGURES_DIR / "stacking_residuals.png"
    )
    
    # Residual statistics
    residuals = y_train - train_preds
    residual_stats = {
        'model': 'Stacking (Ridge Meta-Learner)',
        'rmse': float(stacking_rmse),
        'simple_avg_rmse': float(simple_avg_rmse),
        'improvement': float(simple_avg_rmse - stacking_rmse),
        'improvement_pct': float((simple_avg_rmse - stacking_rmse) / simple_avg_rmse * 100),
        'mean_residual': float(np.mean(residuals)),
        'std_residual': float(np.std(residuals)),
        'min_residual': float(np.min(residuals)),
        'max_residual': float(np.max(residuals)),
        'mae': float(np.mean(np.abs(residuals))),
        'meta_coefficients': {col: float(coef) for col, coef in zip(cols, meta_model.coef_)},
        'meta_intercept': float(meta_model.intercept_),
        'meta_alpha': meta_alpha
    }
    save_results(residual_stats, RESULTS_DIR / "stacking_stats.json", format='json')
    
    # Update model comparison
    comparison_path = RESULTS_DIR / "model_comparison_with_ensemble.csv"
    if comparison_path.exists():
        comparison_df = pd.read_csv(comparison_path)
        
        # Remove old stacking row if exists
        comparison_df = comparison_df[~comparison_df['Model'].str.contains('Stacking', case=False, na=False)]
        
        # Add new stacking result
        new_row = pd.DataFrame([{
            'Model': 'Stacking (Ridge Meta)',
            'CV_RMSE': stacking_rmse,
            'CV_Std': 0.0
        }])
        comparison_df = pd.concat([comparison_df, new_row], ignore_index=True)
        comparison_df = comparison_df.sort_values('CV_RMSE')
        
        save_results(comparison_df, RESULTS_DIR / "model_comparison_with_stacking.csv", format='csv')
        
        print("\nUpdated model comparison:")
        print(comparison_df.to_string(index=False))
    
    print(f"\n{'='*70}")
    print("FILES SAVED:")
    print(f"{'='*70}")
    print(f"  Training predictions: {RESULTS_DIR / 'stacking_train_predictions.csv'}")
    print(f"  Test predictions:     {RESULTS_DIR / 'stacking_test_predictions.csv'}")
    print(f"  Submission file:      {RESULTS_DIR / 'stacking_submission.csv'}")
    print(f"  Residual plot:        {FIGURES_DIR / 'stacking_residuals.png'}")
    print(f"  Statistics:           {RESULTS_DIR / 'stacking_stats.json'}")
    print(f"  Model comparison:     {RESULTS_DIR / 'model_comparison_with_stacking.csv'}")
    print(f"{'='*70}")
    
    print("\n" + "="*70)
    print("STACKING COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train stacking ensemble with Ridge meta-learner"
    )
    parser.add_argument(
        '--retrain', 
        action='store_true',
        help='Retrain base models from scratch (slower)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=1.0,
        help='Ridge regularization parameter (default: 1.0)'
    )
    
    args = parser.parse_args()
    main(retrain=args.retrain, meta_alpha=args.alpha)
