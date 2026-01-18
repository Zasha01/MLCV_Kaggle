"""Cross-validation utilities."""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm
from ..config import N_FOLDS, SEED
from ..evaluation.metrics import calculate_rmse


def cross_validate_model(model_class, X, y, X_test, params=None, n_folds=N_FOLDS):
    """
    Perform cross-validation for a model.
    
    Args:
        model_class: Model class to instantiate
        X: Training features
        y: Training target
        X_test: Test features
        params: Model parameters (optional)
        n_folds: Number of CV folds
        
    Returns:
        dict: Results including OOF predictions, test predictions, and scores
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    oof_predictions = np.zeros(len(X))
    test_predictions = np.zeros(len(X_test))
    fold_scores = []
    feature_importance = None
    
    print(f"\nTraining {model_class.__name__}...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"  Fold {fold}/{n_folds}", end='')
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Initialize and train model
        model = model_class(params=params)
        model.fit(X_train, y_train, X_val, y_val)
        
        # Predictions
        oof_predictions[val_idx] = model.predict(X_val)
        test_predictions += model.predict(X_test) / n_folds
        
        # Score
        fold_score = calculate_rmse(y_val, oof_predictions[val_idx])
        fold_scores.append(fold_score)
        print(f" - RMSE: {fold_score:.5f}")
        
        # Feature importance (average across folds)
        importance = model.get_feature_importance()
        if importance is not None:
            if feature_importance is None:
                feature_importance = importance / n_folds
            else:
                feature_importance += importance / n_folds
    
    cv_mean = np.mean(fold_scores)
    cv_std = np.std(fold_scores)
    
    print(f"  Mean CV RMSE: {cv_mean:.5f} (+/- {cv_std:.5f})")
    
    results = {
        'model_name': model_class.__name__,
        'oof_predictions': oof_predictions,
        'test_predictions': test_predictions,
        'cv_scores': fold_scores,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'feature_importance': feature_importance
    }
    
    return results


def train_all_models(X, y, X_test, model_configs):
    """
    Train multiple models with cross-validation.
    
    Args:
        X: Training features
        y: Training target
        X_test: Test features
        model_configs: List of (model_class, params) tuples
        
    Returns:
        dict: Results for all models
    """
    all_results = {}
    
    print("\n" + "="*70)
    print("TRAINING ALL MODELS WITH CROSS-VALIDATION")
    print("="*70)
    
    for model_class, params in model_configs:
        results = cross_validate_model(model_class, X, y, X_test, params)
        all_results[results['model_name']] = results
    
    # Create summary DataFrame
    summary_df = pd.DataFrame([
        {
            'Model': name,
            'CV_RMSE': results['cv_mean'],
            'CV_Std': results['cv_std']
        }
        for name, results in all_results.items()
    ]).sort_values('CV_RMSE')
    
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    print(summary_df.to_string(index=False))
    print("="*70 + "\n")
    
    return all_results, summary_df

