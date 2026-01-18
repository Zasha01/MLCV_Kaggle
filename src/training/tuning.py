"""Hyperparameter tuning with Optuna."""
import optuna
import numpy as np
from sklearn.model_selection import KFold
from ..config import SEED, N_FOLDS
from ..evaluation.metrics import calculate_rmse
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor


def tune_lightgbm(X, y, n_trials=50, n_folds=3):
    """
    Tune LightGBM hyperparameters using Optuna.
    
    Args:
        X: Training features
        y: Training target
        n_trials: Number of optimization trials
        n_folds: Number of CV folds for evaluation
        
    Returns:
        dict: Best parameters
    """
    print("\n" + "="*70)
    print(f"TUNING LIGHTGBM ({n_trials} trials)")
    print("="*70)
    
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': 5,
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
            'random_state': SEED,
            'verbose': -1
        }
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val)
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
            
            preds = model.predict(X_val, num_iteration=model.best_iteration)
            score = calculate_rmse(y_val, preds)
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    study = optuna.create_study(direction='minimize', study_name='lightgbm_tuning')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest RMSE: {study.best_value:.5f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study.best_params


def tune_xgboost(X, y, n_trials=50, n_folds=3):
    """
    Tune XGBoost hyperparameters using Optuna.
    
    Args:
        X: Training features
        y: Training target
        n_trials: Number of optimization trials
        n_folds: Number of CV folds for evaluation
        
    Returns:
        dict: Best parameters
    """
    print("\n" + "="*70)
    print(f"TUNING XGBOOST ({n_trials} trials)")
    print("="*70)
    
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
            'random_state': SEED,
            'n_estimators': 1000,
            'tree_method': 'hist',
            'early_stopping_rounds': 50
        }
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            preds = model.predict(X_val)
            score = calculate_rmse(y_val, preds)
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    study = optuna.create_study(direction='minimize', study_name='xgboost_tuning')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest RMSE: {study.best_value:.5f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study.best_params


def tune_catboost(X, y, n_trials=50, n_folds=3):
    """
    Tune CatBoost hyperparameters using Optuna.
    
    Args:
        X: Training features
        y: Training target
        n_trials: Number of optimization trials
        n_folds: Number of CV folds for evaluation
        
    Returns:
        dict: Best parameters
    """
    print("\n" + "="*70)
    print(f"TUNING CATBOOST ({n_trials} trials)")
    print("="*70)
    
    def objective(trial):
        params = {
            'loss_function': 'RMSE',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'iterations': 1000,
            'random_seed': SEED,
            'verbose': False,
            'early_stopping_rounds': 50
        }
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = CatBoostRegressor(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            
            preds = model.predict(X_val)
            score = calculate_rmse(y_val, preds)
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    study = optuna.create_study(direction='minimize', study_name='catboost_tuning')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest RMSE: {study.best_value:.5f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study.best_params


def tune_all_models(X, y, n_trials=50):
    """
    Tune all gradient boosting models.
    
    Args:
        X: Training features
        y: Training target
        n_trials: Number of trials per model
        
    Returns:
        dict: Best parameters for each model
    """
    best_params = {}
    
    # Tune LightGBM
    best_params['lightgbm'] = tune_lightgbm(X, y, n_trials)
    
    # Tune XGBoost
    best_params['xgboost'] = tune_xgboost(X, y, n_trials)
    
    # Tune CatBoost
    best_params['catboost'] = tune_catboost(X, y, n_trials)
    
    return best_params

