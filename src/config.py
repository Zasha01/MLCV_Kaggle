"""Global configuration for the project."""
import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"
MODELS_DIR = OUTPUT_DIR / "models"

# Create directories if they don't exist
for dir_path in [FIGURES_DIR, RESULTS_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Random seed for reproducibility
SEED = 42

# Cross-validation settings
N_FOLDS = 5

# Feature columns
NUMERICAL_COLS = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
CATEGORICAL_COLS = [
    'gender', 'course', 'internet_access', 'sleep_quality',
    'study_method', 'facility_rating', 'exam_difficulty'
]

# Target column
TARGET_COL = 'exam_score'

# Model hyperparameters (defaults)
RIDGE_PARAMS = {
    'alpha': 10.0,
    'random_state': SEED
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 15,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'random_state': SEED,
    'n_jobs': -1
}

LIGHTGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': -1,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': SEED,
    'verbose': -1
}

XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': SEED,
    'n_estimators': 10000,
    'tree_method': 'hist',
    'early_stopping_rounds': 100
}

CATBOOST_PARAMS = {
    'loss_function': 'RMSE',
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3,
    'iterations': 10000,
    'random_seed': SEED,
    'verbose': False,
    'early_stopping_rounds': 100
}

# Plotting configuration
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
PLOT_PALETTE = 'husl'
FIG_DPI = 300

