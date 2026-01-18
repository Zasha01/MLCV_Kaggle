# Student Test Score Prediction - Modular ML Pipeline

A comprehensive, modular machine learning pipeline for predicting student test scores. This refactored codebase separates concerns into distinct modules for data processing, model training, hyperparameter tuning, and visualization.

## 📁 Project Structure

```
MLCV_Kaggle/
├── src/                          # Source code modules
│   ├── config.py                 # Global configuration and hyperparameters
│   ├── data/                     # Data loading and feature engineering
│   │   ├── loader.py
│   │   └── features.py
│   ├── models/                   # Model implementations
│   │   ├── base.py               # Abstract base class
│   │   ├── ridge.py
│   │   ├── random_forest.py
│   │   ├── lightgbm_model.py
│   │   ├── xgboost_model.py
│   │   ├── catboost_model.py
│   │   └── ensemble.py
│   ├── training/                 # Training utilities
│   │   ├── cross_validation.py
│   │   └── tuning.py             # Optuna hyperparameter optimization
│   ├── evaluation/               # Metrics and evaluation
│   │   └── metrics.py
│   └── visualization/            # Plotting functions
│       └── plots.py
├── scripts/                      # Executable scripts
│   ├── run_eda.py               # Generate EDA plots
│   ├── run_training.py          # Train all models with CV
│   ├── run_tuning.py            # Hyperparameter tuning
│   └── run_submission.py        # Generate submission file
├── outputs/                      # Generated outputs
│   ├── figures/                 # Saved plots for report
│   ├── results/                 # CSV/JSON results
│   └── models/                  # Model checkpoints (future)
├── data/                        # Raw data files
├── latex/                       # LaTeX report
├── notebooks/                   # Legacy notebooks
└── requirements.txt             # Python dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

```bash
# Step 1: Generate EDA plots
python scripts/run_eda.py

# Step 2: Train all models with cross-validation
python scripts/run_training.py

# Step 3: (Optional) Hyperparameter tuning
python scripts/run_tuning.py --n_trials 50

# Step 4: Generate submission file
python scripts/run_submission.py
```

## 📊 What Each Script Does

### `run_eda.py` - Exploratory Data Analysis
- Loads training data
- Generates and saves EDA plots:
  - `target_distribution.png` - Distribution of exam scores
  - `correlation_matrix.png` - Feature correlation heatmap
  - `scatter_numerical_vs_target.png` - Scatter plots vs target
  - `categorical_boxplots.png` - Categorical feature distributions

**Output:** All plots saved to `outputs/figures/`

### `run_training.py` - Model Training
- Loads and engineers features (44 total features)
- Trains 5 models with 5-fold cross-validation:
  - Ridge Regression (baseline)
  - Random Forest
  - LightGBM
  - XGBoost
  - CatBoost
- Creates ensemble models:
  - Simple average ensemble
  - Optimized weighted ensemble (using scipy.optimize)
- Saves comprehensive results:
  - `model_comparison.csv` - CV RMSE for all models
  - `feature_importance.csv` - Top features
  - `ensemble_weights.json` - Optimal blend weights
  - `oof_predictions.csv` - Out-of-fold predictions
  - `test_predictions.csv` - Test set predictions
  - `residual_stats.json` - Residual analysis
  - Various plots for the report

**Output:** Results in `outputs/results/`, plots in `outputs/figures/`

### `run_tuning.py` - Hyperparameter Optimization
- Uses **Optuna** for Bayesian optimization
- Tunes LightGBM, XGBoost, and CatBoost
- Saves best parameters to `best_params.json`
- Configurable number of trials (default: 50)

**Usage:**
```bash
python scripts/run_tuning.py --n_trials 100  # More thorough search
```

**Output:** `outputs/results/best_params.json`

### `run_submission.py` - Generate Submission
- Loads best model predictions from `run_training.py`
- Creates `submission.csv` with proper format
- Clips predictions to valid range [0, 100]

**Output:** `submission.csv` in project root

## 🔧 Configuration

All configuration is centralized in `src/config.py`:

- **Paths:** Data directories, output directories
- **Random Seed:** `SEED = 42` for reproducibility
- **Cross-Validation:** `N_FOLDS = 5`
- **Feature Lists:** Numerical and categorical columns
- **Model Hyperparameters:** Default parameters for all models

To modify hyperparameters, either:
1. Edit `src/config.py` directly
2. Pass custom parameters when instantiating models
3. Use tuned parameters from `run_tuning.py`

## 🧪 Feature Engineering

The pipeline creates **44 engineered features** from 11 original features:

### Interaction Features
- `study_attendance`, `study_sleep`, `attendance_sleep`, `age_study`

### Ratio Features
- `attendance_per_hour`, `sleep_per_study`

### Polynomial Features
- Squared and cubed terms for key features
- Square root transformations

### Binary Flags
- `is_low_sleep`, `is_high_attendance`, `is_high_study`

### Domain Knowledge Formula
```python
formula_score = 6.0 * study_hours + 0.35 * class_attendance + 1.5 * sleep_hours
```

### Encoding Strategies
- **Label Encoding:** For tree-based models
- **Target Encoding:** CV-safe encoding to prevent leakage
- **Frequency Encoding:** Category frequency as feature

## 📈 Model Performance

Based on 5-fold cross-validation:

| Model | CV RMSE | CV Std |
|-------|---------|--------|
| Ensemble (Optimized) | **8.7568** | 0.0000 |
| CatBoost | 8.7627 | 0.0837 |
| LightGBM | 8.7648 | 0.0838 |
| XGBoost | 8.8211 | 0.0900 |
| Random Forest | 9.2846 | 0.0826 |
| Ridge | 9.4151 | 0.0801 |

**Optimal Ensemble Weights:**
- CatBoost: 62%
- LightGBM: 33%
- XGBoost: 5%

## 🔍 Code Architecture

### Base Model Class
All models inherit from `BaseModel` abstract class:

```python
class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y, X_val=None, y_val=None): pass
    
    @abstractmethod
    def predict(self, X): pass
    
    def get_feature_importance(self): pass
```

This ensures consistent interface across all models.

### Cross-Validation
Standardized CV implementation in `src/training/cross_validation.py`:
- 5-fold KFold with shuffling
- Out-of-fold (OOF) predictions for stacking
- Averaged test predictions across folds
- Per-fold and overall RMSE tracking

### Hyperparameter Tuning
Optuna-based Bayesian optimization:
- Automatic early stopping
- 3-fold CV for each trial
- Log-scale search for regularization parameters
- Study results saved for analysis

## 📝 Adding New Models

To add a new model:

1. Create a new file in `src/models/` (e.g., `my_model.py`)
2. Inherit from `BaseModel`
3. Implement `fit()` and `predict()` methods
4. Add to `src/models/__init__.py`
5. Add configuration to `src/config.py`
6. Add to model list in `scripts/run_training.py`

Example:

```python
from .base import BaseModel

class MyModel(BaseModel):
    def __init__(self, params=None):
        super().__init__(name="MyModel")
        self.params = params or {}
        self.model = None
        
    def fit(self, X, y, X_val=None, y_val=None):
        # Your training code
        return self
    
    def predict(self, X):
        return self.model.predict(X)
```

## 📊 Output Files for Report

### Figures (`outputs/figures/`)
All plots are saved at 300 DPI for publication quality:
- EDA plots (distributions, correlations, relationships)
- Model comparison bar chart
- Feature importance plot
- Residual analysis plots

### Results (`outputs/results/`)
Structured data files for easy report generation:
- `model_comparison.csv` - Full model results table
- `feature_importance.csv` - Top features with scores
- `best_params.json` - Tuned hyperparameters
- `ensemble_weights.json` - Optimal blend weights
- `residual_stats.json` - Comprehensive residual metrics
- `oof_predictions.csv` - For ensemble stacking
- `test_predictions.csv` - For final submission

## 🧪 Testing Individual Modules

You can import and use modules independently:

```python
# Load data
from src.data import load_data, prepare_datasets
train, test, _ = load_data()
X_train, y_train, X_test, test_ids, train_ids = prepare_datasets(train, test)

# Train a single model
from src.models import LightGBMModel
model = LightGBMModel()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluate
from src.evaluation import calculate_rmse
rmse = calculate_rmse(y_val, predictions)

# Plot results
from src.visualization import plot_residuals
plot_residuals(y_val, predictions, "LightGBM", "residuals.png")
```

## 🐛 Troubleshooting

### Import Errors
If you get import errors when running scripts:
```bash
# Make sure you're running from the project root
cd /path/to/MLCV_Kaggle
python scripts/run_training.py
```

### Missing Dependencies
```bash
pip install -r requirements.txt --upgrade
```

### Memory Issues
For large datasets, reduce CV folds or use sampling:
```python
# In src/config.py
N_FOLDS = 3  # Instead of 5
```

## 📚 Key Dependencies

- **pandas, numpy:** Data manipulation
- **scikit-learn:** Preprocessing, Ridge, Random Forest
- **lightgbm, xgboost, catboost:** Gradient boosting models
- **optuna:** Hyperparameter optimization
- **matplotlib, seaborn:** Visualization
- **scipy:** Ensemble weight optimization

## 🎯 Next Steps

1. **Stacking Ensemble:** Implement meta-model stacking
2. **Neural Network:** Add deep learning model
3. **Feature Selection:** Automated feature selection
4. **Model Checkpointing:** Save/load trained models
5. **Interactive Dashboard:** Streamlit app for results
6. **Automated Testing:** Unit tests for modules

## 📄 License

This project is for educational purposes as part of a university project.

## 👥 Contributors

Group project for MLCV course.

---

**For questions or issues, refer to the original notebook:** `student_score_prediction.ipynb`


