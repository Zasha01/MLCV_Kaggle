# Quick Start Guide

## Prerequisites

Activate the conda environment:
```bash
conda activate coding
```

## Running the Pipeline

### Step 1: Generate EDA Plots (5-10 minutes)
```bash
python scripts/run_eda.py
```
**Output:** Saves plots to `outputs/figures/`
- `target_distribution.png`
- `correlation_matrix.png`
- `scatter_numerical_vs_target.png`
- `categorical_boxplots.png`

### Step 2: Train All Models (15-20 minutes)
```bash
python scripts/run_training.py
```
**Output:** 
- Trains Ridge, Random Forest, LightGBM, XGBoost, CatBoost
- Creates ensemble models (simple & optimized)
- Saves results to `outputs/results/`
- Saves plots to `outputs/figures/`

**Key Files Generated:**
- `model_comparison.csv` - Performance metrics
- `feature_importance.csv` - Top features
- `ensemble_weights.json` - Optimal blend weights
- `oof_predictions.csv` - Out-of-fold predictions
- `test_predictions.csv` - Test predictions
- `residual_stats.json` - Residual analysis

### Step 3 (Optional): Hyperparameter Tuning (30-60 minutes)
```bash
# Quick tuning (10 trials per model)
python scripts/run_tuning.py --n_trials 10

# Default tuning (50 trials per model)
python scripts/run_tuning.py --n_trials 50

# Thorough tuning (100 trials per model)
python scripts/run_tuning.py --n_trials 100
```
**Output:** `outputs/results/best_params.json`

**Note:** After tuning, update `src/config.py` with the best parameters and rerun `run_training.py`

### Step 4: Generate Submission File
```bash
python scripts/run_submission.py
```
**Output:** `submission.csv` in project root

## Running Everything at Once

```bash
conda activate coding
python scripts/run_eda.py && \
python scripts/run_training.py && \
python scripts/run_submission.py
```

## Project Structure

```
src/
├── config.py           # Configuration (hyperparameters, paths, seeds)
├── data/               # Data loading & feature engineering
├── models/             # Model implementations (Ridge, RF, LightGBM, XGBoost, CatBoost, Ensemble)
├── training/           # Cross-validation & hyperparameter tuning (Optuna)
├── evaluation/         # Metrics (RMSE, residual analysis)
└── visualization/      # Plotting functions (save to disk)

scripts/
├── run_eda.py         # Exploratory data analysis
├── run_training.py    # Train all models with CV
├── run_tuning.py      # Bayesian hyperparameter optimization
└── run_submission.py  # Generate final submission

outputs/
├── figures/           # All plots (300 DPI PNG)
├── results/           # CSV/JSON results
└── models/            # Model checkpoints (future)
```

## Expected Results

Based on the original notebook:

| Model | CV RMSE |
|-------|---------|
| Ensemble (Optimized) | ~8.757 |
| CatBoost | ~8.763 |
| LightGBM | ~8.765 |
| XGBoost | ~8.821 |
| Random Forest | ~9.285 |
| Ridge | ~9.415 |

## Troubleshooting

### Import Errors
Make sure you're in the project root:
```bash
cd /home/zaka/Documents/MLCV_Kaggle
```

### Missing Conda Environment
```bash
conda activate coding
```

### Check Module Structure
```bash
python -c "from src.data import load_data; print('OK')"
```

## What's Different from the Notebook?

### ✅ Improvements
1. **Modular Code**: Easy to modify individual components
2. **Reusability**: Import modules in other projects
3. **Saved Outputs**: All plots and results saved automatically
4. **Hyperparameter Tuning**: Optuna-based Bayesian optimization
5. **Better Organization**: Clear separation of concerns
6. **Easier Collaboration**: Multiple team members can work on different modules
7. **Version Control Friendly**: Small files instead of large notebook

### 📊 Same Functionality
- All 44 engineered features
- Same models (Ridge, RF, LightGBM, XGBoost, CatBoost)
- Same ensemble methods (simple & optimized)
- Same evaluation metrics
- Same data processing pipeline

## For the Report

All required outputs are automatically saved:

### Figures for LaTeX
```
outputs/figures/
├── target_distribution.png
├── correlation_matrix.png
├── scatter_numerical_vs_target.png
├── categorical_boxplots.png
├── feature_importance.png
├── model_comparison.png
├── model_comparison_with_ensemble.png
└── residual_analysis.png
```

### Results Tables
```
outputs/results/
├── model_comparison.csv              # Main results table
├── feature_importance.csv            # Top features
├── ensemble_weights.json             # Optimal weights
├── residual_stats.json               # Residual metrics
└── best_params.json                  # Tuned hyperparameters (if run)
```

## Next Steps

1. Run the pipeline to generate all outputs
2. Check `outputs/figures/` for plots to include in report
3. Check `outputs/results/` for tables and metrics
4. Optional: Run hyperparameter tuning for better results
5. Update LaTeX report with new figures and results

