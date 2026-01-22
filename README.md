# Student Test Score Prediction - Kaggle Playground Series S6E1

Modular machine learning codebase for the Kaggle Playground Series S6E1 competition. Includes traditional models, gradient boosting, custom SE-ResNet architecture, and an IEEE conference paper.

## 📁 Project Structure

```
MLCV_Kaggle/
├── data/                      # Dataset files
│   ├── train.csv             # Training data
│   ├── test.csv              # Test data
│   └── sample_submission.csv # Submission format
│
├── src/                       # Modular source code
│   ├── config.py             # Global configuration and hyperparameters
│   ├── data/
│   │   ├── loader.py         # Data loading utilities
│   │   └── features.py       # Feature engineering functions
│   ├── models/
│   │   ├── base.py           # Base model class
│   │   ├── ridge.py          # Ridge regression
│   │   ├── random_forest.py  # Random Forest
│   │   ├── lightgbm_model.py # LightGBM implementation
│   │   ├── xgboost_model.py  # XGBoost implementation
│   │   ├── catboost_model.py # CatBoost implementation
│   │   ├── senet.py          # SE-ResNet neural network
│   │   └── ensemble.py       # Ensemble methods
│   ├── training/
│   │   ├── cross_validation.py # CV framework
│   │   └── tuning.py         # Hyperparameter optimization
│   ├── evaluation/
│   │   └── metrics.py        # Evaluation metrics
│   └── visualization/
│       └── plots.py          # Plotting utilities
│
├── scripts/                   # Executable scripts
│   ├── run_eda.py            # Exploratory data analysis
│   ├── run_training.py       # Train models with CV
│   ├── run_tuning.py         # Bayesian hyperparameter tuning
│   ├── run_stacking.py       # Stacking ensemble
│   ├── run_senet.py          # SE-ResNet training
│   ├── run_shap_analysis.py  # SHAP interpretability
│   ├── run_submission.py     # Generate submission file
│
├── outputs/                   # Generated outputs
│   ├── figures/              # Visualizations
│   ├── models/               # Saved model files
│   └── results/              # JSON results and metrics
│
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🚀 Installation

```bash
# Clone repository
git clone <repository-url>
cd MLCV_Kaggle

# Install dependencies
pip install -r requirements.txt
```

## 🔧 Usage

### Running Analysis and Training

```bash
# 1. Exploratory Data Analysis
python scripts/run_eda.py
# Generates visualizations in outputs/figures/

# 2. Train all models with cross-validation
python scripts/run_training.py
# Trains Ridge, Random Forest, LightGBM, XGBoost, CatBoost

# 3. Hyperparameter tuning (optional)
python scripts/run_tuning.py
# Bayesian optimization for gradient boosting models

# 4. Train SE-ResNet neural network
python scripts/run_senet.py
# Custom deep learning model with entity embeddings

# 5. Run SHAP analysis
python scripts/run_shap_analysis.py
# Generate feature importance visualizations

# 6. Generate Kaggle submission
python scripts/run_submission.py
# Creates submission.csv with predictions
```

## 📝 Building the Paper

The `latex/` directory contains the IEEE conference paper source:

```bash
cd latex/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Output: `main.pdf`

## ⚙️ Configuration

All settings are centralized in `src/config.py`:

- **Paths**: Data directories, output locations
- **Hyperparameters**: Model-specific parameters for all algorithms
- **Seeds**: Random seed for reproducibility (default: 42)
- **Cross-validation**: Number of folds (default: 5)
- **Features**: Column definitions (numerical/categorical)
- **Plotting**: DPI, style, colors

## 📦 Code Organization

### Design Principles
- **Modular**: Separated concerns (data, models, training, evaluation)
- **Reusable**: Base classes with consistent interfaces
- **Configurable**: Centralized configuration management
- **Reproducible**: Fixed random seeds, version-controlled
- **Maintainable**: Clean structure with docstrings

### Key Modules

**`src/data/`** - Data handling
- Data loading and preprocessing
- Feature engineering (interactions, polynomials, target encoding)
- CV-safe transformations

**`src/models/`** - Model implementations
- Base class for unified interface
- Sklearn/LightGBM/XGBoost/CatBoost wrappers
- Custom SE-ResNet implementation (PyTorch)

**`src/training/`** - Training utilities
- Cross-validation framework
- Hyperparameter optimization (Bayesian)
- Model checkpointing

**`src/evaluation/`** - Evaluation tools
- RMSE and other metrics
- Residual analysis
- Model comparison

**`src/visualization/`** - Plotting
- EDA visualizations
- Feature importance plots
- SHAP visualizations

## 🎯 Workflow

Standard workflow for reproducing results:

1. Place data files in `data/` directory
2. Run `python scripts/run_eda.py` to understand the data
3. Run `python scripts/run_training.py` to train baseline and boosting models
4. Run `python scripts/run_senet.py` to train deep learning model
5. Run `python scripts/run_submission.py` to generate predictions
6. (Optional) Run `python scripts/run_shap_analysis.py` for interpretability

All outputs (figures, models, results) saved in `outputs/`.

## 👥 Team

University of Porto (FEUP) - Machine Learning Course
- Lars Moen Storvik (up202508437@up.pt)
- Tina Kovačević (up202501724@up.pt)
- Zakariea Sharfeddine (up202501730@up.pt)

## 📄 License

Educational project for University of Porto.

