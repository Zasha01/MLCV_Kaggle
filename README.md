# Student Test Score Prediction - Kaggle Competition

This repository contains a comprehensive solution for the Kaggle Playground Series S6E1 competition to predict student test scores.

## 📁 Project Structure

```
MLCV_Kaggle/
├── data/
│   ├── train.csv              # Training data (630,000 samples)
│   ├── test.csv               # Test data (270,000 samples)
│   └── sample_submission.csv  # Sample submission format
├── kaggle_notebooks/          # Reference notebooks from Kaggle
├── student_score_prediction.ipynb  # Main solution notebook
├── context.md                 # Competition strategy and plan
└── README.md                  # This file
```

## 🎯 Competition Overview

- **Goal**: Predict student exam scores from demographic and behavioral features
- **Metric**: RMSE (Root Mean Squared Error) - lower is better
- **Dataset**: Synthetically generated from deep learning model
- **Features**: 12 features (4 numerical, 7 categorical)

### Features

**Numerical:**
- `age`: Student age (17-24)
- `study_hours`: Weekly study hours
- `class_attendance`: Attendance percentage
- `sleep_hours`: Daily sleep hours

**Categorical:**
- `gender`: Student gender
- `course`: Course enrolled (B.Sc, B.Tech, BCA, etc.)
- `internet_access`: Yes/No
- `sleep_quality`: Good/Average/Poor
- `study_method`: Coaching/Self-study/Mixed/Group study/Online videos
- `facility_rating`: High/Medium/Low
- `exam_difficulty`: Easy/Moderate/Hard

## 📊 Main Notebook Contents

The `student_score_prediction.ipynb` notebook includes:

### 1. Setup and Data Loading
- Import necessary libraries
- Load train/test datasets
- Basic data inspection

### 2. Exploratory Data Analysis (EDA)
- Target distribution analysis
- Feature correlations
- Scatter plots for numerical features
- Box plots for categorical features
- Comprehensive visualizations

### 3. Feature Engineering
- **Interaction features**: `study_hours * class_attendance`, etc.
- **Ratio features**: `attendance_per_hour`, `sleep_per_study`
- **Polynomial features**: Squared and cubed terms
- **Binary flags**: `is_low_sleep`, `is_high_attendance`
- **Domain formula**: From Kaggle discussion insights
- **Target encoding**: CV-safe implementation
- **Frequency encoding**: Category frequency features

### 4. Model Development

#### Baseline Models
- Ridge Regression (with standardization)
- Random Forest Regressor

#### Advanced Models (Gradient Boosting)
- **LightGBM**: Primary model with feature importance
- **XGBoost**: Alternative boosting model
- **CatBoost**: Native categorical handling

All models use 5-fold cross-validation for robust evaluation.

### 5. Ensemble Methods
- **Simple Average**: Equal weights for top 3 models
- **Weighted Ensemble**: Optimized weights using scipy

### 6. Results Summary
- Comparison table of all models
- Visualization of model performance
- Residual analysis
- Feature importance plots

### 7. Final Submission
- Best model selection
- Prediction clipping
- CSV submission file generation

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost catboost scipy
```

### Running the Notebook

1. **Ensure data files are in the `data/` directory:**
   - `train.csv`
   - `test.csv`
   - `sample_submission.csv`

2. **Open the notebook:**
   ```bash
   jupyter notebook student_score_prediction.ipynb
   ```

3. **Run all cells sequentially** (or use "Run All")

4. **Output:**
   - `submission.csv`: Final predictions for Kaggle submission
   - Multiple visualizations and analysis throughout

## 📈 Expected Results

Based on cross-validation, expected RMSE scores:

| Model | Expected CV RMSE |
|-------|------------------|
| Ridge Regression | ~10-11 |
| Random Forest | ~9.5-10 |
| LightGBM | ~8.75-8.80 |
| XGBoost | ~8.75-8.85 |
| CatBoost | ~8.70-8.80 |
| **Ensemble** | **~8.65-8.70** |

## 🔑 Key Findings

### Most Important Features
1. `study_hours` (highest correlation: 0.76)
2. `study_hours * class_attendance` (interaction)
3. `class_attendance`
4. Target-encoded categorical features
5. Formula-based composite feature

### Model Insights
- Gradient boosting models significantly outperform baseline approaches
- Feature engineering provides substantial performance gains
- Ensemble methods offer marginal but consistent improvements
- Target encoding is highly effective for categorical features

## 📝 For University Report

The notebook is structured to support a comprehensive report:

1. **Introduction**: Problem statement and dataset overview
2. **EDA**: Comprehensive analysis with visualizations
3. **Methodology**: Feature engineering and model selection
4. **Results**: Cross-validation scores and comparisons
5. **Analysis**: Feature importance and residual analysis
6. **Conclusion**: Key findings and recommendations

## 🤝 Team Collaboration

The project can be divided into 3 parts:

- **Person 1**: EDA, visualizations, and interpretation
- **Person 2**: Feature engineering and baseline models
- **Person 3**: Gradient boosting, tuning, and ensembling

## 📚 References

- Kaggle Competition: [Playground Series S6E1](https://www.kaggle.com/competitions/playground-series-s6e1)
- Reference notebooks in `kaggle_notebooks/` directory
- Competition discussions for domain insights

## 🎓 Next Steps

1. **Hyperparameter tuning**: Further optimize model parameters
2. **Advanced ensembling**: Implement stacking with meta-learners
3. **Neural networks**: Experiment with TabNet or deep learning
4. **Original dataset**: Incorporate original data for training
5. **Feature selection**: Identify and remove redundant features

## 📄 License

This is a university project for educational purposes.

---

**Good luck with your competition and presentation! 🎉**

