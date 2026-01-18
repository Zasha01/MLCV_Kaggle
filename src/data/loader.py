"""Data loading utilities."""
import pandas as pd
from pathlib import Path
from ..config import DATA_DIR


def load_data():
    """
    Load training, test, and submission template data.
    
    Returns:
        tuple: (train_df, test_df, sample_submission_df)
    """
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    submission_path = DATA_DIR / "sample_submission.csv"
    
    print(f"Loading data from {DATA_DIR}")
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_submission = pd.read_csv(submission_path)
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Sample submission shape: {sample_submission.shape}")
    
    # Basic data validation
    print(f"\nMissing values in train: {train.isnull().sum().sum()}")
    print(f"Missing values in test: {test.isnull().sum().sum()}")
    
    return train, test, sample_submission


def get_data_summary(train):
    """
    Print summary statistics of the training data.
    
    Args:
        train: Training dataframe
    """
    print("\n" + "="*70)
    print("DATA SUMMARY")
    print("="*70)
    
    print(f"\nDataset shape: {train.shape}")
    print(f"\nColumn types:\n{train.dtypes.value_counts()}")
    
    if 'exam_score' in train.columns:
        print(f"\nTarget (exam_score) statistics:")
        print(train['exam_score'].describe())
    
    print("\n" + "="*70)


