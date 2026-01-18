"""Feature engineering pipeline."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from ..config import (
    NUMERICAL_COLS, CATEGORICAL_COLS, TARGET_COL, SEED, N_FOLDS
)


def create_interaction_features(df):
    """
    Create interaction, ratio, polynomial, and derived features.
    
    Args:
        df: Input dataframe
        
    Returns:
        DataFrame with additional engineered features
    """
    df = df.copy()
    
    # Interaction features
    df['study_attendance'] = df['study_hours'] * df['class_attendance']
    df['study_sleep'] = df['study_hours'] * df['sleep_hours']
    df['attendance_sleep'] = df['class_attendance'] * df['sleep_hours']
    df['age_study'] = df['age'] * df['study_hours']
    
    # Ratio features
    df['attendance_per_hour'] = df['class_attendance'] / (df['study_hours'] + 0.001)
    df['sleep_per_study'] = df['sleep_hours'] / (df['study_hours'] + 0.001)
    
    # Polynomial features
    df['study_hours_sq'] = df['study_hours'] ** 2
    df['study_hours_cb'] = df['study_hours'] ** 3
    df['class_attendance_sq'] = df['class_attendance'] ** 2
    df['sleep_hours_sq'] = df['sleep_hours'] ** 2
    
    # Square root features
    df['study_hours_sqrt'] = np.sqrt(df['study_hours'])
    df['class_attendance_sqrt'] = np.sqrt(df['class_attendance'])
    
    # Binary flags
    df['is_low_sleep'] = (df['sleep_hours'] < 6).astype(int)
    df['is_high_attendance'] = (df['class_attendance'] > 80).astype(int)
    df['is_high_study'] = (df['study_hours'] > 6).astype(int)
    
    # Domain knowledge formula (from Kaggle discussion)
    df['formula_score'] = (
        6.0 * df['study_hours'] +
        0.35 * df['class_attendance'] +
        1.5 * df['sleep_hours']
    )
    
    # Binning features
    df['study_hours_bin'] = pd.cut(df['study_hours'], bins=5, labels=False)
    df['class_attendance_bin'] = pd.cut(df['class_attendance'], bins=5, labels=False)
    df['sleep_hours_bin'] = pd.cut(df['sleep_hours'], bins=5, labels=False)
    
    return df


def apply_label_encoding(train, test, categorical_cols):
    """
    Apply label encoding to categorical columns.
    
    Args:
        train: Training dataframe
        test: Test dataframe
        categorical_cols: List of categorical column names
        
    Returns:
        tuple: (train_encoded, test_encoded, encoders_dict)
    """
    train = train.copy()
    test = test.copy()
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        label_encoders[col] = le
        
        # Transform test set (handle unseen categories)
        test[col] = test[col].astype(str).map(
            {label: idx for idx, label in enumerate(le.classes_)}
        ).fillna(-1).astype(int)
    
    return train, test, label_encoders


def apply_target_encoding_cv(train, test, categorical_cols, target_col, n_folds=N_FOLDS):
    """
    Apply CV-safe target encoding to prevent leakage.
    
    Args:
        train: Training dataframe
        test: Test dataframe
        categorical_cols: List of categorical column names
        target_col: Name of target column
        n_folds: Number of folds for CV
        
    Returns:
        tuple: (train_encoded, test_encoded)
    """
    train_encoded = train.copy()
    test_encoded = test.copy()
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    for col in categorical_cols:
        # Initialize with global mean
        global_mean = train[target_col].mean()
        train_encoded[f'{col}_target_enc'] = global_mean
        
        # Encode train with CV
        for train_idx, val_idx in kf.split(train):
            target_mean = train.iloc[train_idx].groupby(col)[target_col].mean()
            train_encoded.loc[val_idx, f'{col}_target_enc'] = \
                train.iloc[val_idx][col].map(target_mean).fillna(global_mean)
        
        # Encode test using full train
        target_mean = train.groupby(col)[target_col].mean()
        test_encoded[f'{col}_target_enc'] = test[col].map(target_mean).fillna(global_mean)
    
    return train_encoded, test_encoded


def apply_frequency_encoding(train, test, categorical_cols):
    """
    Apply frequency encoding to categorical columns.
    
    Args:
        train: Training dataframe
        test: Test dataframe
        categorical_cols: List of categorical column names
        
    Returns:
        tuple: (train_encoded, test_encoded)
    """
    train = train.copy()
    test = test.copy()
    
    for col in categorical_cols:
        freq_map = train[col].value_counts(normalize=True).to_dict()
        train[f'{col}_freq'] = train[col].map(freq_map)
        test[f'{col}_freq'] = test[col].map(freq_map).fillna(0)
    
    return train, test


def prepare_datasets(train, test):
    """
    Full feature engineering pipeline.
    
    Args:
        train: Raw training dataframe
        test: Raw test dataframe
        
    Returns:
        tuple: (X_train, y_train, X_test, test_ids, train_ids)
    """
    print("\n" + "="*70)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*70)
    
    # Store IDs and target
    train_ids = train['id'].copy()
    test_ids = test['id'].copy()
    y_train = train[TARGET_COL].copy()
    
    # Apply interaction features
    print("\n1. Creating interaction features...")
    train_fe = create_interaction_features(train)
    test_fe = create_interaction_features(test)
    print(f"   Shape after interactions: {train_fe.shape}")
    
    # Label encoding
    print("\n2. Applying label encoding...")
    train_fe, test_fe, _ = apply_label_encoding(train_fe, test_fe, CATEGORICAL_COLS)
    print(f"   Label encoding completed")
    
    # Target encoding (CV-safe)
    print("\n3. Applying target encoding (CV-safe)...")
    train_fe, test_fe = apply_target_encoding_cv(train_fe, test_fe, CATEGORICAL_COLS, TARGET_COL)
    print(f"   Shape after target encoding: {train_fe.shape}")
    
    # Frequency encoding
    print("\n4. Applying frequency encoding...")
    train_fe, test_fe = apply_frequency_encoding(train_fe, test_fe, CATEGORICAL_COLS)
    print(f"   Final shape: {train_fe.shape}")
    
    # Prepare final datasets
    X_train = train_fe.drop(['id', TARGET_COL], axis=1)
    X_test = test_fe.drop(['id'], axis=1)
    
    # Ensure test has same columns as train
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    print(f"\n5. Final datasets:")
    print(f"   X_train: {X_train.shape}")
    print(f"   y_train: {y_train.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   Total features: {X_train.shape[1]}")
    print("="*70 + "\n")
    
    return X_train, y_train, X_test, test_ids, train_ids


