#!/usr/bin/env python3
"""
Train SE-ResNet neural network with entity embeddings and attention.

This script trains a deep learning model for tabular data using:
- Entity embeddings for categorical features
- Residual connections for deeper networks
- Squeeze-and-Excitation (SE) attention blocks
- Data augmentation with the original dataset

Usage:
    python scripts/run_senet.py                    # Train with default params
    python scripts/run_senet.py --augment          # Use data augmentation
    python scripts/run_senet.py --epochs 500       # Custom training epochs
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import FIGURES_DIR, RESULTS_DIR, SEED, N_FOLDS, TARGET_COL
from src.data import load_data
from src.models.senet import TabularResNetWithEmbedding
from src.evaluation import calculate_rmse, save_results
from src.visualization import setup_plotting, plot_residuals


def add_engineered_features(df):
    """Add engineered features for neural network."""
    df = df.copy()
    
    # Sine features for cyclical patterns
    df['study_hours_sin'] = np.sin(2 * np.pi * df['study_hours'] / 12).astype('float32')
    df['class_attendance_sin'] = np.sin(2 * np.pi * df['class_attendance'] / 100).astype('float32')
    
    # Log transforms
    num_features = ['study_hours', 'class_attendance', 'sleep_hours']
    for col in num_features:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])
            df[f'{col}_sq'] = df[col] ** 2
    
    # Frequency encoding for categorical features
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        cat_series = df[col].astype(str)
        freq_map = cat_series.value_counts().to_dict()
        df[f"{col}_freq"] = cat_series.map(freq_map).fillna(0).astype(int)
    
    # Domain formula
    df['feature_formula'] = (
        5.9051 * df['study_hours'] +
        0.3454 * df['class_attendance'] +
        1.4235 * df['sleep_hours'] +
        4.7819
    )
    
    return df


def prepare_data_for_nn(train_df, test_df, original_df=None):
    """Prepare data for neural network training."""
    # Add engineered features
    train_eng = add_engineered_features(train_df)
    test_eng = add_engineered_features(test_df)
    
    if original_df is not None:
        original_eng = add_engineered_features(original_df)
    else:
        original_eng = None
    
    # Separate features and target
    y_train = train_eng[TARGET_COL].values
    train_ids = train_eng['id'].values
    test_ids = test_eng['id'].values
    
    # Remove target and id
    X_train = train_eng.drop(columns=[TARGET_COL, 'id'], errors='ignore')
    X_test = test_eng.drop(columns=[TARGET_COL, 'id'], errors='ignore')
    
    if original_eng is not None:
        y_original = original_eng[TARGET_COL].values
        X_original = original_eng.drop(columns=[TARGET_COL, 'id'], errors='ignore')
    else:
        y_original = None
        X_original = None
    
    # Identify numerical and categorical columns
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Convert categorical to string
    for col in cat_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)
        if X_original is not None:
            X_original[col] = X_original[col].astype(str)
    
    return X_train, y_train, X_test, X_original, y_original, train_ids, test_ids, num_cols, cat_cols


def train_fold(model, X_num_train, X_cat_train, y_train, X_num_val, X_cat_val, y_val, 
               device, params):
    """Train model for one fold."""
    # Convert to tensors
    X_num_train_t = torch.tensor(X_num_train, dtype=torch.float32)
    X_cat_train_t = torch.tensor(X_cat_train, dtype=torch.int64)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    
    X_num_val_t = torch.tensor(X_num_val, dtype=torch.float32)
    X_cat_val_t = torch.tensor(X_cat_val, dtype=torch.int64)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    
    # Create data loaders
    from torch.utils.data import TensorDataset, DataLoader
    train_ds = TensorDataset(X_num_train_t, X_cat_train_t, y_train_t)
    val_ds = TensorDataset(X_num_val_t, X_cat_val_t, y_val_t)
    train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)
    
    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=params['factor'], 
        patience=params['patience'] // 2, min_lr=params['min_lr']
    )
    criterion = torch.nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = None
    
    for epoch in range(params['epochs']):
        # Train
        model.train()
        for xb_num, xb_cat, yb in train_loader:
            xb_num, xb_cat, yb = xb_num.to(device), xb_cat.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb_num, xb_cat)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb_num, xb_cat, yb in val_loader:
                xb_num, xb_cat, yb = xb_num.to(device), xb_cat.to(device), yb.to(device)
                pred = model(xb_num, xb_cat)
                loss = criterion(pred, yb)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_rmse = val_loss ** 0.5
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_rmse = val_rmse
            patience_counter = 0
            best_weights = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= params['patience']:
                break
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch + 1}/{params['epochs']} | Val RMSE: {val_rmse:.5f}")
    
    # Load best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)
    
    return model, best_val_rmse


def main(augment=True, epochs=300, batch_size=256, lr=1e-3):
    """Run SE-ResNet training pipeline."""
    setup_plotting()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    print("\n" + "="*70)
    print("SE-RESNET NEURAL NETWORK TRAINING")
    print("="*70)
    
    # Load data
    print("\nLoading and preparing data...")
    train_df, test_df, _ = load_data()
    
    # Load original dataset for augmentation
    original_df = None
    if augment:
        original_path = Path(__file__).parent.parent / "data" / "original.csv"
        if original_path.exists():
            original_df = pd.read_csv(original_path)
            print(f"  Loaded original dataset for augmentation: {len(original_df)} samples")
        else:
            print(f"  Warning: Original dataset not found at {original_path}")
            print("  Continuing without data augmentation...")
    
    # Prepare data
    X_train, y_train, X_test, X_original, y_original, train_ids, test_ids, num_cols, cat_cols = \
        prepare_data_for_nn(train_df, test_df, original_df)
    
    print(f"\nData prepared:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Numerical features: {len(num_cols)}")
    print(f"  Categorical features: {len(cat_cols)}")
    if X_original is not None:
        print(f"  Augmentation samples: {len(X_original)}")
    
    # Fit scalers on full training data
    scaler = StandardScaler()
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    scaler.fit(X_train[num_cols])
    encoder.fit(X_train[cat_cols])
    
    # Get categorical cardinalities
    cat_unique_counts = [int(cat.size) for cat in encoder.categories_]
    print(f"\nCategorical cardinalities: {cat_unique_counts}")
    
    # Cross-validation
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    oof_predictions = np.zeros(len(y_train))
    test_predictions = []
    fold_scores = []
    
    params = {
        'batch_size': batch_size,
        'lr': lr,
        'weight_decay': 1e-4,
        'epochs': epochs,
        'patience': 20,
        'factor': 0.5,
        'min_lr': 1e-6
    }
    
    print(f"\nStarting {N_FOLDS}-fold cross-validation...")
    print(f"Training parameters: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        print(f"\n{'='*70}")
        print(f"Fold {fold}/{N_FOLDS}")
        print(f"{'='*70}")
        
        # Split data
        X_train_fold = X_train.iloc[train_idx]
        y_train_fold = y_train[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_val_fold = y_train[val_idx]
        
        # Process numerical features
        X_num_train = scaler.transform(X_train_fold[num_cols])
        X_num_val = scaler.transform(X_val_fold[num_cols])
        
        # Process categorical features
        X_cat_train = encoder.transform(X_train_fold[cat_cols]).astype(np.int64)
        X_cat_val = encoder.transform(X_val_fold[cat_cols]).astype(np.int64)
        
        # Augment with original data
        if X_original is not None:
            X_num_orig = scaler.transform(X_original[num_cols])
            X_cat_orig = encoder.transform(X_original[cat_cols]).astype(np.int64)
            
            X_num_train = np.vstack([X_num_train, X_num_orig])
            X_cat_train = np.vstack([X_cat_train, X_cat_orig])
            y_train_fold = np.concatenate([y_train_fold, y_original])
            
            print(f"  Augmented training set: {len(y_train_fold)} samples")
        
        # Create model
        model = TabularResNetWithEmbedding(
            num_numerical=len(num_cols),
            cat_unique_counts=cat_unique_counts,
            embedding_dim=8,
            hidden_dim=256,
            n_blocks=3,
            dropout=0.11,
            head_dims=[64, 16]
        ).to(device)
        
        # Train
        model, fold_rmse = train_fold(
            model, X_num_train, X_cat_train, y_train_fold,
            X_num_val, X_cat_val, y_val_fold,
            device, params
        )
        
        # Predict
        model.eval()
        with torch.no_grad():
            # OOF predictions
            X_num_val_t = torch.tensor(X_num_val, dtype=torch.float32).to(device)
            X_cat_val_t = torch.tensor(X_cat_val, dtype=torch.int64).to(device)
            val_pred = model(X_num_val_t, X_cat_val_t).cpu().numpy()
            oof_predictions[val_idx] = val_pred
            
            # Test predictions
            X_num_test = scaler.transform(X_test[num_cols])
            X_cat_test = encoder.transform(X_test[cat_cols]).astype(np.int64)
            X_num_test_t = torch.tensor(X_num_test, dtype=torch.float32).to(device)
            X_cat_test_t = torch.tensor(X_cat_test, dtype=torch.int64).to(device)
            test_pred = model(X_num_test_t, X_cat_test_t).cpu().numpy()
            test_predictions.append(test_pred)
        
        fold_scores.append(fold_rmse)
        print(f"\n  Fold {fold} Best RMSE: {fold_rmse:.5f}")
    
    # Calculate OOF RMSE
    oof_rmse = calculate_rmse(y_train, oof_predictions)
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"OOF RMSE: {oof_rmse:.5f}")
    print(f"Mean Fold RMSE: {np.mean(fold_scores):.5f} (+/- {np.std(fold_scores):.5f})")
    print(f"{'='*70}\n")
    
    # Average test predictions
    final_test_pred = np.mean(test_predictions, axis=0)
    
    # Save results
    print("Saving results...")
    
    # OOF predictions
    oof_df = pd.DataFrame({
        'id': train_ids,
        'actual': y_train,
        'predicted': oof_predictions,
        'residual': y_train - oof_predictions
    })
    save_results(oof_df, RESULTS_DIR / "senet_oof_predictions.csv", format='csv')
    
    # Test predictions
    test_df_pred = pd.DataFrame({
        'id': test_ids,
        TARGET_COL: final_test_pred
    })
    save_results(test_df_pred, RESULTS_DIR / "senet_submission.csv", format='csv')
    
    # Stats
    stats = {
        'oof_rmse': float(oof_rmse),
        'mean_fold_rmse': float(np.mean(fold_scores)),
        'std_fold_rmse': float(np.std(fold_scores)),
        'fold_scores': [float(s) for s in fold_scores],
        'augmented': augment,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr
    }
    save_results(stats, RESULTS_DIR / "senet_stats.json", format='json')
    
    # Plot residuals
    plot_residuals(y_train, oof_predictions, "SE-ResNet", FIGURES_DIR / "senet_residuals.png")
    
    print(f"\nResults saved:")
    print(f"  OOF predictions: {RESULTS_DIR / 'senet_oof_predictions.csv'}")
    print(f"  Submission: {RESULTS_DIR / 'senet_submission.csv'}")
    print(f"  Statistics: {RESULTS_DIR / 'senet_stats.json'}")
    print(f"  Residual plot: {FIGURES_DIR / 'senet_residuals.png'}")
    
    print("\n" + "="*70)
    print("SE-RESNET TRAINING COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SE-ResNet neural network")
    parser.add_argument('--no-augment', action='store_true', help='Disable data augmentation')
    parser.add_argument('--epochs', type=int, default=300, help='Training epochs (default: 300)')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size (default: 256)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    
    args = parser.parse_args()
    
    main(
        augment=not args.no_augment,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

