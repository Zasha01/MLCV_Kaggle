#!/usr/bin/env python3
"""Hyperparameter tuning with Optuna."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DIR
from src.data import load_data, prepare_datasets
from src.training import tune_all_models
from src.evaluation import save_results


def main(n_trials=50):
    """Run hyperparameter tuning."""
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING WITH OPTUNA")
    print("="*70)
    
    # Load and prepare data
    print("\nLoading and preparing data...")
    train, test, _ = load_data()
    X_train, y_train, X_test, test_ids, train_ids = prepare_datasets(train, test)
    
    # Tune all models
    best_params = tune_all_models(X_train, y_train, n_trials=n_trials)
    
    # Save best parameters
    save_results(best_params, RESULTS_DIR / "best_params.json", format='json')
    
    print("\n" + "="*70)
    print("TUNING COMPLETE!")
    print(f"Best parameters saved to: {RESULTS_DIR / 'best_params.json'}")
    print("\nTo use these parameters, update src/config.py or pass them to model constructors.")
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter tuning")
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of optimization trials (default: 50)')
    
    args = parser.parse_args()
    main(n_trials=args.n_trials)


