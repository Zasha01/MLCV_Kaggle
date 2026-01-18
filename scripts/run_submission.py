#!/usr/bin/env python3
"""Generate final submission file."""
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DIR, ROOT_DIR


def main():
    """Generate submission file from saved predictions."""
    print("\n" + "="*70)
    print("GENERATING SUBMISSION FILE")
    print("="*70)
    
    # Load test predictions
    test_pred_path = RESULTS_DIR / "test_predictions.csv"
    
    if not test_pred_path.exists():
        print(f"\nError: Test predictions not found at {test_pred_path}")
        print("Please run 'python scripts/run_training.py' first.")
        return
    
    test_predictions = pd.read_csv(test_pred_path)
    
    # Load sample submission
    sample_submission = pd.read_csv(ROOT_DIR / "data" / "sample_submission.csv")
    
    # Create submission with best model predictions
    submission = sample_submission.copy()
    submission['exam_score'] = test_predictions['best_model'].values
    
    # Clip predictions to valid range
    submission['exam_score'] = submission['exam_score'].clip(0, 100)
    
    # Save submission
    submission_path = ROOT_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)
    
    print(f"\nSubmission file created: {submission_path}")
    print(f"Number of predictions: {len(submission)}")
    print(f"Prediction range: [{submission['exam_score'].min():.2f}, {submission['exam_score'].max():.2f}]")
    print(f"Mean prediction: {submission['exam_score'].mean():.2f}")
    
    print("\n" + "="*70)
    print("SUBMISSION READY!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

