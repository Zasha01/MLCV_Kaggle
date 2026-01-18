"""Metrics and evaluation utilities."""
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error


def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        float: RMSE score
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def get_residual_stats(y_true, y_pred):
    """
    Calculate comprehensive residual statistics.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        dict: Dictionary containing residual statistics
    """
    residuals = y_true - y_pred
    
    stats = {
        'mean': float(np.mean(residuals)),
        'std': float(np.std(residuals)),
        'min': float(np.min(residuals)),
        'max': float(np.max(residuals)),
        'median': float(np.median(residuals)),
        'q25': float(np.percentile(residuals, 25)),
        'q75': float(np.percentile(residuals, 75)),
        'rmse': calculate_rmse(y_true, y_pred)
    }
    
    return stats


def save_results(results_dict, filepath, format='json'):
    """
    Save results to file.
    
    Args:
        results_dict: Dictionary or DataFrame to save
        filepath: Path to save file
        format: 'json' or 'csv'
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
    elif format == 'csv':
        if isinstance(results_dict, dict):
            df = pd.DataFrame([results_dict])
        else:
            df = results_dict
        df.to_csv(filepath, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Results saved to {filepath}")


def load_results(filepath, format='json'):
    """
    Load results from file.
    
    Args:
        filepath: Path to load from
        format: 'json' or 'csv'
        
    Returns:
        Dictionary or DataFrame
    """
    filepath = Path(filepath)
    
    if format == 'json':
        with open(filepath, 'r') as f:
            return json.load(f)
    elif format == 'csv':
        return pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")

