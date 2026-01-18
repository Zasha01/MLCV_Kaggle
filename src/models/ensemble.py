"""Ensemble model for combining predictions."""
import numpy as np
from scipy.optimize import minimize
from .base import BaseModel
from ..evaluation.metrics import calculate_rmse


class EnsembleModel(BaseModel):
    """Ensemble model using weighted averaging."""
    
    def __init__(self, models, mode='simple'):
        """
        Initialize ensemble.
        
        Args:
            models: List of fitted BaseModel instances
            mode: 'simple' for equal weights, 'optimized' for weight optimization
        """
        super().__init__(name="Ensemble")
        self.models = models
        self.mode = mode
        self.weights = None
        
    def optimize_weights(self, X, y):
        """
        Optimize ensemble weights on validation data.
        
        Args:
            X: Features
            y: Target
        """
        # Get predictions from each model
        predictions = np.array([model.predict(X) for model in self.models])
        
        def objective(weights):
            """Objective function to minimize RMSE."""
            weights = weights / weights.sum()  # Normalize
            ensemble_pred = (weights[:, None] * predictions).sum(axis=0)
            return calculate_rmse(y, ensemble_pred)
        
        # Initial equal weights
        initial_weights = np.ones(len(self.models)) / len(self.models)
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='Nelder-Mead',
            bounds=[(0, 1)] * len(self.models)
        )
        
        self.weights = result.x / result.x.sum()
        
        return self
    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit ensemble by optimizing weights if mode is 'optimized'.
        
        Args:
            X: Training features (for weight optimization)
            y: Training target (for weight optimization)
        """
        if self.mode == 'optimized' and self.weights is None:
            self.optimize_weights(X, y)
        elif self.mode == 'simple':
            self.weights = np.ones(len(self.models)) / len(self.models)
        
        return self
    
    def predict(self, X):
        """
        Make ensemble predictions.
        
        Args:
            X: Features
            
        Returns:
            Weighted average predictions
        """
        predictions = np.array([model.predict(X) for model in self.models])
        
        if self.weights is None:
            # Default to equal weights
            self.weights = np.ones(len(self.models)) / len(self.models)
        
        ensemble_pred = (self.weights[:, None] * predictions).sum(axis=0)
        return ensemble_pred
    
    def get_weights(self):
        """Get ensemble weights."""
        return dict(zip([m.name for m in self.models], self.weights))

