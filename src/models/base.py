"""Base model class."""
from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, name="BaseModel"):
        self.name = name
        self.model = None
        
    @abstractmethod
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit the model.
        
        Args:
            X: Training features
            y: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Predictions array
        """
        pass
    
    def get_feature_importance(self):
        """
        Get feature importance if available.
        
        Returns:
            Feature importance array or None
        """
        return None
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


