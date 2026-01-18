"""Ridge Regression model."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from .base import BaseModel
from ..config import RIDGE_PARAMS


class RidgeModel(BaseModel):
    """Ridge Regression with standardization."""
    
    def __init__(self, params=None):
        super().__init__(name="Ridge")
        self.params = params if params is not None else RIDGE_PARAMS.copy()
        self.scaler = StandardScaler()
        self.model = None
        
    def fit(self, X, y, X_val=None, y_val=None):
        """Fit Ridge model with standardized features."""
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = Ridge(**self.params)
        self.model.fit(X_scaled, y)
        
        return self
    
    def predict(self, X):
        """Predict with standardized features."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

