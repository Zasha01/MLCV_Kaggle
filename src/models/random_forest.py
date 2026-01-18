"""Random Forest model."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from .base import BaseModel
from ..config import RANDOM_FOREST_PARAMS


class RandomForestModel(BaseModel):
    """Random Forest Regressor."""
    
    def __init__(self, params=None):
        super().__init__(name="RandomForest")
        self.params = params if params is not None else RANDOM_FOREST_PARAMS.copy()
        self.model = None
        
    def fit(self, X, y, X_val=None, y_val=None):
        """Fit Random Forest model."""
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Get feature importance."""
        if self.model is not None:
            return self.model.feature_importances_
        return None

