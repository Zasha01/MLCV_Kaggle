"""CatBoost model."""
import numpy as np
from catboost import CatBoostRegressor
from .base import BaseModel
from ..config import CATBOOST_PARAMS


class CatBoostModel(BaseModel):
    """CatBoost Regressor."""
    
    def __init__(self, params=None):
        super().__init__(name="CatBoost")
        self.params = params if params is not None else CATBOOST_PARAMS.copy()
        self.model = None
        
    def fit(self, X, y, X_val=None, y_val=None):
        """Fit CatBoost model."""
        # Create a copy of params to avoid modifying the original
        params = self.params.copy()
        
        # Extract early_stopping_rounds (it's a fit parameter for CatBoost)
        early_stopping = params.pop('early_stopping_rounds', None)
        
        if X_val is not None and y_val is not None:
            # Use validation set with early stopping
            if early_stopping is not None:
                params['early_stopping_rounds'] = early_stopping
            self.model = CatBoostRegressor(**params)
            self.model.fit(
                X, y,
                eval_set=(X_val, y_val),
                verbose=False
            )
        else:
            # No validation set - train without early stopping
            # Cap iterations to a reasonable value when no early stopping
            if 'iterations' in params and params['iterations'] > 1000:
                params['iterations'] = 1000
            self.model = CatBoostRegressor(**params)
            self.model.fit(X, y, verbose=False)
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Get feature importance."""
        if self.model is not None:
            return self.model.feature_importances_
        return None


