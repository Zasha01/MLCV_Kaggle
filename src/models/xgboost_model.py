"""XGBoost model."""
import numpy as np
import xgboost as xgb
from .base import BaseModel
from ..config import XGBOOST_PARAMS


class XGBoostModel(BaseModel):
    """XGBoost Regressor."""
    
    def __init__(self, params=None):
        super().__init__(name="XGBoost")
        self.params = params if params is not None else XGBOOST_PARAMS.copy()
        self.model = None
        
    def fit(self, X, y, X_val=None, y_val=None):
        """Fit XGBoost model."""
        # Create a copy of params to avoid modifying the original
        params = self.params.copy()
        
        # Extract early_stopping_rounds (it's a fit parameter, not constructor parameter)
        early_stopping = params.pop('early_stopping_rounds', None)
        
        if X_val is not None and y_val is not None:
            # Use validation set with early stopping
            self.model = xgb.XGBRegressor(**params)
            if early_stopping is not None:
                self.model.fit(
                    X, y,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                self.model.fit(
                    X, y,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
        else:
            # No validation set - train without early stopping
            # Cap n_estimators to a reasonable value when no early stopping
            if 'n_estimators' in params and params['n_estimators'] > 1000:
                params['n_estimators'] = 1000
            self.model = xgb.XGBRegressor(**params)
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


