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
        self.model = xgb.XGBRegressor(**self.params)
        
        if X_val is not None and y_val is not None:
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
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

