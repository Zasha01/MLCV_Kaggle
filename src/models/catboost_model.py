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
        self.model = CatBoostRegressor(**self.params)
        
        if X_val is not None and y_val is not None:
            self.model.fit(
                X, y,
                eval_set=(X_val, y_val),
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


