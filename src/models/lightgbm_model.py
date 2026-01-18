"""LightGBM model."""
import numpy as np
import lightgbm as lgb
from .base import BaseModel
from ..config import LIGHTGBM_PARAMS


class LightGBMModel(BaseModel):
    """LightGBM Regressor."""
    
    def __init__(self, params=None):
        super().__init__(name="LightGBM")
        self.params = params if params is not None else LIGHTGBM_PARAMS.copy()
        self.model = None
        self.best_iteration = None
        
    def fit(self, X, y, X_val=None, y_val=None):
        """Fit LightGBM model."""
        train_data = lgb.Dataset(X, label=y)
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val)
            self.model = lgb.train(
                self.params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=10000,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
            self.best_iteration = self.model.best_iteration
        else:
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=1000
            )
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.best_iteration is not None:
            return self.model.predict(X, num_iteration=self.best_iteration)
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Get feature importance."""
        if self.model is not None:
            return self.model.feature_importance(importance_type='gain')
        return None


