"""Ensemble model for combining predictions."""
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from .base import BaseModel
from ..evaluation.metrics import calculate_rmse
from ..config import SEED


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


class StackingModel(BaseModel):
    """Stacking ensemble with meta-learner trained on out-of-fold predictions."""
    
    def __init__(self, base_model_configs, meta_model_class, meta_params=None, n_folds=5):
        """
        Initialize stacking ensemble.
        
        Args:
            base_model_configs: List of (model_class, params) tuples for base models
            meta_model_class: Meta-learner class (e.g., RidgeModel, ElasticNetModel)
            meta_params: Parameters for meta-learner (optional)
            n_folds: Number of CV folds for generating OOF predictions
        """
        super().__init__(name="StackingModel")
        self.base_model_configs = base_model_configs
        self.meta_model_class = meta_model_class
        self.meta_params = meta_params
        self.n_folds = n_folds
        self.base_models = []  # Will store trained base models (one per base model type)
        self.meta_model = None
        
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit stacking model.
        
        Step 1: Generate OOF predictions for each base model using CV
        Step 2: Train each base model on full training data
        Step 3: Train meta-model on stacked OOF predictions
        
        Args:
            X: Training features
            y: Training target
            X_val: Validation features (optional, not used in stacking)
            y_val: Validation target (optional, not used in stacking)
        """
        print(f"\n{'='*70}")
        print("STACKING MODEL - TRAINING")
        print(f"{'='*70}")
        print(f"Base models: {len(self.base_model_configs)}")
        print(f"Meta-model: {self.meta_model_class.__name__}")
        print(f"CV folds: {self.n_folds}")
        print(f"{'='*70}\n")
        
        # Step 1: Generate OOF predictions for each base model
        oof_predictions_list = []
        
        for idx, (model_class, params) in enumerate(self.base_model_configs, 1):
            print(f"Base Model {idx}/{len(self.base_model_configs)}: {model_class.__name__}")
            
            # Generate OOF predictions via CV
            oof_preds = self._generate_oof_predictions(model_class, X, y, params)
            oof_predictions_list.append(oof_preds)
            
            # Calculate OOF score for this base model
            oof_rmse = calculate_rmse(y, oof_preds)
            print(f"  OOF RMSE: {oof_rmse:.5f}\n")
        
        # Step 2: Train each base model on full training data (for test predictions)
        print("Training base models on full training data...")
        for model_class, params in self.base_model_configs:
            model = model_class(params=params)
            model.fit(X, y)
            self.base_models.append(model)
            print(f"  {model_class.__name__} - trained")
        
        # Step 3: Stack OOF predictions as meta-features
        X_meta = np.column_stack(oof_predictions_list)
        print(f"\nMeta-features shape: {X_meta.shape}")
        
        # Step 4: Train meta-model on stacked OOF predictions
        print(f"Training meta-model: {self.meta_model_class.__name__}...")
        self.meta_model = self.meta_model_class(params=self.meta_params)
        self.meta_model.fit(X_meta, y)
        
        # Calculate stacking OOF predictions (meta-model predictions on OOF)
        stacking_oof = self.meta_model.predict(X_meta)
        stacking_rmse = calculate_rmse(y, stacking_oof)
        
        print(f"{'='*70}")
        print(f"Stacking Model OOF RMSE: {stacking_rmse:.5f}")
        print(f"{'='*70}\n")
        
        return self
    
    def _generate_oof_predictions(self, model_class, X, y, params):
        """
        Generate out-of-fold predictions for a base model.
        
        Args:
            model_class: Model class to instantiate
            X: Training features
            y: Training target
            params: Model parameters
            
        Returns:
            Out-of-fold predictions array
        """
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=SEED)
        oof_predictions = np.zeros(len(X))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            print(f"  Fold {fold}/{self.n_folds}...", end='')
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Initialize and train model
            model = model_class(params=params)
            model.fit(X_train, y_train, X_val, y_val)
            
            # Store OOF predictions
            oof_predictions[val_idx] = model.predict(X_val)
            
            fold_score = calculate_rmse(y_val, oof_predictions[val_idx])
            print(f" RMSE: {fold_score:.5f}")
        
        return oof_predictions
    
    def predict(self, X):
        """
        Make predictions with stacking model.
        
        Step 1: Get predictions from all base models
        Step 2: Stack as meta-features
        Step 3: Predict with meta-model
        
        Args:
            X: Features
            
        Returns:
            Stacking predictions
        """
        # Get predictions from all base models
        base_predictions = np.column_stack([
            model.predict(X) for model in self.base_models
        ])
        
        # Predict with meta-model
        stacking_pred = self.meta_model.predict(base_predictions)
        
        return stacking_pred
    
    def get_base_model_names(self):
        """Get names of base models."""
        return [model.name for model in self.base_models]
    
    def get_meta_model_name(self):
        """Get name of meta-model."""
        return self.meta_model.name if self.meta_model else None


