"""SE-ResNet model for tabular data."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from .base import BaseModel
from ..config import SEED


# ==============================
# SE Block (Squeeze-and-Excitation)
# ==============================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation attention block."""
    
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, channels)
        se = x.mean(dim=0, keepdim=True)  # global avg pool -> (1, channels)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        return x * se  # broadcast


# ==============================
# Residual Block with SE
# ==============================
class ResidualBlock(nn.Module):
    """Residual block with Squeeze-and-Excitation."""
    
    def __init__(self, dim, dropout=0.1, reduction=4):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.se = SEBlock(dim, reduction=reduction)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        # First sub-block
        out = self.norm1(x)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.dropout(out)
        # Second sub-block
        out = self.norm2(out)
        out = self.linear2(out)
        out = self.dropout(out)
        # SE
        out = self.se(out)
        # Residual connection
        out = out + residual
        return out


# ==============================
# Complete Model: Embedding + Concat + ResNet + Head
# ==============================
class TabularResNetWithEmbedding(nn.Module):
    """Tabular ResNet with entity embeddings and SE blocks."""
    
    def __init__(
            self,
            num_numerical,
            cat_unique_counts,
            embedding_dim=8,
            hidden_dim=256,
            n_blocks=3,
            dropout=0.11,
            head_dims=[64, 16]
    ):
        super().__init__()
        self.num_numerical = num_numerical
        self.embedding_dim = embedding_dim

        # Embedding layers for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_cat + 1, embedding_dim, padding_idx=-1)
            for n_cat in cat_unique_counts
        ])

        total_cat_dim = len(cat_unique_counts) * embedding_dim
        input_dim = num_numerical + total_cat_dim

        # Projection to hidden_dim
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.dropout_in = nn.Dropout(dropout)

        # Residual blocks
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)]
        )

        # Prediction head
        layers = []
        prev = hidden_dim
        for h in head_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        """Forward pass."""
        batch_size = x_num.size(0)

        # Embed categorical features
        x_embeds = []
        for i, emb in enumerate(self.embeddings):
            xi = x_cat[:, i]
            # Handle -1 (unknown): map to last embedding index
            xi = torch.where(xi == -1, torch.tensor(emb.num_embeddings - 1, device=xi.device), xi)
            embed_i = emb(xi)  # (B, embedding_dim)
            x_embeds.append(embed_i)

        x_cat_emb = torch.cat(x_embeds, dim=1)  # (B, total_cat_dim)

        # Concat numerical and embedded categorical
        x = torch.cat([x_num, x_cat_emb], dim=1)  # (B, input_dim)

        # Project to hidden space
        x = self.proj(x)
        x = self.dropout_in(x)

        # Residual blocks
        x = self.blocks(x)

        # Prediction head
        out = self.head(x).squeeze(1)
        return out


# ==============================
# Wrapper Model Class
# ==============================
class SENetModel(BaseModel):
    """
    SE-ResNet wrapper compatible with the existing pipeline.
    
    Note: This model requires different preprocessing (separate numerical/categorical).
    """
    
    def __init__(self, params=None):
        super().__init__(name="SENet")
        
        # Default parameters
        self.params = {
            'embedding_dim': 8,
            'hidden_dim': 256,
            'n_blocks': 3,
            'dropout': 0.11,
            'head_dims': [64, 16],
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'batch_size': 256,
            'epochs': 300,
            'patience': 20,
            'factor': 0.5,
            'min_lr': 1e-6
        }
        
        if params:
            self.params.update(params)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = None
        self.encoder = None
        self.numerical_cols = None
        self.categorical_cols = None
        
    def _prepare_data(self, X):
        """Separate numerical and categorical features."""
        # Infer numerical and categorical columns
        if self.numerical_cols is None:
            self.numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        X_num = X[self.numerical_cols].values
        X_cat = X[self.categorical_cols].values
        
        return X_num, X_cat
    
    def fit(self, X, y, X_val=None, y_val=None):
        """Fit SE-ResNet model."""
        # Prepare data
        X_num_train, X_cat_train = self._prepare_data(X)
        
        # Fit scalers and encoders
        self.scaler = StandardScaler()
        X_num_train = self.scaler.fit_transform(X_num_train)
        
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_cat_train = self.encoder.fit_transform(X_cat_train).astype(np.int64)
        
        # Get categorical cardinalities
        cat_unique_counts = [int(cat.size) for cat in self.encoder.categories_]
        
        # Create model
        self.model = TabularResNetWithEmbedding(
            num_numerical=X_num_train.shape[1],
            cat_unique_counts=cat_unique_counts,
            embedding_dim=self.params['embedding_dim'],
            hidden_dim=self.params['hidden_dim'],
            n_blocks=self.params['n_blocks'],
            dropout=self.params['dropout'],
            head_dims=self.params['head_dims']
        ).to(self.device)
        
        # Prepare validation data if provided
        if X_val is not None and y_val is not None:
            X_num_val, X_cat_val = self._prepare_data(X_val)
            X_num_val = self.scaler.transform(X_num_val)
            X_cat_val = self.encoder.transform(X_cat_val).astype(np.int64)
        else:
            # Use 20% of training as validation
            split_idx = int(len(X_num_train) * 0.8)
            X_num_val = X_num_train[split_idx:]
            X_cat_val = X_cat_train[split_idx:]
            y_val = y[split_idx:] if hasattr(y, '__getitem__') else y.iloc[split_idx:]
            
            X_num_train = X_num_train[:split_idx]
            X_cat_train = X_cat_train[:split_idx]
            y = y[:split_idx] if hasattr(y, '__getitem__') else y.iloc[:split_idx]
        
        # Convert to tensors
        X_num_train_t = torch.tensor(X_num_train, dtype=torch.float32)
        X_cat_train_t = torch.tensor(X_cat_train, dtype=torch.int64)
        y_train_t = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.float32)
        
        X_num_val_t = torch.tensor(X_num_val, dtype=torch.float32)
        X_cat_val_t = torch.tensor(X_cat_val, dtype=torch.int64)
        y_val_t = torch.tensor(y_val.values if hasattr(y_val, 'values') else y_val, dtype=torch.float32)
        
        # Create data loaders
        train_ds = TensorDataset(X_num_train_t, X_cat_train_t, y_train_t)
        val_ds = TensorDataset(X_num_val_t, X_cat_val_t, y_val_t)
        
        train_loader = DataLoader(train_ds, batch_size=self.params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)
        
        # Train model
        self._train_loop(train_loader, val_loader)
        
        return self
    
    def _train_loop(self, train_loader, val_loader):
        """Training loop with early stopping."""
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.params['lr'],
            weight_decay=self.params['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.params['factor'],
            patience=self.params['patience'] // 2,
            min_lr=self.params['min_lr']
        )
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        for epoch in range(self.params['epochs']):
            # Train
            self.model.train()
            train_loss = 0.0
            for xb_num, xb_cat, yb in train_loader:
                xb_num, xb_cat, yb = xb_num.to(self.device), xb_cat.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self.model(xb_num, xb_cat)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validate
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb_num, xb_cat, yb in val_loader:
                    xb_num, xb_cat, yb = xb_num.to(self.device), xb_cat.to(self.device), yb.to(self.device)
                    pred = self.model(xb_num, xb_cat)
                    loss = criterion(pred, yb)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_rmse = val_loss ** 0.5
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.params['patience']:
                    break
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{self.params['epochs']} | Val RMSE: {val_rmse:.5f}")
        
        # Load best weights
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted yet!")
        
        # Prepare data
        X_num, X_cat = self._prepare_data(X)
        X_num = self.scaler.transform(X_num)
        X_cat = self.encoder.transform(X_cat).astype(np.int64)
        
        # Convert to tensors
        X_num_t = torch.tensor(X_num, dtype=torch.float32)
        X_cat_t = torch.tensor(X_cat, dtype=torch.int64)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X_num_t.to(self.device), X_cat_t.to(self.device))
            pred = pred.cpu().numpy()
        
        return pred
    
    def get_feature_importance(self):
        """Neural networks don't have direct feature importance."""
        return None

