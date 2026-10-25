"""PyTorch LSTM and CNN-LSTM for RUL prediction."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.config import (
    BATCH_SIZE,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    GRAD_CLIP,
    LR,
    SEED,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LSTMModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class CNNLSTM(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.lstm = nn.LSTM(64, hidden, batch_first=True, dropout=0.0)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, F, T) for Conv1d
        c = self.conv(x.transpose(1, 2)).squeeze(-1)  # (B, 64)
        c = c.unsqueeze(1)  # (B, 1, 64)
        out, _ = self.lstm(c)
        return self.fc(out[:, -1, :]).squeeze(-1)


def _engine_split(
    units: np.ndarray, val_frac: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """Split by engine ID (Group-aware)."""
    unique = np.unique(units)
    rng = np.random.RandomState(SEED)
    rng.shuffle(unique)
    n_val = max(1, int(len(unique) * val_frac))
    val_engines = set(unique[:n_val])
    train_mask = np.array([u not in val_engines for u in units])
    return np.where(train_mask)[0], np.where(~train_mask)[0]


def train_deep(
    X: np.ndarray,
    y: np.ndarray,
    units: np.ndarray,
    model_type: str = "lstm",
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    patience: int = EARLY_STOP_PATIENCE,
) -> nn.Module:
    """
    Train LSTM or CNN-LSTM with engine-level validation split,
    early stopping, Adam, grad clipping, and LR scheduler.
    """
    set_seed()
    n_features = X.shape[-1]

    if model_type == "lstm":
        model = LSTMModel(n_features).to(DEVICE)
    elif model_type == "cnn_lstm":
        model = CNNLSTM(n_features).to(DEVICE)
    else:
        raise ValueError(f"Unknown model_type {model_type}")

    train_idx, val_idx = _engine_split(units)
    X_tr = torch.tensor(X[train_idx], dtype=torch.float32)
    y_tr = torch.tensor(y[train_idx], dtype=torch.float32)
    X_va = torch.tensor(X[val_idx], dtype=torch.float32)
    y_va = torch.tensor(y[val_idx], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_va, y_va), batch_size=batch_size, shuffle=False
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3
    )
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_idx)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * len(xb)
        val_loss /= len(val_idx)
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def predict_deep(model: nn.Module, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i : i + batch_size], dtype=torch.float32).to(DEVICE)
        preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds).astype(np.float32)
