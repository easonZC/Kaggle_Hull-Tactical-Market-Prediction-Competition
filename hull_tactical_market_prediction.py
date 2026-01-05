# train_and_save_artifacts.py

import os
import warnings
from typing import List, Tuple
import optuna
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostRegressor
from gplearn.genetic import SymbolicTransformer
from sklearn.base import BaseEstimator
import joblib
from pathlib import Path
from sklearn.utils.validation import check_X_y, check_array
warnings.filterwarnings("ignore")


# =============================================================================
# 0. Global config
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = os.getenv("HTMP_DATA_DIR", str(Path(__file__).resolve().parent))
ARTIFACT_DIR = os.getenv("HTMP_ARTIFACT_DIR", "./htmp_artifacts")
FINAL_USE_FULL_SEQ = int(os.getenv("HTMP_FINAL_USE_FULL_SEQ", "1"))  # 1=全量训练；0=保留85/15切分

# -----------------------------------------------------------------------------
# [REQUIRED] STRATEGY PARAMETERS
# The following parameters are critical for model performance and must be
# tuned or defined by the user. They are set to None to prevent accidental usage
# of suboptimal defaults.
# -----------------------------------------------------------------------------

LOOKBACK = None
SEQ_LEN = None

MIN_POSITION = 0.0
MAX_POSITION = 2.0
TRADING_DAYS_PER_YEAR = 252

TARGET_COL = "forward_returns"

EXCLUDE_COLS = [
    "date_id",
    "forward_returns",
    "risk_free_rate",
    "is_scored",
]

BASE_FEATURE_COLS: List[str] = []
D_FEATURE_COLS: List[str] = []
FINAL_FEATURE_COLS: List[str] = []

FEATURE_MEAN: np.ndarray | None = None
FEATURE_STD: np.ndarray | None = None

ENSEMBLE_W = None
ALPHA = None

# CatBoost feature selection
TOP_K_FEATURES = None

# Scheme B controls
MINE_FRAC = None
EVAL_ON_TAIL = None
STATS_ON_MINE = None

cat_model: CatBoostRegressor | None = None
gru_model: nn.Module | None = None
trf_model: nn.Module | None = None


# =============================================================================
# 1. Official metric: Adjusted Sharpe (HTMP)
# =============================================================================
def calculate_adjusted_sharpe(solution_df: pd.DataFrame,
                              positions: np.ndarray) -> float:
    """
    Official metric: given forward_returns / risk_free_rate / position, calculate Adjusted Sharpe
    """
    df = solution_df.copy()
    df["position"] = positions

    df["strategy_returns"] = (
        df["risk_free_rate"] * (1 - df["position"]) +
        df["position"] * df["forward_returns"]
    )

    strategy_excess = df["strategy_returns"] - df["risk_free_rate"]
    strategy_excess_cum = (1 + strategy_excess).prod()
    if strategy_excess_cum <= 0:
        return -1000.0

    strategy_mean_excess = strategy_excess_cum ** (1 / len(df)) - 1
    strategy_std = df["strategy_returns"].std()
    if strategy_std == 0:
        return 0.0

    sharpe = (
        strategy_mean_excess / strategy_std *
        np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    market_excess = df["forward_returns"] - df["risk_free_rate"]
    market_cum = (1 + market_excess).prod()
    if market_cum <= 0:
        market_mean_excess = -1.0
    else:
        market_mean_excess = market_cum ** (1 / len(df)) - 1

    market_std = df["forward_returns"].std()
    market_vol = float(market_std * np.sqrt(TRADING_DAYS_PER_YEAR) * 100)
    strategy_vol = float(strategy_std * np.sqrt(TRADING_DAYS_PER_YEAR) * 100)

    if market_vol > 0:
        vol_ratio = strategy_vol / market_vol
        excess_vol = max(0.0, vol_ratio - 1.2)
    else:
        excess_vol = 0.0
    vol_penalty = 1 + excess_vol

    return_gap = max(
        0.0,
        (market_mean_excess - strategy_mean_excess) * 100 * TRADING_DAYS_PER_YEAR,
    )
    return_penalty = 1 + (return_gap ** 2) / 100

    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    return float(min(adjusted_sharpe, 1_000_000))


def score_from_raw_returns(
    y_ret: np.ndarray,
    rf: np.ndarray,
    pred_ret: np.ndarray,
    alpha: float = 120.0,
) -> float:
    """
    Given predicted returns, map to positions and calculate HTMP.
    """
    pos = 2.0 / (1.0 + np.exp(-alpha * pred_ret))
    pos = np.clip(pos, MIN_POSITION, MAX_POSITION)

    df_eval = pd.DataFrame({
        "forward_returns": y_ret,
        "risk_free_rate": rf,
    })
    return calculate_adjusted_sharpe(df_eval, pos)


# =============================================================================
# 2. Feature Engineering (完全一致 with Kaggle version)
# =============================================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced feature engineering: log1p, D-lag/rolling, diff/pct, D-cross stats, z-score
    Use global BASE_FEATURE_COLS / D_FEATURE_COLS
    """
    global BASE_FEATURE_COLS, D_FEATURE_COLS

    base = df[BASE_FEATURE_COLS].astype(float)
    feat_blocks: list[pd.DataFrame] = []

    # Original values
    feat_blocks.append(base)

    # log1p transformation
    log1p_block = base.apply(lambda s: np.sign(s) * np.log1p(np.abs(s)))
    log1p_block.columns = [f"{c}_log1p" for c in base.columns]
    feat_blocks.append(log1p_block)

    # D series time window features
    if D_FEATURE_COLS:
        d_vals = df[D_FEATURE_COLS].astype(float)

        lag1 = d_vals.shift(1)
        lag1.columns = [f"{c}_lag1" for c in D_FEATURE_COLS]
        lag2 = d_vals.shift(2)
        lag2.columns = [f"{c}_lag2" for c in D_FEATURE_COLS]

        roll5_mean = d_vals.rolling(5, min_periods=1).mean()
        roll5_mean.columns = [f"{c}_roll5_mean" for c in D_FEATURE_COLS]
        roll10_mean = d_vals.rolling(10, min_periods=1).mean()
        roll10_mean.columns = [f"{c}_roll10_mean" for c in D_FEATURE_COLS]

        roll5_std = d_vals.rolling(5, min_periods=1).std()
        roll5_std.columns = [f"{c}_roll5_std" for c in D_FEATURE_COLS]
        roll10_std = d_vals.rolling(10, min_periods=1).std()
        roll10_std.columns = [f"{c}_roll10_std" for c in D_FEATURE_COLS]

        feat_blocks += [lag1, lag2, roll5_mean, roll10_mean, roll5_std, roll10_std]

    # First-order difference & relative change of all base columns
    base_vals = df[BASE_FEATURE_COLS].astype(float)
    diff1 = base_vals.diff(1)
    diff1.columns = [f"{c}_diff1" for c in BASE_FEATURE_COLS]
    pct1 = base_vals.pct_change(1).replace([np.inf, -np.inf], 0.0)
    pct1.columns = [f"{c}_pct1" for c in BASE_FEATURE_COLS]
    feat_blocks += [diff1, pct1]

    # D series cross-sectional statistics + z-score
    if D_FEATURE_COLS:
        d_vals = df[D_FEATURE_COLS].astype(float)
        d_mean = d_vals.mean(axis=1)
        d_std = d_vals.std(axis=1)
        d_max = d_vals.max(axis=1)
        d_min = d_vals.min(axis=1)

        agg = pd.DataFrame({
            "D_cross_mean": d_mean,
            "D_cross_std": d_std,
            "D_cross_max": d_max,
            "D_cross_min": d_min,
            "D_cross_ptp": d_max - d_min,
        })
        feat_blocks.append(agg)

        denom = d_std.replace(0.0, np.nan)
        z = (d_vals.sub(d_mean, axis=0)).div(denom, axis=0)
        z = z.fillna(0.0)
        z.columns = [f"{c}_zscore" for c in D_FEATURE_COLS]
        feat_blocks.append(z)

    feats = pd.concat(feat_blocks, axis=1)
    feats = feats.fillna(0.0).astype(np.float32)
    return feats


# =============================================================================
# 3. Sequence切片 (for GRU / Transformer)
# =============================================================================
def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    T = X.shape[0]
    if T <= seq_len:
        return np.empty((0, seq_len, X.shape[1]), dtype=X.dtype), np.empty((0,), dtype=y.dtype)

    seqs = []
    targets = []
    for t in range(seq_len, T):
        seqs.append(X[t - seq_len:t])
        targets.append(y[t])
    return np.array(seqs), np.array(targets)


# =============================================================================
# 4. Model Structure: Enhanced GRU / Transformer
# =============================================================================
class GRUNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        last = self.norm(last)
        last = self.dropout(last)
        return self.fc(last).squeeze(-1)


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        out = self.encoder(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.fc(last).squeeze(-1)


# =============================================================================
# 5. Training function (with some regularization & clipping)
# =============================================================================
def train_gru(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    X_val_seq: np.ndarray | None,
    y_val_seq: np.ndarray | None,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    epochs: int,
    lr: float,
) -> Tuple[nn.Module, np.ndarray]:
    model = GRUNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train_seq, dtype=torch.float32),
        torch.tensor(y_train_seq, dtype=torch.float32),
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)

    # ---- train ----
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    # ---- optional val pred ----
    if X_val_seq is None or y_val_seq is None or len(X_val_seq) == 0:
        return model, np.empty((0,), dtype=np.float32)

    val_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_val_seq, dtype=torch.float32),
        torch.tensor(y_val_seq, dtype=torch.float32),
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False)

    model.eval()
    all_val_preds = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(DEVICE)
            preds = model(xb)
            all_val_preds.append(preds.cpu().numpy())
    val_preds = np.concatenate(all_val_preds) if len(all_val_preds) else np.empty((0,), dtype=np.float32)
    return model, val_preds



def train_transformer(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    X_val_seq: np.ndarray | None,
    y_val_seq: np.ndarray | None,
    input_dim: int,
    d_model: int,
    num_layers: int,
    epochs: int,
    lr: float,
) -> Tuple[nn.Module, np.ndarray]:
    model = TransformerRegressor(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train_seq, dtype=torch.float32),
        torch.tensor(y_train_seq, dtype=torch.float32),
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)

    # ---- train ----
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    # ---- optional val pred ----
    if X_val_seq is None or y_val_seq is None or len(X_val_seq) == 0:
        return model, np.empty((0,), dtype=np.float32)

    val_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_val_seq, dtype=torch.float32),
        torch.tensor(y_val_seq, dtype=torch.float32),
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False)

    model.eval()
    all_val_preds = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(DEVICE)
            preds = model(xb)
            all_val_preds.append(preds.cpu().numpy())
    val_preds = np.concatenate(all_val_preds) if len(all_val_preds) else np.empty((0,), dtype=np.float32)
    return model, val_preds


def train_catboost_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
) -> Tuple[CatBoostRegressor, np.ndarray]:
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    val_pred = model.predict(X_val)
    return model, val_pred


# =============================================================================
# 6. Time-series folds
# =============================================================================
def time_series_folds(
    n_samples: int,
    n_folds: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    fold_sizes = np.full(n_folds, n_samples // n_folds, dtype=int)
    fold_sizes[: n_samples % n_folds] += 1

    indices = np.arange(n_samples)
    current = 0
    folds = []
    for f in range(n_folds):
        start = current
        stop = current + fold_sizes[f]
        current = stop
        if f == 0:
            continue
        train_idx = indices[:start]
        val_idx = indices[start:stop]
        folds.append((train_idx, val_idx))
    return folds


# =============================================================================
# 7. 超参调优：HTMP 上的 CatBoost / GRU / Transformer
# =============================================================================
def tune_catboost_htmp(
    X: np.ndarray,
    y: np.ndarray,
    rf: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    eval_mask: np.ndarray | None = None,
) -> dict:
    """
    Optuna tuning for CatBoost on HTMP adjusted sharpe.
    - Prefer GPU if available; if GPU fails -> auto fallback to CPU (and stick to CPU afterwards).
    - Only score on indices that have OOF predictions (mask_oof).
    - If eval_mask is not None, only validation samples with index where eval_mask is True contribute to the score.
    """
    import optuna
    from optuna.trial import TrialState
    from catboost import CatBoostError
    import gc
    import time

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_trials = N_TRIALS
    gpu_devices = os.getenv("HTMP_CAT_GPU_DEVICES", "1")  # e.g. "0" or "0:1"
    
    # Allow user to force disable GPU via env var
    # Defaulting to 1 (GPU) initially, but I will override this to 0 in my mind or logic if I could.
    # Actually, as per recent step, I want to DEFAULT TO 0 to allow safe run.
    want_gpu = int(os.getenv("HTMP_USE_GPU", "0")) == 1

    print(f"\n[CatBoost HTMP tuning | Optuna | GPU->CPU fallback enabled | Eval Mask Active: {eval_mask is not None}]")


    # -------------------------
    # base params (GPU / CPU)
    # -------------------------
    base_params_gpu = dict(
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        task_type="GPU",
        devices=gpu_devices,
        iterations=None,
        early_stopping_rounds=None,
        verbose=False,
        allow_writing_files=False,
    )

    base_params_cpu = dict(
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        task_type="CPU",
        iterations=None,                 # CPU 可适当给高一点上限，仍会 early stop
        early_stopping_rounds=None,
        verbose=False,
        allow_writing_files=False,
        thread_count=max(1, os.cpu_count() or 1),
    )

    # -------------------------
    # GPU quick test
    # -------------------------
    use_gpu = want_gpu
    gpu_disabled_msg_printed = False

    if use_gpu:
        try:
            # quick test
            n_test = min(2000, len(y))
            if n_test <= 100:
                n_test = len(y)

            # Try to fit dummy model
            m = CatBoostRegressor(**base_params_gpu)
            m.fit(X[:n_test], y[:n_test], verbose=False)
            print(f"[CatBoost] GPU quick test OK. Using GPU devices={gpu_devices}.")
            
            # CLEANUP: Delete model and force GC to release GPU handle immediately
            del m
            gc.collect()
            time.sleep(0.5)
            
        except Exception as e:
            use_gpu = False
            print(f"[CatBoost][WARNING] GPU not available / failed. Switch to CPU. Reason: {str(e)[:200]}")

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8)

    def objective(trial: optuna.Trial) -> float:
        nonlocal use_gpu, gpu_disabled_msg_printed

        # ---- choose base params by current mode ----
        base_params = base_params_gpu if use_gpu else base_params_cpu
        params = base_params.copy()

        # ---- search space ----
        params.update(
            dict(
                depth=trial.suggest_int("depth", None, None),
                learning_rate=trial.suggest_float("learning_rate", None, None, log=True),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", None, None, log=True),
                min_data_in_leaf=trial.suggest_int("min_data_in_leaf", None, None),
                random_strength=trial.suggest_float("random_strength", None, None),
                bagging_temperature=trial.suggest_float("bagging_temperature", None, None),
                border_count=trial.suggest_int("border_count", None, None),
            )
        )

        oof_pred = np.zeros_like(y)
        mask_oof = np.zeros_like(y, dtype=bool)

        if eval_mask is None:
            eval_mask_local = np.ones_like(y, dtype=bool)
        else:
            eval_mask_local = eval_mask

        for step, (tr_idx, val_idx) in enumerate(folds):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            # ---- try train/predict ----
            try:
                _, val_pred = train_catboost_regressor(X_tr, y_tr, X_val, y_val, params)

            except (CatBoostError, Exception) as e:
                # if GPU failed, fallback to CPU
                if use_gpu:
                    use_gpu = False
                    if not gpu_disabled_msg_printed:
                        print(f"[CatBoost][WARNING] GPU failed during tuning. Fallback to CPU for remaining trials. "
                              f"Reason: {str(e)[:200]}")
                        gpu_disabled_msg_printed = True

                    params_cpu = base_params_cpu.copy()
                    # inherit same hyperparams
                    for k in ["depth", "learning_rate", "l2_leaf_reg", "min_data_in_leaf",
                              "random_strength", "bagging_temperature", "border_count"]:
                        params_cpu[k] = params[k]

                    try:
                        _, val_pred = train_catboost_regressor(X_tr, y_tr, X_val, y_val, params_cpu)
                    except Exception as e2:
                        trial.set_user_attr("catboost_error", str(e2)[:200])
                        return -1e9
                else:
                    trial.set_user_attr("catboost_error", str(e)[:200])
                    return -1e9

            # ---- fill oof ----
            oof_pred[val_idx] = val_pred
            mask_oof[val_idx] = True

            # Mask only effectively evaluated parts
            # inside folds loop, after val_pred computed:
            m = eval_mask_local[val_idx]
            if np.any(m):
                # Only score on the intersection of valid OOF and Eval Mask
                valid_so_far_mask = mask_oof & eval_mask_local
                valid_so_far = np.where(valid_so_far_mask)[0]
                
                # Report intermediate score if enough samples
                if len(valid_so_far) > 50:
                    fold_score = score_from_raw_returns(y[valid_so_far], rf[valid_so_far], oof_pred[valid_so_far], alpha=120.0)
                    trial.report(fold_score, step=step)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

        valid_idx = np.where(mask_oof & eval_mask_local)[0]
        if len(valid_idx) == 0:
            return -1e9

        score = score_from_raw_returns(
            y[valid_idx], rf[valid_idx], oof_pred[valid_idx], alpha=120.0
        )
        return float(score)

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    # if no complete trials, fallback to base params
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if len(completed) == 0:
        print("[CatBoost][WARNING] No completed trials. Return base CPU params.")
        return base_params_cpu

    best_params = (base_params_gpu if use_gpu else base_params_cpu).copy()
    best_params.update(study.best_params)

    print(f"Best CatBoost params (HTMP, Optuna): {best_params}")
    print(f"Best CatBoost OOF adjusted Sharpe: {study.best_value}")
    print(f"[CatBoost] Final mode used for best params: {'GPU' if best_params.get('task_type') == 'GPU' else 'CPU'}")
    return best_params

def tune_gru_htmp(
    X: np.ndarray,
    y: np.ndarray,
    rf: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    input_dim: int,
    eval_mask: np.ndarray | None = None,
) -> dict:
    """
    Optuna tuning for GRU on HTMP adjusted sharpe.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_trials = int(os.getenv("HTMP_OPTUNA_TRIALS_GRU", "25"))

    print(f"\n[GRU HTMP tuning | Optuna | Eval Mask Active: {eval_mask is not None}]")

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    def objective(trial: optuna.Trial) -> float:
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 96, 128])
        num_layers = trial.suggest_categorical("num_layers", [1, 2,3])
        lr = trial.suggest_float("lr", 5e-4, 3e-3, log=True)

        oof_pred = np.zeros_like(y)
        mask = np.zeros_like(y, dtype=bool)
        
        if eval_mask is None:
            eval_mask_local = np.ones_like(y, dtype=bool)
        else:
            eval_mask_local = eval_mask

        step = 0
        for tr_idx, val_idx in folds:
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            X_tr_seq, y_tr_seq = make_sequences(X_tr, y_tr, SEQ_LEN)
            X_val_seq, y_val_seq = make_sequences(X_val, y_val, SEQ_LEN)

            if len(X_val_seq) == 0:
                continue

            val_seq_start = val_idx[0] + SEQ_LEN
            val_seq_idx = np.arange(val_seq_start, val_idx[-1] + 1)
            
            # only keep eval_mask points
            keep = eval_mask_local[val_seq_idx]
            if not np.any(keep):
                continue
            
            # Note: We still train on the full X_tr_seq / X_val_seq structure 
            # (validation logic inside train_gru returns all predictions)
            # But we only score/optimize on `keep` indices
            
            model, val_pred = train_gru(
                X_tr_seq,
                y_tr_seq,
                X_val_seq,
                y_val_seq,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                epochs=10,
                lr=lr,
            )
            
            val_seq_idx_kept = val_seq_idx[keep]
            val_pred_kept = val_pred[keep]
            
            oof_pred[val_seq_idx_kept] = val_pred_kept
            mask[val_seq_idx_kept] = True

            # report score on eval mask region
            valid_idx = np.where(mask)[0]
            if len(valid_idx) > 0:
                score_mid = score_from_raw_returns(
                    y[valid_idx], rf[valid_idx], oof_pred[valid_idx], alpha=120.0
                )
                trial.report(score_mid, step=step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                step += 1

        valid_idx = np.where(mask)[0]
        if len(valid_idx) == 0:
            return -1e9

        score = score_from_raw_returns(y[valid_idx], rf[valid_idx], oof_pred[valid_idx], alpha=120.0)
        return float(score)

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    best_cfg = dict(
        hidden_dim=study.best_params["hidden_dim"],
        num_layers=study.best_params["num_layers"],
    )

    print(f"Best GRU config (HTMP, Optuna): {best_cfg} score: {study.best_value}")
    return best_cfg



def tune_transformer_htmp(
    X: np.ndarray,
    y: np.ndarray,
    rf: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    input_dim: int,
    eval_mask: np.ndarray | None = None,
) -> dict:
    """
    Optuna tuning for Transformer on HTMP adjusted sharpe.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_trials = int(os.getenv("HTMP_OPTUNA_TRIALS_TRF", "25"))

    print(f"\n[Transformer HTMP tuning | Optuna | Eval Mask Active: {eval_mask is not None}]")

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    def objective(trial: optuna.Trial) -> float:
        d_model = trial.suggest_categorical("d_model", [64, 96, 128])
        num_layers = trial.suggest_categorical("num_layers", [1, 2,3])
        lr = trial.suggest_float("lr", 5e-4, 3e-3, log=True)

        oof_pred = np.zeros_like(y)
        mask = np.zeros_like(y, dtype=bool)
        
        if eval_mask is None:
            eval_mask_local = np.ones_like(y, dtype=bool)
        else:
            eval_mask_local = eval_mask

        step = 0
        for tr_idx, val_idx in folds:
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            X_tr_seq, y_tr_seq = make_sequences(X_tr, y_tr, SEQ_LEN)
            X_val_seq, y_val_seq = make_sequences(X_val, y_val, SEQ_LEN)

            if len(X_val_seq) == 0:
                continue

            val_seq_start = val_idx[0] + SEQ_LEN
            val_seq_idx = np.arange(val_seq_start, val_idx[-1] + 1)
            
            keep = eval_mask_local[val_seq_idx]
            if not np.any(keep):
                continue

            model, val_pred = train_transformer(
                X_tr_seq,
                y_tr_seq,
                X_val_seq,
                y_val_seq,
                input_dim=input_dim,
                d_model=d_model,
                num_layers=num_layers,
                epochs=10,
                lr=lr,
            )
            
            val_seq_idx_kept = val_seq_idx[keep]
            val_pred_kept = val_pred[keep]
            
            oof_pred[val_seq_idx_kept] = val_pred_kept
            mask[val_seq_idx_kept] = True

            valid_idx = np.where(mask)[0]
            if len(valid_idx) > 0:
                score_mid = score_from_raw_returns(
                    y[valid_idx], rf[valid_idx], oof_pred[valid_idx], alpha=120.0
                )
                trial.report(score_mid, step=step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                step += 1

        valid_idx = np.where(mask)[0]
        if len(valid_idx) == 0:
            return -1e9

        score = score_from_raw_returns(y[valid_idx], rf[valid_idx], oof_pred[valid_idx], alpha=120.0)
        return float(score)

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    best_cfg = dict(
        d_model=study.best_params["d_model"],
        num_layers=study.best_params["num_layers"],
    )

    print(f"Best Transformer config (HTMP, Optuna): {best_cfg} score: {study.best_value}")
    return best_cfg



# =============================================================================
# 8. Stacking: Ridge + L2 + Alpha search
# =============================================================================
def fit_ridge_ensemble(
    oof_cat: np.ndarray,
    oof_gru: np.ndarray,
    oof_trf: np.ndarray,
    y: np.ndarray,
    rf: np.ndarray,
    mask_valid: np.ndarray,
) -> Tuple[Tuple[float, float, float], float, float]:
    """
    Ridge + L2 + Alpha search for HTMP adjusted sharpe.
    """
    idx = np.where(mask_valid)[0]
    Z = np.stack([oof_cat[idx], oof_gru[idx], oof_trf[idx]], axis=1)  # (n_valid, 3)
    y_t = y[idx]
    rf_t = rf[idx]

    # center y & Z (for L2)
    y_center = y_t - y_t.mean()
    Z_center = Z - Z.mean(axis=0, keepdims=True)

    lambda_grid = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    alpha_grid = [80.0, 100.0, 120.0, 140.0]

    best_score = -np.inf
    best_w = np.array([1/3, 1/3, 1/3], dtype=float)
    best_alpha = 120.0

    for lam in lambda_grid:
        # Ridge closed-form solution
        A = Z_center.T @ Z_center + lam * np.eye(Z_center.shape[1])
        b = Z_center.T @ y_center
        w = np.linalg.solve(A, b)  # (3,)

        # force non-negative + normalize, prevent (1,0,0) type of extreme cases
        w = np.maximum(w, 0.0)
        if w.sum() == 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / w.sum()

        pred_ret = (Z @ w).astype(np.float32)

        for alpha in alpha_grid:
            score = score_from_raw_returns(y_t, rf_t, pred_ret, alpha=alpha)
            if score > best_score:
                best_score = score
                best_w = w.copy()
                best_alpha = alpha

    print(f"Best ensemble weights (cat, gru, trf): {tuple(best_w)}")
    print(f"Best alpha: {best_alpha}   OOF adjusted Sharpe: {best_score}")
    return tuple(best_w.tolist()), float(best_alpha), float(best_score)


def fit_metric_ensemble_optuna(
    oof_cat: np.ndarray,
    oof_gru: np.ndarray,
    oof_trf: np.ndarray,
    y: np.ndarray,
    rf: np.ndarray,
    mask_valid: np.ndarray,
) -> Tuple[Tuple[float, float, float], float, float]:
    """
    Maximize (raw_HTMP - w_reg * ||w - uniform||^2) over (weights, alpha),
    but report/return raw_HTMP for interpretability.
    Weights are parameterized via logits -> softmax for stability.
    """
    import optuna

    idx = np.where(mask_valid)[0]
    if len(idx) == 0:
        print("[Stacking][ERROR] mask_valid has 0 samples.")
        return (1/3, 1/3, 1/3), 80.0, -1e9

    Z = np.stack([oof_cat[idx], oof_gru[idx], oof_trf[idx]], axis=1).astype(np.float32)
    y_t = y[idx].astype(np.float32)
    rf_t = rf[idx].astype(np.float32)

    n_trials = int(os.getenv("HTMP_OPTUNA_TRIALS_STACK", "120"))

    alpha_low = float(os.getenv("HTMP_ALPHA_LOW", "40"))
    alpha_high = float(os.getenv("HTMP_ALPHA_HIGH", "120"))
    alpha_step = float(os.getenv("HTMP_ALPHA_STEP", "5"))

    # ---- new: uniform-deviation regularization strength ----
    w_reg = float(os.getenv("HTMP_WEIGHT_UNIFORM_REG", "0.08"))
    # optional: logit range controls how extreme softmax can get
    logit_bound = float(os.getenv("HTMP_WEIGHT_LOGIT_BOUND", "3.0"))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=42)

    uniform = np.array([1/3, 1/3, 1/3], dtype=np.float64)

    def softmax_logits(a: float, b: float, c: float) -> np.ndarray:
        v = np.array([a, b, c], dtype=np.float64)
        v = v - np.max(v)  # stability
        e = np.exp(v)
        return e / np.sum(e)

    print(f"\n[Stacking] alpha in [{alpha_low}, {alpha_high}] step={alpha_step}, trials={n_trials}")
    print(f"[Stacking] uniform L2 reg: coef={w_reg}, logit_bound=±{logit_bound}")

    def objective(trial: optuna.Trial) -> float:
        # logits -> softmax weights (sum=1, all > 0)
        lc = trial.suggest_float("logit_cat", -logit_bound, logit_bound)
        lg = trial.suggest_float("logit_gru", -logit_bound, logit_bound)
        lt = trial.suggest_float("logit_trf", -logit_bound, logit_bound)
        w = softmax_logits(lc, lg, lt)  # (3,)

        alpha = trial.suggest_float("alpha", alpha_low, alpha_high, step=alpha_step)

        pred_ret = (Z @ w).astype(np.float32)
        raw = float(score_from_raw_returns(y_t, rf_t, pred_ret, alpha=alpha))

        # penalize deviation from uniform weights
        penalty = float(np.sum((w - uniform) ** 2))  # 0 at uniform; ~0.666 at (1,0,0)
        obj = raw - w_reg * penalty

        # store for debugging / transparency
        trial.set_user_attr("raw_score", raw)
        trial.set_user_attr("penalty", penalty)
        trial.set_user_attr("objective", float(obj))
        trial.set_user_attr("weights", [float(w[0]), float(w[1]), float(w[2])])
        trial.set_user_attr("alpha", float(alpha))

        return float(obj)

    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    bt = study.best_trial
    bp = bt.params

    wc, wg, wt = softmax_logits(bp["logit_cat"], bp["logit_gru"], bp["logit_trf"])
    alpha = float(bp["alpha"])

    pred_ret = (Z @ np.array([wc, wg, wt], dtype=np.float64)).astype(np.float32)
    raw_best = float(score_from_raw_returns(y_t, rf_t, pred_ret, alpha=alpha))
    penalty_best = float(np.sum((np.array([wc, wg, wt]) - uniform) ** 2))
    obj_best = raw_best - w_reg * penalty_best

    print(f"Best weights (cat, gru, trf): ({wc:.6f}, {wg:.6f}, {wt:.6f})")
    print(f"Best alpha: {alpha:.1f}")
    print(f"Best raw OOF adjusted Sharpe: {raw_best:.6f}")
    print(f"Best penalty: {penalty_best:.6f}   Penalized objective: {obj_best:.6f}")

    return (float(wc), float(wg), float(wt)), float(alpha), float(raw_best)




# =============================================================================
# 2.1 Symbolic Transformer Helpers
# =============================================================================
def _patch_symbolic_tags() -> None:
    def _tags(self):
        try:
            return BaseEstimator.__sklearn_tags__(self)
        except Exception:
            from sklearn.utils._tags import Tags
            return Tags()

    if getattr(SymbolicTransformer, "__patched_tags__", False):
        return
    SymbolicTransformer.__sklearn_tags__ = _tags
    for cls in SymbolicTransformer.mro():
        if cls.__name__ == "BaseSymbolic":
            cls.__sklearn_tags__ = _tags
            break
    SymbolicTransformer.__patched_tags__ = True


def _cross_sectional_rank(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """
    Rank y within each group (e.g. date_id) to [-0.5, 0.5].
    Robust target for Spearman correlation in SymbolicTransformer.
    """
    df = pd.DataFrame({"group": groups, "y": y})
    # If group size < 2, rank makes no sense (becomes avg), so return raw y normalized or just raw
    # Spearman metric acts on rank order anyway, so raw y is fine if unique.
    # We will just skip rank if count is 1.
    def _rank_wrapper(x):
        if len(x) < 2:
            return x
        return x.rank(pct=True, method="average") - 0.5
        
    df["rank"] = df.groupby("group")["y"].transform(_rank_wrapper)
    return df["rank"].to_numpy(dtype=np.float32)


def train_symbolic_transformer(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_components: int = 600,
    generations: int = 20,
    population_size: int = 2000,
    random_state: int = 42
) -> Tuple[SymbolicTransformer, pd.DataFrame]:
    """
    Train a SymbolicTransformer and return it along with the transformed train features.
    """
    _patch_symbolic_tags()
    
    print(f"[Symbolic] Training... n_components={n_components}, pop={population_size}, gens={generations}")
    st = SymbolicTransformer(
        n_components=n_components,
        generations=generations,
        population_size=population_size,
        hall_of_fame=max(400, n_components), # Must be >= n_components
        stopping_criteria=0.0,
        const_range=(-0.5, 0.5),
        init_depth=(2, 6),
        init_method="half and half",
        function_set=("add", "sub", "mul", "div", "sin", "cos", "log", "sqrt"),
        metric="spearman",
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=0.9,
        verbose=1,
        random_state=random_state,
        n_jobs=-1  # Use parallel if possible
    )
    
    # Ensure no NaNs
    X_mat = X_train.fillna(0.0).values
    
    st.fit(X_mat, y_train)
    
    # Transform
    X_sym = st.transform(X_mat)
    sym_cols = [f"sym_{i}" for i in range(X_sym.shape[1])]
    
    df_sym = pd.DataFrame(X_sym, columns=sym_cols, index=X_train.index)
    
    print(f"[Symbolic] Generated {len(sym_cols)} features.")
    
    return st, df_sym


# =============================================================================
# 9. Main workflow: train + save artifacts
# =============================================================================
def main():
    global BASE_FEATURE_COLS, D_FEATURE_COLS, FINAL_FEATURE_COLS
    global FEATURE_MEAN, FEATURE_STD, ENSEMBLE_W, ALPHA

    print(f"Device: {DEVICE}")
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"Train shape: {train.shape}  Test shape: {test.shape}")

    train_noleak = train.iloc[:-LOOKBACK].reset_index(drop=True)
    print(f"Train after leak cut: {train_noleak.shape}")

    N = len(train_noleak)
    mine_end = int(MINE_FRAC * N)
    mine_end = max(mine_end, 1)

    eval_mask = np.ones(N, dtype=bool)
    if EVAL_ON_TAIL == 1:
        eval_mask[:] = False
        eval_mask[mine_end:] = True

    print(f"[Scheme B] mine_end={mine_end}/{N} (front {MINE_FRAC:.0%}); "
          f"eval_on_tail={bool(EVAL_ON_TAIL)} -> eval_count={eval_mask.sum()}")

    train_numeric = [
        c for c in train_noleak.columns
        if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(train_noleak[c])
    ]
    test_numeric = [
        c for c in test.columns
        if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(test[c])
    ]
    common_numeric = sorted(set(train_numeric) & set(test_numeric))
    BASE_FEATURE_COLS = common_numeric
    D_FEATURE_COLS = [c for c in BASE_FEATURE_COLS if c.startswith("D")]

    print(f"Base feature columns: {len(BASE_FEATURE_COLS)}")

    X_train_fe = build_features(train_noleak)
    X_test_fe = build_features(test)

    print(f"Engineered train feature shape: {X_train_fe.shape}")
    print(f"Engineered test feature shape:  {X_test_fe.shape}")

    common_cols = sorted(set(X_train_fe.columns) & set(X_test_fe.columns))
    X_train_fe = X_train_fe[common_cols].copy()
    X_test_fe = X_test_fe[common_cols].copy()
    FINAL_FEATURE_COLS = list(common_cols)

    print(f"Selected feature count (after aligning with test): {len(common_cols)}")

    # =========================================================================
    # Symbolic Transformer Integration (Scheme B: fit only on first mine_end)
    # =========================================================================
    y_raw_for_sym = train_noleak[TARGET_COL].values.astype(np.float32)

    use_cs_rank = False
    if "date_id" in train_noleak.columns:
        # only do cross-sectional rank if same date_id has multiple rows (true截面)
        vc = pd.Series(train_noleak["date_id"]).value_counts()
        if vc.max() > 1:
            use_cs_rank = True

    if use_cs_rank:
        groups_for_sym = train_noleak["date_id"].values
        y_for_sym = _cross_sectional_rank(y_raw_for_sym, groups_for_sym)
        print("[Symbolic] Using cross-sectional rank target (date_id has multi-rows).")
    else:
        y_for_sym = y_raw_for_sym
        print("[Symbolic] Using raw target (single-asset / no cross-section).")

    # ---- fit on first mine_end only ----
    # Save the columns used for Symbolic Transformer input to ensure inference alignment
    SYMBOLIC_INPUT_COLS = X_train_fe.columns.tolist()
    
    st_model, X_train_sym_fit = train_symbolic_transformer(
        X_train_fe.iloc[:mine_end],
        y_for_sym[:mine_end],
        n_components=None,
        generations=None,
        population_size=None,
        random_state=42
    )

    # Save Symbolic Model
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(st_model, os.path.join(ARTIFACT_DIR, "symbolic_model.pkl"))
    print(f"[Symbolic] Model saved to {ARTIFACT_DIR}/symbolic_model.pkl")

    # ---- transform full train & test with the same fitted st_model ----
    # Ensure no NaNs before passing to transform
    X_train_sym_vals = st_model.transform(X_train_fe.fillna(0.0).values)
    sym_cols = [f"sym_{i}" for i in range(X_train_sym_vals.shape[1])]
    X_train_sym = pd.DataFrame(X_train_sym_vals, columns=sym_cols, index=X_train_fe.index)

    X_test_sym_vals = st_model.transform(X_test_fe.fillna(0.0).values)
    X_test_sym = pd.DataFrame(X_test_sym_vals, columns=sym_cols, index=X_test_fe.index)

    X_train_fe = pd.concat([X_train_fe, X_train_sym], axis=1)
    X_test_fe  = pd.concat([X_test_fe,  X_test_sym], axis=1)

    FINAL_FEATURE_COLS = X_train_fe.columns.tolist()
    print(f"Total features after Symbolic Transformer: {len(FINAL_FEATURE_COLS)}")


    # Base arrays (full features)
    X_full = X_train_fe.values.astype(np.float32)
    y = train_noleak[TARGET_COL].values.astype(np.float32)
    rf = train_noleak["risk_free_rate"].values.astype(np.float32)

    # Time series folds
    # Scheme B: Folds cover the whole duration.
    # We train/predict on all folds (expanding window), 
    # but we only score/optimize on the Validation Split (back 40%).
    # This allows models to learn from the Mining Split features but not be evaluated on their formation period.
    folds = time_series_folds(len(train_noleak), n_folds=4)

    # ---- Step A: Run Selection Logic on MINING SPLIT ----
    
    # ---- stats source ----
    if STATS_ON_MINE == 1:
        stats_slice = slice(0, mine_end)
    else:
        stats_slice = slice(0, len(train_noleak))

    mean_full = X_full[stats_slice].mean(axis=0)
    std_full = X_full[stats_slice].std(axis=0)
    std_full[std_full == 0] = 1e-8
    X_std_full = (X_full - mean_full) / std_full
    
    # Create simple folds within mining set for Tuning the Selector
    # Note: We tune CatBoost for Feature Importance using ONLY the mine_end part
    folds_mining = time_series_folds(mine_end, n_folds=3)

    print("\n" + "="*80)
    print("[Step A] Tune CatBoost on MINING split (60% data) for Selection")
    print("="*80 + "\n")
    
    # Pass ONLY minining split to tuning
    best_cat_params_full = tune_catboost_htmp(
        X_std_full[:mine_end], 
        y[:mine_end], 
        rf[:mine_end], 
        folds_mining
    )

    # Train optimal CatBoost on MINING data for feature importance
    cat_for_rank, _ = train_catboost_regressor(
        X_std_full[:mine_end], y[:mine_end], X_std_full[:mine_end], y[:mine_end], best_cat_params_full
    )
    importances = cat_for_rank.get_feature_importance(type="PredictionValuesChange")
    order = np.argsort(importances)[::-1]

    top_k = min(TOP_K_FEATURES, len(FINAL_FEATURE_COLS))
    selected_idx = order[:top_k]
    selected_cols = [FINAL_FEATURE_COLS[i] for i in selected_idx]

    FINAL_FEATURE_COLS = selected_cols
    X_train_fe = X_train_fe[FINAL_FEATURE_COLS].copy()
    X_test_fe = X_test_fe[FINAL_FEATURE_COLS].copy()

    print(f"Feature count after CatBoost top-{top_k} selection: {len(FINAL_FEATURE_COLS)}")

    # ---- All models after this only train on selected features ----
    X = X_train_fe.values.astype(np.float32)

    # Re-calc stats on selected features using the same slice logic
    FEATURE_MEAN = X[stats_slice].mean(axis=0)
    FEATURE_STD = X[stats_slice].std(axis=0)
    FEATURE_STD[FEATURE_STD == 0] = 1e-8

    X_std = (X - FEATURE_MEAN) / FEATURE_STD
    input_dim = X_std.shape[1]

    # ---- Tuning Phase: Tune on Full Folds but Score Only on Back 40% ----
    print("\n" + "="*80)
    print(f"[Step B] Re-tune CatBoost (Eval Mask Active: {EVAL_ON_TAIL})")
    print("="*80 + "\n")
    
    # Pass full X_std, y, rf, folds
    # Pass eval_mask to restrict scoring to the Evaluation Split
    best_cat_params = tune_catboost_htmp(X_std, y, rf, folds, eval_mask=eval_mask)
    
    # ---------- Hyperparameter tuning ----------
    best_gru_cfg = tune_gru_htmp(X_std, y, rf, folds, input_dim, eval_mask=eval_mask)
    best_trf_cfg = tune_transformer_htmp(X_std, y, rf, folds, input_dim, eval_mask=eval_mask)

    # ---------- Re-fit with best hyperparameters (Evaluation Split Scoring) ----------
    print(f"\n[OOF with best hyperparams] (Scoring on Eval Mask)")
    from sklearn.metrics import mean_squared_error

    oof_cat = np.zeros_like(y)
    oof_gru = np.zeros_like(y)
    oof_trf = np.zeros_like(y)

    mask_cat = np.zeros_like(y, dtype=bool)
    mask_gru = np.zeros_like(y, dtype=bool)
    mask_trf = np.zeros_like(y, dtype=bool)

    # Use Full Folds
    for fold_idx, (tr_idx, val_idx) in enumerate(folds, start=1):
        print(f"\n=== Fold {fold_idx} ===")
        print(f"Train idx: [{tr_idx[0]}, {tr_idx[-1] + 1})  Val idx: [{val_idx[0]}, {val_idx[-1] + 1})")

        X_tr, X_val = X_std[tr_idx], X_std[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        # CatBoost
        cb_model, cb_val = train_catboost_regressor(X_tr, y_tr, X_val, y_val, best_cat_params)
        oof_cat[val_idx] = cb_val
        mask_cat[val_idx] = True

        # GRU / Transformer sequences
        X_tr_seq, y_tr_seq = make_sequences(X_tr, y_tr, SEQ_LEN)
        X_val_seq, y_val_seq = make_sequences(X_val, y_val, SEQ_LEN)

        if len(X_val_seq) == 0:
            continue

        val_seq_start = val_idx[0] + SEQ_LEN
        val_seq_idx = np.arange(val_seq_start, val_idx[-1] + 1)
        
        # Scheme B: Only keep points in eval_mask
        keep = eval_mask[val_seq_idx]
        if not np.any(keep):
            continue

        val_seq_idx_kept = val_seq_idx[keep]

        gru_m, gru_val = train_gru(
            X_tr_seq,
            y_tr_seq,
            X_val_seq,
            y_val_seq,
            input_dim=input_dim,
            hidden_dim=best_gru_cfg["hidden_dim"],
            num_layers=best_gru_cfg["num_layers"],
            epochs=12,
        )
        oof_gru[val_seq_idx_kept] = gru_val[keep]
        mask_gru[val_seq_idx_kept] = True

        trf_m, trf_val = train_transformer(
            X_tr_seq,
            y_tr_seq,
            X_val_seq,
            y_val_seq,
            input_dim=input_dim,
            d_model=best_trf_cfg["d_model"],
            num_layers=best_trf_cfg["num_layers"],
            epochs=12,
        )
        oof_trf[val_seq_idx_kept] = trf_val[keep]
        mask_trf[val_seq_idx_kept] = True

        # Report MSE on filtered part
        # y[val_seq_idx_kept] vs gru_val[keep]
        print("GRU val MSE (filtered):", mean_squared_error(y[val_seq_idx_kept], gru_val[keep]))
        print("Transformer val MSE (filtered):", mean_squared_error(y[val_seq_idx_kept], trf_val[keep]))

    # Combine masks with eval_mask for final Stacking
    mask_valid_all = (mask_cat & mask_gru & mask_trf & eval_mask)
    valid_idx = np.where(mask_valid_all)[0]
    print("\nOOF valid count (tail only):", len(valid_idx))

    # ---------- Stacking: Metric-aligned Optuna + Alpha ----------
    ENSEMBLE_W, ALPHA, best_score = fit_metric_ensemble_optuna(
        oof_cat,
        oof_gru,
        oof_trf,
        y,
        rf,
        mask_valid=mask_valid_all,
    )

    # ---------- Train final models on full data ----------
    print("\n[Train final models on full data]")

    # Recalculate stats on FULL data for the final model artifacts (as recommended)
    # This ensures the deployed model uses the most up-to-date distribution center/scale
    # The "Anti-Leakage" was for VALIDATION. Now we are deploying, we use all info.
    FEATURE_MEAN = X.mean(axis=0)
    FEATURE_STD = X.std(axis=0)
    FEATURE_STD[FEATURE_STD == 0] = 1e-8
    X_std = (X - FEATURE_MEAN) / FEATURE_STD
    
    print("[Final] Updated MEAN/STD using full dataset for artifacts.")

    # CatBoost full (on selected features)
    cat_model_full, _ = train_catboost_regressor(X_std, y, X_std, y, best_cat_params)
    # Sequence construction
    X_full_seq, y_full_seq = make_sequences(X_std, y, SEQ_LEN)

    if FINAL_USE_FULL_SEQ == 1:
        print("[Final] GRU/Transformer train on FULL sequences (no 85/15 split).")
        gru_model_full, _ = train_gru(
            X_full_seq, y_full_seq,
            None, None,
            input_dim=input_dim,
            hidden_dim=best_gru_cfg["hidden_dim"],
            num_layers=best_gru_cfg["num_layers"],
            epochs=14,
        )
        trf_model_full, _ = train_transformer(
            X_full_seq, y_full_seq,
            None, None,
            input_dim=input_dim,
            d_model=best_trf_cfg["d_model"],
            num_layers=best_trf_cfg["num_layers"],
            epochs=14,
        )
    else:
        print("[Final][WARNING] GRU/Transformer keep 85/15 split, but NO early stopping is used -> last 15% sequences are NOT used for weight updates (evaluation-only).")
        split = int(0.85 * len(X_full_seq))
        X_seq_tr, y_seq_tr = X_full_seq[:split], y_full_seq[:split]
        X_seq_val, y_seq_val = X_full_seq[split:], y_full_seq[split:]

        gru_model_full, _ = train_gru(
            X_seq_tr, y_seq_tr,
            X_seq_val, y_seq_val,
            input_dim=input_dim,
            hidden_dim=best_gru_cfg["hidden_dim"],
            num_layers=best_gru_cfg["num_layers"],
            epochs=14,
        )
        trf_model_full, _ = train_transformer(
            X_seq_tr, y_seq_tr,
            X_seq_val, y_seq_val,
            input_dim=input_dim,
            d_model=best_trf_cfg["d_model"],
            num_layers=best_trf_cfg["num_layers"],
            epochs=14,
        )

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # Save meta
    meta_path = os.path.join(ARTIFACT_DIR, "meta.npz")
    np.savez_compressed(
        meta_path,
        FEATURE_MEAN=FEATURE_MEAN,
        FEATURE_STD=FEATURE_STD,
        ENSEMBLE_W=np.array(ENSEMBLE_W, dtype=np.float32),
        ALPHA=np.array([ALPHA], dtype=np.float32),
        BASE_FEATURE_COLS=np.array(BASE_FEATURE_COLS, dtype=object),
        D_FEATURE_COLS=np.array(D_FEATURE_COLS, dtype=object),
        FINAL_FEATURE_COLS=np.array(FINAL_FEATURE_COLS, dtype=object),
        GRU_HIDDEN_DIM=np.array([best_gru_cfg["hidden_dim"]], dtype=np.int32),
        GRU_NUM_LAYERS=np.array([best_gru_cfg["num_layers"]], dtype=np.int32),
        TRF_D_MODEL=np.array([best_trf_cfg["d_model"]], dtype=np.int32),
        TRF_NUM_LAYERS=np.array([best_trf_cfg["num_layers"]], dtype=np.int32),
        SYMBOLIC_INPUT_COLS=np.array(SYMBOLIC_INPUT_COLS, dtype=object),
    )

    # CatBoost model
    cat_path = os.path.join(ARTIFACT_DIR, "cat_model.cbm")
    cat_model_full.save_model(cat_path)

    # GRU / Transformer weights
    gru_path = os.path.join(ARTIFACT_DIR, "gru_model.pth")
    trf_path = os.path.join(ARTIFACT_DIR, "trf_model.pth")
    torch.save(gru_model_full.state_dict(), gru_path)
    torch.save(trf_model_full.state_dict(), trf_path)

    print(f"\nArtifacts saved to: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
