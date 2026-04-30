"""
time_series_prediction.py
=========================
Time series forecasting models for process KPI prediction.

All model functions share the signature:
    forecast_*(train, horizon, params) -> np.ndarray  (shape: (horizon,))

Parameters are passed as an optional dict; see each function's docstring for
supported keys and their defaults.  Missing keys always fall back to a
documented default so callers can pass partial dicts.
"""

from __future__ import annotations

import gc
import math
import warnings
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX
import torch
import torch.nn as nn
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler as DartsScaler
from darts.models import NBEATSModel


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build cyclical calendar features from a DatetimeIndex.

    Encodes hour-of-day, day-of-week, and month-of-year as (sin, cos) pairs so
    that the circular distance between adjacent periods is small (e.g. hour 23
    and hour 0 are close).  Columns with zero variance across *index* are
    dropped to prevent StandardScaler from dividing by zero — for example,
    hour features are constant when the index has daily resolution.

    Returns a DataFrame with the same index; may be empty if all features are
    constant (callers should handle this case by disabling time features).
    """
    features = {
        "hour_sin":  np.sin(2 * np.pi * index.hour        / 24.0),
        "hour_cos":  np.cos(2 * np.pi * index.hour        / 24.0),
        "dow_sin":   np.sin(2 * np.pi * index.dayofweek   /  7.0),
        "dow_cos":   np.cos(2 * np.pi * index.dayofweek   /  7.0),
        "month_sin": np.sin(2 * np.pi * index.month       / 12.0),
        "month_cos": np.cos(2 * np.pi * index.month       / 12.0),
    }
    df = pd.DataFrame(features, index=index)
    return df.loc[:, df.std() > 0]   # drop constant columns


def _create_window_dataset(
    values: np.ndarray, n_steps: int, n_future: int
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window over *values* to produce a supervised (X, y) dataset.

        X[i] = values[i : i + n_steps]                    shape (n_steps,)
        y[i] = values[i + n_steps : i + n_steps + n_future]  shape (n_future,)

    Returns two arrays of shape (n_windows, n_steps) and (n_windows, n_future).
    Returns empty arrays (shape (0,)) when the series is shorter than
    n_steps + n_future, which the calling code should treat as a fallback signal.
    """
    n_windows = max(0, len(values) - n_steps - n_future + 1)
    X = np.array([values[i : i + n_steps]                      for i in range(n_windows)], dtype=float)
    y = np.array([values[i + n_steps : i + n_steps + n_future] for i in range(n_windows)], dtype=float)
    return X, y


class _GRUNet(nn.Module):
    """GRU encoder → linear decoder for direct multi-step forecasting.

    Architecture:
        Input  (batch, seq_len, 1)   — one univariate feature per time step
        GRU    (num_layers stacked)
        Linear (hidden_size → n_future)
        Output (batch, n_future)     — predicted values for the next n_future steps
    """

    def __init__(self, hidden_size: int, n_future: int, num_layers: int = 1):
        super().__init__()
        self.gru  = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, n_future)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)       # (batch, seq_len, hidden_size)
        return self.head(out[:, -1, :])   # last hidden state → (batch, n_future)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline models
# ─────────────────────────────────────────────────────────────────────────────

def forecast_naive(
    train: pd.Series, horizon: int, params: Optional[dict] = None
) -> np.ndarray:
    """Persistence baseline: repeat the last observed value for every step.

    Parameters
    ----------
    train   : training series
    horizon : number of steps to forecast
    params  : not used; accepted for API compatibility
    """
    return np.full(horizon, float(train.iloc[-1]), dtype=float)


def forecast_naive_recent(train_val: pd.Series, test: pd.Series) -> np.ndarray:
    """Oracle 1-step-ahead persistence baseline.

    Predicts each test step as the true previous value:
        step 0  →  train_val[-1]
        step i  →  test[i - 1]   (i ≥ 1)

    This is a lower-bound reference, not a deployable model, because it uses
    ground-truth test values as inputs.  It has a different signature from
    the other model functions and cannot be passed to tune_on_val.

    Parameters
    ----------
    train_val : concatenated train + val series
    test      : ground-truth test series
    """
    prev = np.concatenate([[train_val.iloc[-1]], test.to_numpy(dtype=float)[:-1]])
    return prev


def forecast_seasonal_naive(
    train: pd.Series, horizon: int, params: Optional[dict] = None
) -> np.ndarray:
    """Seasonal naive: tile the last complete season over the forecast horizon.

    Parameters
    ----------
    train   : training series
    horizon : number of steps to forecast
    params  : optional dict with keys:
        season (int, default 7): season length in observations; should match the
                                 dominant periodicity of the series (e.g. 7 for
                                 weekly seasonality in daily data)

    Notes
    -----
    Falls back to forecast_naive when len(train) < season, since there is not
    enough history to extract one full season.
    """
    p      = params or {}
    season = int(p.get("season", 7))
    if season <= 0:
        raise ValueError(f"season must be a positive integer, got {season}")
    if len(train) < season:
        return forecast_naive(train, horizon)
    last_season = train.iloc[-season:].to_numpy(dtype=float)
    return np.resize(last_season, horizon).astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# Statistical models
# ─────────────────────────────────────────────────────────────────────────────

def forecast_ets(
    train: pd.Series, horizon: int, params: Optional[dict] = None
) -> np.ndarray:
    """Exponential smoothing (ETS) via statsmodels ETSModel.

    ETS decomposes the series into Error, Trend, and Seasonal components, each
    of which can be additive, multiplicative, or absent.  Parameters are
    estimated by maximum likelihood.

    Parameters
    ----------
    train   : training series; must be strictly positive when any component is
              multiplicative (statsmodels will raise otherwise)
    horizon : number of steps to forecast
    params  : optional dict with keys:
        error            (str, default "add")   : error type — "add" or "mul"
        trend            (str|None, default None): trend type — "add", "mul", or None
        damped_trend     (bool, default False)   : apply damping to the trend;
                                                   only valid when trend is not None
        seasonal         (str|None, default None): seasonal type — "add", "mul", or None
        seasonal_periods (int, default 7)        : season length in observations;
                                                   only used when seasonal is not None
    """
    p = params or {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ETSModel(
            train,
            error=p.get("error", "add"),
            trend=p.get("trend", None),
            damped_trend=bool(p.get("damped_trend", False)),
            seasonal=p.get("seasonal", None),
            seasonal_periods=int(p.get("seasonal_periods", 7)),
        )
        res = model.fit(maxiter=1000, disp=False)
    return np.asarray(res.forecast(horizon), dtype=float)


def forecast_sarimax(
    train: pd.Series, horizon: int, params: Optional[dict] = None
) -> np.ndarray:
    """SARIMA via statsmodels SARIMAX.

    Fits a Seasonal AutoRegressive Integrated Moving Average model.  The
    intercept (trend="c") is always included; exogenous regressors are not used.

    Parameters
    ----------
    train   : training series
    horizon : number of steps to forecast
    params  : optional dict with keys:
        order          (tuple (p, d, q), default (1, 0, 0)):
                           p — AR order (past values)
                           d — differencing order (for non-stationary series)
                           q — MA order (past errors)
        seasonal_order (tuple (P, D, Q, s), default (0, 0, 0, 0)):
                           P, D, Q — seasonal AR/I/MA orders
                           s       — seasonal period in observations (e.g. 7 for weekly)

    Notes
    -----
    - Fitting uses L-BFGS-B (method="lbfgs") with up to 200 iterations.
    - Model data is released after forecasting to limit memory use.
    - Convergence and stationarity warnings are suppressed; use
      statsmodels directly for diagnostic output.
    """
    p = params or {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            train,
            order=p.get("order", (1, 0, 0)),
            seasonal_order=p.get("seasonal_order", (0, 0, 0, 0)),
            trend="c",
        )
        res = model.fit(disp=False, method="lbfgs", maxiter=200)
    yhat = np.asarray(res.get_forecast(steps=horizon).predicted_mean, dtype=float)
    res.remove_data()
    del res, model
    gc.collect()
    return yhat


# ─────────────────────────────────────────────────────────────────────────────
# Machine learning models
# ─────────────────────────────────────────────────────────────────────────────

def forecast_ridge(
    train: pd.Series, horizon: int, params: Optional[dict] = None
) -> np.ndarray:
    """Recursive Ridge regression on lag features with optional calendar features.

    The model is built using sktime's make_reduction (strategy="recursive"):
    a Ridge regressor is trained to predict one step ahead from the last *lags*
    observations; at forecast time it feeds its own predictions back as inputs,
    advancing one step at a time.

    Parameters
    ----------
    train   : training series (DatetimeIndex required for time features)
    horizon : number of steps to forecast
    params  : optional dict with keys:
        lags              (int, default 14)   : number of lag features passed to the
                                                regressor (= the autoregressive window)
        alpha             (float, default 1.0): Ridge L2 penalty; larger values shrink
                                                coefficients more, reducing variance at
                                                the cost of bias
        add_time_features (bool, default True): augment lag features with cyclical
                                                calendar encodings (hour, day-of-week,
                                                month); requires a DatetimeIndex with
                                                an inferable regular frequency

    Notes
    -----
    - Constant calendar columns (e.g. hour on daily data) are dropped automatically
      before fitting to avoid a division-by-zero in StandardScaler.
    - If the index frequency cannot be inferred, time features are silently disabled.
    - Features are z-score scaled inside the sklearn Pipeline before Ridge fitting.
    """
    p                 = params or {}
    lags              = int(p.get("lags", 14))
    alpha             = float(p.get("alpha", 1.0))
    add_time_features = bool(p.get("add_time_features", True))

    y = train.astype(float).copy()

    X_train: Optional[pd.DataFrame] = None
    X_pred:  Optional[pd.DataFrame] = None

    if add_time_features and isinstance(y.index, pd.DatetimeIndex):
        freq = pd.infer_freq(y.index)
        if freq is not None:
            X_train = _make_time_features(y.index)
            future_idx = pd.date_range(
                start=y.index[-1], periods=horizon + 1, freq=freq
            )[1:]
            X_pred = _make_time_features(future_idx)
            # disable if all features turned out constant on this particular index
            if X_train.empty or X_pred.empty:
                X_train = X_pred = None

    regressor = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=alpha, random_state=0)),
    ])
    forecaster = make_reduction(
        estimator=regressor,
        window_length=lags,
        strategy="recursive",
    )
    fh = ForecastingHorizon(np.arange(1, horizon + 1), is_relative=True)
    forecaster.fit(y=y, X=X_train)
    y_pred = forecaster.predict(fh=fh, X=X_pred)
    return np.asarray(y_pred, dtype=float).reshape(-1)


def forecast_gru(
    train: pd.Series, horizon: int, params: Optional[dict] = None
) -> np.ndarray:
    """GRU-based multi-step forecaster trained on a sliding-window dataset.

    The network maps a window of *n_steps* past (z-score scaled) observations
    to the next *n_future* steps in one forward pass.  Multi-step forecasting
    beyond *n_future* is handled recursively: each predicted chunk is appended
    to the history buffer and the window slides forward.

    Parameters
    ----------
    train   : training series
    horizon : number of steps to forecast
    params  : optional dict with keys:
        n_steps     (int, default 14)   : input window length (past observations
                                          given to the GRU at each time step)
        n_future    (int, default 7)    : number of steps predicted per forward pass;
                                          horizons longer than n_future use recursion
        hidden_size (int, default 64)   : GRU hidden state dimension
        num_layers  (int, default 1)    : number of stacked GRU layers
        epochs      (int, default 50)   : training epochs (full passes over dataset)
        lr          (float, default 1e-3): Adam learning rate
        batch_size  (int, default 32)   : mini-batch size for training
        seed        (int, default 0)    : random seed for NumPy and PyTorch

    Notes
    -----
    - Falls back to forecast_naive when there are fewer than 50 usable training
      windows (i.e. len(train) < n_steps + n_future + 49).
    - The series is z-score scaled before training; predictions are
      inverse-transformed at the end using the same scaler.
    - torch.use_deterministic_algorithms is intentionally NOT set here because
      some GRU CUDA kernels have no deterministic implementation and would raise.
      Set *seed* for reproducibility instead.
    """
    p           = params or {}
    n_steps     = int(p.get("n_steps",     14))
    n_future    = int(p.get("n_future",     7))
    hidden_size = int(p.get("hidden_size", 64))
    num_layers  = int(p.get("num_layers",   1))
    epochs      = int(p.get("epochs",      50))
    lr          = float(p.get("lr",       1e-3))
    batch_size  = int(p.get("batch_size",  32))
    seed        = int(p.get("seed",         0))
    device      = "cuda" if torch.cuda.is_available() else "cpu"

    # ── reproducibility ──────────────────────────────────────────────────────
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    # ── scale to zero mean / unit variance ───────────────────────────────────
    values = train.to_numpy(dtype=float).reshape(-1, 1)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values).reshape(-1)

    # ── build supervised dataset ─────────────────────────────────────────────
    X, y = _create_window_dataset(scaled, n_steps=n_steps, n_future=n_future)
    if len(X) < 50:
        # not enough training windows for meaningful gradient-based training
        return forecast_naive(train, horizon)

    g   = torch.Generator().manual_seed(seed)
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (N, n_steps, 1)
    y_t = torch.tensor(y, dtype=torch.float32)                 # (N, n_future)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t, y_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=g,
        num_workers=0,   # 0 keeps DataLoader deterministic on CPU
    )

    # ── train ─────────────────────────────────────────────────────────────────
    net       = _GRUNet(hidden_size=hidden_size, n_future=n_future, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    net.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        print(f"  [GRU] epoch {epoch + 1}/{epochs}  loss={total_loss / len(X):.6f}", flush=True)

    # ── recursive forecast ────────────────────────────────────────────────────
    net.eval()
    hist          = scaled.tolist()
    preds_scaled: list[float] = []
    n_chunks      = math.ceil(horizon / n_future)

    with torch.no_grad():
        for _ in range(n_chunks):
            inp   = torch.tensor(hist[-n_steps:], dtype=torch.float32).view(1, n_steps, 1).to(device)
            chunk = net(inp).cpu().numpy().reshape(-1)   # (n_future,)
            preds_scaled.extend(chunk.tolist())
            hist.extend(chunk.tolist())                  # slide window forward

    preds_scaled = preds_scaled[:horizon]
    return scaler.inverse_transform(
        np.array(preds_scaled, dtype=float).reshape(-1, 1)
    ).reshape(-1)


def forecast_nbeats(
    train: pd.Series, horizon: int, params: Optional[dict] = None
) -> np.ndarray:
    """N-BEATS forecaster via the darts library.

    N-BEATS learns basis expansions (generically or as trend + seasonality stacks)
    that decompose the lookback window into an interpretable forecast.  It is
    trained end-to-end without hand-crafted features.

    Parameters
    ----------
    train   : training series (must have a regular DatetimeIndex for darts)
    horizon : number of steps to forecast; used as output_chunk_length at model
              creation time
    params  : optional dict with keys:
        input_chunk_length    (int, default 14)  : lookback window in observations
        output_chunk_length   (int, default 14)  : steps predicted per forward pass;
                                                   horizons longer than this are handled
                                                   internally by darts via autoregression,
                                                   so the same tuned value is valid at
                                                   both val and test time
        n_stacks              (int, default 2)   : number of basis-expansion stacks
        n_blocks              (int, default 2)   : residual blocks per stack
        n_layers              (int, default 2)   : FC layers inside each block
        layer_width           (int, default 64)  : width of each FC layer
        expansion_coefficient (int, default 32)  : Fourier/polynomial expansion dim
                                                   (higher = more flexible basis)
        dropout               (float, default 0.0): dropout rate applied inside blocks
        lr                    (float, default 1e-3): Adam learning rate
        batch_size            (int, default 32)  : training batch size
        epochs                (int, default 50)  : training epochs
        seed                  (int, default 0)   : random seed

    Notes
    -----
    - The series is z-score scaled with darts Scaler before fitting and
      inverse-transformed after prediction.
    - output_chunk_length is a tunable hyperparameter (not tied to horizon) so the
      same architecture is used at val-tuning time and at test-refit time.  darts
      handles horizons longer than output_chunk_length internally via autoregression.
    - output_chunk_length is clamped to min(output_chunk_length, len(train) // 2)
      to avoid darts raising when the series is very short.
    """
    p                     = params or {}
    input_chunk_length    = int(p.get("input_chunk_length",    14))
    output_chunk_length   = int(p.get("output_chunk_length",   14))
    output_chunk_length   = min(output_chunk_length, max(1, len(train) // 2))
    n_stacks              = int(p.get("n_stacks",               2))
    n_blocks              = int(p.get("n_blocks",               2))
    n_layers              = int(p.get("n_layers",               2))
    layer_width           = int(p.get("layer_width",           64))
    expansion_coefficient = int(p.get("expansion_coefficient", 32))
    dropout               = float(p.get("dropout",            0.0))
    lr                    = float(p.get("lr",                 1e-3))
    batch_size            = int(p.get("batch_size",            32))
    epochs                = int(p.get("epochs",                50))
    seed                  = int(p.get("seed",                   0))

    ts      = TimeSeries.from_series(train.astype("float32"))
    scaler  = DartsScaler()
    ts      = scaler.fit_transform(ts)

    model = NBEATSModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        num_stacks=n_stacks,
        num_blocks=n_blocks,
        num_layers=n_layers,
        layer_widths=layer_width,
        expansion_coefficient_dim=expansion_coefficient,
        dropout=dropout,
        optimizer_kwargs={"lr": lr},
        batch_size=batch_size,
        n_epochs=epochs,
        random_state=seed,
    )
    model.fit(ts)
    pred = scaler.inverse_transform(model.predict(horizon))
    return pred.values().squeeze(-1).astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# Default parameter grids
# ─────────────────────────────────────────────────────────────────────────────

SEASONAL_NAIVE_GRID: list[dict] = [
    {"season": s} for s in [7, 14, 30]
]

# ETS: cross error ∈ {add, mul} × trend ∈ {None, add, mul} × damped × seasonal
ETS_GRID: list[dict] = [
    {"error": e, "trend": t, "damped_trend": d, "seasonal": s, "seasonal_periods": 7}
    for e, t, d, s in product(
        ["add", "mul"],
        [None, "add", "mul"],
        [False, True],
        [None, "add", "mul"],
    )
    if not (t is None and d is True)   # damped_trend requires a trend component
]

SARIMAX_GRID: list[dict] = [
    {"order": order, "seasonal_order": seas}
    for order in [(1, 0, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1)]
    for seas  in [(0, 0, 0, 0), (1, 0, 0, 7), (0, 1, 1, 7), (1, 1, 1, 7)]
]

RIDGE_GRID: list[dict] = [
    {"lags": lags, "alpha": alpha, "add_time_features": True}
    for lags  in [7, 14, 28]
    for alpha in [0.01, 0.1, 1.0, 10.0]
]

# GRU grid is kept small because each candidate requires a full training run
GRU_GRID: list[dict] = [
    {
        "n_steps": ns, "n_future": nf,
        "hidden_size": hs, "num_layers": nl,
        "epochs": 50, "lr": 1e-3, "batch_size": 32, "seed": 0,
    }
    for ns, nf in [(14, 7), (28, 14)]
    for hs, nl in [(32, 1), (64, 2)]
]

# N-BEATS grid is kept small for the same reason as GRU
NBEATS_GRID: list[dict] = [
    {
        "input_chunk_length": icl,
        "n_stacks": 2, "n_blocks": 2, "n_layers": 2,
        "layer_width": lw, "expansion_coefficient": 32,
        "dropout": 0.0, "lr": 1e-3, "batch_size": 32, "epochs": 50, "seed": 0,
    }
    for icl in [7, 14, 28]
    for lw  in [64, 128]
]


# ─────────────────────────────────────────────────────────────────────────────
# Large parameter grids  (thorough tuning — slower)
# ─────────────────────────────────────────────────────────────────────────────

SEASONAL_NAIVE_GRID_LARGE: list[dict] = [
    {"season": s} for s in [7, 14, 21, 28, 30]
]

# Same component combinations as the small grid, but seasonal_periods also
# varies over [7, 14, 30] whenever a seasonal component is active.
ETS_GRID_LARGE: list[dict] = [
    {"error": e, "trend": t, "damped_trend": d, "seasonal": s, "seasonal_periods": sp}
    for e, t, d, s in product(
        ["add", "mul"],
        [None, "add", "mul"],
        [False, True],
        [None, "add", "mul"],
    )
    if not (t is None and d is True)
    for sp in ([7] if s is None else [7, 14, 30])   # period only matters when seasonal ≠ None
]

SARIMAX_GRID_LARGE: list[dict] = [
    {"order": order, "seasonal_order": seas}
    for order in [(1, 0, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1),
                  (2, 1, 0), (0, 1, 2), (2, 1, 2)]
    for seas  in [(0, 0, 0, 0), (1, 0, 0, 7), (0, 1, 1, 7), (1, 1, 1, 7),
                  (1, 0, 0, 14)]
]

RIDGE_GRID_LARGE: list[dict] = [
    {"lags": lags, "alpha": alpha, "add_time_features": True}
    for lags  in [7, 14, 21, 28, 42]
    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
]

GRU_GRID_LARGE: list[dict] = [
    {
        "n_steps": ns, "n_future": nf,
        "hidden_size": hs, "num_layers": nl,
        "epochs": 50, "lr": 1e-3, "batch_size": 32, "seed": 0,
    }
    for ns, nf in [(7, 7), (14, 7), (14, 14), (28, 14)]
    for hs, nl in [(32, 1), (64, 2), (128, 2)]
]

NBEATS_GRID_LARGE: list[dict] = [
    {
        "input_chunk_length": icl,
        "n_stacks": ns, "n_blocks": 2, "n_layers": 2,
        "layer_width": lw, "expansion_coefficient": 32,
        "dropout": 0.0, "lr": 1e-3, "batch_size": 32, "epochs": 50, "seed": 0,
    }
    for icl in [7, 14, 28]
    for ns  in [2, 3]
    for lw  in [64, 128, 256]
]


# ─────────────────────────────────────────────────────────────────────────────
# Tuning and pipeline
# ─────────────────────────────────────────────────────────────────────────────

def tune_on_val(
    fn, train: pd.Series, val: pd.Series, grid: list
) -> tuple[Optional[dict], float]:
    """Grid-search *fn* over *grid*; return (best_params, best_val_MSE).

    Each candidate is evaluated by calling fn(train, len(val), params) and
    computing MSE against val.  Candidates that raise an exception or produce
    non-finite MSE are silently skipped.  Returns (None, inf) if no candidate
    succeeds.
    """
    y_val                   = val.to_numpy()
    best_params, best_mse   = None, float("inf")
    for params in grid:
        try:
            yhat = fn(train, len(val), params)
            mse  = mean_squared_error(y_val, yhat)
            if np.isfinite(mse) and mse < best_mse:
                best_mse, best_params = mse, params
        except Exception:
            pass
    return best_params, best_mse


def run_pipeline(
    train: pd.Series,
    val: pd.Series,
    test: pd.Series,
    label: str,
    models: Optional[list[str]] = None,
    tuning: str = "small",
) -> dict[str, np.ndarray]:
    """Tune each model on val, refit on train+val, evaluate on test.

    Parameters
    ----------
    train  : training series
    val    : validation series used only for hyperparameter selection
    test   : held-out test series used only for final evaluation
    label  : string printed in progress messages (e.g. "ConcurrentCases")
    models : list of model names to run; None runs all default models.
             Default set: "naive", "seasonal_naive", "ets", "sarimax",
                          "ridge", "gru", "nbeats".
             "naive_recent" is excluded from the default because it is an
             oracle baseline (uses ground-truth test values as input) and
             is not meaningful in batch evaluation; pass it explicitly if
             needed for reference purposes.
    tuning : "small" (default) or "large".
             "small" uses compact grids (fast, few candidates).
             "large" uses expanded grids with more hyperparameter combinations.

    Returns
    -------
    dict mapping model name → np.ndarray of test-set predictions (shape: (horizon,))
    """
    if tuning not in ("small", "large"):
        raise ValueError(f"tuning must be 'small' or 'large', got {tuning!r}")

    ALL_MODELS = [
        "naive", "seasonal_naive",
        "ets", "sarimax", "ridge", "gru", "nbeats",
    ]
    # naive_recent is always valid if requested explicitly
    VALID_MODELS = ALL_MODELS + ["naive_recent"]
    if models is None:
        models = ALL_MODELS
    else:
        unknown = [m for m in models if m not in VALID_MODELS]
        if unknown:
            raise ValueError(f"Unknown model(s): {unknown}. Choose from {VALID_MODELS}")

    grids = {
        "seasonal_naive": SEASONAL_NAIVE_GRID       if tuning == "small" else SEASONAL_NAIVE_GRID_LARGE,
        "ets":            ETS_GRID                  if tuning == "small" else ETS_GRID_LARGE,
        "sarimax":        SARIMAX_GRID              if tuning == "small" else SARIMAX_GRID_LARGE,
        "ridge":          RIDGE_GRID                if tuning == "small" else RIDGE_GRID_LARGE,
        "gru":            GRU_GRID                  if tuning == "small" else GRU_GRID_LARGE,
        "nbeats":         NBEATS_GRID               if tuning == "small" else NBEATS_GRID_LARGE,
    }

    train_val = pd.concat([train, val])
    horizon   = len(test)
    y_test    = test.to_numpy()
    preds: dict[str, np.ndarray] = {}

    def _tune_and_fit(model_name: str, fn, grid: list) -> np.ndarray:
        print(f"[{label}] Tuning {model_name} ({len(grid)} candidates) …")
        best, val_mse = tune_on_val(fn, train, val, grid)
        print(f"  best: {best}  val_MSE={val_mse:.4f}")
        return fn(train_val, horizon, best)

    if "naive" in models:
        preds["naive"] = forecast_naive(train_val, horizon)

    if "naive_recent" in models:
        preds["naive_recent"] = forecast_naive_recent(train_val, test)

    if "seasonal_naive" in models:
        preds["seasonal_naive"] = _tune_and_fit("seasonal_naive", forecast_seasonal_naive, grids["seasonal_naive"])

    if "ets" in models:
        preds["ets"] = _tune_and_fit("ets", forecast_ets, grids["ets"])

    if "sarimax" in models:
        preds["sarimax"] = _tune_and_fit("sarimax", forecast_sarimax, grids["sarimax"])

    if "ridge" in models:
        preds["ridge"] = _tune_and_fit("ridge", forecast_ridge, grids["ridge"])

    if "gru" in models:
        preds["gru"] = _tune_and_fit("gru", forecast_gru, grids["gru"])

    if "nbeats" in models:
        preds["nbeats"] = _tune_and_fit("nbeats", forecast_nbeats, grids["nbeats"])

    # ── test metrics table ───────────────────────────────────────────────────
    print(f"\n{'Model':<16}  {'MSE':>12}  {'MAE':>10}")
    print("-" * 44)
    for name, yhat in preds.items():
        try:
            mse = mean_squared_error(y_test, yhat)
            mae = mean_absolute_error(y_test, yhat)
            print(f"{name:<16}  {mse:>12.4f}  {mae:>10.4f}")
        except Exception as e:
            print(f"{name:<16}  ERROR: {e}")

    return preds
