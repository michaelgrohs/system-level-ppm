import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def trim_tail_pct(series: pd.Series, pct: float = 0.10) -> pd.Series:
    """Drop the last `pct` fraction of observations (default 10%)."""
    keep = max(1, int(len(series) * (1 - pct)))
    return series.iloc[:keep]


def trim_tail_magnitude(
    series: pd.Series,
    k: float = 1.5,
    window: int = 7,
) -> pd.Series:
    """Drop a fading-out tail using adaptive moving statistics.

    Threshold at t: expanding_mean[t] - k * expanding_std[t].
    Cuts after the last index where the rolling mean exceeds this threshold.
    """
    rolling_mean   = series.rolling(window=window, min_periods=1).mean()
    expanding_mean = series.expanding(min_periods=1).mean()
    expanding_std  = series.expanding(min_periods=1).std().fillna(0)
    threshold      = expanding_mean - k * expanding_std
    above = rolling_mean[rolling_mean >= threshold]
    if above.empty:
        return series
    cut = series.index.get_loc(above.index[-1])
    return series.iloc[: cut + 1]


def trim_tail_peak(
    series: pd.Series,
    frac: float = 0.60,
    window: int = 7,
) -> pd.Series:
    """Drop a fading-out tail by comparing to the global peak.

    Threshold: frac * series.max() — a fixed, stable anchor that the tail
    can't drag down.  Cuts after the last index where the rolling mean still
    exceeds this fraction of the peak.  Tune `frac` between 0 and 1:
    higher = more aggressive trimming.
    """
    peak      = series.max()
    threshold = frac * peak
    rolling   = series.rolling(window=window, min_periods=1).mean()
    above = rolling[rolling >= threshold]
    if above.empty:
        return series
    cut = series.index.get_loc(above.index[-1])
    return series.iloc[: cut + 1]


def apply_trim(series: pd.Series, method: str, **kw) -> pd.Series:
    if method == "pct":
        return trim_tail_pct(series, pct=kw.get("pct", 0.10))
    if method == "magnitude":
        return trim_tail_magnitude(series, k=kw.get("k", 1.5), window=kw.get("window", 7))
    if method == "peak":
        return trim_tail_peak(series, frac=kw.get("frac", 0.60), window=kw.get("window", 7))
    return series   # method=None → no trimming

class Split3WayConfig:
    """
    Three-way split configuration for train/validation/test set.

    """
    train_frac: float = 0.7
    val_frac: float = 0.1
    test_frac: float = 0.2

    def __init__(self, train_frac, val_frac, test_frac):
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        s = self.train_frac + self.val_frac + self.test_frac
        if s != 1.0:
            raise ValueError(f"train_frac+val_frac+test_frac must sum to 1.0; got {s}")
        if min(self.train_frac, self.val_frac, self.test_frac) <= 0:
            raise ValueError("All split fractions must be positive.")

def split_timeseries(ts: pd.Series, cfg: Split3WayConfig):
    n = len(ts)
    train_end = int(n * cfg.train_frac)
    val_end   = train_end + int(n * cfg.val_frac)
    return ts.iloc[:train_end], ts.iloc[train_end:val_end], ts.iloc[val_end:]