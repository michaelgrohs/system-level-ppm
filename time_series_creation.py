from typing import Literal

import pandas as pd
import matplotlib.pyplot as plt

_VALID_WINDOWS = ("hours", "days", "weeks", "months")

_RANGE_FREQ = {
    "hours":  "h",
    "days":   "D",
    "weeks":  "W-MON",
    "months": "MS",
}

_TITLES_CC = {
    "hours":  "Concurrent Cases per Hour",
    "days":   "Concurrent Cases per Day",
    "weeks":  "Concurrent Cases per Week",
    "months": "Concurrent Cases per Month",
}

_TITLES_TT = {
    "hours":  "Avg Throughput Time per Hour",
    "days":   "Avg Throughput Time per Day",
    "weeks":  "Avg Throughput Time per Week",
    "months": "Avg Throughput Time per Month",
}


def _floor_to_window(ts: pd.Series, window: str) -> pd.Series:
    if window == "hours":
        return ts.dt.floor("h")
    if window == "days":
        return ts.dt.normalize()
    if window == "weeks":
        return ts.dt.to_period("W").dt.start_time
    # months
    return ts.dt.to_period("M").dt.start_time


def create_concurrent_cases_timeseries(
    log: pd.DataFrame,
    time_col: str = "time:timestamp",
    case_col: str = "case:concept:name",
    window: str = "days",
    plot: bool = True,
) -> pd.Series:
    """
    Create a time series of concurrently active cases per time window.

    Parameters
    ----------
    log      : event log as a DataFrame
    time_col : timestamp column name
    case_col : case ID column name
    window   : granularity – one of 'hours', 'days', 'weeks', 'months'
    plot     : whether to plot the result

    Returns
    -------
    pd.Series with the bucket timestamps as index and active-case counts as values
    """
    if window not in _VALID_WINDOWS:
        raise ValueError(f"window must be one of {_VALID_WINDOWS}, got {window!r}")

    t = pd.to_datetime(log[time_col])
    if t.dt.tz is not None:
        t = t.dt.tz_convert(None)

    first_last = (
        log.assign(_t=t)
        .groupby(case_col)["_t"]
        .agg(first="min", last="max")
    )

    first_bucket = _floor_to_window(first_last["first"], window)
    last_bucket  = _floor_to_window(first_last["last"],  window)

    buckets = pd.date_range(
        start=first_bucket.min(),
        end=last_bucket.max(),
        freq=_RANGE_FREQ[window],
    )

    counts = pd.Series(
        [(first_bucket <= b).sum() - (last_bucket < b).sum() for b in buckets],
        index=buckets,
        name="concurrent_cases",
    )

    if plot:
        counts.plot(figsize=(12, 4), title=_TITLES_CC[window])
        plt.xlabel("Date")
        plt.ylabel("Active cases")
        plt.tight_layout()
        plt.show()
        #print(counts.describe())

    return counts


def create_avg_throughtput_time_timeseries(
    log: pd.DataFrame,
    time_col: str = "time:timestamp",
    case_col: str = "case:concept:name",
    window: str = "days",
    fill: Literal["ffill", "interpolate", None] = "ffill",
    plot: bool = True,
) -> pd.Series:
    """
    Create a time series of average throughput time per time window.

    For each bucket, the average throughput time (in hours) is computed
    over all cases whose last event falls within that bucket.
    Buckets with no completing cases are filled according to `fill`.

    Parameters
    ----------
    log      : event log as a DataFrame
    time_col : timestamp column name
    case_col : case ID column name
    window   : granularity – one of 'hours', 'days', 'weeks', 'months'
    fill     : how to handle empty buckets – 'ffill', 'interpolate', or None (keep NaN)
    plot     : whether to plot the result

    Returns
    -------
    pd.Series with bucket timestamps as index and average throughput time (hours) as values
    """
    if window not in _VALID_WINDOWS:
        raise ValueError(f"window must be one of {_VALID_WINDOWS}, got {window!r}")

    t = pd.to_datetime(log[time_col])
    if t.dt.tz is not None:
        t = t.dt.tz_convert(None)

    first_last = (
        log.assign(_t=t)
        .groupby(case_col)["_t"]
        .agg(first="min", last="max")
    )

    first_last["duration_h"] = (
        (first_last["last"] - first_last["first"]).dt.total_seconds() / 3600
    )
    first_last["last_bucket"] = _floor_to_window(first_last["last"], window)

    avg_tt = (
        first_last
        .groupby("last_bucket")["duration_h"]
        .mean()
        .rename("avg_throughput_time_h")
    )

    buckets = pd.date_range(
        start=avg_tt.index.min(),
        end=avg_tt.index.max(),
        freq=_RANGE_FREQ[window],
    )
    avg_tt = avg_tt.reindex(buckets)

    if fill == "ffill":
        avg_tt = avg_tt.ffill()
    elif fill == "interpolate":
        avg_tt = avg_tt.interpolate(method="time")

    if plot:
        avg_tt.plot(figsize=(12, 4), title=_TITLES_TT[window])
        plt.xlabel("Date")
        plt.ylabel("Avg throughput time (hours)")
        plt.tight_layout()
        plt.show()
        #print(avg_tt.describe())

    return avg_tt