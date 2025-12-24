from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error


def rmsle(y_true, y_pred, clip: float = 0.0) -> float:
    """
    Root Mean Squared Log Error (Kaggle metric).
    Works with numpy arrays, Series, or DataFrames.
    Clips predictions (and targets) at `clip` to avoid negative values.
    """
    
    if isinstance(y_true, (pd.Series, pd.DataFrame)):
        y_true = y_true.to_numpy()
    if isinstance(y_pred, (pd.Series, pd.DataFrame)):
        y_pred = y_pred.to_numpy()

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if clip is not None:
        y_true = np.clip(y_true, clip, None)
        y_pred = np.clip(y_pred, clip, None)

    return mean_squared_log_error(y_true, y_pred) ** 0.5


def time_train_valid_split(index, valid_start):
    """
    Simple cutoff split by date.

    Parameters
    ----------
    index : pandas Index (PeriodIndex or DatetimeIndex)
        Index of the time axis (e.g., y.index).
    valid_start : str or Period/Timestamp
        First date of validation set (inclusive).

    Returns
    -------
    train_mask, valid_mask : np.ndarray(bool), np.ndarray(bool)
    """
    if isinstance(index, pd.PeriodIndex):
        valid_start = pd.Period(valid_start, freq=index.freqstr)
    else:
        valid_start = pd.Timestamp(valid_start)

    train_mask = index < valid_start
    valid_mask = index >= valid_start
    return train_mask, valid_mask


def rolling_origin_splits(index, start, horizon: int, step: int = 7, n_splits: int = 4):
    """
    Rolling-origin evaluation for time series.

    Each split:
      train = [index[0] .. cutoff]
      valid = (cutoff+1 .. cutoff+horizon)

    Parameters
    ----------
    index : PeriodIndex or DatetimeIndex (sorted)
    start : str
        First cutoff date for the first split.
    horizon : int
        Number of days in validation window (for Kaggle, usually 16).
    step : int
        How many days to move the cutoff each split.
    n_splits : int
        Number of splits.

    Yields
    ------
    (train_mask, valid_mask)
    """
    if isinstance(index, pd.PeriodIndex):
        start = pd.Period(start, freq=index.freqstr)
    else:
        start = pd.Timestamp(start)

    index = pd.Index(index)

    # convert to positions via boolean masks per split
    for k in range(n_splits):
        cutoff = start + k * step
        train_mask = index <= cutoff
        valid_mask = (index > cutoff) & (index <= cutoff + horizon)
        yield train_mask.to_numpy(), valid_mask.to_numpy()
