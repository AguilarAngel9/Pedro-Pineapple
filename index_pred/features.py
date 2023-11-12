# Features library.
# Author: @THEFFTKID

import pandas as pd
import numpy as np

def n_rolling_mean(
    x_: pd.Series, n: int
) -> pd.Series:
    return x_.rolling(window=n).mean().fillna(0)

def weigted_n_moving_avg(
    x_: pd.Series, n: int    
) -> pd.Series:
    weights = np.arange(1, n + 1)
    wma = x_.rolling(window=n).apply(
        lambda x: np.dot(x, weights) / weights.sum(),
        raw=True
    ).fillna(0)
    return wma

def momentum(x_: pd.Series, n: int) -> pd.Series:
    """
    """
    hare = np.arange(n, len(x_))
    v_ = [0 for i in range(n)]
    for i in hare:
        v_.append(
            x_[i] - x_[i - n]
        )
    return pd.Series(v_)