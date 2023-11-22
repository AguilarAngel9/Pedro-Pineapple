# Features library.
# Author: @THEFFTKID

import pandas as pd
import numpy as np


def n_rolling_mean(
    x_: pd.Series, n: int
) -> pd.Series:
    """
    Calculated the N day rolling mean.
    """
    return x_.rolling(window=n).mean().fillna(0)


def weighted_n_moving_avg(
    x_: pd.Series, n: int
) -> pd.Series:
    """
    Calculates the N day weighted rolling mean.
    """
    weights = np.arange(1, n + 1)
    wma = x_.rolling(window=n).apply(
        lambda x: np.dot(x, weights) / weights.sum(),
        raw=True
    ).fillna(0)
    return wma


def momentum(
    x_: pd.Series, n: int
) -> pd.Series:
    """
    Calculates the n day momentum.
    """
    hare = np.arange(n, len(x_))
    # Momentum.
    m = [0] * n
    for i in hare:
        m.append(
            x_[i] - x_[i - n]
        )
    return pd.Series(m)


def tendency_removal(
        df_close: pd.Series,
        n: int
) -> pd.Series:
    '''
    Calculate the closing price minus the n day moving
    average to remove the tendency.
    '''
    n_moving_average = df_close.rolling(
        window=n
    ).mean().fillna(0)
    eliminate_tendency = df_close - n_moving_average
    eliminate_tendency.iloc[:n] = 0
    return eliminate_tendency


def volume_perc_rate_of_change(
        df_volume: pd.Series
) -> pd.Series:
    '''
    Calculate the relative rate of change for the volume. 
    '''
    v_proc = df_volume.pct_change().fillna(0) * 100
    return v_proc


def williams_range(
        data: pd.DataFrame,
        days: int = 14,

):
    '''
    Calculate the Williams Percent Range, a momentum indicator that 
    measures underbought and oversold. 
    '''
    
    highest_high = data['high'].rolling(window=days).max().fillna(0)
    lowest_low = data['low'].rolling(window=days).min().fillna(0)
    is_equal = highest_high[days::] == lowest_low[days::]
    
    williams_prange = (highest_high - data['close']) / (highest_high - lowest_low)*-100
    williams_prange[:days] = 0

    if is_equal.any() == True:
         williams_prange[days::] = williams_prange[days::].replace([np.inf, -np.inf], -100) #Highest_high == lowest_low

    return williams_prange, highest_high, lowest_low


def stochastic_oscillator(
        data:pd.DataFrame,
        days:int = 14
) -> pd.Series:
    s_o = (data['close']-data['n_lowest_low'])/(data['n_highest_high']-data['n_lowest_low'])*100
    s_o[:days] = 0 
    return s_o
