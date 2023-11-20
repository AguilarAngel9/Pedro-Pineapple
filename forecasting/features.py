# Features library.
# Author: @THEFFTKID

import pandas as pd
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

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
        x_: pd.Series,
        n: int
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

def plot_tdist(
        dof: float,
        loc: float,
        scale: float,
        data:pd.Series
):
    x = np.linspace(min(data), max(data), 1000)
    y = t.pdf(x, dof, loc, scale)
    plt.hist(data, bins=30, density=True, alpha=0.5, label='Data Histogram')
    plt.plot(x, y, label='Fitted t-Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('t-Distribution Fit to Data')
    plt.legend()
    plt.show()


def define_threshold(
    df:pd.Series,
    lower_bound: float,
    upper_bound: float,
    override_plot: bool = False        
)-> tuple:
    
    '''
    This function fits a t-distribution to a pandas Series (df) and calculates upper
    and lower thresholds based on specified percentiles (upper_bound and lower_bound).
    It utilizes the t-distribution's percent point function (ppf) to determine these 
    thresholds, effectively identifying the range of values corresponding to the given 
    percentiles in the data's distribution

    Parameters: 

    - df (pd.Series): The data series to which the t-distribution is to be fitted.
    - lower_bound (float): The lower percentile for calculating the lower threshold. 
      Value should be between 0 and 1.
    - upper_bound (float): The upper percentile for calculating the upper threshold. 
      Value should be between 0 and 1.
    - override_plot (bool): If set to True, a plot of the t-distribution is generated.
    
    Returns:
    - tuple: A tuple containing the calculated upper and lower thresholds.
    '''

    dof, loc, scale = t.fit(df)

    if override_plot:
        plot_tdist(dof,loc,scale,df)

    threshold_upper = t.ppf(upper_bound, dof, loc, scale)
    threshold_lower = t.ppf(lower_bound, dof, loc, scale)

    return threshold_upper, threshold_lower