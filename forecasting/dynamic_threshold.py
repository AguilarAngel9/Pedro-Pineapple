import pandas as pd
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt


def plot_tdist(
    dof: float,
    loc: float,
    scale: float,
    data: pd.Series
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
    df: pd.Series,
    lower_bound: float,
    upper_bound: float,
    override_plot: bool = False
) -> tuple:
    '''
    Fits a t-distribution to a pandas Series (df) and calculates upper
    and lower thresholds based on specified percentiles.
    It utilizes t-distribution's percent point function (ppf) to determine
    thresholds, effectively identifying the range of values corresponding
    percentiles in the data's distribution

    Parameters:

    - df (pd.Series):
        The data series to which the t-distribution is to be fitted.
    - lower_bound (float):
        The lower percentile for calculating the lower threshold.
      Value should be between 0 and 1.
    - upper_bound (float):
      The upper percentile for calculating the upper threshold.
      Value should be between 0 and 1.
    - override_plot (bool):
        If set to True, a plot of the t-distribution is generated.

    Returns:
    - tuple: A tuple containing the calculated upper and lower thresholds.
    '''

    dof, loc, scale = t.fit(df)

    if override_plot:
        plot_tdist(dof, loc, scale, df)

    if upper_bound >= lower_bound:
        upper_threshold = t.ppf(upper_bound, dof, loc, scale)
        lower_threshold = t.ppf(lower_bound, dof, loc, scale)
        if lower_threshold == upper_threshold or lower_threshold > 0:
            lower_threshold = -1 * lower_threshold
    else:
        raise ValueError(
            "Error: bounds are not consecutive. Modify extremities."
        )

    return upper_threshold, lower_threshold
