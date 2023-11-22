import pandas as pd
import dynamic_threshold
import matplotlib.pyplot as plt

from typing import Tuple, Union, List
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix
)


def data_splitter(
    raw_data: pd.DataFrame,
    proportion: int = 0.7,
    init: Union[int, Tuple[int, int]] = None,
    end: Union[int, Tuple[int, int]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the raw time series data set.
    """
    if isinstance(init, int) and isinstance(end, int):
        train = raw_data.iloc[:init]
        test = raw_data.iloc[end:]

    if isinstance(init, tuple) and isinstance(end, tuple):
        train = raw_data.iloc[init[0]:init[1]]
        test = raw_data.iloc[end[0]:end[1]]

    if not init and not end:
        splitter = round(raw_data.shape[0] * proportion)
        train, test = raw_data.iloc[:splitter], raw_data.iloc[splitter:]

    return train, test


def evaluation_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    target_names: List[Union[int, str, float]]
) -> Union[str, dict]:
    """
    Creates the confusion matrix from Scikit-learn.
    """
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=target_names
    )
    disp.plot()
    plt.show()

    report = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        target_names=target_names,
        output_dict=True
    )
    return report


def create_labels(
    x: pd.Series,
    labels: List[Union[str, float, int]],
    perc_bounds: List[float],
    override_plot: bool = False
) -> Tuple[pd.Series, pd.Series]:
    """
    Create the labels based on a given pd.Series.
    """

    # Limit for bins.
    # Relative differences.
    relative_diff = x.pct_change(periods=1).fillna(value=0)

    # Percentual.
    perc_relative_diff = relative_diff * 100

    # Cut labels.
    threshold_up, threshold_low = dynamic_threshold.define_threshold(
        df=perc_relative_diff,
        lower_bound=perc_bounds[0],
        upper_bound=perc_bounds[1],
        override_plot=override_plot
    )
    bins = [-float('inf'), threshold_low, threshold_up, float('inf')]

    all_labels = pd.cut(
        x=perc_relative_diff,
        bins=bins,
        labels=labels,
        right=False
    )

    return all_labels, perc_relative_diff
