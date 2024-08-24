
from typing import Union
import pandas as pd
import numpy as np

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    # TODO: Write here
    assert isinstance(y_hat, pd.Series) and isinstance(y, pd.Series)
    assert y_hat.size>0
    assert y.size>0
    assert y_hat.dtype == y.dtype
    assert y_hat.isna().sum()==0
    assert y.isna().sum()==0
    tot_eq = (y_hat==y).sum()
    tot_sum = y.size
    return tot_eq/tot_sum
    


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    num = ((y==cls) & (y_hat==y)).sum()
    den = (y_hat==cls).sum()
    if(den==0):
        return 0
    return num/den

def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    num = ((y==cls) & (y_hat==y)).sum()
    den = (y==cls).sum()
    if(den==0):
        return 0
    return num/den

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    mean_squared_difference = np.sqrt(((y-y_hat)**2).mean())

    return mean_squared_difference

def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    mae = abs(y-y_hat).mean()
    return mae








