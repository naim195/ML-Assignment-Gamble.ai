
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
    # assert y_hat.size == y.size
    # # TODO: Write here
    # assert isinstance(y_hat, pd.Series) and isinstance(y, pd.Series)
    # assert y_hat.size>0
    # assert y.size>0
    # assert y_hat.dtype == y.dtype
    # assert y_hat.isna().sum()==0
    # assert y.isna().sum()==0
    # tot_eq = (y_hat==y).sum()
    # tot_sum = y.size
    # return tot_eq/tot_sum
    assert y_hat.size == y.size
    return np.sum(y_hat == y)/y.size

    


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """

    
    
    assert y_hat.size == y.size
    cls_series = pd.Series([cls] * len(y_hat))
    true_positive = np.sum((y == y_hat) & (y == cls_series))
    true_predicted = np.sum(y_hat == cls_series)
    prec = float(true_positive / true_predicted) if true_predicted > 0 else 0.0
    return prec
     
    

def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    # num = ((y==cls) & (y_hat==y)).sum()
    # den = (y==cls).sum()
    # if(den==0):
    #     return 0
    # return num/den
    assert y_hat.size == y.size
    cls_series = pd.Series([cls] * len(y_hat))
    true_positive = np.sum((y == y_hat) & (y == cls_series))
    true_actual = np.sum(y == cls_series)
    rec = float(true_positive / true_actual) if true_actual > 0 else 0.0
    return rec

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    # mean_squared_difference = np.sqrt(((y-y_hat)**2).mean())

    # return mean_squared_difference
    assert y_hat.size == y.size
    return np.sqrt(np.mean((y_hat - y)**2))

def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size
    return np.mean(np.abs(y_hat - y))








