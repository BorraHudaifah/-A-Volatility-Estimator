import math
import numpy as np


def get_estimator(price_data, lambda_param=0.94, trading_periods=252, clean=True):
    """
    Exponentially Weighted Moving Average (EWMA) Volatility Estimator
    
    Captures recent market shocks faster than standard rolling window estimators
    by applying exponential weights to historical returns.
    
    Parameters
    ----------
    price_data :  pandas.DataFrame
        DataFrame with 'Close' column
    lambda_param : float, optional
        Decay factor (default: 0.94 for daily data)
        Higher values give more weight to older observations
    trading_periods : int, optional
        Trading periods per year (default: 252 for daily data)
    clean : bool, optional
        If True, drop NaN values (default: True)
    
    Returns
    -------
    pandas.Series
        EWMA volatility estimates annualized
    """
    log_return = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
    
    # Calculate squared returns
    squared_returns = log_return ** 2
    
    # Initialize EWMA variance with first squared return
    ewma_var = squared_returns.copy()
    
    # Apply exponential weighting:  Var_t = lambda * Var_{t-1} + (1-lambda) * r_t^2
    for i in range(1, len(squared_returns)):
        if not np.isnan(squared_returns.iloc[i-1]) and not np.isnan(ewma_var.iloc[i-1]):
            ewma_var.iloc[i] = (lambda_param * ewma_var.iloc[i-1] + 
                               (1 - lambda_param) * squared_returns.iloc[i])
    
    # Convert variance to volatility and annualize
    result = np.sqrt(ewma_var) * math.sqrt(trading_periods)
    
    if clean:
        return result.dropna()
    else:
        return result