"""
Macro Event Analyzer for Volatility Estimation

Analyzes realized volatility before and after macroeconomic events
(e.g., CPI releases, central bank meetings) to understand how events
impact market volatility.
"""

import numpy as np
import pandas as pd
from datetime import timedelta


class MacroEventAnalyzer: 
    """
    Analyzes the impact of macroeconomic events on realized volatility. 
    """
    
    def __init__(self, price_data, volatility_series):
        """
        Initialize the analyzer with price data and pre-calculated volatility. 
        
        Parameters
        ----------
        price_data : pandas. DataFrame
            DataFrame with DatetimeIndex and 'Close' column
        volatility_series : pandas.Series
            Pre-calculated volatility series with DatetimeIndex
        """
        self.price_data = price_data
        self.volatility = volatility_series
        
    def calculate_log_returns(self):
        """Calculate log returns from price data."""
        return np.log(self.price_data['Close'] / self.price_data['Close'].shift(1))
    
    def realized_volatility_window(self, start_date, end_date):
        """
        Calculate realized volatility for a specific date window.
        
        Parameters
        ----------
        start_date : datetime or str
            Start date of the window
        end_date :  datetime or str
            End date of the window
        
        Returns
        -------
        float
            Realized volatility (annualized standard deviation of log returns)
        """
        mask = (self.price_data. index >= start_date) & (self.price_data.index <= end_date)
        log_returns = self.calculate_log_returns()[mask]
        
        if len(log_returns) < 2:
            return np.nan
        
        # Annualize by sqrt(252) for daily data
        realized_vol = log_returns.std() * np.sqrt(252)
        return realized_vol
    
    def event_impact_analysis(self, events, pre_window_days=30, post_window_days=30):
        """
        Analyze volatility changes around macro events.
        
        Parameters
        ----------
        events : list of dict
            List of events with structure: 
            [
                {'date': '2024-01-15', 'name': 'CPI Release', 'type': 'inflation'},
                {'date': '2024-02-20', 'name': 'FOMC Meeting', 'type':  'monetary_policy'},
                ...
            ]
        pre_window_days : int, optional
            Days before event to analyze (default: 30)
        post_window_days : int, optional
            Days after event to analyze (default: 30)
        
        Returns
        -------
        pandas.DataFrame
            Summary of pre/post event volatility metrics
        """
        results = []
        
        for event in events:
            event_date = pd.to_datetime(event['date'])
            pre_start = event_date - timedelta(days=pre_window_days)
            post_end = event_date + timedelta(days=post_window_days)
            
            pre_vol = self.realized_volatility_window(pre_start, event_date)
            post_vol = self.realized_volatility_window(event_date, post_end)
            
            vol_change = post_vol - pre_vol
            vol_change_pct = (vol_change / pre_vol * 100) if pre_vol != 0 else np.nan
            
            results.append({
                'event_date': event_date,
                'event_name': event['name'],
                'event_type': event. get('type', 'unknown'),
                'pre_volatility': pre_vol,
                'post_volatility': post_vol,
                'volatility_change': vol_change,
                'volatility_change_pct': vol_change_pct,
                'pre_window_days': pre_window_days,
                'post_window_days': post_window_days
            })
        
        return pd.DataFrame(results).sort_values('event_date')
    
    def event_summary_statistics(self, event_impacts_df):
        """
        Calculate summary statistics for event impacts.
        
        Parameters
        ----------
        event_impacts_df : pandas.DataFrame
            Output from event_impact_analysis()
        
        Returns
        -------
        dict
            Summary statistics by event type
        """
        summary = {}
        
        for event_type in event_impacts_df['event_type'].unique():
            subset = event_impacts_df[event_impacts_df['event_type'] == event_type]
            
            summary[event_type] = {
                'count': len(subset),
                'avg_pre_vol': subset['pre_volatility'].mean(),
                'avg_post_vol': subset['post_volatility'].mean(),
                'avg_vol_change': subset['volatility_change'].mean(),
                'avg_vol_change_pct': subset['volatility_change_pct'].mean(),
                'vol_increase_frequency': (subset['volatility_change'] > 0).sum() / len(subset)
            }
        
        return summary