"""
Visualization utilities for volatility analysis. 
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class VolatilityVisualizer: 
    """
    Visualization tools for volatility estimators and event analysis.
    """
    
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer with style.
        
        Parameters
        ----------
        style : str, optional
            Matplotlib style (default: 'seaborn-v0_8-darkgrid')
        """
        try:
            plt.style.use(style)
        except: 
            plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_multi_window_volatility(self, volatility_df, figsize=(14, 6), title=None):
        """
        Plot volatility estimates from multiple rolling windows.
        
        Parameters
        ----------
        volatility_df : pandas.DataFrame
            Output from multi_window_volatility() or compare_volatility_measures()
        figsize : tuple, optional
            Figure size (default: (14, 6))
        title : str, optional
            Plot title
        
        Returns
        -------
        tuple
            (figure, axes)
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for col in volatility_df.columns:
            ax.plot(volatility_df.index, volatility_df[col], label=col. replace('_', ' ').title(), linewidth=2)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Volatility (Annualized)', fontsize=12)
        ax.set_title(title or 'Multi-Window Volatility Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_volatility_vs_ewma(self, vol_window_df, ewma_df, window_name='20d', figsize=(14, 6)):
        """
        Compare rolling window volatility with EWMA. 
        
        Parameters
        ----------
        vol_window_df : pandas.Series
            Rolling window volatility
        ewma_df : pandas.Series
            EWMA volatility
        window_name : str, optional
            Name of rolling window (default: '20d')
        figsize : tuple, optional
            Figure size (default: (14, 6))
        
        Returns
        -------
        tuple
            (figure, axes)
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(vol_window_df.index, vol_window_df. values, label=f'{window_name} Rolling Window', 
                linewidth=2, alpha=0.8)
        ax.plot(ewma_df. index, ewma_df.values, label='EWMA', 
                linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Volatility (Annualized)', fontsize=12)
        ax.set_title('Rolling Window vs EWMA Volatility', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_event_impact(self, event_impacts_df, figsize=(14, 6)):
        """
        Visualize pre/post event volatility comparison.
        
        Parameters
        ----------
        event_impacts_df : pandas.DataFrame
            Output from MacroEventAnalyzer.event_impact_analysis()
        figsize : tuple, optional
            Figure size (default: (14, 6))
        
        Returns
        -------
        tuple
            (figure, axes)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot absolute volatility
        x = np.arange(len(event_impacts_df))
        width = 0.35
        
        ax1.bar(x - width/2, event_impacts_df['pre_volatility'], width, 
                label='Pre-Event', alpha=0.8)
        ax1.bar(x + width/2, event_impacts_df['post_volatility'], width, 
                label='Post-Event', alpha=0.8)
        
        ax1.set_xlabel('Event', fontsize=12)
        ax1.set_ylabel('Volatility (Annualized)', fontsize=12)
        ax1.set_title('Pre vs Post-Event Volatility', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"Event {i+1}" for i in range(len(event_impacts_df))], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot volatility change percentage
        colors = ['green' if x > 0 else 'red' for x in event_impacts_df['volatility_change_pct']]
        ax2.bar(x, event_impacts_df['volatility_change_pct'], color=colors, alpha=0.7)
        
        ax2.set_xlabel('Event', fontsize=12)
        ax2.set_ylabel('Volatility Change (%)', fontsize=12)
        ax2.set_title('Volatility Change Around Events', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"Event {i+1}" for i in range(len(event_impacts_df))], rotation=45)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    def plot_volatility_distribution(self, volatility_series, figsize=(12, 5), bins=50):
        """
        Plot distribution of volatility estimates.
        
        Parameters
        ----------
        volatility_series : pandas.Series
            Volatility series to analyze
        figsize : tuple, optional
            Figure size (default: (12, 5))
        bins : int, optional
            Number of histogram bins (default: 50)
        
        Returns
        -------
        tuple
            (figure, axes)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        ax1.hist(volatility_series. dropna(), bins=bins, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Volatility (Annualized)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Volatility Distribution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Time series
        ax2.plot(volatility_series.index, volatility_series.values, linewidth=1. 5, alpha=0.8)
        ax2.fill_between(volatility_series. index, volatility_series.values, alpha=0.3)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Volatility (Annualized)', fontsize=12)
        ax2.set_title('Volatility Over Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, (ax1, ax2)