# Dynamic Volatility Estimator

A comprehensive Python tool for computing, comparing, and analyzing different volatility measures on financial assets.  This tool extends the `jasonstrimpel/volatility-trading` framework with advanced dynamic volatility estimation and macro event analysis. 

## Features

### 1. **Multiple Volatility Measures**
   - **Historical Volatility (Rolling Window)**: Standard deviation of log returns over fixed windows (20, 60, 120 days)
   - **Exponentially Weighted Moving Average (EWMA)**: Responsive volatility that emphasizes recent market shocks
   - **Multi-Window Comparison**: Side-by-side analysis of different time horizons

### 2. **Macro Event Analysis**
   - Compare realized volatility **before** and **after** macroeconomic events
   - Support for multiple event types (CPI releases, FOMC meetings, earnings announcements, etc.)
   - Statistical summary of event impacts by type

### 3. **Visualization Tools**
   - Multi-window volatility comparison plots
   - Rolling window vs.  EWMA comparison
   - Pre/post event volatility analysis charts
   - Volatility distribution histograms

### 4. **Modular Architecture**
   - Clean separation of concerns (estimation, analysis, visualization)
   - Extensible design for future models (GARCH, regime-switching, etc.)
   - Easy integration with existing volest framework

## Installation

### Dependencies
```bash
pip install pandas numpy scipy yfinance matplotlib seaborn statsmodels