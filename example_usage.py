"""
Example usage of the Dynamic Volatility Estimator. 

Demonstrates:
1. Calculating historical volatility with multiple rolling windows
2. Computing EWMA volatility for shock detection
3. Analyzing volatility changes around macro events
4. Visualizing results
"""

import pandas as pd
import yfinance as yf
from dynamic_volatility import DynamicVolatilityEstimator
from MacroEventAnalyzer import MacroEventAnalyzer
from visualization import VolatilityVisualizer


def main():
    """
    Main example:  Analyze volatility of SPY (S&P 500 ETF)
    """
    
    print("=" * 60)
    print("Dynamic Volatility Estimator - Example Analysis")
    print("=" * 60)
    
    # 1. FETCH DATA
    print("\n[1] Fetching historical price data for SPY (5 years)...")
    spy = yf.Ticker('SPY')
    price_data = spy.history(period="5y")
    price_data = price_data[['Open', 'High', 'Low', 'Close']]
    print(f"Data shape: {price_data.shape}")
    print(f"Date range: {price_data.index[0]. date()} to {price_data.index[-1].date()}")
    
    # 2. INITIALIZE VOLATILITY ESTIMATOR
    print("\n[2] Initializing Dynamic Volatility Estimator...")
    estimator = DynamicVolatilityEstimator(price_data, trading_periods=252)
    
    # 3. CALCULATE MULTI-WINDOW VOLATILITY
    print("\n[3] Calculating rolling window volatility (20, 60, 120 days)...")
    multi_vol = estimator.multi_window_volatility(windows=[20, 60, 120])
    print("\nLatest volatility estimates:")
    print(multi_vol.tail())
    
    # 4. CALCULATE EWMA VOLATILITY
    print("\n[4] Calculating EWMA volatility (lambda=0.94)...")
    ewma_vol = estimator. ewma_volatility(lambda_param=0.94)
    print(f"\nEWMA volatility - Latest 5 values:")
    print(ewma_vol.tail())
    
    # 5. COMPARE ALL MEASURES
    print("\n[5] Comparing all volatility measures...")
    comparison = estimator.compare_volatility_measures(windows=[20, 60, 120], lambda_param=0.94)
    print("\nComparison DataFrame (latest 5 rows):")
    print(comparison.tail())
    
    # Print summary statistics
    print("\n[6] Summary Statistics:")
    print("\n" + comparison.describe().to_string())
    
    # 7. MACRO EVENT ANALYSIS
    print("\n[7] Analyzing volatility around macro events...")
    
    # Define some example macro events
    events = [
        {
            'date': '2023-03-22',
            'name': 'FOMC Meeting',
            'type': 'monetary_policy'
        },
        {
            'date': '2023-09-20',
            'name': 'FOMC Decision',
            'type': 'monetary_policy'
        },
        {
            'date': '2024-01-31',
            'name': 'CPI Release',
            'type':  'inflation'
        },
    ]
    
    analyzer = MacroEventAnalyzer(price_data, ewma_vol)
    event_impacts = analyzer.event_impact_analysis(events, pre_window_days=30, post_window_days=30)
    
    print("\nEvent Impact Analysis:")
    print(event_impacts. to_string())
    
    # Summary by event type
    print("\n[8] Summary Statistics by Event Type:")
    summary_stats = analyzer.event_summary_statistics(event_impacts)
    for event_type, stats in summary_stats.items():
        print(f"\n{event_type. replace('_', ' ').title()}:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}:  {value}")
    
    # 9. VISUALIZATION
    print("\n[9] Generating visualizations...")
    visualizer = VolatilityVisualizer()
    
    # Plot multi-window comparison
    fig1, ax1 = visualizer. plot_multi_window_volatility(
        multi_vol,
        title="SPY:  Multi-Window Volatility (20, 60, 120 days)"
    )
    print("✓ Multi-window volatility plot saved")
    
    # Plot rolling window vs EWMA
    fig2, ax2 = visualizer.plot_volatility_vs_ewma(
        multi_vol['vol_20d'],
        ewma_vol,
        window_name='20d'
    )
    print("✓ Rolling window vs EWMA plot saved")
    
    # Plot event impacts
    fig3, ax3 = visualizer.plot_event_impact(event_impacts)
    print("✓ Event impact plot saved")
    
    # Plot volatility distribution
    fig4, ax4 = visualizer.plot_volatility_distribution(ewma_vol)
    print("✓ Volatility distribution plot saved")
    
    # 10. SAVE RESULTS
    print("\n[10] Saving results to CSV...")
    comparison.to_csv('spy_volatility_comparison.csv')
    event_impacts.to_csv('spy_event_impacts.csv', index=False)
    print("✓ Results saved")
    
    print("\n" + "=" * 60)
    print("Analysis complete!  Check output CSV files and plots.")
    print("=" * 60)
    
    return estimator, comparison, event_impacts, visualizer


if __name__ == '__main__':
    estimator, comparison, events, visualizer = main()