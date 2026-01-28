"""
Unit tests for Dynamic Volatility Estimator modules. 

Run with: pytest tests_volatility. py -v
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

from dynamic_volatility import DynamicVolatilityEstimator
from MacroEventAnalyzer import MacroEventAnalyzer


class TestDynamicVolatilityEstimator(unittest.TestCase):
    """Test suite for DynamicVolatilityEstimator class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests."""
        # Fetch 1 year of SPY data
        spy = yf.Ticker('SPY')
        cls.price_data = spy.history(period="1y")[['Open', 'High', 'Low', 'Close']]
    
    def test_initialization(self):
        """Test estimator initialization."""
        estimator = DynamicVolatilityEstimator(self.price_data)
        self.assertEqual(estimator.trading_periods, 252)
        self.assertIsNotNone(estimator.price_data)
    
    def test_invalid_input_missing_close(self):
        """Test that initialization fails without Close column."""
        bad_data = self.price_data[['Open', 'High', 'Low']].copy()
        with self.assertRaises(ValueError):
            DynamicVolatilityEstimator(bad_data)
    
    def test_invalid_input_wrong_type(self):
        """Test that initialization fails with wrong data type."""
        with self.assertRaises(TypeError):
            DynamicVolatilityEstimator([[1, 2, 3, 4]])
    
    def test_log_returns_calculation(self):
        """Test log returns are calculated correctly."""
        estimator = DynamicVolatilityEstimator(self.price_data)
        log_returns = estimator.log_returns
        
        # Check shape
        self.assertEqual(len(log_returns), len(self.price_data))
        
        # First return should be NaN
        self.assertTrue(np.isnan(log_returns. iloc[0]))
        
        # Subsequent returns should be valid
        self.assertTrue(np.isfinite(log_returns.iloc[1:]).all())
    
    def test_historical_volatility_shape(self):
        """Test historical volatility output shape."""
        estimator = DynamicVolatilityEstimator(self.price_data)
        vol = estimator.historical_volatility(window=30, clean=True)
        
        # Should have fewer values than input due to window
        self.assertLess(len(vol), len(self.price_data))
        
        # All values should be positive
        self. assertTrue((vol > 0).all())
    
    def test_historical_volatility_window_effect(self):
        """Test that larger windows produce smoother volatility."""
        estimator = DynamicVolatilityEstimator(self.price_data)
        vol_20 = estimator.historical_volatility(window=20, clean=True)
        vol_60 = estimator.historical_volatility(window=60, clean=True)
        
        # 60-day volatility should be less volatile (smoother)
        vol_20_vol = vol_20.std()
        vol_60_vol = vol_60.std()
        
        self.assertLess(vol_60_vol, vol_20_vol)
    
    def test_multi_window_volatility(self):
        """Test multi-window volatility calculation."""
        estimator = DynamicVolatilityEstimator(self.price_data)
        windows = [20, 60, 120]
        multi_vol = estimator.multi_window_volatility(windows=windows, clean=True)
        
        # Should have correct number of columns
        self.assertEqual(len(multi_vol.columns), len(windows))
        
        # Check column names
        for window in windows:
            self. assertIn(f'vol_{window}d', multi_vol.columns)
        
        # All values should be positive
        for col in multi_vol.columns:
            self.assertTrue((multi_vol[col] > 0).all())
    
    def test_ewma_volatility_responsiveness(self):
        """Test that EWMA is more responsive than rolling window."""
        estimator = DynamicVolatilityEstimator(self.price_data)
        
        vol_rolling = estimator.historical_volatility(window=30, clean=True)
        ewma_vol = estimator.ewma_volatility(lambda_param=0.94, clean=True)
        
        # Align indices for comparison
        common_idx = vol_rolling.index.intersection(ewma_vol.index)
        vol_rolling = vol_rolling.loc[common_idx]
        ewma_vol = ewma_vol.loc[common_idx]
        
        # EWMA should have higher volatility of volatility (more responsive)
        ewma_vol_of_vol = ewma_vol. std()
        rolling_vol_of_vol = vol_rolling.std()
        
        # EWMA responsiveness should be detectable
        self.assertGreater(ewma_vol_of_vol, 0)
        self.assertGreater(rolling_vol_of_vol, 0)
    
    def test_realized_volatility(self):
        """Test realized volatility calculation for a period."""
        estimator = DynamicVolatilityEstimator(self.price_data)
        
        start_date = self.price_data.index[0]
        end_date = self.price_data.index[0] + timedelta(days=30)
        
        realized_vol = estimator.realized_volatility(start_date, end_date)
        
        # Should be positive and reasonable
        self.assertGreater(realized_vol, 0)
        self.assertLess(realized_vol, 1)  # Annual volatility rarely exceeds 100%
    
    def test_event_volatility_comparison(self):
        """Test event volatility comparison calculation."""
        estimator = DynamicVolatilityEstimator(self.price_data)
        
        # Use a date in the middle of the data
        event_date = self.price_data.index[len(self.price_data) // 2]
        
        result = estimator.event_volatility_comparison(
            event_date,
            pre_days=30,
            post_days=30
        )
        
        # Check structure
        self.assertIn('pre_volatility', result)
        self.assertIn('post_volatility', result)
        self.assertIn('volatility_change', result)
        self.assertIn('volatility_change_pct', result)
        
        # All should be numeric
        self.assertIsInstance(result['pre_volatility'], (float, np.floating))
        self.assertIsInstance(result['post_volatility'], (float, np.floating))


class TestMacroEventAnalyzer(unittest.TestCase):
    """Test suite for MacroEventAnalyzer class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        spy = yf.Ticker('SPY')
        cls.price_data = spy.history(period="2y")[['Open', 'High', 'Low', 'Close']]
        cls.estimator = DynamicVolatilityEstimator(cls.price_data)
        cls.volatility = cls.estimator.ewma_volatility()
    
    def test_analyzer_initialization(self):
        """Test MacroEventAnalyzer initialization."""
        analyzer = MacroEventAnalyzer(self.price_data, self.volatility)
        self.assertIsNotNone(analyzer.price_data)
        self.assertIsNotNone(analyzer.volatility)
    
    def test_realized_volatility_window(self):
        """Test realized volatility calculation for a window."""
        analyzer = MacroEventAnalyzer(self.price_data, self.volatility)
        
        start_date = self.price_data.index[0]
        end_date = self.price_data.index[30]
        
        realized_vol = analyzer.realized_volatility_window(start_date, end_date)
        
        self.assertGreater(realized_vol, 0)
        self.assertLess(realized_vol, 1)
    
    def test_event_impact_analysis(self):
        """Test event impact analysis."""
        analyzer = MacroEventAnalyzer(self.price_data, self. volatility)
        
        # Create test events
        events = [
            {
                'date': self.price_data.index[100]. strftime('%Y-%m-%d'),
                'name': 'Test Event 1',
                'type': 'monetary_policy'
            },
            {
                'date':  self.price_data.index[200].strftime('%Y-%m-%d'),
                'name':  'Test Event 2',
                'type': 'inflation'
            }
        ]
        
        impacts = analyzer.event_impact_analysis(events, pre_window_days=30, post_window_days=30)
        
        # Check structure
        self.assertEqual(len(impacts), 2)
        self.assertIn('event_date', impacts. columns)
        self.assertIn('pre_volatility', impacts. columns)
        self.assertIn('post_volatility', impacts. columns)
        self.assertIn('volatility_change_pct', impacts.columns)
    
    def test_event_summary_statistics(self):
        """Test summary statistics calculation."""
        analyzer = MacroEventAnalyzer(self.price_data, self.volatility)
        
        events = [
            {
                'date': self. price_data.index[100]. strftime('%Y-%m-%d'),
                'name': 'Event 1',
                'type':  'monetary_policy'
            },
            {
                'date': self.price_data.index[150].strftime('%Y-%m-%d'),
                'name': 'Event 2',
                'type': 'monetary_policy'
            }
        ]
        
        impacts = analyzer.event_impact_analysis(events, pre_window_days=30, post_window_days=30)
        summary = analyzer.event_summary_statistics(impacts)
        
        self.assertIn('monetary_policy', summary)
        self.assertIn('count', summary['monetary_policy'])
        self.assertIn('avg_pre_vol', summary['monetary_policy'])


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        spy = yf.Ticker('SPY')
        cls.price_data = spy.history(period="1y")[['Open', 'High', 'Low', 'Close']]
    
    def test_complete_workflow(self):
        """Test complete volatility analysis workflow."""
        # Initialize estimator
        estimator = DynamicVolatilityEstimator(self.price_data)
        
        # Calculate different measures
        vol_20 = estimator.historical_volatility(window=20)
        vol_60 = estimator.historical_volatility(window=60)
        ewma = estimator.ewma_volatility()
        
        # All should have data
        self.assertGreater(len(vol_20), 0)
        self.assertGreater(len(vol_60), 0)
        self.assertGreater(len(ewma), 0)
        
        # Initialize analyzer
        analyzer = MacroEventAnalyzer(self.price_data, ewma)
        
        # Create and analyze events
        events = [
            {
                'date': self.price_data.index[100].strftime('%Y-%m-%d'),
                'name': 'Test Event',
                'type': 'monetary_policy'
            }
        ]
        
        impacts = analyzer.event_impact_analysis(events)
        self.assertEqual(len(impacts), 1)


if __name__ == '__main__':
    unittest.main()