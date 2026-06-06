import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.backtester import Backtester
from src.indicators import calculate_indicators
from src.strategy import Strategy
from src.paper_trader import PaperTrader

def test_backtest_signal_at_t_executes_at_t_plus_1_open():
    """Verify that a signal generated at candle t executes at t+1 open."""
    # Create 205 candles to ensure indicators are calculated
    dates = pd.date_range(start="2026-01-01", periods=205, freq="H")
    df = pd.DataFrame(index=dates)
    df['open'] = 100.0
    df['high'] = 100.0
    df['low'] = 100.0
    df['close'] = 100.0
    df['volume'] = 1000.0
    
    df = calculate_indicators(df)
    
    # Set market state bullish
    df['market_state'] = 'bullish'
    
    # Manually configure prices at index 200 (t) to trigger a BUY signal
    # Close > EMA 50, RSI < 70, market_state == bullish
    df.loc[df.index[200], 'close'] = 100.0
    df.loc[df.index[200], 'ema_50'] = 90.0
    df.loc[df.index[200], 'rsi'] = 50.0
    
    # Configure t+1 open (index 201) to verify fill price
    df.loc[df.index[201], 'open'] = 105.0
    df.loc[df.index[201], 'close'] = 106.0
    df.loc[df.index[201], 'high'] = 107.0
    df.loc[df.index[201], 'low'] = 104.0
    df.loc[df.index[201], 'ema_50'] = 90.0
    df.loc[df.index[201], 'rsi'] = 50.0
    
    # Run backtester (Rule-only baseline)
    backtester = Backtester(df, "SPY")
    results = backtester.run_backtest(ml_model=None)
    
    trades = results['trades']
    assert len(trades) > 0
    
    # Find the trade entered
    # The first trade should have entry_price = 105.0 and entry_time = index 201
    entry_trade = trades[0]
    assert entry_trade['entry_price'] == 105.0
    assert entry_trade['entry_time'] == df.index[201]

def test_paper_trader_stale_closed_market():
    """Verify stale/closed market checks force HOLD and block paper execution."""
    pt = PaperTrader("TEST_MOCK_SYMBOL")
    pt.reset_account()
    
    # 1. Closed market check
    signal = {
        "action": "BUY",
        "confidence": 0.85,
        "explanation": "Buy signal"
    }
    latest_row = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}
    
    # Process signal when market is closed
    pt.process_signal(signal, latest_row, "2026-06-06 10:00:00", is_market_open=False, is_data_stale=False)
    
    # Signal action should be forced to HOLD
    assert signal["action"] == "HOLD"
    assert "blocked" in signal["explanation"]
    # Account state should not have any position or queued signal
    assert pt.position is None
    assert pt.queued_signal is None
    
    # 2. Stale market check
    signal_stale = {
        "action": "BUY",
        "confidence": 0.85,
        "explanation": "Buy signal"
    }
    
    # Process signal when data is stale
    pt.process_signal(signal_stale, latest_row, "2026-06-06 11:00:00", is_market_open=True, is_data_stale=True)
    
    # Signal action should be forced to HOLD
    assert signal_stale["action"] == "HOLD"
    assert "blocked" in signal_stale["explanation"]
    assert pt.position is None
    assert pt.queued_signal is None
    
    # Clean up state file
    if os.path.exists(pt.state_path):
        os.remove(pt.state_path)

def test_train_uses_time_series_split():
    """Verify that TimeSeriesSplit is used instead of KFold during training."""
    from src.ml_model import MarketMLModel
    
    X = pd.DataFrame(index=pd.date_range("2026-01-01", periods=50))
    X['feat1'] = 1.0
    y = pd.Series([0, 1] * 25, index=X.index)
    
    model = MarketMLModel("TEST_MOCK")
    
    with patch("src.ml_model.TimeSeriesSplit") as mock_tss:
        mock_split = MagicMock()
        mock_split.split.return_value = [([0], [1])]
        mock_tss.return_value = mock_split
        
        with patch("src.ml_model.XGBClassifier") as mock_xgb, \
             patch("src.ml_model.joblib.dump") as mock_dump:
            
            import numpy as np
            clf_mock = MagicMock()
            clf_mock.predict_proba.side_effect = lambda X: np.array([[0.5, 0.5]] * len(X))
            mock_xgb.return_value = clf_mock
            
            model.train(X, y)
            
            # Confirm TimeSeriesSplit was used
            mock_tss.assert_called_once_with(n_splits=5)
            mock_split.split.assert_called()

def test_rule_only_baseline_no_ml():
    """Verify rule-only baseline generates signals without ML confidence checks."""
    # min_confidence is very high (0.99), so any ML-guided check would block the trade
    strat = Strategy(min_confidence=0.99)
    
    row_buy = {
        'close': 100.0,
        'ema_50': 90.0,
        'rsi': 50.0,
        'market_state': 'bullish'
    }
    
    # Even if ML probs are low, use_ml=False should generate BUY
    res = strat.generate_signal(row_buy, prob_up=0.10, prob_down=0.90, use_ml=False)
    assert res['action'] == "BUY"
    assert "Rule-Only" in res['explanation']
    
    # SELL signal when RSI is overbought (> 80)
    row_sell = {
        'close': 100.0,
        'ema_50': 90.0,
        'rsi': 85.0,
        'market_state': 'bullish'
    }
    res_sell = strat.generate_signal(row_sell, prob_up=0.90, prob_down=0.10, use_ml=False)
    assert res_sell['action'] == "SELL"
    assert "Rule-Only" in res_sell['explanation']
