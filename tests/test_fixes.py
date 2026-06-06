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


def test_diagnostics_low_sample_robustness():
    """Verify that ModelDiagnostics handles empty predictions, single-class targets, or zero trades robustly."""
    from src.diagnostics import ModelDiagnostics
    
    # Case 1: Empty results
    ml_results_empty = {
        'df': pd.DataFrame(),
        'pred_prob_up': pd.Series(dtype=float),
        'pred_prob_down': pd.Series(dtype=float),
        'trades': []
    }
    diag_empty = ModelDiagnostics(ml_results_empty, {}, "SPY")
    metrics_empty = diag_empty.get_oos_classification_metrics()
    assert metrics_empty['accuracy'] == 'N/A'
    assert metrics_empty['precision'] == 'N/A'
    
    # Case 2: Only 1 class present in targets (e.g., all 0s)
    # Target logic expects rises >= 0.05% over next 4 hours
    dates = pd.date_range("2026-01-01", periods=10, freq="h")
    df = pd.DataFrame(index=dates)
    df['close'] = 100.0 # flat price means no rise, target series will be all 0.0
    
    pred_prob_up = pd.Series([0.5] * 10, index=dates)
    pred_prob_down = pd.Series([0.5] * 10, index=dates)
    
    ml_results_one_class = {
        'df': df,
        'pred_prob_up': pred_prob_up,
        'pred_prob_down': pred_prob_down,
        'trades': []
    }
    diag_one_class = ModelDiagnostics(ml_results_one_class, {}, "SPY")
    metrics_one_class = diag_one_class.get_oos_classification_metrics(threshold=0.60)
    
    assert metrics_one_class['accuracy'] == 1.0 # 0 prediction for all 0 targets -> 100% correct
    assert "No trades predicted" in metrics_one_class['precision']
    assert "Only 1 class present" in metrics_one_class['roc_auc']
    assert "Only 1 class present" in metrics_one_class['pr_auc']
    
    # Case 3: Zero trades in trade diagnostics
    trade_diag = diag_one_class.get_trade_diagnostics()
    assert trade_diag['strategy_exposure_pct'] == 0.0
    assert trade_diag['avg_holding_period_hours'] == 0.0
    assert trade_diag['total_losing_trades'] == 0
    assert trade_diag['bad_entries_count'] == 0
    assert trade_diag['bad_exits_count'] == 0


def test_mae_mfe_intratrade_calculation():
    """Verify that ModelDiagnostics calculates trade MAE and MFE correctly using exact intratrade paths."""
    from src.diagnostics import ModelDiagnostics
    
    # Create 5 hourly candles
    dates = pd.date_range("2026-01-01 09:30:00", periods=5, freq="h")
    df = pd.DataFrame(index=dates)
    df['open'] =  [100, 100, 101, 102, 96]
    df['high'] =  [100, 102, 105, 104, 110]
    df['low'] =   [100,  98,  97,  95,  90]
    df['close'] = [100, 101, 102,  99,  96]
    
    # Mock a single trade:
    # Entry at Candle 1 (open/entry price 100)
    # Exit at Candle 4 (open/exit price 96)
    # Sliced path is candles 1, 2, 3, 4. Mid candles are 1, 2, 3.
    # Entry price = 100. Exit price = 96.
    # Max price in [entry_price, exit_price, mid_candles high] = max(100, 96, 102, 105, 104) = 105.0
    # Min price in [entry_price, exit_price, mid_candles low] = min(100, 96, 98, 97, 95) = 95.0
    # Expected MFE = (105 - 100) / 100 * 100 = 5.0%
    # Expected MAE = (100 - 95) / 100 * 100 = 5.0%
    
    trade = {
        'entry_time': dates[1],
        'exit_time': dates[4],
        'entry_price': 100.0,
        'exit_price': 96.0,
        'net_pnl': -4.0
    }
    
    ml_results = {
        'df': df,
        'pred_prob_up': pd.Series([0.5] * 5, index=dates),
        'pred_prob_down': pd.Series([0.5] * 5, index=dates),
        'trades': [trade]
    }
    
    diag = ModelDiagnostics(ml_results, {}, "SPY")
    trade_diag = diag.get_trade_diagnostics()
    
    # We also check that the trade_mae_mfe_list populated by get_trade_diagnostics contains the correct MFE/MAE
    assert len(diag.trade_mae_mfe_list) == 1
    t_res = diag.trade_mae_mfe_list[0]
    assert t_res['mfe'] == 5.0
    assert t_res['mae'] == 5.0
    assert t_res['is_win'] is False
    
    # Verify Bad Exit classification:
    # A bad exit is a loss where MFE >= 0.5% (meaning it was in profit but failed to take profits and ended in loss).
    # Since MFE is 5.0% and it ended in a loss, bad_exits_count should be 1, bad_entries_count should be 0.
    assert trade_diag['bad_exits_count'] == 1
    assert trade_diag['bad_entries_count'] == 0


def test_regime_classification_by_entry():
    """Verify that trades are classified by market state at the time of entry only."""
    from src.diagnostics import ModelDiagnostics
    
    dates = pd.date_range("2026-01-01 09:30:00", periods=5, freq="h")
    df = pd.DataFrame(index=dates)
    df['close'] = 100.0
    df['market_state'] = ['sideways', 'bullish', 'bullish', 'sideways', 'volatile']
    
    trade = {
        'entry_time': dates[1],
        'exit_time': dates[4],
        'entry_price': 100.0,
        'exit_price': 101.0,
        'net_pnl': 1.0
    }
    
    ml_results = {
        'df': df,
        'pred_prob_up': pd.Series([0.5] * 5, index=dates),
        'pred_prob_down': pd.Series([0.5] * 5, index=dates),
        'trades': [trade]
    }
    
    diag = ModelDiagnostics(ml_results, {}, "SPY")
    regimes = diag.get_regime_analysis()
    
    # The trade should be grouped under 'bullish' (its entry state), NOT 'volatile' (its exit state)
    assert 'bullish' in regimes
    assert 'volatile' not in regimes
    assert regimes['bullish']['trades_count'] == 1
    assert regimes['bullish']['wins'] == 1
    assert regimes['bullish']['net_pnl'] == 1.0
