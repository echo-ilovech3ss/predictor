import pytest
from src.strategy import Strategy

def test_strategy_actions_valid():
    """Verify that strategy returns only BUY, SELL, or HOLD."""
    strat = Strategy(min_confidence=0.60)
    
    # Check return values on dummy data
    dummy_row = {
        'close': 100.0,
        'ema_50': 98.0,
        'rsi': 55.0,
        'market_state': 'bullish'
    }
    
    # 1. Bullish buy signal
    res = strat.generate_signal(dummy_row, prob_up=0.65, prob_down=0.35)
    assert res['action'] == "BUY"
    assert res['confidence'] == 0.65
    
    # 2. Bearish exit signal
    res_sell = strat.generate_signal(dummy_row, prob_up=0.30, prob_down=0.70)
    assert res_sell['action'] == "SELL"
    
    # 3. Conflicting market state (bearish state overrides buy signal)
    bearish_row = dummy_row.copy()
    bearish_row['market_state'] = 'bearish'
    res_block = strat.generate_signal(bearish_row, prob_up=0.65, prob_down=0.35)
    assert res_block['action'] == "HOLD"
    assert "blocked" in res_block['explanation']
    
    # 4. Low confidence defaults to HOLD
    res_hold = strat.generate_signal(dummy_row, prob_up=0.55, prob_down=0.45)
    assert res_hold['action'] == "HOLD"
    assert "below the threshold" in res_hold['explanation']

def test_strategy_sell_is_exit_only():
    """Verify that SELL is recommended under bearish or downward signals, and never BUY."""
    strat = Strategy(min_confidence=0.60)
    
    # Even if market state is bearish and prob_down is high, strategy should recommend SELL (exit), not short entry
    bearish_row = {
        'close': 95.0,
        'ema_50': 100.0,
        'rsi': 40.0,
        'market_state': 'bearish'
    }
    
    res = strat.generate_signal(bearish_row, prob_up=0.20, prob_down=0.80)
    assert res['action'] == "SELL"
    assert "SELL" in res['explanation']
