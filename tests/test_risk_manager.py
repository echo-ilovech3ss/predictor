import pytest
from src.risk_manager import RiskManager

def test_risk_manager_stop_loss():
    """Verify stop loss triggers correctly."""
    rm = RiskManager(stop_loss_pct=0.02) # 2% stop loss
    
    # Entry at 100.0, price drops to 98.0 -> triggers SL
    assert rm.is_stop_loss_triggered(100.0, 98.0) is True
    assert rm.is_stop_loss_triggered(100.0, 97.9) is True
    
    # Entry at 100.0, price at 99.0 -> does not trigger
    assert rm.is_stop_loss_triggered(100.0, 99.0) is False
    assert rm.is_stop_loss_triggered(100.0, 101.0) is False

def test_risk_manager_daily_limits():
    """Verify that daily loss limits and trade counts block entries."""
    rm = RiskManager(
        daily_loss_limit_pct=0.05,  # 5% daily limit
        max_trades_per_day=3
    )
    
    # 1. Standard allowed state
    # Starting: 100k, Equity: 99k (1% loss), trades: 1
    assert rm.can_open_position(100000.0, 99000.0, today_trades=1) is True
    
    # 2. Blocked by trade count (3 trades executed, limit is 3)
    assert rm.can_open_position(100000.0, 99000.0, today_trades=3) is False
    
    # 3. Blocked by daily loss limit (Starting: 100k, Current Equity: 94k -> 6% loss)
    assert rm.can_open_position(100000.0, 94000.0, today_trades=0) is False
    
    # 4. Check daily loss limit flag directly
    assert rm.check_daily_loss_limit(100000.0, 94000.0) is True
    assert rm.check_daily_loss_limit(100000.0, 96000.0) is False

def test_risk_manager_position_sizing():
    """Verify position sizing calculations."""
    rm = RiskManager(max_position_size_pct=0.20) # 20% max size
    
    # Capital 100k, stock price 10.0 -> max size 20k -> 2000 shares
    shares = rm.calculate_position_size(10.0, 100000.0)
    assert shares == 2000.0
    
    # Invalid price should return 0
    assert rm.calculate_position_size(0.0, 100000.0) == 0.0
    assert rm.calculate_position_size(-5.0, 100000.0) == 0.0
