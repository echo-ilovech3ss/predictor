import pytest
from config import Config
from src.costs import CostCalculator

def test_costs_slippage_and_brokerage():
    """Verify entry and exit costs calculations."""
    price = 100.0
    quantity = 10.0
    notional = price * quantity  # 1000.0
    
    # US Preset
    entry_us = CostCalculator.calculate_entry_costs(price, quantity, market="US")
    expected_slippage = notional * Config.SLIPPAGE_PCT
    expected_brokerage = notional * Config.BROKERAGE_PCT
    assert entry_us == pytest.approx(expected_slippage + expected_brokerage)
    
    # India Preset should be higher due to STT, GST, and Stamp duty
    entry_in = CostCalculator.calculate_entry_costs(price, quantity, market="INDIA")
    assert entry_in > entry_us

def test_costs_tax_calculation():
    """Verify tax is only applied to positive net profits."""
    # Scenario 1: Profitable trade
    # Entry: 1000.0, Exit: 1200.0, Entry costs: 10.0, Exit costs: 10.0
    # Net profit: 200.0 - 20.0 = 180.0
    # US tax = 180.0 * 15% = 27.0
    tax_us = CostCalculator.calculate_tax_on_profit(
        entry_val=1000.0, exit_val=1200.0, entry_costs=10.0, exit_costs=10.0, market="US"
    )
    assert tax_us == pytest.approx(180.0 * Config.TAX_ON_PROFIT_PCT)
    
    # Scenario 2: Unprofitable trade
    # Entry: 1000.0, Exit: 950.0
    # Net profit is negative, tax should be 0.0
    tax_loss = CostCalculator.calculate_tax_on_profit(
        entry_val=1000.0, exit_val=950.0, entry_costs=10.0, exit_costs=10.0, market="US"
    )
    assert tax_loss == 0.0
    
    # Scenario 3: Profitable trade under India market rules
    # STCG is at least 20%
    tax_in = CostCalculator.calculate_tax_on_profit(
        entry_val=1000.0, exit_val=1200.0, entry_costs=10.0, exit_costs=10.0, market="INDIA"
    )
    assert tax_in == pytest.approx(180.0 * 0.20)
