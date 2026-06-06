import pandas as pd
from config import Config
from src.logger import logger

# Strict legal disclaimer required by functional spec
TAX_DISCLAIMER = (
    "WARNING: Tax calculations in this application are highly simplified "
    "theoretical assumptions for backtesting purposes only. They do NOT "
    "represent actual tax liabilities, nor do they constitute legal or "
    "professional tax advice. Please consult a qualified tax professional "
    "for real-world trading implications."
)

class CostCalculator:
    """Computes realistic transaction costs, slippage, and taxes on realized trades."""
    
    @staticmethod
    def calculate_entry_costs(price: float, quantity: float, market: str = "US") -> float:
        """
        Calculate cost overhead when entering a trade.
        Overhead consists of brokerage fee + slippage penalty.
        """
        notional_value = price * quantity
        
        # 1. Slippage (buying at a slightly worse price)
        slippage_cost = notional_value * Config.SLIPPAGE_PCT
        
        # 2. Brokerage / fees
        if market.upper() == "INDIA":
            # Indian fees: Brokerage + STT (0.1% buy/sell) + GST (18% on brokerage) + Stamp Duty (0.015% buy)
            brokerage = notional_value * Config.BROKERAGE_PCT
            stt = notional_value * 0.001  # Securities Transaction Tax: 0.1% on delivery
            gst = brokerage * 0.18
            stamp_duty = notional_value * 0.00015  # Stamp duty: 0.015% on buy
            other_charges = notional_value * 0.00003  # Exchange transaction charges, SEBI turnover fees
            total_fees = brokerage + stt + gst + stamp_duty + other_charges
        else:
            # US fees: Brokerage (often 0, but use config value) + regulatory fees (SEC, FINRA)
            brokerage = notional_value * Config.BROKERAGE_PCT
            # SEC fees are buy-exempt (only charged on sell), FINRA TAF is sell-only
            total_fees = brokerage
            
        total_cost = slippage_cost + total_fees
        return total_cost
        
    @staticmethod
    def calculate_exit_costs(price: float, quantity: float, market: str = "US") -> float:
        """
        Calculate cost overhead when exiting a trade.
        """
        notional_value = price * quantity
        
        # 1. Slippage (selling at a slightly worse price)
        slippage_cost = notional_value * Config.SLIPPAGE_PCT
        
        # 2. Brokerage / fees
        if market.upper() == "INDIA":
            # Indian exit fees: Brokerage + STT (0.1% sell) + GST (18% on brokerage) + SEBI/Exchange
            brokerage = notional_value * Config.BROKERAGE_PCT
            stt = notional_value * 0.001
            gst = brokerage * 0.18
            other_charges = notional_value * 0.00003
            total_fees = brokerage + stt + gst + other_charges
        else:
            # US exit fees: Brokerage + SEC section 31 fee (approx 0.0000278 of value) + FINRA TAF
            brokerage = notional_value * Config.BROKERAGE_PCT
            sec_fee = notional_value * 0.0000278
            finra_taf = quantity * 0.000166  # $0.000166 per share (capped at $8.30)
            finra_taf = min(finra_taf, 8.30)
            total_fees = brokerage + sec_fee + finra_taf
            
        total_cost = slippage_cost + total_fees
        return total_cost

    @staticmethod
    def calculate_tax_on_profit(entry_val: float, exit_val: float, entry_costs: float, exit_costs: float, market: str = "US") -> float:
        """
        Calculate simplified tax on profit if the trade was profitable.
        Profit is calculated net of entry and exit costs.
        """
        gross_profit = exit_val - entry_val
        net_profit = gross_profit - entry_costs - exit_costs
        
        if net_profit <= 0:
            return 0.0
            
        # Apply simplified short term capital gains tax (STCG)
        # 15% default or config-defined tax rate
        tax_rate = Config.TAX_ON_PROFIT_PCT
        
        # Indian short term capital gains tax on equity is currently 20% (since July 2024)
        if market.upper() == "INDIA":
            tax_rate = max(tax_rate, 0.20)  # Indian STCG is at least 20%
            
        tax_amount = net_profit * tax_rate
        return tax_amount
