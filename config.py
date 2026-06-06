import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Starting balance
    STARTING_CAPITAL = float(os.getenv("STARTING_CAPITAL", 100000.0))
    
    # Timeframe
    TIMEFRAME = os.getenv("TIMEFRAME", "1h")
    
    # Risk Management
    STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", 2.0)) / 100.0  # Convert to decimal
    DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", 5.0)) / 100.0
    MAX_POSITION_SIZE_PCT = float(os.getenv("MAX_POSITION_SIZE_PCT", 20.0)) / 100.0
    MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", 3))
    MIN_CONFIDENCE_FOR_TRADE = float(os.getenv("MIN_CONFIDENCE_FOR_TRADE", 0.60))
    
    # Transaction Costs
    SLIPPAGE_PCT = float(os.getenv("SLIPPAGE_PCT", 0.05)) / 100.0
    BROKERAGE_PCT = float(os.getenv("BROKERAGE_PCT", 0.03)) / 100.0
    TAX_ON_PROFIT_PCT = float(os.getenv("TAX_ON_PROFIT_PCT", 15.0)) / 100.0
    
    # Safety
    PAPER_TRADING_ONLY = os.getenv("PAPER_TRADING_ONLY", "true").lower() == "true"
    
    @classmethod
    def print_config(cls):
        print("--- Loaded Configuration ---")
        print(f"Starting Capital: {cls.STARTING_CAPITAL}")
        print(f"Timeframe: {cls.TIMEFRAME}")
        print(f"Stop Loss Pct: {cls.STOP_LOSS_PCT * 100:.2f}%")
        print(f"Daily Loss Limit Pct: {cls.DAILY_LOSS_LIMIT_PCT * 100:.2f}%")
        print(f"Max Position Size Pct: {cls.MAX_POSITION_SIZE_PCT * 100:.2f}%")
        print(f"Max Trades Per Day: {cls.MAX_TRADES_PER_DAY}")
        print(f"Min Confidence for Trade: {cls.MIN_CONFIDENCE_FOR_TRADE * 100:.2f}%")
        print(f"Slippage Pct: {cls.SLIPPAGE_PCT * 100:.2f}%")
        print(f"Brokerage Pct: {cls.BROKERAGE_PCT * 100:.2f}%")
        print(f"Tax on Profit Pct: {cls.TAX_ON_PROFIT_PCT * 100:.2f}%")
        print(f"Paper Trading Only: {cls.PAPER_TRADING_ONLY}")
        print("----------------------------")
