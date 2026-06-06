import datetime
from config import Config
from src.logger import logger

class RiskManager:
    """Enforces safety rules: daily loss limits, max trade counts, stop losses, and position sizing."""
    
    def __init__(self, 
                 stop_loss_pct: float = None,
                 daily_loss_limit_pct: float = None,
                 max_position_size_pct: float = None,
                 max_trades_per_day: int = None):
                 
        self.stop_loss_pct = stop_loss_pct if stop_loss_pct is not None else Config.STOP_LOSS_PCT
        self.daily_loss_limit_pct = daily_loss_limit_pct if daily_loss_limit_pct is not None else Config.DAILY_LOSS_LIMIT_PCT
        self.max_position_size_pct = max_position_size_pct if max_position_size_pct is not None else Config.MAX_POSITION_SIZE_PCT
        self.max_trades_per_day = max_trades_per_day if max_trades_per_day is not None else Config.MAX_TRADES_PER_DAY
        
    def check_daily_loss_limit(self, starting_capital: float, current_equity: float) -> bool:
        """
        Check if the daily loss limit has been breached.
        Loss is calculated relative to starting capital.
        """
        pnl = current_equity - starting_capital
        if pnl < 0:
            loss_pct = abs(pnl) / starting_capital
            if loss_pct >= self.daily_loss_limit_pct:
                logger.warning(f"Daily loss limit breached: -{loss_pct*100:.2f}% (Limit: {self.daily_loss_limit_pct*100:.2f}%)")
                return True
        return False
        
    def can_open_position(self, 
                          starting_capital: float, 
                          current_equity: float, 
                          today_trades: int) -> bool:
        """
        Determine if a new trade can be opened based on daily loss limits
        and daily trade limits.
        """
        # 1. Check daily trade count limit
        if today_trades >= self.max_trades_per_day:
            logger.info(f"Trade limit reached for today: {today_trades} trades executed. Max limit is {self.max_trades_per_day}.")
            return False
            
        # 2. Check daily loss limit
        if self.check_daily_loss_limit(starting_capital, current_equity):
            logger.info("Cannot open new position: Daily loss limit is breached.")
            return False
            
        return True
        
    def calculate_position_size(self, price: float, current_capital: float) -> float:
        """
        Calculate maximum position size in number of shares.
        Uses max_position_size_pct of the current capital.
        """
        if price <= 0:
            return 0.0
            
        max_allocated_cash = current_capital * self.max_position_size_pct
        shares = max_allocated_cash / price
        return float(shares)
        
    def is_stop_loss_triggered(self, entry_price: float, current_price: float) -> bool:
        """
        Evaluate if stop loss is triggered (long-only).
        Triggers if current_price <= entry_price * (1 - STOP_LOSS_PCT)
        """
        if entry_price <= 0:
            return False
            
        trigger_price = entry_price * (1.0 - self.stop_loss_pct)
        if current_price <= trigger_price:
            logger.warning(f"Stop loss triggered! Entry: {entry_price:.2f}, Current: {current_price:.2f}, Trigger Price: {trigger_price:.2f}")
            return True
            
        return False
        
    def get_stop_loss_price(self, entry_price: float) -> float:
        """Calculate the stop loss price for a given entry price."""
        return entry_price * (1.0 - self.stop_loss_pct)
