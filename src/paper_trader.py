import os
import json
import datetime
from config import Config
from src.logger import logger
from src.costs import CostCalculator
from src.risk_manager import RiskManager

STATE_FILE = "paper_trader_state.json"

class PaperTrader:
    """Simulates live paper trading by reading/writing state to a persistent JSON file."""
    
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.market = "INDIA" if "NSEI" in self.symbol or "NIFTY" in self.symbol else "US"
        self.risk_manager = RiskManager()
        self.state_path = f"{self.symbol.lower()}_{STATE_FILE}"
        self.cash = Config.STARTING_CAPITAL
        self.position = None  # Holds active position dictionary or None
        self.trades = []      # Closed trades list
        
        self.load_state()
        
    def load_state(self):
        """Load paper trading state from disk, initializing if not present."""
        if not os.path.exists(self.state_path):
            self.cash = Config.STARTING_CAPITAL
            self.position = None
            self.trades = []
            self.save_state()
            logger.info(f"Initialized new paper trader state for {self.symbol} at {self.state_path}")
            return
            
        try:
            with open(self.state_path, "r") as f:
                state = json.load(f)
            self.cash = float(state.get("cash", Config.STARTING_CAPITAL))
            self.position = state.get("position", None)
            self.trades = state.get("trades", [])
            logger.info(f"Loaded paper trader state for {self.symbol}. Balance: {self.cash:.2f}")
        except Exception as e:
            logger.error(f"Error loading paper trader state: {e}. Reverting to defaults.")
            self.cash = Config.STARTING_CAPITAL
            self.position = None
            self.trades = []
            
    def save_state(self):
        """Save current state to file."""
        state = {
            "cash": self.cash,
            "position": self.position,
            "trades": self.trades
        }
        try:
            with open(self.state_path, "w") as f:
                json.dump(state, f, indent=4)
            logger.debug(f"Saved paper trader state for {self.symbol}.")
        except Exception as e:
            logger.error(f"Failed to save paper trader state: {e}")
            
    def get_equity(self, current_price: float) -> float:
        """Calculate total equity (cash + open position value)."""
        equity = self.cash
        if self.position is not None:
            equity += self.position["qty"] * current_price
        return equity
        
    def check_and_apply_stop_loss(self, current_price: float, current_time: str) -> bool:
        """Check if active position has hit its stop loss."""
        if self.position is None:
            return False
            
        if self.risk_manager.is_stop_loss_triggered(self.position["entry_price"], current_price):
            logger.warning(f"Stop Loss triggered in paper trading for {self.symbol} at price {current_price:.2f}!")
            self._close_position(current_price, current_time, reason="STOP_LOSS")
            return True
            
        return False
        
    def process_signal(self, signal_res: dict, current_price: float, current_time: str):
        """Process a BUY, SELL, or HOLD signal in paper trading."""
        action = signal_res.get("action", "HOLD")
        
        # 1. Update stop loss first
        stop_loss_hit = self.check_and_apply_stop_loss(current_price, current_time)
        if stop_loss_hit:
            # Stopped out, do not enter again immediately in the same cycle
            return
            
        # 2. Check risk manager limits (today's trade count)
        today_date = datetime.datetime.now().date()
        today_trades_count = 0
        for t in self.trades:
            exit_time_str = t.get("exit_time", "")
            try:
                # Expecting format 'YYYY-MM-DD HH:MM:SS' or ISO format
                exit_date = datetime.datetime.fromisoformat(exit_time_str).date()
                if exit_date == today_date:
                    today_trades_count += 1
            except ValueError:
                # Try parsing with split in case of space
                if " " in exit_time_str:
                    try:
                        exit_date = datetime.datetime.strptime(exit_time_str.split(" ")[0], "%Y-%m-%d").date()
                        if exit_date == today_date:
                            today_trades_count += 1
                    except Exception:
                        pass
        
        # Calculate daily starting capital (approximate using cash at start of day)
        # For simplicity, we can use Config.STARTING_CAPITAL or self.get_equity(current_price)
        current_equity = self.get_equity(current_price)
        
        if action == "BUY" and self.position is None:
            # Can we trade?
            if self.risk_manager.can_open_position(self.cash, current_equity, today_trades_count):
                self._open_position(current_price, current_time)
            else:
                logger.info(f"BUY signal rejected by Risk Manager for {self.symbol}.")
                
        elif action == "SELL" and self.position is not None:
            self._close_position(current_price, current_time, reason="STRATEGY_EXIT")
            
        else:
            logger.debug(f"Paper trading action for {self.symbol} is HOLD.")
            
        self.save_state()
        
    def _open_position(self, price: float, time_str: str):
        """Open a new simulated long position."""
        shares = self.risk_manager.calculate_position_size(price, self.cash)
        if shares <= 0:
            logger.warning(f"Calculated position size is 0 for price {price:.2f}.")
            return
            
        entry_costs = CostCalculator.calculate_entry_costs(price, shares, self.market)
        total_cost = (price * shares) + entry_costs
        
        # Readjust shares if we exceed cash
        if total_cost > self.cash:
            shares = (self.cash - entry_costs) / price
            entry_costs = CostCalculator.calculate_entry_costs(price, shares, self.market)
            total_cost = (price * shares) + entry_costs
            
        if shares > 0 and self.cash >= total_cost:
            self.cash -= total_cost
            stop_loss = self.risk_manager.get_stop_loss_price(price)
            
            self.position = {
                "entry_price": float(price),
                "qty": float(shares),
                "entry_time": time_str,
                "entry_costs": float(entry_costs),
                "stop_loss": float(stop_loss)
            }
            logger.info(f"Simulated BUY of {shares:.2f} shares of {self.symbol} at {price:.2f}. SL: {stop_loss:.2f}")
        else:
            logger.warning("Insufficient funds to open position in paper trading.")
            
    def _close_position(self, price: float, time_str: str, reason: str):
        """Close an active long position."""
        if self.position is None:
            return
            
        exit_costs = CostCalculator.calculate_exit_costs(price, self.position["qty"], self.market)
        exit_val = price * self.position["qty"]
        entry_val = self.position["entry_price"] * self.position["qty"]
        
        # Tax calculation
        tax = CostCalculator.calculate_tax_on_profit(
            entry_val, exit_val, self.position["entry_costs"], exit_costs, self.market
        )
        
        realized_cash = exit_val - exit_costs - tax
        self.cash += realized_cash
        
        net_pnl = realized_cash - (entry_val + self.position["entry_costs"])
        gross_pnl = exit_val - entry_val
        
        trade_record = {
            "entry_time": self.position["entry_time"],
            "exit_time": time_str,
            "entry_price": self.position["entry_price"],
            "exit_price": float(price),
            "qty": self.position["qty"],
            "gross_pnl": float(gross_pnl),
            "fees": float(self.position["entry_costs"] + exit_costs),
            "tax": float(tax),
            "net_pnl": float(net_pnl),
            "reason": reason
        }
        
        self.trades.append(trade_record)
        logger.info(f"Simulated SELL of {self.position['qty']:.2f} shares of {self.symbol} at {price:.2f}. Net PnL: {net_pnl:.2f}. Reason: {reason}.")
        
        self.position = None
        self.save_state()
        
    def reset_account(self):
        """Reset the paper trading account capital and logs."""
        self.cash = Config.STARTING_CAPITAL
        self.position = None
        self.trades = []
        self.save_state()
        logger.info(f"Paper trading account for {self.symbol} has been reset.")
