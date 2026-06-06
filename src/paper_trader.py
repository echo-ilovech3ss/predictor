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
        self.queued_signal = None
        self.last_processed_time = None
        
        self.load_state()
        
    def load_state(self):
        """Load paper trading state from disk, initializing if not present."""
        if not os.path.exists(self.state_path):
            self.cash = Config.STARTING_CAPITAL
            self.position = None
            self.trades = []
            self.queued_signal = None
            self.last_processed_time = None
            self.save_state()
            logger.info(f"Initialized new paper trader state for {self.symbol} at {self.state_path}")
            return
            
        try:
            with open(self.state_path, "r") as f:
                state = json.load(f)
            self.cash = float(state.get("cash", Config.STARTING_CAPITAL))
            self.position = state.get("position", None)
            self.trades = state.get("trades", [])
            self.queued_signal = state.get("queued_signal", None)
            self.last_processed_time = state.get("last_processed_time", None)
            logger.info(f"Loaded paper trader state for {self.symbol}. Balance: {self.cash:.2f}")
        except Exception as e:
            logger.error(f"Error loading paper trader state: {e}. Reverting to defaults.")
            self.cash = Config.STARTING_CAPITAL
            self.position = None
            self.trades = []
            self.queued_signal = None
            self.last_processed_time = None
            
    def save_state(self):
        """Save current state to file."""
        state = {
            "cash": self.cash,
            "position": self.position,
            "trades": self.trades,
            "queued_signal": self.queued_signal,
            "last_processed_time": self.last_processed_time
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
        
    def process_signal(self, signal_res: dict, latest_row: dict, current_time: str, is_market_open: bool, is_data_stale: bool):
        """Process a BUY, SELL, or HOLD signal in paper trading using next-bar open execution logic."""
        if not is_market_open or is_data_stale:
            logger.warning(f"Paper trading execution blocked: Market is closed or data is stale. Forcing HOLD.")
            signal_res["action"] = "HOLD"
            signal_res["explanation"] = f"HOLD: Execution blocked due to closed or stale market."
            return
            
        if self.last_processed_time == current_time:
            logger.info(f"Skipping prediction processing: candle at {current_time} was already processed.")
            return

        # 1. Execute any queued signal at the current candle's open
        if self.queued_signal is not None:
            logger.info("Simulated next-candle open execution filled. True real-time next-open execution would require a live quote/broker feed.")
            
            entry_price = latest_row['open']
            
            # Check risk manager limits for today
            today_date = datetime.datetime.now().date()
            today_trades_count = 0
            for t in self.trades:
                exit_time_str = t.get("exit_time", "")
                try:
                    exit_date = datetime.datetime.fromisoformat(exit_time_str).date()
                    if exit_date == today_date:
                        today_trades_count += 1
                except Exception:
                    pass
            
            current_equity = self.get_equity(latest_row['close'])
            
            if self.queued_signal == "BUY" and self.position is None:
                if self.risk_manager.can_open_position(self.cash, current_equity, today_trades_count):
                    self._open_position(entry_price, current_time)
                else:
                    logger.info(f"Queued BUY signal rejected by Risk Manager for {self.symbol}.")
            elif self.queued_signal == "SELL" and self.position is not None:
                self._close_position(entry_price, current_time, reason="STRATEGY_EXIT")
                
            self.queued_signal = None

        # 2. Check Stop Loss on the current candle (using low price)
        if self.position is not None:
            if latest_row['low'] <= self.position['stop_loss']:
                exit_price = min(latest_row['open'], self.position['stop_loss'])
                logger.warning(f"Stop Loss triggered in paper trading for {self.symbol} at price {exit_price:.2f}!")
                self._close_position(exit_price, current_time, reason="STOP_LOSS")

        # 3. Queue the new signal generated from the close of this candle
        action = signal_res.get("action", "HOLD")
        if action in ("BUY", "SELL"):
            if action == "BUY" and self.position is None:
                self.queued_signal = "BUY"
                logger.info(f"Queued BUY signal for {self.symbol} to execute at next candle open.")
            elif action == "SELL" and self.position is not None:
                self.queued_signal = "SELL"
                logger.info(f"Queued SELL signal for {self.symbol} to execute at next candle open.")
        else:
            self.queued_signal = None

        self.last_processed_time = current_time
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
                "stop_loss": float(stop_loss),
                "execution_note": "Simulated next-candle open fill. True real-time next-open execution requires a live quote/broker feed."
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
            "reason": reason,
            "execution_note": self.position.get("execution_note", "")
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
        self.queued_signal = None
        self.last_processed_time = None
        self.save_state()
        logger.info(f"Paper trading account for {self.symbol} has been reset.")
