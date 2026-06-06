import pandas as pd
import numpy as np
import datetime
from config import Config
from src.logger import logger
from src.costs import CostCalculator, TAX_DISCLAIMER
from src.risk_manager import RiskManager
from src.strategy import Strategy

class Backtester:
    """
    Backtests a trading strategy on historical candle data.
    Ensures next-bar execution (signal at t, entry at t+1 Open).
    """
    
    def __init__(self, df: pd.DataFrame, symbol: str):
        self.df = df.copy()
        self.symbol = symbol.upper()
        self.market = "INDIA" if "NSEI" in self.symbol or "NIFTY" in self.symbol else "US"
        self.risk_manager = RiskManager()
        self.strategy = Strategy()
        
    def run_backtest(self, ml_model=None) -> dict:
        """
        Run backtest loop. If ml_model is provided, uses ML strategy.
        Otherwise, runs rule-only baseline.
        """
        logger.info(f"Starting backtest for {self.symbol} ({self.market} market) on {len(self.df)} candles...")
        
        if ml_model is not None and hasattr(ml_model, 'optimal_threshold') and ml_model.optimal_threshold is not None:
            self.strategy.min_confidence = ml_model.optimal_threshold
            logger.info(f"Using model's saved optimal threshold: {self.strategy.min_confidence:.3f}")
            
        # Prepare structures
        cash = Config.STARTING_CAPITAL
        position = None  # None or dict: {'entry_price': X, 'qty': Y, 'entry_time': T, 'stop_loss': SL}
        trades = []      # List of closed trade records
        equity_curve = []
        timestamps = []
        
        # Track daily stats for risk management
        current_day = None
        day_start_equity = cash
        today_trades_count = 0
        
        # We start the backtest after we have enough historical bars for indicators (200 bars)
        start_idx = 200
        if len(self.df) <= start_idx:
            logger.error("Not enough data to run backtest. Need > 200 bars.")
            return {}
            
        # If we are using ML model, generate probabilities first
        if ml_model is not None:
            from src.features import extract_features
            features = extract_features(self.df)
            # Predict probabilities
            prob_up_list = []
            prob_down_list = []
            
            # Predict row-by-row or batch. Batch is faster.
            # However, scaling is already fitted on training set.
            # We can run predict_proba on the whole set (excluding NaNs) and align.
            # To prevent any lookahead, we do it safely:
            # ml_model.predict_proba will transform the features.
            # Filter features to match the trained features list
            X_feats = features.loc[self.df.index[start_idx:]]
            # Fill NaNs to prevent model errors
            X_feats = X_feats.fillna(0)
            
            # Predict batch
            probas = []
            for i in range(len(X_feats)):
                row_df = X_feats.iloc[[i]]
                try:
                    p = ml_model.predict_proba(row_df)
                    probas.append(p)
                except Exception as e:
                    # Fallback to neutral if prediction fails
                    probas.append([0.5, 0.5])
                    
            probas = np.array(probas)
            prob_down_series = pd.Series(probas[:, 0], index=X_feats.index)
            prob_up_series = pd.Series(probas[:, 1], index=X_feats.index)
        else:
            # Rule-only baseline has no ML probabilities (neutral 50/50)
            prob_down_series = pd.Series(0.5, index=self.df.index[start_idx:])
            prob_up_series = pd.Series(0.5, index=self.df.index[start_idx:])
            
        pending_signal = None  # BUY, SELL, or None
        
        # Loop candle by candle
        for i in range(start_idx, len(self.df) - 1):
            t_now = self.df.index[i]
            t_next = self.df.index[i+1]
            
            candle_now = self.df.iloc[i]
            candle_next = self.df.iloc[i+1]
            
            # 1. Update daily reset for risk manager
            candle_date = t_now.date()
            if current_day != candle_date:
                current_day = candle_date
                # Calculate current equity
                current_equity = cash
                if position is not None:
                    current_equity += position['qty'] * candle_now['close']
                day_start_equity = current_equity
                today_trades_count = 0
                
            # 2. Check Stop Loss on current candle
            if position is not None:
                # Did price touch or fall below stop loss?
                # We check the low of the current candle
                if candle_now['low'] <= position['stop_loss']:
                    # Exit at stop loss price (or open if open was already lower)
                    exit_price = min(candle_now['open'], position['stop_loss'])
                    
                    # Calculate exit costs
                    exit_costs = CostCalculator.calculate_exit_costs(exit_price, position['qty'], self.market)
                    exit_val = exit_price * position['qty']
                    entry_val = position['entry_price'] * position['qty']
                    
                    # Calculate tax
                    tax = CostCalculator.calculate_tax_on_profit(
                        entry_val, exit_val, position['entry_costs'], exit_costs, self.market
                    )
                    
                    realized_cash = exit_val - exit_costs - tax
                    cash += realized_cash
                    
                    net_pnl = (exit_val - exit_costs - tax) - (entry_val + position['entry_costs'])
                    gross_pnl = exit_val - entry_val
                    
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': t_now,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'qty': position['qty'],
                        'gross_pnl': gross_pnl,
                        'fees': position['entry_costs'] + exit_costs,
                        'tax': tax,
                        'net_pnl': net_pnl,
                        'reason': 'STOP_LOSS'
                    })
                    
                    position = None
                    pending_signal = None  # Cancel any pending exits
                    
            # 3. Execute Pending Signal at t+1 Open (candle_next['open'])
            if pending_signal == "BUY" and position is None:
                # Check if risk manager allows trade
                current_equity = cash
                if self.risk_manager.can_open_position(day_start_equity, current_equity, today_trades_count):
                    # Enter trade
                    entry_price = candle_next['open']
                    # Calculate shares to buy using max size
                    shares = self.risk_manager.calculate_position_size(entry_price, cash)
                    if shares > 0:
                        entry_costs = CostCalculator.calculate_entry_costs(entry_price, shares, self.market)
                        total_outflow = (entry_price * shares) + entry_costs
                        
                        # Adjust shares if we don't have enough cash for fees
                        if total_outflow > cash:
                            shares = (cash - entry_costs) / entry_price
                            entry_costs = CostCalculator.calculate_entry_costs(entry_price, shares, self.market)
                            total_outflow = (entry_price * shares) + entry_costs
                            
                        if shares > 0 and cash >= total_outflow:
                            cash -= total_outflow
                            stop_loss_val = self.risk_manager.get_stop_loss_price(entry_price)
                            
                            position = {
                                'entry_price': entry_price,
                                'qty': shares,
                                'entry_time': t_next,
                                'entry_costs': entry_costs,
                                'stop_loss': stop_loss_val
                            }
                            today_trades_count += 1
                            logger.debug(f"Executed BUY at {t_next} - Price: {entry_price:.2f}, Qty: {shares:.2f}")
                            
                pending_signal = None
                
            elif pending_signal == "SELL" and position is not None:
                # Exit trade at open of t+1
                exit_price = candle_next['open']
                exit_costs = CostCalculator.calculate_exit_costs(exit_price, position['qty'], self.market)
                exit_val = exit_price * position['qty']
                entry_val = position['entry_price'] * position['qty']
                
                # Tax on profit
                tax = CostCalculator.calculate_tax_on_profit(
                    entry_val, exit_val, position['entry_costs'], exit_costs, self.market
                )
                
                realized_cash = exit_val - exit_costs - tax
                cash += realized_cash
                
                net_pnl = (exit_val - exit_costs - tax) - (entry_val + position['entry_costs'])
                gross_pnl = exit_val - entry_val
                
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': t_next,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'qty': position['qty'],
                    'gross_pnl': gross_pnl,
                    'fees': position['entry_costs'] + exit_costs,
                    'tax': tax,
                    'net_pnl': net_pnl,
                    'reason': 'STRATEGY_EXIT'
                })
                
                position = None
                pending_signal = None
                logger.debug(f"Executed SELL at {t_next} - Price: {exit_price:.2f}")
                
            # 4. Generate next signal at close of current candle t
            row_dict = {
                'close': candle_now['close'],
                'ema_20': candle_now['ema_20'],
                'ema_50': candle_now['ema_50'],
                'ema_200': candle_now['ema_200'],
                'rsi': candle_now['rsi'],
                'macd': candle_now['macd'],
                'macd_signal': candle_now['macd_signal'],
                'atr': candle_now['atr'],
                'bb_upper': candle_now['bb_upper'],
                'bb_lower': candle_now['bb_lower'],
                'bb_middle': candle_now['bb_middle'],
                'market_state': candle_now['market_state']
            }
            
            p_up = prob_up_series.loc[t_now]
            p_down = prob_down_series.loc[t_now]
            
            # Evaluate strategy
            signal_res = self.strategy.generate_signal(row_dict, p_up, p_down)
            
            if signal_res['action'] == "BUY" and position is None:
                pending_signal = "BUY"
            elif signal_res['action'] == "SELL" and position is not None:
                pending_signal = "SELL"
                
            # Record equity
            current_equity = cash
            if position is not None:
                current_equity += position['qty'] * candle_now['close']
            equity_curve.append(current_equity)
            timestamps.append(t_now)
            
        # Close out any remaining position at final bar close for clean statistics
        if position is not None:
            last_idx = len(self.df) - 1
            t_last = self.df.index[last_idx]
            candle_last = self.df.iloc[last_idx]
            
            exit_price = candle_last['close']
            exit_costs = CostCalculator.calculate_exit_costs(exit_price, position['qty'], self.market)
            exit_val = exit_price * position['qty']
            entry_val = position['entry_price'] * position['qty']
            
            tax = CostCalculator.calculate_tax_on_profit(
                entry_val, exit_val, position['entry_costs'], exit_costs, self.market
            )
            
            realized_cash = exit_val - exit_costs - tax
            cash += realized_cash
            
            net_pnl = (exit_val - exit_costs - tax) - (entry_val + position['entry_costs'])
            gross_pnl = exit_val - entry_val
            
            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': t_last,
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'qty': position['qty'],
                'gross_pnl': gross_pnl,
                'fees': position['entry_costs'] + exit_costs,
                'tax': tax,
                'net_pnl': net_pnl,
                'reason': 'FORCE_CLOSE_BACKTEST'
            })
            
            # Append final equity point
            equity_curve.append(cash)
            timestamps.append(t_last)
            
        # Compute performance stats
        equity_series = pd.Series(equity_curve, index=timestamps)
        
        # Buy and Hold Baseline
        bh_start_price = self.df['open'].iloc[start_idx]
        bh_end_price = self.df['close'].iloc[-1]
        bh_qty = Config.STARTING_CAPITAL / bh_start_price
        bh_gross_return = (bh_end_price - bh_start_price) * bh_qty
        
        # Calculate B&H net return (subject to entry, exit, and tax if profitable)
        bh_entry_costs = CostCalculator.calculate_entry_costs(bh_start_price, bh_qty, self.market)
        bh_exit_costs = CostCalculator.calculate_exit_costs(bh_end_price, bh_qty, self.market)
        bh_tax = CostCalculator.calculate_tax_on_profit(
            bh_start_price * bh_qty, bh_end_price * bh_qty, bh_entry_costs, bh_exit_costs, self.market
        )
        bh_net_return = bh_gross_return - bh_entry_costs - bh_exit_costs - bh_tax
        bh_final_equity = Config.STARTING_CAPITAL + bh_net_return
        
        # Strategy returns calculations
        total_net_pnl = sum([t['net_pnl'] for t in trades])
        total_gross_pnl = sum([t['gross_pnl'] for t in trades])
        total_fees = sum([t['fees'] for t in trades])
        total_tax = sum([t['tax'] for t in trades])
        
        win_trades = [t for t in trades if t['net_pnl'] > 0]
        loss_trades = [t for t in trades if t['net_pnl'] <= 0]
        win_rate = len(win_trades) / len(trades) if trades else 0.0
        
        # Max drawdown
        peaks = equity_series.cummax()
        drawdowns = (equity_series - peaks) / peaks
        max_drawdown = drawdowns.min()
        
        # Profit factor
        gross_profits = sum([t['net_pnl'] for t in win_trades])
        gross_losses = abs(sum([t['net_pnl'] for t in loss_trades]))
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else (np.inf if gross_profits > 0 else 1.0)
        
        # Avg trade return pct
        avg_trade_return_pct = np.mean([t['net_pnl'] / (t['entry_price'] * t['qty']) for t in trades]) * 100 if trades else 0.0
        
        metrics = {
            'starting_capital': Config.STARTING_CAPITAL,
            'final_capital_after_costs': cash,
            'total_return_before_costs_pct': (total_gross_pnl / Config.STARTING_CAPITAL) * 100,
            'total_return_after_costs_pct': (total_net_pnl / Config.STARTING_CAPITAL) * 100,
            'fees_paid': total_fees,
            'taxes_paid': total_tax,
            'trade_count': len(trades),
            'win_rate': win_rate,
            'max_drawdown_pct': max_drawdown * 100,
            'profit_factor': profit_factor,
            'avg_trade_return_pct': avg_trade_return_pct,
            'trades': trades,
            'equity_curve': equity_series,
            'bh_return_after_costs_pct': (bh_net_return / Config.STARTING_CAPITAL) * 100,
            'bh_final_capital': bh_final_equity
        }
        
        logger.info(f"Backtest completed. Final Capital: {cash:.2f}, Trades: {len(trades)}, Win Rate: {win_rate*100:.2f}%, Max Drawdown: {max_drawdown*100:.2f}%")
        return metrics
