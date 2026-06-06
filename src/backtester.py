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
        
    def run_backtest(self, ml_model=None, test_start_date=None,
                     pred_prob_up=None, pred_prob_down=None,
                     dynamic_thresholds=None) -> dict:
        """
        Run backtest loop. If ml_model is provided or pre-calculated predictions are passed, uses ML strategy.
        Otherwise, runs rule-only baseline.
        """
        logger.info(f"Starting backtest for {self.symbol} ({self.market} market) on {len(self.df)} candles...")
        
        # Determine confidence threshold
        use_ml = (ml_model is not None or pred_prob_up is not None)
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
        
        # Track OOS start equity
        equity_at_oos_start = None
        
        # We start the backtest after we have enough historical bars for indicators (200 bars)
        start_idx = 200
        if len(self.df) <= start_idx:
            logger.error("Not enough data to run backtest. Need > 200 bars.")
            return {}
            
        # If we are using ML model, generate probabilities first
        if pred_prob_up is not None and pred_prob_down is not None:
            prob_up_series = pred_prob_up
            prob_down_series = pred_prob_down
        elif use_ml:
            from src.features import extract_features
            features = extract_features(self.df)
            X_feats = features.loc[self.df.index[start_idx:]]
            X_feats = X_feats.fillna(0)
            
            # Predict batch
            probas = []
            for i in range(len(X_feats)):
                row_df = X_feats.iloc[[i]]
                try:
                    p = ml_model.predict_proba(row_df)
                    probas.append(p)
                except Exception as e:
                    probas.append([0.5, 0.5])
                    
            probas = np.array(probas)
            prob_down_series = pd.Series(probas[:, 0], index=X_feats.index)
            prob_up_series = pd.Series(probas[:, 1], index=X_feats.index)
        else:
            # Rule-only baseline has no ML probabilities
            prob_down_series = pd.Series(0.5, index=self.df.index[start_idx:])
            prob_up_series = pd.Series(0.5, index=self.df.index[start_idx:])
            
        pending_signal = None  # BUY, SELL, or None
        
        # Loop candle by candle
        for i in range(start_idx, len(self.df)):
            t_now = self.df.index[i]
            candle_now = self.df.iloc[i]
            
            # 1. Update daily reset for risk manager
            candle_date = t_now.date()
            if current_day != candle_date:
                current_day = candle_date
                current_equity = cash
                if position is not None:
                    current_equity += position['qty'] * candle_now['close']
                day_start_equity = current_equity
                today_trades_count = 0
                
            # Track portfolio equity at the start of the Out-of-Sample period
            if test_start_date is not None and t_now >= test_start_date and equity_at_oos_start is None:
                current_equity = cash
                if position is not None:
                    current_equity += position['qty'] * candle_now['close']
                equity_at_oos_start = current_equity
                logger.info(f"OOS period started at {t_now}. Initial OOS Equity: {equity_at_oos_start:.2f}")

            # 2. Execute Pending Signal at t+1 Open (which is candle_now['open'] of this iteration)
            if pending_signal == "BUY" and position is None:
                current_equity = cash
                if self.risk_manager.can_open_position(day_start_equity, current_equity, today_trades_count):
                    entry_price = candle_now['open']
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
                                'entry_time': t_now,
                                'entry_costs': entry_costs,
                                'stop_loss': stop_loss_val
                            }
                            today_trades_count += 1
                            logger.debug(f"Executed BUY at {t_now} (Next-Bar Open) - Price: {entry_price:.2f}, Qty: {shares:.2f}")
                pending_signal = None
                
            elif pending_signal == "SELL" and position is not None:
                exit_price = candle_now['open']
                exit_costs = CostCalculator.calculate_exit_costs(exit_price, position['qty'], self.market)
                exit_val = exit_price * position['qty']
                entry_val = position['entry_price'] * position['qty']
                
                tax = CostCalculator.calculate_tax_on_profit(
                    entry_val, exit_val, position['entry_costs'], exit_costs, self.market
                )
                
                realized_cash = exit_val - exit_costs - tax
                cash += realized_cash
                
                net_pnl = realized_cash - (entry_val + position['entry_costs'])
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
                    'reason': 'STRATEGY_EXIT'
                })
                
                position = None
                pending_signal = None
                logger.debug(f"Executed SELL at {t_now} (Next-Bar Open) - Price: {exit_price:.2f}")

            # 3. Check Stop Loss on current candle_now (after next-bar entry executes)
            if position is not None:
                if candle_now['low'] <= position['stop_loss']:
                    exit_price = min(candle_now['open'], position['stop_loss'])
                    exit_costs = CostCalculator.calculate_exit_costs(exit_price, position['qty'], self.market)
                    exit_val = exit_price * position['qty']
                    entry_val = position['entry_price'] * position['qty']
                    
                    tax = CostCalculator.calculate_tax_on_profit(
                        entry_val, exit_val, position['entry_costs'], exit_costs, self.market
                    )
                    
                    realized_cash = exit_val - exit_costs - tax
                    cash += realized_cash
                    
                    net_pnl = realized_cash - (entry_val + position['entry_costs'])
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
                    pending_signal = None
                    
            # 4. Generate next signal at the close of current candle t_now
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
            
            # Dynamic threshold override for walk-forward OOS period
            if dynamic_thresholds is not None and t_now in dynamic_thresholds.index:
                self.strategy.min_confidence = float(dynamic_thresholds.loc[t_now])
                
            signal_res = self.strategy.generate_signal(row_dict, p_up, p_down, use_ml=use_ml)
            
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
            
            net_pnl = realized_cash - (entry_val + position['entry_costs'])
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
            
            equity_curve.append(cash)
            timestamps.append(t_last)
            
        equity_series = pd.Series(equity_curve, index=timestamps)
        
        # 5. Calculate Partitioned Performance Metrics
        def compute_sub_metrics(sub_trades, start_capital, final_equity_curve=None):
            if not sub_trades:
                return {
                    'trade_count': 0,
                    'win_rate': 0.0,
                    'total_return_pct': 0.0,
                    'profit_factor': 1.0,
                    'avg_win_cash': 0.0,
                    'avg_loss_cash': 0.0,
                    'avg_win_pct': 0.0,
                    'avg_loss_pct': 0.0,
                    'max_drawdown_pct': 0.0,
                    'warning_msg': "WARNING: Trade count is too low (N < 5) to draw statistically meaningful conclusions."
                }
            
            win_trades = [t for t in sub_trades if t['net_pnl'] > 0]
            loss_trades = [t for t in sub_trades if t['net_pnl'] <= 0]
            win_rate = len(win_trades) / len(sub_trades)
            
            total_net_pnl = sum([t['net_pnl'] for t in sub_trades])
            total_return_pct = (total_net_pnl / start_capital) * 100
            
            gross_profits = sum([t['net_pnl'] for t in win_trades])
            gross_losses = abs(sum([t['net_pnl'] for t in loss_trades]))
            profit_factor = gross_profits / gross_losses if gross_losses > 0 else (np.inf if gross_profits > 0 else 1.0)
            
            avg_win_cash = np.mean([t['net_pnl'] for t in win_trades]) if win_trades else 0.0
            avg_loss_cash = np.mean([t['net_pnl'] for t in loss_trades]) if loss_trades else 0.0
            
            avg_win_pct = np.mean([t['net_pnl'] / (t['entry_price'] * t['qty']) for t in win_trades]) * 100 if win_trades else 0.0
            avg_loss_pct = np.mean([t['net_pnl'] / (t['entry_price'] * t['qty']) for t in loss_trades]) * 100 if loss_trades else 0.0
            
            max_dd_pct = 0.0
            if final_equity_curve is not None and len(final_equity_curve) > 0:
                peaks = final_equity_curve.cummax()
                drawdowns = (final_equity_curve - peaks) / peaks
                max_dd_pct = drawdowns.min() * 100
                
            warning_msg = None
            if len(sub_trades) < 5:
                warning_msg = "WARNING: Trade count is too low (N < 5) to draw statistically meaningful conclusions."
                
            return {
                'trade_count': len(sub_trades),
                'win_rate': win_rate,
                'total_return_pct': total_return_pct,
                'profit_factor': profit_factor,
                'avg_win_cash': avg_win_cash,
                'avg_loss_cash': avg_loss_cash,
                'avg_win_pct': avg_win_pct,
                'avg_loss_pct': avg_loss_pct,
                'max_drawdown_pct': max_dd_pct,
                'warning_msg': warning_msg
            }
            
        # Parse IS vs OOS trades
        if test_start_date is not None:
            is_trades = [t for t in trades if t['entry_time'] < test_start_date]
            oos_trades = [t for t in trades if t['entry_time'] >= test_start_date]
            
            # Divide equity curve
            is_equity = equity_series.loc[:test_start_date]
            oos_equity = equity_series.loc[test_start_date:]
        else:
            is_trades = trades
            oos_trades = []
            is_equity = equity_series
            oos_equity = pd.Series(dtype=float)
            
        is_cap = Config.STARTING_CAPITAL
        oos_cap = equity_at_oos_start if equity_at_oos_start is not None else Config.STARTING_CAPITAL
        
        is_metrics = compute_sub_metrics(is_trades, is_cap, is_equity)
        oos_metrics = compute_sub_metrics(oos_trades, oos_cap, oos_equity)
        full_metrics = compute_sub_metrics(trades, Config.STARTING_CAPITAL, equity_series)
        
        # 6. Buy and Hold Baselines
        def calculate_bh_net_return(start_p, end_p, cap):
            bh_qty = cap / start_p
            bh_gross_return = (end_p - start_p) * bh_qty
            bh_entry_costs = CostCalculator.calculate_entry_costs(start_p, bh_qty, self.market)
            bh_exit_costs = CostCalculator.calculate_exit_costs(end_p, bh_qty, self.market)
            bh_tax = CostCalculator.calculate_tax_on_profit(
                start_p * bh_qty, end_p * bh_qty, bh_entry_costs, bh_exit_costs, self.market
            )
            bh_net_return = bh_gross_return - bh_entry_costs - bh_exit_costs - bh_tax
            return (bh_net_return / cap) * 100
            
        bh_start_full = self.df['open'].iloc[start_idx]
        bh_end_full = self.df['close'].iloc[-1]
        bh_full_return = calculate_bh_net_return(bh_start_full, bh_end_full, Config.STARTING_CAPITAL)
        
        if test_start_date is not None:
            # Out-of-sample buy and hold starts at test_start_date open
            # Find closest index to test_start_date
            oos_indices = self.df.index[self.df.index >= test_start_date]
            if len(oos_indices) > 0:
                bh_start_oos = self.df.loc[oos_indices[0], 'open']
                bh_end_oos = self.df['close'].iloc[-1]
                bh_oos_return = calculate_bh_net_return(bh_start_oos, bh_end_oos, oos_cap)
            else:
                bh_oos_return = 0.0
                
            # In-sample buy and hold ends at test_start_date open
            is_indices = self.df.index[self.df.index < test_start_date]
            if len(is_indices) > 0:
                bh_start_is = self.df['open'].iloc[start_idx]
                bh_end_is = self.df.loc[is_indices[-1], 'close']
                bh_is_return = calculate_bh_net_return(bh_start_is, bh_end_is, Config.STARTING_CAPITAL)
            else:
                bh_is_return = 0.0
        else:
            bh_is_return = bh_full_return
            bh_oos_return = 0.0
            
        metrics = {
            'starting_capital': Config.STARTING_CAPITAL,
            'final_capital_after_costs': cash,
            'trades': trades,
            'equity_curve': equity_series,
            'is_metrics': is_metrics,
            'oos_metrics': oos_metrics,
            'full_metrics': full_metrics,
            'bh_full_return_pct': bh_full_return,
            'bh_is_return_pct': bh_is_return,
            'bh_oos_return_pct': bh_oos_return,
            'oos_start_date': test_start_date
        }
        
        logger.info(f"Backtest completed. Final Capital: {cash:.2f}, Trades: {len(trades)}, Win Rate: {full_metrics['win_rate']*100:.2f}%, Max Drawdown: {full_metrics['max_drawdown_pct']:.2f}%")
        return metrics
