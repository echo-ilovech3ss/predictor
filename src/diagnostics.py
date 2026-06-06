import pandas as pd
import numpy as np
from src.logger import logger
from src.backtester import Backtester
from config import Config

class ModelDiagnostics:
    """
    Computes advanced research diagnostics using Out-of-Sample (OOS) walk-forward data only.
    Provides safety guards for low-sample edge cases and handles MAE/MFE and regime calculations.
    """
    
    def __init__(self, ml_results: dict, rule_results: dict, symbol: str = None):
        self.ml_results = ml_results
        self.rule_results = rule_results
        self.df = ml_results.get('df', pd.DataFrame())
        self.pred_prob_up = ml_results.get('pred_prob_up', pd.Series(dtype=float))
        self.pred_prob_down = ml_results.get('pred_prob_down', pd.Series(dtype=float))
        self.symbol = symbol.upper() if symbol else 'ASSET'
        
        # Prepare clean aligned OOS targets and predictions
        self.y_true_clean = pd.Series(dtype=int)
        self.y_prob_clean = pd.Series(dtype=float)
        self.trade_mae_mfe_list = []
        self._align_oos_data()
        
    def _align_oos_data(self):
        """Align OOS predictions with true forward lookahead labels, dropping boundary NaNs."""
        if self.df.empty or self.pred_prob_up.empty:
            return
            
        # Target logic: rises >= 0.05% over next 4 hours
        target_series = (self.df['close'].shift(-4) > self.df['close'] * 1.0005)
        # Convert to float and map NaNs for the last 4 rows where future close is unknown
        target_series = target_series.map({True: 1.0, False: 0.0})
        
        # Align with the prediction index
        aligned = pd.DataFrame({'y_true': target_series, 'y_prob': self.pred_prob_up}).dropna()
        self.y_true_clean = aligned['y_true'].astype(int)
        self.y_prob_clean = aligned['y_prob']
        
    def get_oos_classification_metrics(self, threshold: float = 0.60) -> dict:
        """Compute standard classification metrics, handling low-sample edge cases safely."""
        metrics = {
            'accuracy': 'N/A',
            'precision': 'N/A',
            'recall': 'N/A',
            'roc_auc': 'N/A',
            'pr_auc': 'N/A',
            'brier_score': 'N/A',
            'calibration_curve': ([], [])
        }
        
        if self.y_true_clean.empty or self.y_prob_clean.empty:
            return metrics
            
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score, brier_score_loss
            from sklearn.calibration import calibration_curve
            
            y_true = self.y_true_clean.values
            y_prob = self.y_prob_clean.values
            y_pred = (y_prob >= threshold).astype(int)
            
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
            
            # Precision is N/A if no trades are predicted
            if y_pred.sum() > 0:
                metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
            else:
                metrics['precision'] = 'N/A (No trades predicted)'
                
            metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
            metrics['brier_score'] = float(brier_score_loss(y_true, y_prob))
            
            # ROC and PR AUC require both classes in the ground truth
            if len(np.unique(y_true)) > 1:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
                metrics['pr_auc'] = float(average_precision_score(y_true, y_prob))
                
                # Calibration curve (uniform binning)
                prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5, strategy='uniform')
                metrics['calibration_curve'] = (prob_true.tolist(), prob_pred.tolist())
            else:
                metrics['roc_auc'] = 'N/A (Only 1 class present in targets)'
                metrics['pr_auc'] = 'N/A (Only 1 class present in targets)'
                
        except Exception as e:
            logger.warning(f"Error calculating OOS classification metrics: {e}")
            
        return metrics
        
    def get_probability_distribution(self) -> dict:
        """Compute predicted probability histogram counts and threshold frequencies."""
        dist = {
            'histogram_bins': [],
            'histogram_counts': [],
            'threshold_frequencies': {}
        }
        
        if self.y_prob_clean.empty:
            return dist
            
        # Histogram counts
        counts, bins = np.histogram(self.y_prob_clean, bins=10, range=(0.0, 1.0))
        dist['histogram_bins'] = bins.tolist()
        dist['histogram_counts'] = counts.tolist()
        
        # Frequencies at key thresholds
        thresholds = [0.52, 0.55, 0.60, 0.65, 0.70]
        total_samples = len(self.y_prob_clean)
        
        for t in thresholds:
            count = int((self.y_prob_clean >= t).sum())
            pct = (count / total_samples) * 100
            dist['threshold_frequencies'][t] = {'count': count, 'percentage': pct}
            
        return dist
        
    def get_threshold_sensitivity(self) -> list:
        """Evaluate backtest returns and trade counts across static thresholds without retraining."""
        sensitivity = []
        if self.pred_prob_up.empty or self.df.empty:
            return sensitivity
            
        thresholds = [0.52, 0.55, 0.58, 0.60, 0.65]
        first_oos_time = self.pred_prob_up.index[0]
        oos_start_idx = self.df.index.get_loc(first_oos_time)
        last_oos_time = self.pred_prob_up.index[-1]
        oos_end_idx = self.df.index.get_loc(last_oos_time)
        
        df_backtest = self.df.iloc[max(0, oos_start_idx - 200) : oos_end_idx + 1]
        
        for t in thresholds:
            backtester = Backtester(df_backtest, self.symbol)
            # Create a static threshold series for OOS
            dynamic_thresholds = pd.Series(t, index=self.pred_prob_up.index)
            
            try:
                res = backtester.run_backtest(
                    ml_model=None,
                    test_start_date=first_oos_time,
                    pred_prob_up=self.pred_prob_up,
                    pred_prob_down=self.pred_prob_down,
                    dynamic_thresholds=dynamic_thresholds
                )
                
                oos_metrics = res.get('oos_metrics', {})
                sensitivity.append({
                    'threshold': t,
                    'trade_count': oos_metrics.get('trade_count', 0),
                    'win_rate': oos_metrics.get('win_rate', 0.0) * 100,
                    'net_return': oos_metrics.get('total_return_pct', 0.0),
                    'max_drawdown': oos_metrics.get('max_drawdown_pct', 0.0)
                })
            except Exception as e:
                logger.error(f"Error evaluating threshold {t}: {e}")
                
        return sensitivity
        
    def get_feature_importances(self) -> dict:
        """Retrieve aggregated feature importances across walk-forward folds, checking for availability."""
        importances = self.ml_results.get('feature_importances', {})
        if not importances:
            return {'top_15': [], 'cross_market': 'N/A'}
            
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        top_15 = sorted_imp[:15]
        
        # Check cross market features
        cross_market_features = ['spy_returns_lag', 'spy_close_dist']
        cross_market_sum = 0.0
        found_cross = False
        
        for feat in cross_market_features:
            if feat in importances:
                cross_market_sum += importances[feat]
                found_cross = True
                
        cross_market_val = f"{cross_market_sum * 100:.2f}%" if found_cross else "N/A (Not present)"
        
        return {
            'top_15': top_15,
            'cross_market': cross_market_val
        }
        
    def get_trade_diagnostics(self) -> dict:
        """Compute MAE/MFE statistics, holding periods, time in market, and bad entries vs exits."""
        diagnostics = {
            'strategy_exposure_pct': 0.0,
            'bh_exposure_pct': 100.0,
            'avg_holding_period_hours': 0.0,
            'total_losing_trades': 0,
            'bad_entries_count': 0,
            'bad_exits_count': 0,
            'avg_win_mfe_pct': 0.0,
            'avg_win_mae_pct': 0.0,
            'avg_loss_mfe_pct': 0.0,
            'avg_loss_mae_pct': 0.0
        }
        
        # Filter trades that occurred during the OOS period
        if self.pred_prob_up.empty or self.df.empty:
            return diagnostics
            
        oos_start_date = self.pred_prob_up.index[0]
        oos_trades = [t for t in self.ml_results.get('trades', []) if t.get('entry_time') >= oos_start_date]
        
        if not oos_trades:
            # Time in market is 0
            return diagnostics
            
        # Calculate strategy exposure (Time-in-market)
        total_oos_bars = len(self.pred_prob_up)
        bars_in_market = 0
        holding_periods = []
        
        win_mfes, win_maes = [], []
        loss_mfes, loss_maes = [], []
        
        bad_entries = 0
        bad_exits = 0
        total_losses = 0
        
        for t in oos_trades:
            try:
                entry_time = t['entry_time']
                exit_time = t['exit_time']
                entry_price = t['entry_price']
                exit_price = t['exit_price']
                
                # Slices candles safely
                entry_idx = self.df.index.get_loc(entry_time)
                exit_idx = self.df.index.get_loc(exit_time)
                
                duration = exit_idx - entry_idx
                bars_in_market += duration
                holding_periods.append(duration) # Assuming 1-hour candles, duration is in hours
                
                # Fetch intratrade candles safely
                sliced_candles = self.df.iloc[entry_idx : exit_idx + 1]
                
                if len(sliced_candles) > 1:
                    # Exclude the exit candle's high/low since we exit at its open
                    mid_candles = sliced_candles.iloc[:-1]
                    highest_price = float(max(entry_price, exit_price, mid_candles['high'].max()))
                    lowest_price = float(min(entry_price, exit_price, mid_candles['low'].min()))
                else:
                    highest_price = float(max(entry_price, exit_price))
                    lowest_price = float(min(entry_price, exit_price))
                    
                mfe = ((highest_price - entry_price) / entry_price) * 100
                mae = ((entry_price - lowest_price) / entry_price) * 100
                
                is_win = t['net_pnl'] > 0
                if is_win:
                    win_mfes.append(mfe)
                    win_maes.append(mae)
                else:
                    loss_mfes.append(mfe)
                    loss_maes.append(mae)
                    total_losses += 1
                    
                    # Classify Bad Entry vs Bad Exit
                    # Bad exit: rose > 0.5% but still ended in loss (failure to take profits)
                    if mfe >= 0.5:
                        bad_exits += 1
                    # Bad entry: went straight down without ever rising above 0.1%
                    elif mae >= (Config.STOP_LOSS_PCT * 100) and mfe < 0.1:
                        bad_entries += 1
                        
                # Store detail for plotting
                self.trade_mae_mfe_list.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'net_pnl': t['net_pnl'],
                    'mae': mae,
                    'mfe': mfe,
                    'is_win': is_win
                })
            except Exception as e:
                logger.error(f"Error calculating MAE/MFE for trade: {e}")
                
        diagnostics['strategy_exposure_pct'] = (bars_in_market / total_oos_bars) * 100
        diagnostics['avg_holding_period_hours'] = np.mean(holding_periods) if holding_periods else 0.0
        diagnostics['total_losing_trades'] = total_losses
        diagnostics['bad_entries_count'] = bad_entries
        diagnostics['bad_exits_count'] = bad_exits
        
        diagnostics['avg_win_mfe_pct'] = np.mean(win_mfes) if win_mfes else 0.0
        diagnostics['avg_win_mae_pct'] = np.mean(win_maes) if win_maes else 0.0
        diagnostics['avg_loss_mfe_pct'] = np.mean(loss_mfes) if loss_mfes else 0.0
        diagnostics['avg_loss_mae_pct'] = np.mean(loss_maes) if loss_maes else 0.0
        
        return diagnostics
        
    def get_regime_analysis(self) -> dict:
        """Analyze strategy performance grouped by market regime at the time of entry only."""
        regimes = {}
        if self.pred_prob_up.empty or self.df.empty:
            return regimes
            
        oos_start_date = self.pred_prob_up.index[0]
        oos_trades = [t for t in self.ml_results.get('trades', []) if t.get('entry_time') >= oos_start_date]
        
        for t in oos_trades:
            try:
                entry_time = t['entry_time']
                entry_state = self.df.loc[entry_time, 'market_state']
                
                if entry_state not in regimes:
                    regimes[entry_state] = {'trades_count': 0, 'wins': 0, 'net_pnl': 0.0}
                    
                regimes[entry_state]['trades_count'] += 1
                if t['net_pnl'] > 0:
                    regimes[entry_state]['wins'] += 1
                regimes[entry_state]['net_pnl'] += t['net_pnl']
            except Exception as e:
                logger.error(f"Regime calculation error: {e}")
                
        # Calculate win rates
        for state, data in regimes.items():
            data['win_rate'] = (data['wins'] / data['trades_count']) * 100 if data['trades_count'] > 0 else 0.0
            
        return regimes
        
    def get_missed_rallies(self) -> list:
        """Identify distinct, non-overlapping rolling 48-bar buy-and-hold rallies (>2.0%) where strategy was flat."""
        missed_rallies = []
        if self.pred_prob_up.empty or self.df.empty:
            return missed_rallies
            
        # Define OOS slice
        oos_start_date = self.pred_prob_up.index[0]
        oos_df = self.df.loc[oos_start_date:]
        
        # 1. Create a boolean mask of whether the strategy is in an active position for each bar in OOS
        in_market = pd.Series(False, index=oos_df.index)
        oos_trades = [t for t in self.ml_results.get('trades', []) if t.get('entry_time') >= oos_start_date]
        
        for t in oos_trades:
            try:
                in_market.loc[t['entry_time'] : t['exit_time']] = True
            except Exception:
                pass
                
        # 2. Find all 48-bar windows with returns > 2.0% where strategy was completely flat
        close_series = oos_df['close']
        n_bars = 48
        
        flat_rally_windows = []
        for i in range(n_bars, len(oos_df)):
            t_end = oos_df.index[i]
            t_start = oos_df.index[i - n_bars]
            
            # Buy & hold return
            bh_ret = (close_series.iloc[i] - close_series.iloc[i - n_bars]) / close_series.iloc[i - n_bars]
            
            if bh_ret > 0.02:
                # Check if strategy was completely flat (no active positions during the entire window)
                if not in_market.iloc[i - n_bars : i + 1].any():
                    flat_rally_windows.append((t_start, t_end, bh_ret * 100))
                    
        # 3. Merge overlapping missed rally windows into distinct non-overlapping intervals
        if not flat_rally_windows:
            return missed_rallies
            
        merged_rallies = []
        current_start, current_end, max_ret = flat_rally_windows[0]
        
        for next_start, next_end, next_ret in flat_rally_windows[1:]:
            if next_start <= current_end:
                # Overlap! Merge windows
                current_end = max(current_end, next_end)
                max_ret = max(max_ret, next_ret)
            else:
                # No overlap! Save current and start a new interval
                merged_rallies.append((current_start, current_end, max_ret))
                current_start, current_end, max_ret = next_start, next_end, next_ret
                
        merged_rallies.append((current_start, current_end, max_ret))
        
        # Format results
        for start, end, ret in merged_rallies:
            missed_rallies.append({
                'start_time': start.strftime('%Y-%m-%d %H:%M'),
                'end_time': end.strftime('%Y-%m-%d %H:%M'),
                'return': ret
            })
            
        return missed_rallies
        
    def generate_markdown_report(self) -> str:
        """Generate a complete Markdown diagnostics report for CLI output."""
        metrics = self.get_oos_classification_metrics()
        dist = self.get_probability_distribution()
        sensitivity = self.get_threshold_sensitivity()
        importances = self.get_feature_importances()
        trade_diag = self.get_trade_diagnostics()
        regimes = self.get_regime_analysis()
        rallies = self.get_missed_rallies()
        
        # Format metric values safely
        acc = metrics['accuracy']
        acc_str = f"{acc * 100:.2f}%" if isinstance(acc, float) else str(acc)
        
        prec = metrics['precision']
        prec_str = f"{prec * 100:.2f}%" if isinstance(prec, float) else str(prec)
        
        rec = metrics['recall']
        rec_str = f"{rec * 100:.2f}%" if isinstance(rec, float) else str(rec)
        
        auc = metrics['roc_auc']
        auc_str = f"{auc:.4f}" if isinstance(auc, float) else str(auc)
        
        pr_auc = metrics['pr_auc']
        pr_auc_str = f"{pr_auc:.4f}" if isinstance(pr_auc, float) else str(pr_auc)
        
        brier = metrics['brier_score']
        brier_str = f"{brier:.4f}" if isinstance(brier, float) else str(brier)
        
        rep = []
        rep.append("==================================================")
        rep.append(f"       MODEL DIAGNOSTICS REPORT: {self.symbol}")
        rep.append("==================================================")
        rep.append("NOTE: This report analyzes Out-of-Sample (OOS) walk-forward data only.")
        rep.append("--------------------------------------------------")
        rep.append("1. OOS CLASSIFICATION PERFORMANCE METRICS")
        rep.append(f"  Accuracy (threshold 0.60):   {acc_str}")
        rep.append(f"  Precision (threshold 0.60):  {prec_str}")
        rep.append(f"  Recall (threshold 0.60):     {rec_str}")
        rep.append(f"  ROC AUC Score:               {auc_str}")
        rep.append(f"  Precision-Recall AUC:        {pr_auc_str}")
        rep.append(f"  Brier Score Loss:            {brier_str}")
        
        rep.append("--------------------------------------------------")
        rep.append("2. PROBABILITY DISTRIBUTION ANALYSIS")
        rep.append("  Confidence Threshold Frequencies:")
        for t, data in dist['threshold_frequencies'].items():
            rep.append(f"    Exceeds {t:.2f}:             {data['count']} times ({data['percentage']:.2f}%)")
            
        rep.append("--------------------------------------------------")
        rep.append("3. THRESHOLD SENSITIVITY SCAN (DIAGNOSTIC ONLY)")
        rep.append("  Threshold | Trades Count | Win Rate | Net Return | Max DD")
        for s in sensitivity:
            rep.append(f"    {s['threshold']:.2f}    |      {s['trade_count']:<7} | {s['win_rate']:>7.2f}% | {s['net_return']:>9.2f}% | {s['max_drawdown']:>6.2f}%")
            
        rep.append("--------------------------------------------------")
        rep.append("4. FEATURE IMPORTANCE (AGGREGATED OVER WF FOLDS)")
        rep.append("  Top Features:")
        for rank, (name, val) in enumerate(importances['top_15'], 1):
            rep.append(f"    Rank {rank:<2} | {name:<20} : {val*100:.2f}%")
        rep.append(f"  Cross-Market Feature Weight: {importances['cross_market']}")
        
        rep.append("--------------------------------------------------")
        rep.append("5. TRADE & EXPOSURE DIAGNOSTICS")
        rep.append(f"  Strategy Time-in-Market:     {trade_diag['strategy_exposure_pct']:.2f}%")
        rep.append(f"  Buy & Hold Time-in-Market:   {trade_diag['bh_exposure_pct']:.2f}%")
        rep.append(f"  Average Holding Period:      {trade_diag['avg_holding_period_hours']:.1f} hours")
        rep.append(f"  Total Losing Trades:         {trade_diag['total_losing_trades']}")
        rep.append(f"  - Bad Entries Count:         {trade_diag['bad_entries_count']} (MAE triggered Stop Loss, no upward excursion)")
        rep.append(f"  - Bad Exits Count:           {trade_diag['bad_exits_count']} (MFE > 0.5%, failed to capture profits)")
        rep.append(f"  Average Win  - MFE: {trade_diag['avg_win_mfe_pct']:.2f}% | MAE: {trade_diag['avg_win_mae_pct']:.2f}%")
        rep.append(f"  Average Loss - MFE: {trade_diag['avg_loss_mfe_pct']:.2f}% | MAE: {trade_diag['avg_loss_mae_pct']:.2f}%")
        if trade_diag['strategy_exposure_pct'] < 5.0:
            rep.append("  WARNING: Strategy exposure is extremely low (< 5%). Returns are mostly a cash-holding avoidance strategy.")
            
        rep.append("--------------------------------------------------")
        rep.append("6. REGIME PERFORMANCE ANALYSIS")
        rep.append("  Entry Regime | Trades Count | Win Rate | Net PnL")
        for state, data in regimes.items():
            rep.append(f"    {state:<10} |      {data['trades_count']:<7} | {data['win_rate']:>7.2f}% | {data['net_pnl']:>7.2f}")
            
        rep.append("--------------------------------------------------")
        rep.append("7. MISSED BUY-AND-HOLD RALLIES (>2.0% over 48 bars)")
        rep.append(f"  Total Missed Rallies:        {len(rallies)}")
        for i, r in enumerate(rallies[:5], 1):
            rep.append(f"    Rally {i}: {r['start_time']} to {r['end_time']} | Return: +{r['return']:.2f}% (Bot was flat)")
        if len(rallies) > 5:
            rep.append(f"    ... and {len(rallies) - 5} more missed rallies.")
        rep.append("==================================================")
        
        return "\n".join(rep)
