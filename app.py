import streamlit as st
import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

from config import Config
from src.logger import logger
from src.alerts import AlertSystem
from src.data_fetcher import YFinanceProvider
from src.indicators import calculate_indicators
from src.features import prepare_data_for_training, extract_features
from src.market_state import classify_market_state_row, classify_market_states
from src.ml_model import MarketMLModel
from src.strategy import Strategy
from src.backtester import Backtester
from src.paper_trader import PaperTrader
from src.costs import TAX_DISCLAIMER
from main import run_walk_forward

# Page Configuration
st.set_page_config(
    page_title="AlphaPredict | Market Prediction & Paper Trading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Injection
st.markdown("""
<style>
    /* Font style overrides */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        background: linear-gradient(135deg, #a78bfa, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }
    
    .card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: #f8fafc;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Recommendations styling */
    .recommendation-buy {
        background-color: rgba(16, 185, 129, 0.15);
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .recommendation-sell {
        background-color: rgba(239, 68, 68, 0.15);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .recommendation-hold {
        background-color: rgba(100, 116, 139, 0.15);
        border: 2px solid #64748b;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .rec-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 2.2rem;
        margin: 0;
    }
</style>
""", unsafe_allowed_html=True)

# Helper function to load recent data
@st.cache_data(ttl=300)  # cache data for 5 minutes
def fetch_recent_data(symbol: str):
    provider = YFinanceProvider()
    # Fetch 60 days to ensure we have enough points for indicators (need 200 candles)
    start = datetime.datetime.now() - datetime.timedelta(days=60)
    end = datetime.datetime.now()
    try:
        df = provider.fetch_data(symbol, start, end)
        if df.empty:
            return None
        df_with_ind = calculate_indicators(df)
        return df_with_ind
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None

# Helper function to cache walk-forward simulation
@st.cache_data(show_spinner="Running Walk-Forward Simulation (expensive, will be cached)...")
def get_cached_walk_forward(symbol: str):
    return run_walk_forward(symbol)


# App Layout
st.markdown("<div class='main-title'>AlphaPredict Bot</div>", unsafe_allowed_html=True)
st.markdown("<div class='sub-title'>Modular ML-guided Market Prediction & Cautious Paper Trading</div>", unsafe_allowed_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.image("https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?auto=format&fit=crop&q=80&w=200&h=100", use_container_width=True)
st.sidebar.header("Navigation & Settings")

# 1. Market Selection
symbol_selection = st.sidebar.selectbox(
    "Select Target Market",
    options=["SPY", "NIFTY"],
    index=0,
    help="SPY (US S&P 500 ETF) or NIFTY (India NIFTY 50 Index)"
)

# Sidebar action buttons
st.sidebar.subheader("Actions")

# 2. Reset Paper Trading
if st.sidebar.button("Reset Paper Portfolio", type="secondary", help="Restores simulated account balance to starting capital and drops positions."):
    pt = PaperTrader(symbol_selection)
    pt.reset_account()
    st.sidebar.success(f"Paper portfolio for {symbol_selection} reset to defaults!")

# 3. Train Model
if st.sidebar.button("Train ML Model Now", type="primary"):
    with st.spinner(f"Fetching 730 days of data and training model for {symbol_selection}..."):
        try:
            # Import train function logic or run directly
            provider = YFinanceProvider()
            start = datetime.datetime.now() - datetime.timedelta(days=729)
            end = datetime.datetime.now()
            df = provider.fetch_data(symbol_selection, start, end)
            if not df.empty:
                df_with_ind = calculate_indicators(df)
                X, y = prepare_data_for_training(df_with_ind)
                ml_model = MarketMLModel(symbol_selection)
                res = ml_model.train(X, y)
                st.sidebar.success(f"Model trained! Accuracy: {res['accuracy']*100:.2f}%")
            else:
                st.sidebar.error("Failed to fetch historical data for training.")
        except Exception as e:
            st.sidebar.error(f"Training failed: {e}")

# Configuration Parameter Panel
with st.sidebar.expander("System Configuration Parameters"):
    st.write(f"**Capital**: ${Config.STARTING_CAPITAL:,.2f}")
    st.write(f"**Trade Threshold**: {Config.MIN_CONFIDENCE_FOR_TRADE*100:.1f}%")
    st.write(f"**Stop Loss Limit**: {Config.STOP_LOSS_PCT*100:.1f}%")
    st.write(f"**Daily Loss Limit**: {Config.DAILY_LOSS_LIMIT_PCT*100:.1f}%")
    st.write(f"**Slippage Assumption**: {Config.SLIPPAGE_PCT*100:.2f}%")
    st.write(f"**Brokerage Rate**: {Config.BROKERAGE_PCT*100:.2f}%")
    st.write(f"**Short Term Tax Rate**: {Config.TAX_ON_PROFIT_PCT*100:.1f}%")

# Disclaimer Display
st.sidebar.warning(f"**Disclaimer & Risk Warning**\n\n{TAX_DISCLAIMER}\n\nThis is a simulation platform. Past performance does not guarantee future results.")

# ----------------- MAIN PANEL -----------------

# Fetch latest data
df_data = fetch_recent_data(symbol_selection)

if df_data is None or df_data.empty:
    st.error("❌ Data Fetching Failed: Unable to retrieve market data. Check your internet connection or API availability.")
    st.info("System operations are halted because recent market information is invalid or stale.")
    st.stop()

# Check active alerts
active_alerts = AlertSystem.get_active_alerts()
if active_alerts:
    for alert in active_alerts:
        st.warning(f"⚠️ {alert}")
    if AlertSystem.is_halted():
        st.error("⛔ SYSTEM OPERATIONS HALTED: Please resolve the issues above to resume predictions.")
        st.stop()

# Extract latest state details
latest_row = df_data.iloc[-1]
latest_time = df_data.index[-1]
latest_price = latest_row['close']

# Classify market state
market_state = classify_market_state_row(latest_row.to_dict())
latest_row_dict = latest_row.to_dict()
latest_row_dict['market_state'] = market_state

# Load ML Model
ml_model = MarketMLModel(symbol_selection)
model_exists = ml_model.load()

# Create interactive "Update / Predict" button
col_header, col_btn = st.columns([4, 1])
with col_header:
    st.write(f"### Latest Market Snapshot ({symbol_selection})")
with col_btn:
    update_data = st.button("Update Prediction", use_container_width=True)
    if update_data:
        st.cache_data.clear()
        st.rerun()

# Check market status
provider = YFinanceProvider()
is_open, is_stale, status_msg = provider.check_market_status(symbol_selection, df_data)

# Show alert banner based on market status
if not is_open:
    st.info(f"ℹ️ {status_msg}")
elif is_stale:
    st.warning(f"⚠️ {status_msg}")
else:
    st.success(f"🟢 {status_msg}")

# Run prediction if model exists
if model_exists:
    features_df = extract_features(df_data)
    latest_features = features_df.iloc[[-1]]
    try:
        probas = ml_model.predict_proba(latest_features)
        prob_down, prob_up = probas[0], probas[1]
    except Exception as e:
        st.error(f"Error predicting probabilities: {e}")
        prob_down, prob_up = 0.5, 0.5
else:
    st.warning("⚠️ No trained ML model found. Showing indicator-only rules. Please click 'Train ML Model Now' in the sidebar.")
    prob_down, prob_up = 0.5, 0.5

# Strategy generation
strat = Strategy(min_confidence=ml_model.optimal_threshold if model_exists else None)

if not is_open or is_stale:
    signal = {
        "action": "HOLD",
        "confidence": 0.50,
        "prob_up": 0.50,
        "prob_down": 0.50,
        "market_state": market_state,
        "explanation": f"HOLD: Prediction forced to HOLD because market is closed or data is stale ({status_msg})."
    }
else:
    signal = strat.generate_signal(latest_row_dict, prob_up, prob_down, use_ml=model_exists)

# Update paper trader log
paper_trader = PaperTrader(symbol_selection)
time_str = latest_time.strftime("%Y-%m-%d %H:%M:%S")
paper_trader.process_signal(signal, latest_row_dict, time_str, is_market_open=is_open, is_data_stale=is_stale)

# Row 1: KPI metrics
col_rec, col_state, col_probs = st.columns([2, 1, 2])

with col_rec:
    action = signal['action']
    if action == "BUY":
        st.markdown(f"""
        <div class='recommendation-buy'>
            <div class='metric-label' style='color:#a7f3d0;'>Recommended Action</div>
            <div class='rec-title' style='color:#10b981;'>BUY</div>
        </div>
        """, unsafe_allowed_html=True)
    elif action == "SELL":
        st.markdown(f"""
        <div class='recommendation-sell'>
            <div class='metric-label' style='color:#fca5a5;'>Recommended Action</div>
            <div class='rec-title' style='color:#ef4444;'>SELL</div>
        </div>
        """, unsafe_allowed_html=True)
    else:
        st.markdown(f"""
        <div class='recommendation-hold'>
            <div class='metric-label' style='color:#cbd5e1;'>Recommended Action</div>
            <div class='rec-title' style='color:#94a3b8;'>HOLD</div>
        </div>
        """, unsafe_allowed_html=True)

with col_state:
    state_color = "#3b82f6"
    if market_state == "bullish":
        state_color = "#10b981"
    elif market_state == "bearish":
        state_color = "#ef4444"
    elif market_state == "volatile":
        state_color = "#f59e0b"
    elif market_state == "sideways":
        state_color = "#a855f7"
        
    st.markdown(f"""
    <div class='card' style='text-align:center;'>
        <div class='metric-label'>Market State</div>
        <div class='metric-value' style='color:{state_color};'>{market_state.upper()}</div>
    </div>
    """, unsafe_allowed_html=True)

with col_probs:
    # Display confidence and probabilities
    st.markdown(f"""
    <div class='card'>
        <div class='metric-label'>Model Probabilities</div>
        <div style='display:flex; justify-content:space-between; margin-top:0.5rem;'>
            <div><span style='color:#10b981; font-weight:600;'>Probability UP:</span> {prob_up*100:.1f}%</div>
            <div><span style='color:#ef4444; font-weight:600;'>Probability DOWN:</span> {prob_down*100:.1f}%</div>
        </div>
        <div style='margin-top:0.8rem; font-size:0.9rem; color:#94a3b8;'>
            Confidence Score: <b>{signal['confidence']*100:.1f}%</b>
        </div>
    </div>
    """, unsafe_allowed_html=True)

# Explanation Card
st.markdown(f"""
<div class='card'>
    <div class='metric-label'>Signal Explanation</div>
    <div style='margin-top:0.5rem; font-size:1.05rem; line-height:1.5; color:#cbd5e1;'>
        {signal['explanation']}
    </div>
</div>
""", unsafe_allowed_html=True)

# Row 2: Price Chart & Indicators
st.markdown("### Interactive Market Chart (Recent 150 Candles)")

# Take last 150 candles for readability
chart_df = df_data.iloc[-150:]

# Create Plotly figure with secondary y-axis for volume/indicators
fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.08, 
    row_heights=[0.7, 0.3],
    subplot_titles=("Price & EMAs", "RSI / MACD")
)

# 1. Candlestick
fig.add_trace(
    go.Candlestick(
        x=chart_df.index,
        open=chart_df['open'],
        high=chart_df['high'],
        low=chart_df['low'],
        close=chart_df['close'],
        name="Candles"
    ),
    row=1, col=1
)

# 2. EMAs
fig.add_trace(
    go.Scatter(x=chart_df.index, y=chart_df['ema_20'], line=dict(color='#f43f5e', width=1.5), name="EMA 20"),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=chart_df.index, y=chart_df['ema_50'], line=dict(color='#3b82f6', width=1.5), name="EMA 50"),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=chart_df.index, y=chart_df['ema_200'], line=dict(color='#10b981', width=1.5), name="EMA 200"),
    row=1, col=1
)

# 3. Bollinger Bands (shaded area)
fig.add_trace(
    go.Scatter(x=chart_df.index, y=chart_df['bb_upper'], line=dict(dash='dash', color='rgba(148, 163, 184, 0.3)'), name="BB Upper"),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=chart_df.index, y=chart_df['bb_lower'], line=dict(dash='dash', color='rgba(148, 163, 184, 0.3)'), fill='tonexty', fillcolor='rgba(148, 163, 184, 0.05)', name="BB Lower"),
    row=1, col=1
)

# 4. RSI (row 2)
fig.add_trace(
    go.Scatter(x=chart_df.index, y=chart_df['rsi'], line=dict(color='#a855f7', width=1.5), name="RSI"),
    row=2, col=1
)
# Overbought/Oversold lines
fig.add_trace(
    go.Scatter(x=chart_df.index, y=[70]*len(chart_df), line=dict(color='#ef4444', dash='dot', width=1), name="Overbought (70)"),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=chart_df.index, y=[30]*len(chart_df), line=dict(color='#10b981', dash='dot', width=1), name="Oversold (30)"),
    row=2, col=1
)

fig.update_layout(
    height=650,
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    margin=dict(l=40, r=40, t=40, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# Row 3: Paper Portfolio Status
st.markdown("### Simulated Paper Portfolio")
st.warning("⚠️ **Simulated Next-Bar Execution**: When a signal is generated at candle $t$ close, the simulated fill occurs at candle $t+1$ open. True real-time next-open execution would require a live quote/broker feed.")

pt_col1, pt_col2, pt_col3 = st.columns(3)
with pt_col1:
    st.markdown(f"""
    <div class='card'>
        <div class='metric-label'>Simulated Cash Balance</div>
        <div class='metric-value'>${paper_trader.cash:,.2f}</div>
    </div>
    """, unsafe_allowed_html=True)
with pt_col2:
    if paper_trader.position:
        qty = paper_trader.position['qty']
        entry_price = paper_trader.position['entry_price']
        pos_val = qty * latest_price
        st.markdown(f"""
        <div class='card'>
            <div class='metric-label'>Open Position Value</div>
            <div class='metric-value' style='color:#10b981;'>${pos_val:,.2f}</div>
            <div style='font-size:0.85rem; color:#cbd5e1; margin-top:0.25rem;'>
                {qty:.2f} shares entered at ${entry_price:,.2f} | Stop Loss: ${paper_trader.position['stop_loss']:,.2f}
            </div>
        </div>
        """, unsafe_allowed_html=True)
    else:
        st.markdown(f"""
        <div class='card'>
            <div class='metric-label'>Open Position Value</div>
            <div class='metric-value'>$0.00</div>
            <div style='font-size:0.85rem; color:#94a3b8; margin-top:0.25rem;'>Currently Flat (No active holdings)</div>
        </div>
        """, unsafe_allowed_html=True)
with pt_col3:
    pt_total_value = paper_trader.get_equity(latest_price)
    return_pct = ((pt_total_value - Config.STARTING_CAPITAL) / Config.STARTING_CAPITAL) * 100
    pt_color = "#10b981" if return_pct >= 0 else "#ef4444"
    st.markdown(f"""
    <div class='card'>
        <div class='metric-label'>Total Portfolio Value</div>
        <div class='metric-value' style='color:{pt_color};'>${pt_total_value:,.2f}</div>
        <div style='font-size:0.85rem; color:#cbd5e1; margin-top:0.25rem;'>
            Total Return: {return_pct:+.2f}% since init
        </div>
    </div>
    """, unsafe_allowed_html=True)

# Trade History
with st.expander("Show Paper Trading Logs & Completed Trades"):
    if paper_trader.trades:
        trades_df = pd.DataFrame(paper_trader.trades)
        # Re-format headers
        trades_df = trades_df.rename(columns={
            "entry_time": "Entry Time",
            "exit_time": "Exit Time",
            "entry_price": "Entry Price",
            "exit_price": "Exit Price",
            "qty": "Quantity",
            "gross_pnl": "Gross PnL",
            "fees": "Fees Paid",
            "tax": "Taxes Paid",
            "net_pnl": "Net PnL",
            "reason": "Exit Reason"
        })
        st.dataframe(trades_df.sort_index(ascending=False), use_container_width=True)
    else:
        st.info("No completed paper trades recorded yet.")

# Row 4: Historical Backtest Engine
st.markdown("### Historical Backtest Analysis")
run_backtest_btn = st.button("Run Backtest on Historical Data", type="secondary")

if run_backtest_btn:
    with st.spinner("Executing historical backtests. Please wait..."):
        try:
            # Fetch 730 days
            provider = YFinanceProvider()
            start_back = datetime.datetime.now() - datetime.timedelta(days=729)
            end_back = datetime.datetime.now()
            
            backtest_data = provider.fetch_data(symbol_selection, start_back, end_back, is_live=False)
            if not backtest_data.empty:
                backtest_data = calculate_indicators(backtest_data)
                backtest_data['market_state'] = classify_market_states(backtest_data)
                
                # Setup backtester
                backtester = Backtester(backtest_data, symbol_selection)
                
                # Split boundary (last 20%)
                split_idx = int(len(backtest_data) * 0.8)
                test_start_date = backtest_data.index[split_idx]
                
                # Rule only
                rule_results = backtester.run_backtest(ml_model=None, test_start_date=test_start_date)
                
                # ML only (if model exists)
                ml_results = None
                if model_exists:
                    ml_results = backtester.run_backtest(ml_model, test_start_date=test_start_date)
                    
                # Display statistics side by side
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Performance Metrics by Period")
                    
                    # Display statistics in tabs
                    tab_is, tab_oos, tab_full = st.tabs(["In-Sample (IS) Period", "Out-of-Sample (OOS) Period", "Full Backtest Period"])
                    
                    with tab_is:
                        st.write(f"**IS Period**: Start to {test_start_date.date()}")
                        rule_is = rule_results['is_metrics']
                        bh_is_ret = rule_results['bh_is_return_pct']
                        
                        if rule_is.get('warning_msg'):
                            st.warning(f"⚠️ **Rule-Only Baseline**: {rule_is['warning_msg']}")
                        
                        metrics_is_data = {
                            "Metric": [
                                "Net Strategy Return",
                                "Buy & Hold Return",
                                "Trades Count",
                                "Win Rate",
                                "Max Drawdown"
                            ],
                            "Rule-Only Strategy": [
                                f"{rule_is.get('total_return_pct', 0):+.2f}%",
                                f"{bh_is_ret:+.2f}%",
                                f"{rule_is.get('trade_count', 0)}",
                                f"{rule_is.get('win_rate', 0)*100:.1f}%",
                                f"{rule_is.get('max_drawdown_pct', 0):.2f}%"
                            ]
                        }
                        
                        if ml_results:
                            ml_is = ml_results['is_metrics']
                            if ml_is.get('warning_msg'):
                                st.warning(f"⚠️ **ML-Guided Strategy**: {ml_is['warning_msg']}")
                            
                            metrics_is_data["ML-Guided Strategy"] = [
                                f"{ml_is.get('total_return_pct', 0):+.2f}%",
                                f"{bh_is_ret:+.2f}%",
                                f"{ml_is.get('trade_count', 0)}",
                                f"{ml_is.get('win_rate', 0)*100:.1f}%",
                                f"{ml_is.get('max_drawdown_pct', 0):.2f}%"
                            ]
                            
                            # Add advanced metrics
                            metrics_is_data["Metric"].extend(["Profit Factor", "Avg Win", "Avg Loss"])
                            metrics_is_data["Rule-Only Strategy"].extend(["N/A", "N/A", "N/A"])
                            metrics_is_data["ML-Guided Strategy"].extend([
                                f"{ml_is.get('profit_factor', 0):.2f}",
                                f"${ml_is.get('avg_win_cash', 0):,.2f} ({ml_is.get('avg_win_pct', 0):+.2f}%)",
                                f"${ml_is.get('avg_loss_cash', 0):,.2f} ({ml_is.get('avg_loss_pct', 0):+.2f}%)"
                            ])
                            
                        st.table(pd.DataFrame(metrics_is_data).set_index("Metric"))
                        
                    with tab_oos:
                        st.write(f"**OOS Period**: {test_start_date.date()} to End")
                        rule_oos = rule_results['oos_metrics']
                        bh_oos_ret = rule_results['bh_oos_return_pct']
                        
                        if rule_oos.get('warning_msg'):
                            st.warning(f"⚠️ **Rule-Only Baseline**: {rule_oos['warning_msg']}")
                            
                        metrics_oos_data = {
                            "Metric": [
                                "Net Strategy Return",
                                "Buy & Hold Return",
                                "Trades Count",
                                "Win Rate",
                                "Max Drawdown"
                            ],
                            "Rule-Only Strategy": [
                                f"{rule_oos.get('total_return_pct', 0):+.2f}%",
                                f"{bh_oos_ret:+.2f}%",
                                f"{rule_oos.get('trade_count', 0)}",
                                f"{rule_oos.get('win_rate', 0)*100:.1f}%",
                                f"{rule_oos.get('max_drawdown_pct', 0):.2f}%"
                            ]
                        }
                        
                        if ml_results:
                            ml_oos = ml_results['oos_metrics']
                            if ml_oos.get('warning_msg'):
                                st.warning(f"⚠️ **ML-Guided Strategy**: {ml_oos['warning_msg']}")
                                
                            metrics_oos_data["ML-Guided Strategy"] = [
                                f"{ml_oos.get('total_return_pct', 0):+.2f}%",
                                f"{bh_oos_ret:+.2f}%",
                                f"{ml_oos.get('trade_count', 0)}",
                                f"{ml_oos.get('win_rate', 0)*100:.1f}%",
                                f"{ml_oos.get('max_drawdown_pct', 0):.2f}%"
                            ]
                            
                            # Add advanced metrics
                            metrics_oos_data["Metric"].extend(["Profit Factor", "Avg Win", "Avg Loss"])
                            metrics_oos_data["Rule-Only Strategy"].extend(["N/A", "N/A", "N/A"])
                            metrics_oos_data["ML-Guided Strategy"].extend([
                                f"{ml_oos.get('profit_factor', 0):.2f}",
                                f"${ml_oos.get('avg_win_cash', 0):,.2f} ({ml_oos.get('avg_win_pct', 0):+.2f}%)",
                                f"${ml_oos.get('avg_loss_cash', 0):,.2f} ({ml_oos.get('avg_loss_pct', 0):+.2f}%)"
                            ])
                            
                        st.table(pd.DataFrame(metrics_oos_data).set_index("Metric"))
                        
                    with tab_full:
                        st.write("**Full Period**: Combined IS + OOS")
                        rule_full = rule_results['full_metrics']
                        bh_full_ret = rule_results['bh_full_return_pct']
                        
                        if rule_full.get('warning_msg'):
                            st.warning(f"⚠️ **Rule-Only Baseline**: {rule_full['warning_msg']}")
                            
                        metrics_full_data = {
                            "Metric": [
                                "Starting Capital",
                                "Net Strategy Return",
                                "Buy & Hold Return",
                                "Trades Count",
                                "Win Rate",
                                "Max Drawdown"
                            ],
                            "Rule-Only Strategy": [
                                f"${Config.STARTING_CAPITAL:,.2f}",
                                f"{rule_full.get('total_return_pct', 0):+.2f}%",
                                f"{bh_full_ret:+.2f}%",
                                f"{rule_full.get('trade_count', 0)}",
                                f"{rule_full.get('win_rate', 0)*100:.1f}%",
                                f"{rule_full.get('max_drawdown_pct', 0):.2f}%"
                            ]
                        }
                        
                        if ml_results:
                            ml_full = ml_results['full_metrics']
                            if ml_full.get('warning_msg'):
                                st.warning(f"⚠️ **ML-Guided Strategy**: {ml_full['warning_msg']}")
                                
                            metrics_full_data["ML-Guided Strategy"] = [
                                f"${Config.STARTING_CAPITAL:,.2f}",
                                f"{ml_full.get('total_return_pct', 0):+.2f}%",
                                f"{bh_full_ret:+.2f}%",
                                f"{ml_full.get('trade_count', 0)}",
                                f"{ml_full.get('win_rate', 0)*100:.1f}%",
                                f"{ml_full.get('max_drawdown_pct', 0):.2f}%"
                            ]
                            
                            # Add advanced metrics
                            metrics_full_data["Metric"].extend(["Profit Factor", "Avg Win", "Avg Loss"])
                            metrics_full_data["Rule-Only Strategy"].extend(["N/A", "N/A", "N/A"])
                            metrics_full_data["ML-Guided Strategy"].extend([
                                f"{ml_full.get('profit_factor', 0):.2f}",
                                f"${ml_full.get('avg_win_cash', 0):,.2f} ({ml_full.get('avg_win_pct', 0):+.2f}%)",
                                f"${ml_full.get('avg_loss_cash', 0):,.2f} ({ml_full.get('avg_loss_pct', 0):+.2f}%)"
                            ])
                            
                        st.table(pd.DataFrame(metrics_full_data).set_index("Metric"))
                        
                with col2:
                    st.markdown("#### Equity Curve Comparison")
                    # Plot equity curves
                    eq_fig = go.Figure()
                    
                    # Rule curve
                    rule_eq = rule_results['equity_curve']
                    eq_fig.add_trace(go.Scatter(
                        x=rule_eq.index, y=rule_eq.values,
                        mode='lines', name='Rule-Only Baseline',
                        line=dict(color='#a855f7', width=1.5)
                    ))
                    
                    # ML curve
                    if ml_results:
                        ml_eq = ml_results['equity_curve']
                        eq_fig.add_trace(go.Scatter(
                            x=ml_eq.index, y=ml_eq.values,
                            mode='lines', name='ML-Guided Strategy',
                            line=dict(color='#10b981', width=2)
                        ))
                        
                    # Buy and Hold (Simulated curve from buy to end)
                    bh_final_cap = Config.STARTING_CAPITAL * (1 + bh_full_ret / 100)
                    eq_fig.add_trace(go.Scatter(
                        x=[rule_eq.index[0], rule_eq.index[-1]],
                        y=[Config.STARTING_CAPITAL, bh_final_cap],
                        mode='lines+markers', name='Buy & Hold (Net)',
                        line=dict(color='#3b82f6', width=1, dash='dash')
                    ))
                    
                    # Add vertical line for OOS boundary
                    eq_fig.add_vline(x=test_start_date, line_width=1.5, line_dash="dash", line_color="#f59e0b", annotation_text="OOS Start")
                    
                    eq_fig.update_layout(
                        height=400,
                        template="plotly_dark",
                        margin=dict(l=20, r=20, t=25, b=20),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(eq_fig, use_container_width=True)
                    
                # Trade log comparison
                with st.expander("Compare Backtest Strategy Trades"):
                    # Display ML trade log if available, else Rule
                    log_to_show = ml_results['trades'] if ml_results else rule_results['trades']
                    if log_to_show:
                        log_df = pd.DataFrame(log_to_show)
                        log_df = log_df.rename(columns={
                            "entry_time": "Entry Time",
                            "exit_time": "Exit Time",
                            "entry_price": "Entry Price",
                            "exit_price": "Exit Price",
                            "qty": "Quantity",
                            "gross_pnl": "Gross PnL",
                            "fees": "Fees Paid",
                            "tax": "Taxes Paid",
                            "net_pnl": "Net PnL",
                            "reason": "Exit Reason",
                            "execution_note": "Execution Note"
                        })
                        st.dataframe(log_df.sort_index(ascending=False), use_container_width=True)
                    else:
                        st.info("No trades executed in the backtest.")
                        
            else:
                st.error("Failed to load historical data for backtesting.")
        except Exception as e:
            st.error(f"Backtesting error: {e}")
            logger.error(f"Backtest error: {e}", exc_info=True)

# Row 5: Walk-Forward Validation Engine
st.markdown("---")
st.markdown("### Walk-Forward Validation (Time-Series Out-of-Sample)")
st.write("Walk-forward validation simulates real trading by iteratively training a model on historical data, tuning the threshold, and predicting the next out-of-sample window chronologically. This completely eliminates lookahead leakage.")
run_wf_btn = st.button("Run Walk-Forward Validation (730 days)", type="secondary", key="run_wf")

if run_wf_btn:
    with st.spinner("Executing walk-forward validation (training multiple rolling models). Please wait..."):
        try:
            ml_results, rule_results = run_walk_forward(symbol_selection)
            
            # Display results
            st.success("Walk-forward validation completed successfully!")
            
            col_wf1, col_wf2 = st.columns(2)
            with col_wf1:
                st.markdown("#### Walk-Forward Performance Metrics (Out-of-Sample)")
                
                ml_oos = ml_results['oos_metrics']
                rule_oos = rule_results['oos_metrics']
                bh_oos_ret = ml_results['bh_oos_return_pct']
                
                if ml_oos.get('warning_msg'):
                    st.warning(f"⚠️ **ML-Guided Walk-Forward**: {ml_oos['warning_msg']}")
                if rule_oos.get('warning_msg'):
                    st.warning(f"⚠️ **Rule-Only Baseline**: {rule_oos['warning_msg']}")
                    
                wf_metrics_data = {
                    "Metric": [
                        "Net Strategy Return",
                        "Buy & Hold Return",
                        "Trades Count",
                        "Win Rate",
                        "Max Drawdown",
                        "Profit Factor",
                        "Avg Win",
                        "Avg Loss"
                    ],
                    "Rule-Only Baseline (OOS)": [
                        f"{rule_oos.get('total_return_pct', 0):+.2f}%",
                        f"{bh_oos_ret:+.2f}%",
                        f"{rule_oos.get('trade_count', 0)}",
                        f"{rule_oos.get('win_rate', 0)*100:.1f}%",
                        f"{rule_oos.get('max_drawdown_pct', 0):.2f}%",
                        "N/A",
                        "N/A",
                        "N/A"
                    ],
                    "ML-Guided Walk-Forward (OOS)": [
                        f"{ml_oos.get('total_return_pct', 0):+.2f}%",
                        f"{bh_oos_ret:+.2f}%",
                        f"{ml_oos.get('trade_count', 0)}",
                        f"{ml_oos.get('win_rate', 0)*100:.1f}%",
                        f"{ml_oos.get('max_drawdown_pct', 0):.2f}%",
                        f"{ml_oos.get('profit_factor', 0):.2f}",
                        f"${ml_oos.get('avg_win_cash', 0):,.2f} ({ml_oos.get('avg_win_pct', 0):+.2f}%)",
                        f"${ml_oos.get('avg_loss_cash', 0):,.2f} ({ml_oos.get('avg_loss_pct', 0):+.2f}%)"
                    ]
                }
                
                st.table(pd.DataFrame(wf_metrics_data).set_index("Metric"))
                
            with col_wf2:
                st.markdown("#### Out-of-Sample Equity Curve Comparison")
                wf_fig = go.Figure()
                
                rule_eq = rule_results['equity_curve']
                oos_start_date = ml_results['oos_start_date']
                rule_oos_eq = rule_eq.loc[oos_start_date:]
                
                wf_fig.add_trace(go.Scatter(
                    x=rule_oos_eq.index, y=rule_oos_eq.values,
                    mode='lines', name='Rule-Only Baseline',
                    line=dict(color='#a855f7', width=1.5)
                ))
                
                ml_eq = ml_results['equity_curve']
                ml_oos_eq = ml_eq.loc[oos_start_date:]
                wf_fig.add_trace(go.Scatter(
                    x=ml_oos_eq.index, y=ml_oos_eq.values,
                    mode='lines', name='ML-Guided Walk-Forward',
                    line=dict(color='#10b981', width=2)
                ))
                
                bh_final_cap = ml_results['starting_capital'] * (1 + bh_oos_ret / 100)
                wf_fig.add_trace(go.Scatter(
                    x=[oos_start_date, rule_oos_eq.index[-1]],
                    y=[ml_results['starting_capital'], bh_final_cap],
                    mode='lines+markers', name='Buy & Hold (Net)',
                    line=dict(color='#3b82f6', width=1, dash='dash')
                ))
                
                wf_fig.update_layout(
                    height=400,
                    template="plotly_dark",
                    margin=dict(l=20, r=20, t=25, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(wf_fig, use_container_width=True)
                
            with st.expander("Show Walk-Forward Completed Trades"):
                if ml_results['trades']:
                    wf_trades_df = pd.DataFrame(ml_results['trades'])
                    wf_trades_df = wf_trades_df[wf_trades_df['entry_time'] >= oos_start_date]
                    if not wf_trades_df.empty:
                        wf_trades_df = wf_trades_df.rename(columns={
                            "entry_time": "Entry Time",
                            "exit_time": "Exit Time",
                            "entry_price": "Entry Price",
                            "exit_price": "Exit Price",
                            "qty": "Quantity",
                            "gross_pnl": "Gross PnL",
                            "fees": "Fees Paid",
                            "tax": "Taxes Paid",
                            "net_pnl": "Net PnL",
                            "reason": "Exit Reason",
                            "execution_note": "Execution Note"
                        })
                        st.dataframe(wf_trades_df.sort_index(ascending=False), use_container_width=True)
                    else:
                        st.info("No trades executed in the Out-of-Sample Walk-Forward period.")
                else:
                    st.info("No trades executed in the Walk-Forward period.")
        except Exception as e:
            st.error(f"Walk-Forward validation error: {e}")
            logger.error(f"Walk-Forward validation error: {e}", exc_info=True)

# Row 6: Model Diagnostics Report
st.markdown("---")
st.markdown("### 🔍 Walk-Forward Model Diagnostics Report")
st.write("Analyze Out-of-Sample (OOS) walk-forward predictions to understand predictive capability, Brier calibration, sensitivity to thresholds, feature importances, trade MAE/MFE excursions, and regime exposures.")

run_diag_btn = st.button("Generate Model Diagnostics Report (Walk-Forward OOS)", type="primary", key="run_diag_report")

if run_diag_btn:
    with st.spinner("Analyzing walk-forward OOS predictions..."):
        try:
            # 1. Fetch walk-forward OOS results (cached)
            ml_results, rule_results = get_cached_walk_forward(symbol_selection)
            
            if ml_results is None or rule_results is None:
                st.error("Walk-forward simulation returned empty results.")
            else:
                from src.diagnostics import ModelDiagnostics
                diag = ModelDiagnostics(ml_results, rule_results, symbol_selection)
                
                # Compute metrics
                metrics = diag.get_oos_classification_metrics()
                dist = diag.get_probability_distribution()
                sensitivity = diag.get_threshold_sensitivity()
                importances = diag.get_feature_importances()
                trade_diag = diag.get_trade_diagnostics()
                regimes = diag.get_regime_analysis()
                rallies = diag.get_missed_rallies()
                
                # Show KPIs
                st.success("Diagnostics report generated successfully!")
                
                # Section 1: Classification Performance
                st.markdown("#### 1. OOS Classification Performance Metrics")
                
                kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
                
                acc = metrics['accuracy']
                acc_str = f"{acc * 100:.2f}%" if isinstance(acc, float) else str(acc)
                kpi_col1.metric("Accuracy (threshold 0.60)", acc_str)
                
                prec = metrics['precision']
                prec_str = f"{prec * 100:.2f}%" if isinstance(prec, float) else str(prec)
                kpi_col2.metric("Precision (threshold 0.60)", prec_str)
                
                rec = metrics['recall']
                rec_str = f"{rec * 100:.2f}%" if isinstance(rec, float) else str(rec)
                kpi_col3.metric("Recall (threshold 0.60)", rec_str)
                
                kpi_col4, kpi_col5, kpi_col6 = st.columns(3)
                
                auc = metrics['roc_auc']
                auc_str = f"{auc:.4f}" if isinstance(auc, float) else str(auc)
                kpi_col4.metric("ROC AUC Score", auc_str)
                
                pr_auc = metrics['pr_auc']
                pr_auc_str = f"{pr_auc:.4f}" if isinstance(pr_auc, float) else str(pr_auc)
                kpi_col5.metric("Precision-Recall AUC", pr_auc_str)
                
                brier = metrics['brier_score']
                brier_str = f"{brier:.4f}" if isinstance(brier, float) else str(brier)
                kpi_col6.metric("Brier Score Loss", brier_str)
                
                # Section 2: Probability Distribution & Calibration
                col_dist, col_cal = st.columns(2)
                
                with col_dist:
                    st.markdown("##### Predicted Probability Distribution")
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Bar(
                        x=[f"{b:.2f}" for b in dist['histogram_bins'][:-1]],
                        y=dist['histogram_counts'],
                        marker_color='#10b981',
                        name="Count"
                    ))
                    # Add vertical lines at key thresholds
                    for t in [0.55, 0.60, 0.65]:
                        fig_hist.add_vline(x=f"{t:.2f}" if f"{t:.2f}" in [f"{b:.2f}" for b in dist['histogram_bins'][:-1]] else len(dist['histogram_bins'])*t, line_dash="dash", line_color="#ef4444")
                    
                    fig_hist.update_layout(
                        template="plotly_dark",
                        height=350,
                        margin=dict(l=20, r=20, t=20, b=20),
                        xaxis_title="Predicted Prob UP",
                        yaxis_title="Frequency"
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Threshold table
                    thresh_df = pd.DataFrame([
                        {"Threshold": f"Exceeds {t:.2f}", "Count": data['count'], "Percentage": f"{data['percentage']:.2f}%"}
                        for t, data in dist['threshold_frequencies'].items()
                    ])
                    st.dataframe(thresh_df.set_index("Threshold"), use_container_width=True)
                    
                with col_cal:
                    st.markdown("##### Brier Calibration Curve")
                    prob_true, prob_pred = metrics['calibration_curve']
                    fig_cal = go.Figure()
                    
                    if prob_true and prob_pred:
                        # Reference line
                        fig_cal.add_trace(go.Scatter(
                            x=[0, 1], y=[0, 1],
                            line=dict(dash='dash', color='rgba(255, 255, 255, 0.3)'),
                            name="Perfect Calibration"
                        ))
                        # Calibration plot
                        fig_cal.add_trace(go.Scatter(
                            x=prob_pred, y=prob_true,
                            mode='lines+markers',
                            line=dict(color='#3b82f6', width=2),
                            marker=dict(size=8),
                            name="Model Calibration"
                        ))
                    else:
                        st.info("Calibration curve not available (low samples).")
                        
                    fig_cal.update_layout(
                        template="plotly_dark",
                        height=350,
                        margin=dict(l=20, r=20, t=20, b=20),
                        xaxis_title="Mean Predicted Probability",
                        yaxis_title="Fraction of Positives",
                        xaxis=dict(range=[0, 1]),
                        yaxis=dict(range=[0, 1])
                    )
                    st.plotly_chart(fig_cal, use_container_width=True)
                    
                # Section 3: Threshold Sensitivity & Feature Importance
                col_sens, col_feat = st.columns(2)
                
                with col_sens:
                    st.markdown("##### Threshold Sensitivity (Diagnostic Only)")
                    sens_df = pd.DataFrame(sensitivity)
                    
                    if not sens_df.empty:
                        fig_sens = make_subplots(specs=[[{"secondary_y": True}]])
                        fig_sens.add_trace(
                            go.Bar(x=sens_df['threshold'].astype(str), y=sens_df['net_return'], name="Net Return (%)", marker_color='#10b981'),
                            secondary_y=False
                        )
                        fig_sens.add_trace(
                            go.Scatter(x=sens_df['threshold'].astype(str), y=sens_df['trade_count'], name="Trade Count", mode='lines+markers', line=dict(color='#a855f7', width=2)),
                            secondary_y=True
                        )
                        
                        fig_sens.update_layout(
                            template="plotly_dark",
                            height=350,
                            margin=dict(l=20, r=20, t=20, b=20),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        fig_sens.update_xaxes(title_text="Decision Threshold")
                        fig_sens.update_yaxes(title_text="Net Return (%)", secondary_y=False)
                        fig_sens.update_yaxes(title_text="Trade Count", secondary_y=True)
                        st.plotly_chart(fig_sens, use_container_width=True)
                        
                        # Formatted sensitivity table
                        st.table(sens_df.rename(columns={
                            'threshold': 'Threshold',
                            'trade_count': 'Trades Count',
                            'win_rate': 'Win Rate',
                            'net_return': 'Net Return (%)',
                            'max_drawdown': 'Max Drawdown (%)'
                        }).set_index("Threshold"))
                    else:
                        st.info("Sensitivity metrics not available.")
                        
                with col_feat:
                    st.markdown("##### Feature Importance (Aggregated WF Folds)")
                    top_15 = importances['top_15']
                    
                    if top_15:
                        feat_names = [f[0] for f in top_15][::-1]
                        feat_vals = [f[1] * 100 for f in top_15][::-1]
                        
                        fig_feat = go.Figure()
                        fig_feat.add_trace(go.Bar(
                            x=feat_vals, y=feat_names,
                            orientation='h',
                            marker_color='#3b82f6'
                        ))
                        fig_feat.update_layout(
                            template="plotly_dark",
                            height=480,
                            margin=dict(l=20, r=20, t=20, b=20),
                            xaxis_title="Average Importance Weight (%)"
                        )
                        st.plotly_chart(fig_feat, use_container_width=True)
                        st.markdown(f"**Cross-Market Feature Weight**: `{importances['cross_market']}`")
                    else:
                        st.info("Feature importances not available.")
                        
                # Section 4: Trade & Exposure Diagnostics
                st.markdown("#### 4. Trade & Exposure Diagnostics")
                
                exposure_col1, exposure_col2, exposure_col3 = st.columns(3)
                
                # Exposure metric
                exposure_col1.metric("Strategy Time-in-Market", f"{trade_diag['strategy_exposure_pct']:.2f}%")
                exposure_col2.metric("Buy & Hold Time-in-Market", f"{trade_diag['bh_exposure_pct']:.2f}%")
                exposure_col3.metric("Average Holding Period", f"{trade_diag['avg_holding_period_hours']:.1f} hours")
                
                if trade_diag['strategy_exposure_pct'] < 5.0:
                    st.warning("⚠️ **Strategy exposure is extremely low (< 5%)**: The bot acts primarily as a cash-holding avoidance strategy rather than an active predictor.")
                    
                st.markdown(f"""
                *   **Total Losing Trades**: {trade_diag['total_losing_trades']}
                *   **Bad Entries (No excursion, straight to Stop Loss)**: {trade_diag['bad_entries_count']}
                *   **Bad Exits (MFE > 0.5%, failed to capture profit before reversing)**: {trade_diag['bad_exits_count']}
                """)
                
                # MAE / MFE Scatter Plot
                st.markdown("##### Intratrade Excursions (MAE vs MFE)")
                mae_mfe_points = diag.trade_mae_mfe_list
                
                if mae_mfe_points:
                    points_df = pd.DataFrame(mae_mfe_points)
                    fig_scatter = go.Figure()
                    
                    # Wins
                    wins = points_df[points_df['is_win'] == True]
                    if not wins.empty:
                        fig_scatter.add_trace(go.Scatter(
                            x=wins['mae'], y=wins['mfe'],
                            mode='markers',
                            marker=dict(color='#10b981', size=10, symbol='circle'),
                            name="Winning Trades",
                            text=[f"PnL: ${p:,.2f}" for p in wins['net_pnl']]
                        ))
                        
                    # Losses
                    losses = points_df[points_df['is_win'] == False]
                    if not losses.empty:
                        fig_scatter.add_trace(go.Scatter(
                            x=losses['mae'], y=losses['mfe'],
                            mode='markers',
                            marker=dict(color='#ef4444', size=10, symbol='x'),
                            name="Losing Trades",
                            text=[f"PnL: ${p:,.2f}" for p in losses['net_pnl']]
                        ))
                        
                    # Add diagonal or stop loss lines
                    fig_scatter.add_vline(x=Config.STOP_LOSS_PCT * 100, line_dash="dash", line_color="rgba(239, 68, 68, 0.5)", annotation_text="SL Limit")
                    fig_scatter.add_hline(y=0.5, line_dash="dot", line_color="rgba(255, 255, 255, 0.4)", annotation_text="Profit Excursion Boundary")
                    
                    fig_scatter.update_layout(
                        template="plotly_dark",
                        height=400,
                        margin=dict(l=20, r=20, t=20, b=20),
                        xaxis_title="Maximum Adverse Excursion (MAE %)",
                        yaxis_title="Maximum Favorable Excursion (MFE %)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("No trade details available for MAE/MFE scatter plot.")
                    
                # Section 5: Regime & Missed Rallies
                col_regime, col_rallies = st.columns(2)
                
                with col_regime:
                    st.markdown("##### Regime Performance (At Entry State)")
                    if regimes:
                        regime_data = []
                        for state, rdata in regimes.items():
                            regime_data.append({
                                "Market State": state.upper(),
                                "Trades Count": rdata['trades_count'],
                                "Win Rate": f"{rdata['win_rate']:.2f}%",
                                "Net PnL ($)": f"{rdata['net_pnl']:+,.2f}"
                            })
                        st.table(pd.DataFrame(regime_data).set_index("Market State"))
                    else:
                        st.info("No trades executed during OOS regime state.")
                        
                with col_rallies:
                    st.markdown(f"##### Missed Buy-and-Hold Rallies ({len(rallies)} Total)")
                    if rallies:
                        rally_data = []
                        for idx, r in enumerate(rallies[:5], 1):
                            rally_data.append({
                                "Rally": f"Rally {idx}",
                                "Start Date": r['start_time'],
                                "End Date": r['end_time'],
                                "B&H Return": f"+{r['return']:.2f}%"
                            })
                        st.table(pd.DataFrame(rally_data).set_index("Rally"))
                        if len(rallies) > 5:
                            st.info(f"... and {len(rallies) - 5} more missed rallies (bot remained flat).")
                    else:
                        st.success("No significant buy-and-hold rallies were missed by the strategy.")
                        
        except Exception as e:
            st.error(f"Diagnostics generation error: {e}")
            logger.error(f"Diagnostics generation error: {e}", exc_info=True)
