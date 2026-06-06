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
from src.market_state import classify_market_state_row
from src.ml_model import MarketMLModel
from src.strategy import Strategy
from src.backtester import Backtester
from src.paper_trader import PaperTrader
from src.costs import TAX_DISCLAIMER

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
signal = strat.generate_signal(latest_row_dict, prob_up, prob_down)

# Update paper trader log
paper_trader = PaperTrader(symbol_selection)
if model_exists:
    time_str = latest_time.strftime("%Y-%m-%d %H:%M:%S")
    paper_trader.process_signal(signal, latest_price, time_str)

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
            
            backtest_data = provider.fetch_data(symbol_selection, start_back, end_back)
            if not backtest_data.empty:
                backtest_data = calculate_indicators(backtest_data)
                backtest_data['market_state'] = classify_market_states(backtest_data)
                
                # Setup backtester
                backtester = Backtester(backtest_data, symbol_selection)
                
                # Rule only
                rule_results = backtester.run_backtest(ml_model=None)
                
                # ML only (if model exists)
                ml_results = None
                if model_exists:
                    ml_results = backtester.run_backtest(ml_model)
                    
                # Display statistics side by side
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Performance Metrics")
                    metrics_data = {
                        "Metric": [
                            "Starting Capital",
                            "Trades Count",
                            "Win Rate",
                            "Max Drawdown",
                            "Taxes Paid",
                            "Fees Paid",
                            "Net Strategy Return",
                            "Buy & Hold Return"
                        ],
                        "Rule-Only Strategy": [
                            f"${Config.STARTING_CAPITAL:,.2f}",
                            f"{rule_results.get('trade_count', 0)}",
                            f"{rule_results.get('win_rate', 0)*100:.1f}%",
                            f"{rule_results.get('max_drawdown_pct', 0):.2f}%",
                            f"${rule_results.get('taxes_paid', 0):,.2f}",
                            f"${rule_results.get('fees_paid', 0):,.2f}",
                            f"{rule_results.get('total_return_after_costs_pct', 0):+.2f}%",
                            f"{rule_results.get('bh_return_after_costs_pct', 0):+.2f}%"
                        ]
                    }
                    
                    if ml_results:
                        metrics_data["ML-Guided Strategy"] = [
                            f"${Config.STARTING_CAPITAL:,.2f}",
                            f"{ml_results.get('trade_count', 0)}",
                            f"{ml_results.get('win_rate', 0)*100:.1f}%",
                            f"{ml_results.get('max_drawdown_pct', 0):.2f}%",
                            f"${ml_results.get('taxes_paid', 0):,.2f}",
                            f"${ml_results.get('fees_paid', 0):,.2f}",
                            f"{ml_results.get('total_return_after_costs_pct', 0):+.2f}%",
                            f"{ml_results.get('bh_return_after_costs_pct', 0):+.2f}%"
                        ]
                        
                    metrics_df = pd.DataFrame(metrics_data)
                    st.table(metrics_df.set_index("Metric"))
                    
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
                    # For a simple baseline, draw a line from starting capital to final B&H equity
                    eq_fig.add_trace(go.Scatter(
                        x=[rule_eq.index[0], rule_eq.index[-1]],
                        y=[Config.STARTING_CAPITAL, rule_results['bh_final_capital']],
                        mode='lines+markers', name='Buy & Hold (Net)',
                        line=dict(color='#3b82f6', width=1, dash='dash')
                    ))
                    
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
                            "reason": "Exit Reason"
                        })
                        st.dataframe(log_df.sort_index(ascending=False), use_container_width=True)
                    else:
                        st.info("No trades executed in the backtest.")
                        
            else:
                st.error("Failed to load historical data for backtesting.")
        except Exception as e:
            st.error(f"Backtesting error: {e}")
            logger.error(f"Backtest error: {e}", exc_info=True)
