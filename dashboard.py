#!/usr/bin/env python3
"""
SPY Trading Strategies Dashboard
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(
    page_title="SPY Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# DATA & SIGNAL FUNCTIONS
# ============================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data(ticker='SPY', period='2y'):
    """Load and prepare price data."""
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data is None or len(data) == 0:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Calculate indicators
        data['MA20'] = data['Close'].rolling(20).mean()
        data['MA50'] = data['Close'].rolling(50).mean()
        data['MA200'] = data['Close'].rolling(200).mean()
        data['MA200_Slope'] = data['MA200'].pct_change(periods=20)
        
        # RSI
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        data['RSI'] = 100 - (100 / (1 + gain/loss))
        
        # Volatility
        data['Vol'] = data['Close'].pct_change().rolling(20).std() * (252 ** 0.5)
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def get_signals(row):
    """Calculate all 5 signals for a given row."""
    signals = {
        'Trend Filter': (row['Close'] > row['MA50']) and (row['MA200_Slope'] > 0),
        'Golden Cross': row['MA50'] > row['MA200'],
        'Momentum': row['Close'] > row['MA20'],
        'RSI OK': row['RSI'] < 70,
        'Low Vol': row['Vol'] < 0.25
    }
    return signals

def get_strategy_position(signals):
    """Get position for each strategy based on signals."""
    count = sum(signals.values())
    tf = signals['Trend Filter']
    gc = signals['Golden Cross']
    mom = signals['Momentum']
    
    positions = {
        'Trend Rider': 1.0 if tf else 0.0,
        'Golden Boost': 1.5 if (tf and gc) else (1.0 if tf else 0.0),
        'Signal Stacker': 2.0 if count >= 5 else (1.5 if count >= 4 else (1.0 if count >= 3 else 0.0)),
        'Steady Eddie': 2.0 if count == 5 else (1.5 if count >= 4 else (1.0 if count >= 3 else 0.0)),
        'Full Send': 2.0 if (tf and gc and mom) else 0.0
    }
    return positions, count

# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(data, strategy_name, initial_capital=100000):
    """Run backtest for a specific strategy."""
    df = data.copy()
    df = df.dropna()
    
    # Calculate daily returns
    df['Return'] = df['Close'].pct_change()
    
    # Get positions for each day
    positions = []
    for idx, row in df.iterrows():
        signals = get_signals(row)
        strat_positions, _ = get_strategy_position(signals)
        positions.append(strat_positions.get(strategy_name, 0))
    
    df['Position'] = positions
    df['Position'] = df['Position'].shift(1)  # Trade on next day
    df['Position'] = df['Position'].fillna(0)
    
    # Calculate strategy returns (with leverage effect)
    df['Strategy_Return'] = df['Position'] * df['Return']
    
    # Calculate cumulative returns
    df['Buy_Hold'] = (1 + df['Return']).cumprod() * initial_capital
    df['Strategy'] = (1 + df['Strategy_Return']).cumprod() * initial_capital
    
    # Calculate metrics
    total_days = len(df)
    years = total_days / 252
    
    strategy_total_return = (df['Strategy'].iloc[-1] / initial_capital - 1) * 100
    buyhold_total_return = (df['Buy_Hold'].iloc[-1] / initial_capital - 1) * 100
    
    strategy_cagr = ((df['Strategy'].iloc[-1] / initial_capital) ** (1/years) - 1) * 100
    buyhold_cagr = ((df['Buy_Hold'].iloc[-1] / initial_capital) ** (1/years) - 1) * 100
    
    # Max Drawdown
    rolling_max = df['Strategy'].cummax()
    drawdown = (df['Strategy'] - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100
    
    # Sharpe Ratio (assuming 4% risk-free rate)
    excess_returns = df['Strategy_Return'] - 0.04/252
    sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
    
    # Win rate
    winning_days = (df['Strategy_Return'] > 0).sum()
    trading_days = (df['Position'] != 0).sum()
    win_rate = (winning_days / trading_days * 100) if trading_days > 0 else 0
    
    metrics = {
        'Total Return': f"{strategy_total_return:.1f}%",
        'CAGR': f"{strategy_cagr:.1f}%",
        'Max Drawdown': f"{max_dd:.1f}%",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Win Rate': f"{win_rate:.1f}%",
        'Buy & Hold Return': f"{buyhold_total_return:.1f}%",
        'Buy & Hold CAGR': f"{buyhold_cagr:.1f}%"
    }
    
    return df, metrics

# ============================================================
# DASHBOARD UI
# ============================================================

st.title("📈 SPY Trading Strategies Dashboard")

# Sidebar
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker", value="SPY")
period = st.sidebar.selectbox("Data Period", ['1y', '2y', '3y', '5y', 'max'], index=1)

# Load data
with st.spinner("Loading data..."):
    data = load_data(ticker, period)

if data is None or len(data) == 0:
    st.error("⚠️ Unable to load market data. Please try again in a moment.")
    st.info("This can happen due to API rate limits or network issues. Try refreshing the page.")
    st.stop()

latest = data.iloc[-1]
signals = get_signals(latest)
positions, signal_count = get_strategy_position(signals)

# ============================================================
# CURRENT SIGNALS SECTION
# ============================================================

st.header("🚦 Current Signals")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Price", f"${latest['Close']:.2f}")
with col2:
    st.metric("RSI", f"{latest['RSI']:.1f}")
with col3:
    st.metric("Volatility", f"{latest['Vol']*100:.1f}%")
with col4:
    st.metric("Signal Count", f"{signal_count}/5", 
              delta="Bullish" if signal_count >= 4 else ("Neutral" if signal_count >= 3 else "Bearish"))

st.subheader("Signal Status")
signal_cols = st.columns(5)
for i, (name, active) in enumerate(signals.items()):
    with signal_cols[i]:
        if active:
            st.success(f"✅ {name}")
        else:
            st.error(f"❌ {name}")

# ============================================================
# STRATEGY POSITIONS SECTION
# ============================================================

st.header("📊 Strategy Positions")

strategy_info = {
    'Trend Rider': ('🟢', 'Basic trend following - long when above 50 DMA & 200 DMA rising'),
    'Golden Boost': ('🟡', 'Adds leverage when Golden Cross confirms'),
    'Signal Stacker': ('📊', 'Scales 1x→1.5x→2x based on signal count'),
    'Steady Eddie': ('⭐', 'Conservative - best risk-adjusted returns'),
    'Full Send': ('🚀', 'Aggressive - 2x when trend+cross+momentum align')
}

strat_cols = st.columns(5)
for i, (name, pos) in enumerate(positions.items()):
    icon, desc = strategy_info[name]
    with strat_cols[i]:
        if pos == 0:
            st.warning(f"{icon} **{name}**\n\n💵 CASH")
        elif pos == 1.0:
            st.info(f"{icon} **{name}**\n\n📈 1x SPY")
        elif pos == 1.5:
            st.success(f"{icon} **{name}**\n\n📈 1.5x (50/50)")
        else:
            st.success(f"{icon} **{name}**\n\n🔥 2x SSO")

# ============================================================
# PRICE CHART SECTION
# ============================================================

st.header("📉 Price Chart")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.05, row_heights=[0.7, 0.3])

# Price and MAs
fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
                              low=data['Low'], close=data['Close'], name='Price'), row=1, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], name='MA20', line=dict(color='orange', width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], name='MA50', line=dict(color='blue', width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data['MA200'], name='MA200', line=dict(color='red', width=1)), row=1, col=1)

# RSI
fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

fig.update_layout(height=600, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, width="stretch")

# ============================================================
# BACKTEST SECTION
# ============================================================

st.header("🔬 Backtest")

backtest_col1, backtest_col2 = st.columns([1, 3])

with backtest_col1:
    selected_strategy = st.selectbox("Strategy", list(positions.keys()))
    initial_capital = st.number_input("Initial Capital", value=100000, step=10000)
    run_btn = st.button("Run Backtest", type="primary")

if run_btn:
    with st.spinner("Running backtest..."):
        bt_data, metrics = run_backtest(data, selected_strategy, initial_capital)
    
    # Metrics
    st.subheader(f"📈 {selected_strategy} Performance")
    
    metric_cols = st.columns(4)
    metric_items = list(metrics.items())
    for i, (name, value) in enumerate(metric_items[:4]):
        with metric_cols[i]:
            st.metric(name, value)
    
    metric_cols2 = st.columns(4)
    for i, (name, value) in enumerate(metric_items[4:]):
        with metric_cols2[i]:
            st.metric(name, value)
    
    # Equity curve
    st.subheader("Equity Curve")
    
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=bt_data.index, y=bt_data['Strategy'], 
                                 name=selected_strategy, line=dict(color='green', width=2)))
    fig_bt.add_trace(go.Scatter(x=bt_data.index, y=bt_data['Buy_Hold'], 
                                 name='Buy & Hold', line=dict(color='blue', width=1, dash='dash')))
    fig_bt.update_layout(height=400, hovermode='x unified')
    st.plotly_chart(fig_bt, width="stretch")

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data from Yahoo Finance")
