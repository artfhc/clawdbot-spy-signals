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

@st.cache_data(ttl=300)
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
        
        # RSI (14-day)
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        data['RSI'] = 100 - (100 / (1 + gain/loss))
        
        # RSI (5-day) for mean reversion
        gain5 = delta.where(delta > 0, 0).rolling(5).mean()
        loss5 = (-delta.where(delta < 0, 0)).rolling(5).mean()
        data['RSI5'] = 100 - (100 / (1 + gain5/loss5))
        
        # Volatility
        data['Vol'] = data['Close'].pct_change().rolling(20).std() * (252 ** 0.5)
        
        # 12-month momentum (for Dual Momentum)
        data['Mom12M'] = data['Close'].pct_change(periods=252)
        
        # End of month flag
        data['MonthEnd'] = data.index.is_month_end
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data(ttl=300)
def load_intl_data(period='2y'):
    """Load international equity data for Dual Momentum."""
    try:
        data = yf.download('VEU', period=period, progress=False)  # Vanguard FTSE All-World ex-US
        if data is None or len(data) == 0:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data['Mom12M'] = data['Close'].pct_change(periods=252)
        return data
    except:
        return None

@st.cache_data(ttl=300)
def load_bond_data(period='2y'):
    """Load bond data for safe haven."""
    try:
        data = yf.download('BND', period=period, progress=False)  # Vanguard Total Bond
        if data is None or len(data) == 0:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except:
        return None

def get_signals(row):
    """Calculate all 5 original signals for a given row."""
    signals = {
        'Trend Filter': (row['Close'] > row['MA50']) and (row['MA200_Slope'] > 0),
        'Golden Cross': row['MA50'] > row['MA200'],
        'Momentum': row['Close'] > row['MA20'],
        'RSI OK': row['RSI'] < 70,
        'Low Vol': row['Vol'] < 0.25
    }
    return signals

def get_strategy_position(signals, row=None, intl_mom=None, rsi_in_state=None):
    """Get position for each strategy based on signals."""
    count = sum(signals.values())
    tf = signals['Trend Filter']
    gc = signals['Golden Cross']
    mom = signals['Momentum']
    
    # Original strategies
    positions = {
        'Trend Rider': 1.0 if tf else 0.0,
        'Golden Boost': 1.5 if (tf and gc) else (1.0 if tf else 0.0),
        'Signal Stacker': 2.0 if count >= 5 else (1.5 if count >= 4 else (1.0 if count >= 3 else 0.0)),
        'Steady Eddie': 2.0 if count == 5 else (1.5 if count >= 4 else (1.0 if count >= 3 else 0.0)),
        'Full Send': 2.0 if (tf and gc and mom) else 0.0,
    }
    
    # New research-backed strategies
    if row is not None:
        above_200 = row['Close'] > row['MA200'] if pd.notna(row['MA200']) else False
        rsi5 = row['RSI5'] if 'RSI5' in row and pd.notna(row['RSI5']) else 50
        rsi_oversold = rsi5 < 30
        rsi_recovered = rsi5 > 50
        vol = row['Vol'] if 'Vol' in row and pd.notna(row['Vol']) else 0.2
        
        # 200 DMA Monthly
        positions['200 DMA Monthly'] = 1.0 if above_200 else 0.0
        
        # RSI Bounce
        if above_200 and rsi_oversold:
            positions['RSI Bounce'] = 1.0
        elif rsi_recovered:
            positions['RSI Bounce'] = 0.0
        else:
            positions['RSI Bounce'] = 0.5
        
        # Dual Momentum
        spy_mom = row['Mom12M'] if pd.notna(row['Mom12M']) else 0
        intl_mom_val = intl_mom if intl_mom is not None else 0
        if spy_mom > intl_mom_val and spy_mom > 0:
            positions['Dual Momentum'] = 1.0
        elif intl_mom_val > spy_mom and intl_mom_val > 0:
            positions['Dual Momentum'] = 0.8
        else:
            positions['Dual Momentum'] = 0.0
        
        # ============================================================
        # HYBRID STRATEGIES (Tuned combinations)
        # ============================================================
        
        # Hybrid B: Voting system - need 3+ signals AND above 200 DMA
        if count >= 4 and above_200:
            positions['Voting System'] = 1.5
        elif count >= 3 and above_200:
            positions['Voting System'] = 1.0
        else:
            positions['Voting System'] = 0.0
        
        # Hybrid D: Vol-adaptive position sizing
        if tf and gc:
            if vol < 0.15:
                positions['Vol Adaptive'] = 2.0
            elif vol < 0.20:
                positions['Vol Adaptive'] = 1.5
            elif vol < 0.25:
                positions['Vol Adaptive'] = 1.0
            else:
                positions['Vol Adaptive'] = 0.5
        elif above_200:
            positions['Vol Adaptive'] = 0.5 if vol < 0.25 else 0.0
        else:
            positions['Vol Adaptive'] = 0.0
        
        # Hybrid F: Ensemble (Trend + RSI combined) - BEST PERFORMER
        trend_pos = 1.0 if tf else 0.0
        rsi_pos = 1.0 if (above_200 and rsi_oversold) else (0.0 if rsi_recovered else 0.5)
        
        if trend_pos > 0 and rsi_pos >= 1.0:
            positions['Ensemble'] = 1.5  # Both agree strongly
        elif trend_pos > 0 or rsi_pos >= 1.0:
            positions['Ensemble'] = 1.0  # Either says buy
        else:
            positions['Ensemble'] = 0.0
    
    return positions, count

# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(data, strategy_name, initial_capital=100000, intl_data=None, bond_data=None):
    """Run backtest for a specific strategy."""
    df = data.copy()
    df = df.dropna(subset=['Close', 'MA200'])
    
    # Calculate daily returns
    df['Return'] = df['Close'].pct_change()
    
    # Prepare international data if available
    if intl_data is not None:
        intl_df = intl_data.copy()
        intl_df = intl_df[['Mom12M', 'Close']].rename(columns={'Mom12M': 'Intl_Mom12M', 'Close': 'Intl_Close'})
        intl_df['Intl_Return'] = intl_df['Intl_Close'].pct_change()
        df = df.join(intl_df, how='left')
    
    # Get positions for each day
    positions = []
    position_types = []  # For Dual Momentum: 'SPY', 'INTL', 'BOND'
    
    # State tracking for RSI Bounce
    rsi_in_position = False
    
    for idx, row in df.iterrows():
        signals = get_signals(row)
        intl_mom = row['Intl_Mom12M'] if 'Intl_Mom12M' in row and pd.notna(row['Intl_Mom12M']) else None
        
        if strategy_name == '200 DMA Monthly':
            # Only change position at month end
            if len(positions) == 0:
                pos = 1.0 if row['Close'] > row['MA200'] else 0.0
            elif row['MonthEnd']:
                pos = 1.0 if row['Close'] > row['MA200'] else 0.0
            else:
                pos = positions[-1]
            positions.append(pos)
            position_types.append('SPY')
            
        elif strategy_name == 'RSI Bounce':
            above_200 = row['Close'] > row['MA200']
            rsi5 = row['RSI5'] if pd.notna(row['RSI5']) else 50
            
            if not rsi_in_position:
                if above_200 and rsi5 < 30:
                    rsi_in_position = True
                    positions.append(1.0)
                else:
                    positions.append(0.0)
            else:
                if rsi5 > 50:
                    rsi_in_position = False
                    positions.append(0.0)
                else:
                    positions.append(1.0)
            position_types.append('SPY')
            
        elif strategy_name == 'Dual Momentum':
            spy_mom = row['Mom12M'] if pd.notna(row['Mom12M']) else 0
            intl_mom_val = intl_mom if intl_mom is not None else 0
            
            if spy_mom > intl_mom_val and spy_mom > 0:
                positions.append(1.0)
                position_types.append('SPY')
            elif intl_mom_val > spy_mom and intl_mom_val > 0:
                positions.append(1.0)
                position_types.append('INTL')
            else:
                positions.append(0.0)
                position_types.append('BOND')
        else:
            # Original strategies
            strat_positions, _ = get_strategy_position(signals, row, intl_mom)
            positions.append(strat_positions.get(strategy_name, 0))
            position_types.append('SPY')
    
    df['Position'] = positions
    df['Position_Type'] = position_types
    df['Position'] = df['Position'].shift(1).fillna(0)
    df['Position_Type'] = df['Position_Type'].shift(1).fillna('SPY')
    
    # Calculate strategy returns
    if strategy_name == 'Dual Momentum' and 'Intl_Return' in df.columns:
        df['Strategy_Return'] = df.apply(
            lambda r: r['Position'] * (r['Intl_Return'] if r['Position_Type'] == 'INTL' else r['Return']),
            axis=1
        )
    else:
        df['Strategy_Return'] = df['Position'] * df['Return']
    
    # Handle leverage for original strategies
    if strategy_name in ['Golden Boost', 'Signal Stacker', 'Steady Eddie', 'Full Send']:
        df['Strategy_Return'] = df['Position'] * df['Return']
    
    # Calculate cumulative returns
    df['Buy_Hold'] = (1 + df['Return']).cumprod() * initial_capital
    df['Strategy'] = (1 + df['Strategy_Return']).cumprod() * initial_capital
    
    # Calculate metrics
    total_days = len(df)
    years = total_days / 252
    
    strategy_total_return = (df['Strategy'].iloc[-1] / initial_capital - 1) * 100
    buyhold_total_return = (df['Buy_Hold'].iloc[-1] / initial_capital - 1) * 100
    
    strategy_cagr = ((df['Strategy'].iloc[-1] / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
    buyhold_cagr = ((df['Buy_Hold'].iloc[-1] / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
    
    # Max Drawdown
    rolling_max = df['Strategy'].cummax()
    drawdown = (df['Strategy'] - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100
    
    # Buy & Hold Max Drawdown
    bh_rolling_max = df['Buy_Hold'].cummax()
    bh_drawdown = (df['Buy_Hold'] - bh_rolling_max) / bh_rolling_max
    bh_max_dd = bh_drawdown.min() * 100
    
    # Sharpe Ratio
    excess_returns = df['Strategy_Return'] - 0.04/252
    sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
    
    # Win rate
    winning_days = (df['Strategy_Return'] > 0).sum()
    trading_days = (df['Position'] != 0).sum()
    win_rate = (winning_days / trading_days * 100) if trading_days > 0 else 0
    
    # Time in market
    time_in_market = (df['Position'] > 0).mean() * 100
    
    metrics = {
        'Total Return': f"{strategy_total_return:.1f}%",
        'CAGR': f"{strategy_cagr:.1f}%",
        'Max Drawdown': f"{max_dd:.1f}%",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Win Rate': f"{win_rate:.1f}%",
        'Time in Market': f"{time_in_market:.1f}%",
        'Buy & Hold Return': f"{buyhold_total_return:.1f}%",
        'B&H Max Drawdown': f"{bh_max_dd:.1f}%"
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
    intl_data = load_intl_data(period)
    bond_data = load_bond_data(period)

if data is None or len(data) == 0:
    st.error("⚠️ Unable to load market data. Please try again in a moment.")
    st.info("This can happen due to API rate limits or network issues. Try refreshing the page.")
    st.stop()

latest = data.iloc[-1]
signals = get_signals(latest)
intl_mom = intl_data['Mom12M'].iloc[-1] if intl_data is not None and len(intl_data) > 0 else None
positions, signal_count = get_strategy_position(signals, latest, intl_mom)

# ============================================================
# CURRENT SIGNALS SECTION
# ============================================================

st.header("🚦 Current Signals")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Price", f"${latest['Close']:.2f}")
with col2:
    st.metric("RSI (14)", f"{latest['RSI']:.1f}")
with col3:
    st.metric("RSI (5)", f"{latest['RSI5']:.1f}")
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

# Separate original and new strategies
st.subheader("Original Strategies")
strategy_info_original = {
    'Trend Rider': ('🟢', 'Basic trend following'),
    'Golden Boost': ('🟡', 'Trend + Golden Cross leverage'),
    'Signal Stacker': ('📊', 'Scales with signal count'),
    'Steady Eddie': ('⭐', 'Conservative, best risk-adjusted'),
    'Full Send': ('🚀', 'Aggressive max leverage')
}

strat_cols = st.columns(5)
for i, (name, (icon, desc)) in enumerate(strategy_info_original.items()):
    pos = positions.get(name, 0)
    with strat_cols[i]:
        if pos == 0:
            st.warning(f"{icon} **{name}**\n\n💵 CASH")
        elif pos == 1.0:
            st.info(f"{icon} **{name}**\n\n📈 1x SPY")
        elif pos == 1.5:
            st.success(f"{icon} **{name}**\n\n📈 1.5x")
        else:
            st.success(f"{icon} **{name}**\n\n🔥 2x SSO")

st.subheader("Research-Backed Strategies")
strategy_info_new = {
    'Dual Momentum': ('🌍', 'Gary Antonacci GEM - US vs Intl, 12mo momentum'),
    '200 DMA Monthly': ('📅', 'Paul Tudor Jones - End of month trend filter'),
    'RSI Bounce': ('📉', 'Mean reversion - Buy RSI<30 dips in uptrends')
}

new_cols = st.columns(3)
for i, (name, (icon, desc)) in enumerate(strategy_info_new.items()):
    pos = positions.get(name, 0)
    with new_cols[i]:
        if name == 'Dual Momentum':
            if pos == 1.0:
                st.success(f"{icon} **{name}**\n\n🇺🇸 US Equities (SPY)")
            elif pos == 0.8:
                st.info(f"{icon} **{name}**\n\n🌏 International (VEU)")
            else:
                st.warning(f"{icon} **{name}**\n\n🏦 Bonds (BND)")
        elif name == 'RSI Bounce':
            if pos == 1.0:
                st.success(f"{icon} **{name}**\n\n🎯 BUY SIGNAL")
            elif pos == 0.5:
                st.info(f"{icon} **{name}**\n\n⏳ Waiting...")
            else:
                st.warning(f"{icon} **{name}**\n\n💵 CASH")
        else:
            if pos == 1.0:
                st.success(f"{icon} **{name}**\n\n📈 LONG SPY")
            else:
                st.warning(f"{icon} **{name}**\n\n💵 CASH")
        st.caption(desc)

st.subheader("🏆 Hybrid Strategies (Tuned)")
strategy_info_hybrid = {
    'Ensemble': ('🎯', 'BEST: Trend + RSI combined'),
    'Voting System': ('🗳️', 'Need 3+ signals to enter'),
    'Vol Adaptive': ('📊', 'Size based on volatility')
}

hybrid_cols = st.columns(3)
for i, (name, (icon, desc)) in enumerate(strategy_info_hybrid.items()):
    pos = positions.get(name, 0)
    with hybrid_cols[i]:
        if pos == 0:
            st.warning(f"{icon} **{name}**\n\n💵 CASH")
        elif pos == 0.5:
            st.info(f"{icon} **{name}**\n\n📈 0.5x")
        elif pos == 1.0:
            st.info(f"{icon} **{name}**\n\n📈 1x SPY")
        elif pos == 1.5:
            st.success(f"{icon} **{name}**\n\n📈 1.5x")
        else:
            st.success(f"{icon} **{name}**\n\n🔥 2x")
        st.caption(desc)

# ============================================================
# PRICE CHART SECTION
# ============================================================

st.header("📉 Price Chart")

# Strategy selector for signals on chart
all_strategies = list(strategy_info_original.keys()) + list(strategy_info_new.keys()) + list(strategy_info_hybrid.keys())
chart_strategy = st.selectbox("Show buy/sell signals for:", all_strategies, index=len(all_strategies)-3)  # Default to Ensemble

# Calculate buy/sell signals for the selected strategy
def get_trade_signals(data, strategy_name, intl_data=None):
    """Get buy and sell signal dates and prices."""
    buy_dates, buy_prices = [], []
    sell_dates, sell_prices = [], []
    
    prev_position = 0
    rsi_in_position = False
    
    for i, (idx, row) in enumerate(data.iterrows()):
        if pd.isna(row['MA200']):
            continue
            
        signals = get_signals(row)
        intl_mom = None
        if intl_data is not None and idx in intl_data.index:
            intl_mom = intl_data.loc[idx, 'Mom12M'] if pd.notna(intl_data.loc[idx, 'Mom12M']) else None
        
        # Get position based on strategy
        if strategy_name == '200 DMA Monthly':
            if row['MonthEnd']:
                current_position = 1.0 if row['Close'] > row['MA200'] else 0.0
            else:
                current_position = prev_position
        elif strategy_name == 'RSI Bounce':
            above_200 = row['Close'] > row['MA200']
            rsi5 = row['RSI5'] if pd.notna(row['RSI5']) else 50
            if not rsi_in_position:
                if above_200 and rsi5 < 30:
                    rsi_in_position = True
                    current_position = 1.0
                else:
                    current_position = 0.0
            else:
                if rsi5 > 50:
                    rsi_in_position = False
                    current_position = 0.0
                else:
                    current_position = 1.0
        elif strategy_name == 'Dual Momentum':
            spy_mom = row['Mom12M'] if pd.notna(row['Mom12M']) else 0
            intl_mom_val = intl_mom if intl_mom is not None else 0
            if spy_mom > intl_mom_val and spy_mom > 0:
                current_position = 1.0
            elif intl_mom_val > spy_mom and intl_mom_val > 0:
                current_position = 0.8
            else:
                current_position = 0.0
        else:
            strat_positions, _ = get_strategy_position(signals, row, intl_mom)
            current_position = strat_positions.get(strategy_name, 0)
        
        # Detect transitions
        if prev_position == 0 and current_position > 0:
            buy_dates.append(idx)
            buy_prices.append(row['Low'] * 0.98)
        elif prev_position > 0 and current_position == 0:
            sell_dates.append(idx)
            sell_prices.append(row['High'] * 1.02)
        
        prev_position = current_position
    
    return buy_dates, buy_prices, sell_dates, sell_prices

buy_dates, buy_prices, sell_dates, sell_prices = get_trade_signals(data, chart_strategy, intl_data)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.05, row_heights=[0.7, 0.3])

# Price and MAs
fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
                              low=data['Low'], close=data['Close'], name='Price'), row=1, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], name='MA20', line=dict(color='orange', width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], name='MA50', line=dict(color='blue', width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data['MA200'], name='MA200', line=dict(color='red', width=2)), row=1, col=1)

# Buy signals
fig.add_trace(go.Scatter(
    x=buy_dates, y=buy_prices,
    mode='markers',
    marker=dict(symbol='triangle-up', size=12, color='green'),
    name='Buy Signal'
), row=1, col=1)

# Sell signals
fig.add_trace(go.Scatter(
    x=sell_dates, y=sell_prices,
    mode='markers',
    marker=dict(symbol='triangle-down', size=12, color='red'),
    name='Sell Signal'
), row=1, col=1)

# RSI
fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI(14)', line=dict(color='purple')), row=2, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data['RSI5'], name='RSI(5)', line=dict(color='cyan', dash='dot')), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

fig.update_layout(height=600, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# BACKTEST SECTION
# ============================================================

st.header("🔬 Backtest")

backtest_col1, backtest_col2 = st.columns([1, 3])

with backtest_col1:
    selected_strategy = st.selectbox("Strategy", all_strategies)
    initial_capital = st.number_input("Initial Capital", value=100000, step=10000)
    run_btn = st.button("Run Backtest", type="primary")

if run_btn:
    with st.spinner("Running backtest..."):
        bt_data, metrics = run_backtest(data, selected_strategy, initial_capital, intl_data, bond_data)
    
    st.subheader(f"📈 {selected_strategy} Performance")
    
    # Metrics in two rows
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
                                 name='Buy & Hold SPY', line=dict(color='blue', width=1, dash='dash')))
    fig_bt.update_layout(height=400, hovermode='x unified')
    st.plotly_chart(fig_bt, use_container_width=True)

# ============================================================
# STRATEGY COMPARISON
# ============================================================

st.header("📊 Strategy Comparison")

if st.button("Compare All Strategies"):
    with st.spinner("Running backtests for all strategies..."):
        comparison_data = []
        for strat in all_strategies:
            _, metrics = run_backtest(data, strat, 100000, intl_data, bond_data)
            comparison_data.append({
                'Strategy': strat,
                'Total Return': metrics['Total Return'],
                'CAGR': metrics['CAGR'],
                'Max Drawdown': metrics['Max Drawdown'],
                'Sharpe': metrics['Sharpe Ratio'],
                'Time in Market': metrics['Time in Market']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data from Yahoo Finance")
st.caption("⚠️ This is for educational purposes only. Past performance does not guarantee future results.")
