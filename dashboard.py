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

@st.cache_data(ttl=600)
def load_sector_data():
    """Load sector ETF data for heatmap."""
    sectors = {
        'XLK': 'Technology',
        'XLF': 'Financials',
        'XLV': 'Healthcare',
        'XLE': 'Energy',
        'XLI': 'Industrials',
        'XLY': 'Consumer Disc',
        'XLP': 'Consumer Staples',
        'XLU': 'Utilities',
        'XLB': 'Materials',
        'XLRE': 'Real Estate',
        'XLC': 'Communication'
    }
    
    results = {}
    try:
        for ticker, name in sectors.items():
            data = yf.download(ticker, period='6mo', progress=False)
            if data is not None and len(data) > 0:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                # Calculate metrics
                latest_close = data['Close'].iloc[-1]
                ma50 = data['Close'].rolling(50).mean().iloc[-1]
                ma200 = data['Close'].rolling(200).mean().iloc[-1] if len(data) >= 200 else ma50
                
                # Returns
                ret_1d = data['Close'].pct_change().iloc[-1] * 100
                ret_1w = (data['Close'].iloc[-1] / data['Close'].iloc[-5] - 1) * 100 if len(data) >= 5 else 0
                ret_1m = (data['Close'].iloc[-1] / data['Close'].iloc[-21] - 1) * 100 if len(data) >= 21 else 0
                
                results[ticker] = {
                    'name': name,
                    'price': latest_close,
                    'above_50ma': latest_close > ma50,
                    'above_200ma': latest_close > ma200,
                    'ret_1d': ret_1d,
                    'ret_1w': ret_1w,
                    'ret_1m': ret_1m
                }
    except Exception as e:
        pass
    
    return results

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

# Quick ticker selection
st.sidebar.subheader("📈 Ticker")
preset_tickers = ['SPY', 'QQQ', 'IWM', 'DIA', 'Custom']
ticker_choice = st.sidebar.selectbox("Select Index", preset_tickers, index=0)

if ticker_choice == 'Custom':
    ticker = st.sidebar.text_input("Enter Ticker", value="SPY")
else:
    ticker = ticker_choice

period = st.sidebar.selectbox("Data Period", ['1y', '2y', '3y', '5y', 'max'], index=1)

# Ticker descriptions
ticker_info = {
    'SPY': 'S&P 500 ETF',
    'QQQ': 'Nasdaq 100 ETF', 
    'IWM': 'Russell 2000 ETF',
    'DIA': 'Dow Jones ETF'
}
if ticker in ticker_info:
    st.sidebar.caption(f"📊 {ticker_info[ticker]}")

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
# MARKET REGIME INDICATOR
# ============================================================

st.header("🌡️ Market Regime")

# Calculate regime indicators
above_200 = latest['Close'] > latest['MA200']
above_50 = latest['Close'] > latest['MA50']
ma50_above_200 = latest['MA50'] > latest['MA200']
rsi = latest['RSI']
vol = latest['Vol']

# Determine regime
if above_200 and ma50_above_200 and rsi > 50:
    regime = "🟢 BULL MARKET"
    regime_color = "green"
    regime_desc = "Strong uptrend - all systems go"
elif above_200 and above_50:
    regime = "🟡 BULLISH"
    regime_color = "orange" 
    regime_desc = "Uptrend intact but watch for weakness"
elif not above_200 and not ma50_above_200:
    regime = "🔴 BEAR MARKET"
    regime_color = "red"
    regime_desc = "Downtrend - consider defensive positions"
else:
    regime = "⚪ TRANSITIONAL"
    regime_color = "gray"
    regime_desc = "Mixed signals - be cautious"

# Vol regime
if vol < 0.12:
    vol_regime = "😴 Very Low Vol"
elif vol < 0.18:
    vol_regime = "😊 Low Vol"
elif vol < 0.25:
    vol_regime = "😐 Normal Vol"
else:
    vol_regime = "😰 High Vol"

regime_cols = st.columns(3)
with regime_cols[0]:
    st.metric("Market Regime", regime)
    st.caption(regime_desc)
with regime_cols[1]:
    st.metric("Volatility", f"{vol*100:.1f}%")
    st.caption(vol_regime)
with regime_cols[2]:
    trend_strength = signal_count / 5 * 100
    st.metric("Trend Strength", f"{trend_strength:.0f}%")
    st.caption(f"{signal_count}/5 signals bullish")

# ============================================================
# SECTOR HEATMAP
# ============================================================

with st.expander("📊 Sector Heatmap", expanded=False):
    sector_data = load_sector_data()
    
    if sector_data:
        sector_list = []
        for ticker, info in sector_data.items():
            trend = "🟢" if info['above_50ma'] and info['above_200ma'] else ("🟡" if info['above_50ma'] else "🔴")
            sector_list.append({
                'Sector': info['name'],
                'Ticker': ticker,
                'Trend': trend,
                '1D': f"{info['ret_1d']:+.1f}%",
                '1W': f"{info['ret_1w']:+.1f}%",
                '1M': f"{info['ret_1m']:+.1f}%"
            })
        
        sector_df = pd.DataFrame(sector_list)
        st.dataframe(sector_df, use_container_width=True, hide_index=True)
        st.caption("🟢 Above 50 & 200 MA | 🟡 Above 50 MA only | 🔴 Below both MAs")
    else:
        st.info("Unable to load sector data")

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
# POSITION SIZER
# ============================================================

st.header("💰 Position Sizer")

ps_col1, ps_col2 = st.columns(2)

with ps_col1:
    portfolio_size = st.number_input("Portfolio Size ($)", value=100000, step=5000, min_value=1000)
    spy_price = latest['Close']
    sso_price_approx = spy_price * 0.15  # SSO is roughly 15% of SPY price
    
with ps_col2:
    st.metric("Current SPY Price", f"${spy_price:.2f}")
    st.caption("SSO price estimated for calculations")

st.subheader("📋 Position Recommendations")

# Calculate positions for each strategy
position_data = []
for strat in ['Ensemble', 'Voting System', 'Vol Adaptive', 'Steady Eddie', 'Golden Boost']:
    pos = positions.get(strat, 0)
    
    if pos == 0:
        allocation = 0
        etf = "CASH (SGOV/BIL)"
        shares = 0
    elif pos <= 1.0:
        allocation = portfolio_size * pos
        etf = "SPY"
        shares = int(allocation / spy_price)
    elif pos == 1.5:
        # 50% SPY + 50% SSO
        spy_alloc = portfolio_size * 0.5
        sso_alloc = portfolio_size * 0.5
        etf = f"SPY + SSO (50/50)"
        shares = f"{int(spy_alloc/spy_price)} SPY + {int(sso_alloc/sso_price_approx)} SSO"
        allocation = portfolio_size
    else:  # 2x
        allocation = portfolio_size
        etf = "SSO"
        shares = int(allocation / sso_price_approx)
    
    position_data.append({
        'Strategy': strat,
        'Signal': f"{pos}x" if pos > 0 else "CASH",
        'ETF': etf,
        'Shares': shares if isinstance(shares, str) else f"{shares:,}",
        'Allocation': f"${allocation:,.0f}"
    })

position_df = pd.DataFrame(position_data)
st.dataframe(position_df, use_container_width=True, hide_index=True)

st.caption("💡 Tip: SSO is 2x leveraged SPY. For 1.5x exposure, split 50/50 between SPY and SSO.")

# ============================================================
# SIGNAL HISTORY LOG
# ============================================================

st.header("📅 Signal History")

# Get signal history from recent data
history_strat = st.selectbox("Show history for:", ['Ensemble', 'Voting System', 'Vol Adaptive', 'Steady Eddie'], key='history_strat')

def get_signal_history(data, strategy_name):
    """Get historical signal changes."""
    history = []
    prev_pos = None
    entry_price = None
    entry_date = None
    
    for idx, row in data.iterrows():
        if pd.isna(row['MA200']):
            continue
        
        signals = get_signals(row)
        strat_positions, _ = get_strategy_position(signals, row, None)
        current_pos = strat_positions.get(strategy_name, 0)
        
        if prev_pos is not None and current_pos != prev_pos:
            if current_pos > 0 and prev_pos == 0:
                # Buy signal
                history.append({
                    'Date': idx.strftime('%Y-%m-%d'),
                    'Action': '🟢 BUY',
                    'Price': f"${row['Close']:.2f}",
                    'Position': f"{current_pos}x",
                    'Return': ''
                })
                entry_price = row['Close']
                entry_date = idx
            elif current_pos == 0 and prev_pos > 0:
                # Sell signal
                ret = ((row['Close'] - entry_price) / entry_price * 100) if entry_price else 0
                history.append({
                    'Date': idx.strftime('%Y-%m-%d'),
                    'Action': '🔴 SELL',
                    'Price': f"${row['Close']:.2f}",
                    'Position': 'CASH',
                    'Return': f"{ret:+.1f}%"
                })
                entry_price = None
        
        prev_pos = current_pos
    
    return history[-10:]  # Last 10 signals

history = get_signal_history(data, history_strat)
if history:
    history_df = pd.DataFrame(history)
    st.dataframe(history_df, use_container_width=True, hide_index=True)
else:
    st.info("No signal changes in the selected period.")

# ============================================================
# PORTFOLIO TRACKER
# ============================================================

st.header("📈 Portfolio Tracker")

with st.expander("Track Your Portfolio", expanded=False):
    st.subheader("Enter Your Holdings")
    
    track_col1, track_col2, track_col3 = st.columns(3)
    
    with track_col1:
        spy_shares = st.number_input("SPY Shares", value=0, min_value=0, step=1)
        spy_avg_cost = st.number_input("SPY Avg Cost ($)", value=0.0, min_value=0.0, step=1.0)
    
    with track_col2:
        sso_shares = st.number_input("SSO Shares", value=0, min_value=0, step=1)
        sso_avg_cost = st.number_input("SSO Avg Cost ($)", value=0.0, min_value=0.0, step=1.0)
    
    with track_col3:
        cash_balance = st.number_input("Cash Balance ($)", value=0.0, min_value=0.0, step=100.0)
    
    if spy_shares > 0 or sso_shares > 0 or cash_balance > 0:
        # Calculate current values
        spy_price = latest['Close']
        sso_price = spy_price * 0.15  # Approximate
        
        spy_value = spy_shares * spy_price
        sso_value = sso_shares * sso_price
        total_value = spy_value + sso_value + cash_balance
        
        # Calculate P&L
        spy_cost_basis = spy_shares * spy_avg_cost if spy_avg_cost > 0 else spy_value
        sso_cost_basis = sso_shares * sso_avg_cost if sso_avg_cost > 0 else sso_value
        total_cost = spy_cost_basis + sso_cost_basis + cash_balance
        
        total_pnl = (spy_value + sso_value) - (spy_cost_basis + sso_cost_basis)
        total_pnl_pct = (total_pnl / (spy_cost_basis + sso_cost_basis) * 100) if (spy_cost_basis + sso_cost_basis) > 0 else 0
        
        # Current allocation
        spy_pct = (spy_value / total_value * 100) if total_value > 0 else 0
        sso_pct = (sso_value / total_value * 100) if total_value > 0 else 0
        cash_pct = (cash_balance / total_value * 100) if total_value > 0 else 0
        
        st.subheader("Portfolio Summary")
        
        port_cols = st.columns(4)
        with port_cols[0]:
            st.metric("Total Value", f"${total_value:,.2f}")
        with port_cols[1]:
            st.metric("Total P&L", f"${total_pnl:,.2f}", delta=f"{total_pnl_pct:+.1f}%")
        with port_cols[2]:
            # Current effective leverage
            leverage = (spy_pct + sso_pct * 2) / 100
            st.metric("Effective Leverage", f"{leverage:.2f}x")
        with port_cols[3]:
            st.metric("Cash", f"{cash_pct:.1f}%")
        
        # Allocation breakdown
        st.caption(f"Allocation: SPY {spy_pct:.1f}% | SSO {sso_pct:.1f}% | Cash {cash_pct:.1f}%")

# ============================================================
# REBALANCE REMINDER
# ============================================================

st.header("🔔 Rebalance Checker")

with st.expander("Check Rebalancing Needs", expanded=False):
    st.subheader("Current vs Target Allocation")
    
    reb_col1, reb_col2 = st.columns(2)
    
    with reb_col1:
        target_strategy = st.selectbox("Target Strategy", ['Ensemble', 'Voting System', 'Vol Adaptive', 'Steady Eddie'], key='rebal_strat')
        current_spy_pct = st.slider("Current SPY %", 0, 100, 50)
        current_sso_pct = st.slider("Current SSO %", 0, 100, 0)
        current_cash_pct = 100 - current_spy_pct - current_sso_pct
        st.caption(f"Cash: {current_cash_pct}%")
    
    with reb_col2:
        # Get target allocation based on strategy signal
        target_pos = positions.get(target_strategy, 0)
        
        if target_pos == 0:
            target_spy = 0
            target_sso = 0
            target_cash = 100
        elif target_pos == 1.0:
            target_spy = 100
            target_sso = 0
            target_cash = 0
        elif target_pos == 1.5:
            target_spy = 50
            target_sso = 50
            target_cash = 0
        else:  # 2x
            target_spy = 0
            target_sso = 100
            target_cash = 0
        
        st.metric("Target SPY", f"{target_spy}%", delta=f"{target_spy - current_spy_pct:+d}%")
        st.metric("Target SSO", f"{target_sso}%", delta=f"{target_sso - current_sso_pct:+d}%")
        st.metric("Target Cash", f"{target_cash}%", delta=f"{target_cash - current_cash_pct:+d}%")
    
    # Calculate drift
    total_drift = abs(target_spy - current_spy_pct) + abs(target_sso - current_sso_pct) + abs(target_cash - current_cash_pct)
    
    if total_drift > 20:
        st.error(f"⚠️ REBALANCE RECOMMENDED - Total drift: {total_drift/2:.0f}%")
    elif total_drift > 10:
        st.warning(f"🟡 Consider rebalancing - Total drift: {total_drift/2:.0f}%")
    else:
        st.success(f"✅ Portfolio aligned - Total drift: {total_drift/2:.0f}%")

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data from Yahoo Finance")
st.caption("⚠️ This is for educational purposes only. Past performance does not guarantee future results.")
