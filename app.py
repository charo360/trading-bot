import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import ta
import time
import threading

# Initialize refresh-specific session state variables
if 'refresh_timestamp' not in st.session_state:
    st.session_state.refresh_timestamp = datetime.now()
if 'refresh_count' not in st.session_state:
    st.session_state.refresh_count = 0
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 10  # Default refresh time in seconds

# Page configuration
st.set_page_config(
    page_title="ABIBOT Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .indicator-success {
        color: #4CAF50;
        font-weight: bold;
    }
    .indicator-waiting {
        color: #FFC107;
        font-weight: bold;
    }
    .indicator-fail {
        color: #F44336;
        font-weight: bold;
    }
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .symbol-card {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .buy-signal {
        background-color: rgba(76, 175, 80, 0.2);
    }
    .sell-signal {
        background-color: rgba(244, 67, 54, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = datetime.now()
if 'auto_trading' not in st.session_state:
    st.session_state.auto_trading = False
if 'symbols_data' not in st.session_state:
    st.session_state.symbols_data = {}
if 'active_symbol' not in st.session_state:
    st.session_state.active_symbol = None
if 'trading_log' not in st.session_state:
    st.session_state.trading_log = []

# Sidebar settings
st.sidebar.title("ABIBOT Settings")

# Dashboard Settings
st.sidebar.header("Dashboard Settings")
new_interval = st.sidebar.slider("Refresh Interval (seconds)", 
                          min_value=5, 
                          max_value=60, 
                          value=st.session_state.refresh_interval,
                          step=5)

if new_interval != st.session_state.refresh_interval:
    st.session_state.refresh_interval = new_interval
    st.sidebar.success(f"Refresh interval updated to {new_interval} seconds")

# Add a manual refresh button
if st.sidebar.button("Force Refresh Now"):
    st.session_state.refresh_count += 1
    st.session_state.refresh_timestamp = datetime.now()
    st.rerun()

# MT5 Connection Settings
st.sidebar.header("MT5 Connection")
account_number = st.sidebar.text_input("Account Number", "")
password = st.sidebar.text_input("Password", "", type="password")
server = st.sidebar.text_input("Server", "")
path = st.sidebar.text_input("Path to MT5 terminal.exe", "C:/Program Files/MetaTrader 5/terminal64.exe")

# Strategy Parameters
st.sidebar.header("Strategy Parameters")
timeframe = st.sidebar.selectbox("Timeframe", ["5m", "15m", "30m", "1h", "4h"], index=0)
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
rsi_overbought = st.sidebar.slider("RSI Overbought", 50, 90, 75)
rsi_oversold = st.sidebar.slider("RSI Oversold", 10, 50, 35)

# Symbol Selection - Multiple symbols
available_symbols = ["USDJPY", "EURJPY", "GBPJPY", "XAUUSD", "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"]
watch_symbols = st.sidebar.multiselect("Watch Symbols", available_symbols, default=["USDJPY", "EURUSD"])

# Auto-trading settings
st.sidebar.header("Auto Trading")
auto_trading_enabled = st.sidebar.checkbox("Enable Auto Trading", False)
risk_percent = st.sidebar.slider("Risk per trade (%)", 0.5, 10.0, 2.0, 0.5)
take_profit_pips = st.sidebar.slider("Take Profit (pips)", 10, 200, 60, 5)
stop_loss_pips = st.sidebar.slider("Stop Loss (pips)", 10, 200, 30, 5)

# Function to connect to MT5
@st.cache_resource
def connect_mt5(account, password, server, path):
    if not mt5.initialize(path=path):
        st.error(f"MT5 initialization failed: {mt5.last_error()}")
        return False

    # Connect to account
    if not mt5.login(int(account), password, server):
        st.error(f"MT5 login failed: {mt5.last_error()}")
        return False

    return True

# MT5 timeframe mapping
timeframe_map = {
    "5m": mt5.TIMEFRAME_M5,
    "15m": mt5.TIMEFRAME_M15,
    "30m": mt5.TIMEFRAME_M30,
    "1h": mt5.TIMEFRAME_H1,
    "4h": mt5.TIMEFRAME_H4
}

def get_data(symbol, timeframe_mt5, bars=500):
    rates = mt5.copy_rates_from_pos(symbol, timeframe_mt5, 0, bars)
    if rates is None:
        st.error(f"Failed to get data for {symbol}: {mt5.last_error()}")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    # Calculate RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=rsi_period).rsi()

    # Calculate SuperTrend (custom implementation)
    atr_period = 10
    multiplier = 3.0

    # Calculate ATR
    df['tr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=atr_period)
    df['atr'] = df['tr'].rolling(window=atr_period).mean()

    # Calculate SuperTrend
    df['upperband'] = ((df['high'] + df['low']) / 2) + (multiplier * df['atr'])
    df['lowerband'] = ((df['high'] + df['low']) / 2) - (multiplier * df['atr'])

    df['in_uptrend'] = True

    for i in range(1, len(df)):
        current_close = df['close'].iloc[i]
        prev_close = df['close'].iloc[i-1]

        prev_upperband = df['upperband'].iloc[i-1]
        prev_lowerband = df['lowerband'].iloc[i-1]
        prev_uptrend = df['in_uptrend'].iloc[i-1]

        # Current upper and lower bands
        current_upperband = df['upperband'].iloc[i]
        current_lowerband = df['lowerband'].iloc[i]

        # Adjust upper and lower bands based on previous trend
        if prev_uptrend:
            if current_close < prev_lowerband:
                df.loc[df.index[i], 'in_uptrend'] = False
            else:
                df.loc[df.index[i], 'in_uptrend'] = True
        else:
            if current_close > prev_upperband:
                df.loc[df.index[i], 'in_uptrend'] = True
            else:
                df.loc[df.index[i], 'in_uptrend'] = False

    # Calculate SuperTrend line
    df['supertrend'] = np.nan
    for i in range(len(df)):
        if df['in_uptrend'].iloc[i]:
            df.loc[df.index[i], 'supertrend'] = df['lowerband'].iloc[i]
        else:
            df.loc[df.index[i], 'supertrend'] = df['upperband'].iloc[i]

    # Calculate SuperTrend direction (-1 for bullish, 1 for bearish)
    df['supertrend_direction'] = np.where(df['in_uptrend'], -1, 1)

    # Calculate VFI (Volume Flow Indicator) - Simplified version
    # In a real implementation, this would be more complex
    # Here it's just using a simple money flow calculation
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['money_flow'] = df['typical_price'] * df['tick_volume']
    df['money_flow_positive'] = np.where(df['close'] > df['close'].shift(1), df['money_flow'], 0)
    df['money_flow_negative'] = np.where(df['close'] < df['close'].shift(1), df['money_flow'], 0)

    df['vfi'] = df['money_flow_positive'].rolling(window=14).sum() - df['money_flow_negative'].rolling(window=14).sum()
    df['vfi_normalized'] = df['vfi'] / df['money_flow'].rolling(window=14).sum()

    return df

# Function to check for signals
def check_signals(df):
    if df is None or len(df) < 2:
        return {
            "rsi_value": None,
            "supertrend_value": None,
            "supertrend_direction": None,
            "vfi_value": None,
            "supertrend_buy_signal": False,
            "supertrend_sell_signal": False,
            "vfi_buy_confirm": False,
            "vfi_sell_confirm": False,
            "rsi_buy_confirm": False,
            "rsi_sell_confirm": False,
            "final_buy_signal": False,
            "final_sell_signal": False,
            "conditions_met": 0,
            "total_conditions": 3,
            "signal_progress": 0.0,
            "current_trend": "neutral",
            "last_checked": datetime.now()
        }

    # Get the latest values
    latest = df.iloc[-1]
    previous = df.iloc[-2]

    # Determine the current trend based on SuperTrend
    current_trend = "uptrend" if latest['supertrend_direction'] < 0 else "downtrend"

    # Check SuperTrend signal (crossover)
    supertrend_buy_signal = latest['supertrend_direction'] < 0 and previous['supertrend_direction'] >= 0
    supertrend_sell_signal = latest['supertrend_direction'] > 0 and previous['supertrend_direction'] <= 0

    # Check VFI conditions
    vfi_buy_confirm = latest['vfi_normalized'] < 0
    vfi_sell_confirm = latest['vfi_normalized'] > 0

    # Check RSI conditions
    rsi_buy_confirm = latest['rsi'] <= rsi_oversold
    rsi_sell_confirm = latest['rsi'] >= rsi_overbought

    # Track how many conditions are met
    conditions_met = 0
    total_conditions = 3

    # For buy signal
    buy_conditions_met = 0
    if supertrend_buy_signal:
        buy_conditions_met += 1
    if vfi_buy_confirm:
        buy_conditions_met += 1
    if rsi_buy_confirm:
        buy_conditions_met += 1

    # For sell signal
    sell_conditions_met = 0
    if supertrend_sell_signal:
        sell_conditions_met += 1
    if vfi_sell_confirm:
        sell_conditions_met += 1
    if rsi_sell_confirm:
        sell_conditions_met += 1

    # Use the higher of the two for the progress indicator
    conditions_met = max(buy_conditions_met, sell_conditions_met)
    signal_progress = (conditions_met / total_conditions) * 100.0

    # Final signals
    final_buy_signal = supertrend_buy_signal and vfi_buy_confirm and rsi_buy_confirm
    final_sell_signal = supertrend_sell_signal and vfi_sell_confirm and rsi_sell_confirm

    return {
        "rsi_value": latest['rsi'],
        "supertrend_value": latest['supertrend'],
        "supertrend_direction": latest['supertrend_direction'],
        "vfi_value": latest['vfi_normalized'],
        "supertrend_buy_signal": supertrend_buy_signal,
        "supertrend_sell_signal": supertrend_sell_signal,
        "vfi_buy_confirm": vfi_buy_confirm,
        "vfi_sell_confirm": vfi_sell_confirm,
        "rsi_buy_confirm": rsi_buy_confirm,
        "rsi_sell_confirm": rsi_sell_confirm,
        "final_buy_signal": final_buy_signal,
        "final_sell_signal": final_sell_signal,
        "conditions_met": conditions_met,
        "total_conditions": total_conditions,
        "signal_progress": signal_progress,
        "current_trend": current_trend,
        "last_checked": datetime.now()
    }

# Function to display indicator status
def display_indicator_status(label, condition, value=None, threshold=None):
    if condition:
        icon = "✅"
        status_class = "indicator-success"
    else:
        icon = "❌"
        status_class = "indicator-fail"

    if value is not None and threshold is not None:
        st.markdown(f"<div><b>{label}:</b> <span class='{status_class}'>{icon} {value:.2f} (Threshold: {threshold})</span></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div><b>{label}:</b> <span class='{status_class}'>{icon}</span></div>", unsafe_allow_html=True)

# Function to place a trade
# Modify the place_trade function to include better error handling
def test_simple_trade(symbol):
    """
    Function to test minimal trade placement with different volume formats
    """
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        st.error(f"Symbol {symbol} not found")
        return False
        
    # Print symbol properties for debugging
    st.write(f"Symbol properties:")
    st.write(f"- Volume step: {symbol_info.volume_step}")
    st.write(f"- Min volume: {symbol_info.volume_min}")
    st.write(f"- Max volume: {symbol_info.volume_max}")
    
    # Try different volume formats
    volume_formats = [0.01, 0.1, 1.0, 0.02, 0.05]
    
    for vol in volume_formats:
        # Format volume according to symbol requirements
        volume = round(vol / symbol_info.volume_step) * symbol_info.volume_step
        
        # Create minimal request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY,
            "price": mt5.symbol_info_tick(symbol).ask,
            "deviation": 10,
            "magic": 123456,
            "comment": "ABIBOT Test",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC
        }
        
        st.write(f"Testing with volume: {volume}")
        st.write(f"Request: {request}")
        
        try:
            result = mt5.order_send(request)
            if result is None:
                st.write(f"Failed with error: {mt5.last_error()}")
                continue
                
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                st.success(f"Success! Trade placed with volume {volume}")
                return volume  # Return the working volume format
            else:
                st.write(f"Failed with retcode: {result.retcode}, comment: {result.comment}")
        except Exception as e:
            st.write(f"Exception: {str(e)}")
    
    st.error("All volume formats failed")
    return None

# Function to place a trade
def place_trade(symbol, order_type, lot_size, sl_points, tp_points):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        st.error(f"Symbol {symbol} not found")
        return False

    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            st.error(f"Symbol {symbol} not available for trading")
            return False

    # Get current price
    point = symbol_info.point

    if order_type == "BUY":
        price = symbol_info.ask
        sl = price - sl_points * point
        tp = price + tp_points * point
        trade_type = mt5.ORDER_TYPE_BUY
    else:  # SELL
        price = symbol_info.bid
        sl = price + sl_points * point
        tp = price - tp_points * point
        trade_type = mt5.ORDER_TYPE_SELL

    # Prepare the trade request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": trade_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": 123456,
        "comment": "ABIBOT Auto Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,  # Use IOC instead of FOK as it worked
    }

    try:
        # Send the order
        result = mt5.order_send(request)
        
        # Check if result exists
        if result is None:
            st.error(f"MT5 order_send returned None. Last error: {mt5.last_error()}")
            return False
            
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            st.error(f"Failed to place order: {result.comment}, code: {result.retcode}")
            return False

        # Add to trading log
        log_entry = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "action": order_type,
            "lots": lot_size,
            "price": price,
            "sl": sl,
            "tp": tp
        }
        st.session_state.trading_log.append(log_entry)
        st.success(f"{order_type} order for {symbol} placed successfully at {price}")
        return True
        
    except Exception as e:
        st.error(f"Error placing trade: {str(e)}")
        return False

# Function to calculate lot size based on risk percentage
def calculate_lot_size(symbol, risk_percent, stop_loss_pips):
    account_info = mt5.account_info()
    if account_info is None:
        return 0.01  # Default minimum lot

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return 0.01

    # Get point value in account currency
    point_value = symbol_info.trade_tick_value

    # Calculate risk amount
    risk_amount = account_info.balance * (risk_percent / 100)

    # Calculate lot size based on risk (FIXED - was recursive before)
    lot_size = risk_amount / (stop_loss_pips * point_value)

    # Make sure lot size is valid
    if np.isnan(lot_size) or lot_size <= 0:
        return 0.01

    # Set maximum lot size
    max_lot_size = 2.0
    
    # Cap the lot size to maximum
    if lot_size > max_lot_size:
        st.warning(f"Calculated lot size ({lot_size}) exceeds maximum ({max_lot_size}). Using maximum allowed.")
        lot_size = max_lot_size

    # Round to standard lot sizes
    if lot_size < 0.01:
        lot_size = 0.01
    elif lot_size < 0.1:
        lot_size = round(lot_size, 2)  # Round to 0.01 lots
    elif lot_size < 1:
        lot_size = round(lot_size, 1)  # Round to 0.1 lots
    else:
        lot_size = round(lot_size)  # Round to whole lots

    return lot_size
# Function to update data for all symbols
def update_all_symbols_data():
    timeframe_mt5 = timeframe_map[timeframe]
    symbols_data = {}

    for symbol in watch_symbols:
        df = get_data(symbol, timeframe_mt5)
        if df is not None:
            signals = check_signals(df)
            symbols_data[symbol] = {
                "df": df,
                "signals": signals,
                "last_update": datetime.now()
            }

            # Auto-trading logic
            if st.session_state.auto_trading and auto_trading_enabled:
                # Check if there's an open position for this symbol
                positions = mt5.positions_get(symbol=symbol)
                has_open_position = positions and len(positions) > 0

                # Only place trades if no open position exists
                if not has_open_position:
                    if signals["final_buy_signal"]:
                        lot_size = calculate_lot_size(symbol, risk_percent, stop_loss_pips)
                        place_trade(symbol, "BUY", lot_size, stop_loss_pips, take_profit_pips)
                    elif signals["final_sell_signal"]:
                        lot_size = calculate_lot_size(symbol, risk_percent, stop_loss_pips)
                        place_trade(symbol, "SELL", lot_size, stop_loss_pips, take_profit_pips)

    st.session_state.symbols_data = symbols_data
    st.session_state.last_update_time = datetime.now()

# Main application
st.markdown("<h1 class='main-header'>ABIBOT Trading Dashboard</h1>", unsafe_allow_html=True)

# Connect button
if st.sidebar.button("Connect to MT5"):
    if account_number and password and server:
        with st.spinner("Connecting to MT5..."):
            connected = connect_mt5(int(account_number), password, server, path)
            if connected:
                st.success("Successfully connected to MT5!")
            else:
                st.error("Failed to connect to MT5. Check credentials and try again.")
    else:
        st.warning("Please fill in all MT5 connection fields")

# Auto-trading toggle
if auto_trading_enabled:
    if st.sidebar.button("Start Auto Trading" if not st.session_state.auto_trading else "Stop Auto Trading"):
        st.session_state.auto_trading = not st.session_state.auto_trading
        if st.session_state.auto_trading:
            st.sidebar.success("Auto trading started!")
        else:
            st.sidebar.info("Auto trading stopped.")

# Main dashboard (only show if MT5 is initialized)
if mt5.terminal_info() is not None:
    # Check if it's time to update data (every 10 seconds)
    current_time = datetime.now()
    time_diff = (current_time - st.session_state.last_update_time).total_seconds()

    if time_diff >= 10 or not st.session_state.symbols_data:
        with st.spinner("Updating data..."):
            update_all_symbols_data()

    # Display account info
    account_info = mt5.account_info()
    if account_info is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='sub-header'>Account Info</div>", unsafe_allow_html=True)
            st.write(f"Login: {account_info.login}")
            st.write(f"Server: {account_info.server}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='sub-header'>Balance</div>", unsafe_allow_html=True)
            st.write(f"Balance: ${account_info.balance:.2f}")
            st.write(f"Equity: ${account_info.equity:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='sub-header'>Profit</div>", unsafe_allow_html=True)
            st.write(f"Profit: ${account_info.profit:.2f}")
            profit_color = "green" if account_info.profit >= 0 else "red"
            st.markdown(f"<span style='color:{profit_color};font-weight:bold;font-size:24px;'>{account_info.profit:.2f}</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col4:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='sub-header'>Auto Trading</div>", unsafe_allow_html=True)
            if st.session_state.auto_trading:
                st.markdown("<span class='indicator-success'>✅ ACTIVE</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='indicator-fail'>❌ INACTIVE</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Multi-symbol summary view
    st.markdown("<div class='sub-header'>Symbol Overview</div>", unsafe_allow_html=True)
    symbol_cols = st.columns(len(watch_symbols) if watch_symbols else 1)

    for i, symbol in enumerate(watch_symbols):
        if symbol in st.session_state.symbols_data:
            data = st.session_state.symbols_data[symbol]
            signals = data["signals"]

            # Determine if there's a buy or sell signal
            signal_class = ""
            if signals["final_buy_signal"]:
                signal_class = "buy-signal"
            elif signals["final_sell_signal"]:
                signal_class = "sell-signal"

            with symbol_cols[i]:
                st.markdown(f"<div class='symbol-card {signal_class}'>", unsafe_allow_html=True)
                st.markdown(f"<h3>{symbol}</h3>", unsafe_allow_html=True)

                # Symbol info
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info:
                    # Add a small indicator to show real-time updates
                    refresh_indicator = "🔄" if (datetime.now() - st.session_state.last_update_time).total_seconds() < 3 else ""

                    st.markdown(f"""
                    <div style='display:flex;align-items:center;'>
                        <div style='flex-grow:1;'>
                            <p style='margin:0;'>Bid: <b>{symbol_info.bid:.5f}</b></p>
                            <p style='margin:0;'>Ask: <b>{symbol_info.ask:.5f}</b></p>
                        </div>
                        <div style='font-size:18px;'>{refresh_indicator}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # RSI value with timestamp to show it's updating
                rsi_value = signals["rsi_value"]
                last_updated = signals["last_checked"].strftime("%H:%M:%S")
                next_refresh = signals.get("next_refresh", "")
                if rsi_value is not None:
                    rsi_color = "green" if rsi_value <= rsi_oversold else "red" if rsi_value >= rsi_overbought else "black"
                    st.markdown(f"""
                    <div>
                        <p style='margin:0;'>RSI: <span style='color:{rsi_color};font-weight:bold;'>{rsi_value:.2f}</span></p>
                        <p style='margin:0;font-size:10px;color:gray;'>Updated: {last_updated}</p>
                        <p style='margin:0;font-size:11px;color:#E63946;font-weight:bold;'>Refresh in: {next_refresh}s</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Signal status
                if signals["final_buy_signal"]:
                    st.markdown("<div style='background-color:#4CAF50;color:white;padding:5px;text-align:center;'>BUY</div>", unsafe_allow_html=True)
                elif signals["final_sell_signal"]:
                    st.markdown("<div style='background-color:#F44336;color:white;padding:5px;text-align:center;'>SELL</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='background-color:#9E9E9E;color:white;padding:5px;text-align:center;'>WAIT</div>", unsafe_allow_html=True)

                # Button to view detailed chart
                if st.button(f"View {symbol} Details", key=f"btn_{symbol}"):
                    st.session_state.active_symbol = symbol

                st.markdown("</div>", unsafe_allow_html=True)

    # Detailed view for selected symbol
    if st.session_state.active_symbol and st.session_state.active_symbol in st.session_state.symbols_data:
        symbol = st.session_state.active_symbol
        data = st.session_state.symbols_data[symbol]
        df = data["df"]
        signals = data["signals"]

        st.markdown(f"<div class='sub-header'>{symbol} Detailed Analysis</div>", unsafe_allow_html=True)

        # Display chart
        fig = go.Figure()

        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ))

        # Add SuperTrend
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['supertrend'],
            mode='lines',
            line=dict(color='purple', width=2),
            name='SuperTrend'
        ))

        # Set chart layout
        fig.update_layout(
            title=f'{symbol} - {timeframe}',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            height=500,
            margin=dict(l=10, r=10, t=40, b=10),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display indicators chart
        col1, col2 = st.columns(2)

        with col1:
            # RSI Chart
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=df.index,
                y=df['rsi'],
                mode='lines',
                name='RSI',
                line=dict(color='blue', width=2)
            ))

            # Add overbought and oversold lines
            fig_rsi.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", annotation_text=f"Overbought ({rsi_overbought})")
            fig_rsi.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", annotation_text=f"Oversold ({rsi_oversold})")

            fig_rsi.update_layout(
                title="RSI Indicator",
                xaxis_title="Date",
                yaxis_title="RSI Value",
                height=300,
                margin=dict(l=10, r=10, t=40, b=10),
            )

            st.plotly_chart(fig_rsi, use_container_width=True)

        with col2:
            # VFI Chart
            fig_vfi = go.Figure()
            fig_vfi.add_trace(go.Scatter(
                x=df.index,
                y=df['vfi_normalized'],
                mode='lines',
                name='VFI',
                line=dict(color='orange', width=2)
            ))

            # Add zero line
            fig_vfi.add_hline(y=0, line_dash="solid", line_color="black")

            fig_vfi.update_layout(
                title="Volume Flow Indicator",
                xaxis_title="Date",
                yaxis_title="VFI Value",
                height=300,
                margin=dict(l=10, r=10, t=40, b=10),
            )

            st.plotly_chart(fig_vfi, use_container_width=True)

        # Display Signal Status
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>Signal Status</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<b>SuperTrend Status</b>", unsafe_allow_html=True)
            display_indicator_status("Buy Signal", signals["supertrend_buy_signal"])
            display_indicator_status("Sell Signal", signals["supertrend_sell_signal"])

            st.markdown("<b>VFI Status</b>", unsafe_allow_html=True)
            display_indicator_status("Buy Confirmation (VFI < 0)", signals["vfi_buy_confirm"], signals["vfi_value"], 0)
            display_indicator_status("Sell Confirmation (VFI > 0)", signals["vfi_sell_confirm"], signals["vfi_value"], 0)

        with col2:
            st.markdown("<b>RSI Status</b>", unsafe_allow_html=True)
            display_indicator_status("Buy Confirmation (RSI <= Oversold)", signals["rsi_buy_confirm"], signals["rsi_value"], rsi_oversold)
            display_indicator_status("Sell Confirmation (RSI >= Overbought)", signals["rsi_sell_confirm"], signals["rsi_value"], rsi_overbought)

            st.markdown("<b>Final Signal</b>", unsafe_allow_html=True)
            if signals["final_buy_signal"]:
                st.markdown("<div style='padding: 10px; background-color: #4CAF50; color: white; font-weight: bold; text-align: center; border-radius: 5px;'>BUY SIGNAL CONFIRMED!</div>", unsafe_allow_html=True)
            elif signals["final_sell_signal"]:
                st.markdown("<div style='padding: 10px; background-color: #F44336; color: white; font-weight: bold; text-align: center; border-radius: 5px;'>SELL SIGNAL CONFIRMED!</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='padding: 10px; background-color: #9E9E9E; color: white; font-weight: bold; text-align: center; border-radius: 5px;'>NO CONFIRMED SIGNALS</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Manual trading buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"BUY {symbol}"):
                lot_size = calculate_lot_size(symbol, risk_percent, stop_loss_pips)
                if place_trade(symbol, "BUY", lot_size, stop_loss_pips, take_profit_pips):
                    st.success(f"BUY order placed for {symbol}")

        with col2:
            if st.button(f"SELL {symbol}"):
                lot_size = calculate_lot_size(symbol, risk_percent, stop_loss_pips)
                if place_trade(symbol, "SELL", lot_size, stop_loss_pips, take_profit_pips):
                    st.success(f"SELL order placed for {symbol}")

    # Open positions across all symbols
    st.markdown("<div class='sub-header'>Open Positions</div>", unsafe_allow_html=True)
    positions = mt5.positions_get()

    if positions and len(positions) > 0:
        positions_df = pd.DataFrame(list(positions), columns=positions[0]._asdict().keys())
        positions_df['time'] = pd.to_datetime(positions_df['time'], unit='s')
        positions_df['time_update'] = pd.to_datetime(positions_df['time_update'], unit='s')
        positions_df['type'] = positions_df['type'].apply(lambda x: 'Buy' if x == 0 else 'Sell')

        # Display as a table
        st.dataframe(positions_df[['symbol', 'type', 'volume', 'price_open', 'tp', 'sl', 'profit', 'time']])
    else:
        st.info("No open positions")

    # Trading log
    if st.session_state.trading_log:
        st.markdown("<div class='sub-header'>Trading Log</div>", unsafe_allow_html=True)
        trading_log_df = pd.DataFrame(st.session_state.trading_log)
        st.dataframe(trading_log_df)

    # Automatic refresh using st.empty() and sleep
    placeholder = st.empty()
    time_to_refresh = max(0, 10 - time_diff)

    # Update the time display for all symbols
    for symbol in watch_symbols:
        if symbol in st.session_state.symbols_data:
            signals = st.session_state.symbols_data[symbol]['signals']
            # Add timestamp to the signals data
            signals['updated_time'] = datetime.now().strftime('%H:%M:%S')
            signals['next_refresh'] = f"{time_to_refresh:.1f}"

    # Show detailed scanner status with progress toward signals
    with placeholder.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>Bot Status - Scanning for Opportunities</div>", unsafe_allow_html=True)

        # Add a pulsing indicator to show active scanning with prominent countdown
        scan_indicator = "🔄" if time_to_refresh < 5 else "⏳"
        st.markdown(f"""
        <div style='display:flex;align-items:center;margin-bottom:10px;'>
            <div style='font-size:24px;margin-right:10px;'>{scan_indicator}</div>
            <div>
                <p style='margin:0;font-weight:bold;font-size:18px;'>LIVE SCANNING ACTIVE</p>
                <p style='margin:0;'>Last update: {st.session_state.last_update_time.strftime('%H:%M:%S')}</p>
                <p style='margin:0;color:#E63946;font-weight:bold;font-size:20px;'>Next refresh in: {time_to_refresh:.1f} seconds</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Show scanning progress for each symbol
        for symbol in watch_symbols:
            if symbol in st.session_state.symbols_data:
                signals = st.session_state.symbols_data[symbol]['signals']
                conditions_met = 0
                total_conditions = 3  # SuperTrend, RSI, VFI

                # Check how many conditions are met
                if signals["supertrend_buy_signal"] or signals["supertrend_sell_signal"]:
                    conditions_met += 1
                if signals["rsi_buy_confirm"] or signals["rsi_sell_confirm"]:
                    conditions_met += 1
                if signals["vfi_buy_confirm"] or signals["vfi_sell_confirm"]:
                    conditions_met += 1

                progress_pct = (conditions_met / total_conditions) * 100

                # Set color based on progress
                if progress_pct == 100:
                    bar_color = "green"
                    status_text = "SIGNAL READY!"
                elif progress_pct > 50:
                    bar_color = "orange"
                    status_text = "Almost there..."
                else:
                    bar_color = "gray"
                    status_text = "Scanning..."

                st.markdown(f"<p><b>{symbol}</b>: {progress_pct:.0f}% conditions met - {status_text}</p>", unsafe_allow_html=True)
                st.progress(progress_pct/100)

        st.markdown("</div>", unsafe_allow_html=True)

    # Update the refresh timestamp when data is actually refreshed
  
    
    #

else:
    st.warning("Please connect to MT5 first")


# Add this at the end of your script, just before the "Cleanup on app close" section
# Add a test trade button to diagnose MT5 issues
# Create a dedicated container for the countdown
refresh_countdown = st.empty()

# Start countdown
current_time = datetime.now()
time_diff = (current_time - st.session_state.last_update_time).total_seconds()
time_to_refresh = max(0, 10 - time_diff)

# If it's already time to refresh, do it immediately
if time_diff >= 10:
    st.write("Refreshing now...")
    time.sleep(0.5)  # Small delay
    st.rerun()  # Force a Streamlit rerun
else:
    # Otherwise, show countdown
    for i in range(int(time_to_refresh), 0, -1):
        refresh_countdown.write(f"Refreshing in {i} seconds...")
        time.sleep(1)
    
    # After countdown completes, refresh
    refresh_countdown.write("Refreshing now...")
    st.rerun()
        


# Cleanup on app close
def on_shutdown():
    if mt5.terminal_info() is not None:
        mt5.shutdown()

# Register the shutdown function to be called when the script exits
import atexit
atexit.register(on_shutdown)

# Function to setup auto-refresh

