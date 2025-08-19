import alpaca_trade_api as tradeapi
import pandas as pd
import time
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("APCA_API_BASE_URL")

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

symbol = "AAPL"
timeframe = "1Min"  # Use minute-level bars for volatility (to see it works)
short_window = 3
long_window = 7
qty = 1

def get_historical_data(symbol, timeframe, limit=50):
    bars = api.get_bars(symbol, timeframe, limit=limit).df
    return bars

def calculate_moving_averages(df):
    df['short_mavg'] = df['close'].rolling(window=short_window, min_periods=1).mean()
    df['long_mavg'] = df['close'].rolling(window=long_window, min_periods=1).mean()
    return df


# Basic example strategy
def generate_signal(df):
    # More aggressive signals: small windows on minute bars
    if df['short_mavg'].iloc[-1] > df['long_mavg'].iloc[-1]:
        return "BUY"
    elif df['short_mavg'].iloc[-1] < df['long_mavg'].iloc[-1]:
        return "SELL"
    return "HOLD"

def get_account_balance():
    account = api.get_account()
    return float(account.cash)

def calculate_order_cost(symbol, qty):
    bar = api.get_bars(symbol, '1Min', limit=1).df
    price = bar['close'].iloc[-1]
    return price * qty

def place_order(signal, symbol, qty):
    balance = get_account_balance()
    order_cost = calculate_order_cost(symbol, qty)
    if order_cost > balance:
        print("Insufficient balance")
        return

    if signal == "BUY":
        api.submit_order(symbol=symbol, qty=qty, side='buy', type='market', time_in_force='gtc')
        print(f"Placed BUY order for {symbol}")
    elif signal == "SELL":
        api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='gtc')
        print(f"Placed SELL order for {symbol}")
    else:
        print("HOLD signal, no order placed")

def run_trading_bot():
    df = get_historical_data(symbol, timeframe)
    df = calculate_moving_averages(df)
    signal = generate_signal(df)
    print(f"Latest signal: {signal}")
    place_order(signal, symbol, qty)

# Run bot every minute
while True:
    run_trading_bot()
    time.sleep(60)
