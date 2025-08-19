# Alpaca Custom Trading Bot - Strategy Testing

This section demonstrates testing a **custom trading strategy** using the [Alpaca API](https://alpaca.markets/). The bot fetches historical market data, generates trading signals based on custom logic, and executes real-time trades for a specific ticker. It is designed for **paper trading**, allowing safe testing without risking real capital.

---

## Screenshots 

<img width="1470" height="956" alt="image" src="https://github.com/user-attachments/assets/2aeeee7d-6138-4659-a466-24125b2ed4d5" /> 

<img width="1470" height="874" alt="image" src="https://github.com/user-attachments/assets/83168b73-a2eb-4ba1-9c8e-1cae73f9e58d" />

---

## Features

- Fetch historical market data for a given stock ticker using Alpaca API.
- Generate simple buy, sell, or hold signals based on custom logic.
- Execute market orders automatically based on generated signals.
- Configurable:
  - Ticker symbol (e.g., `AAPL`)
  - Timeframe for bars (`1Min`, `5Min`, `1Day`, etc.)
  - Trade quantity per order
- Runs at **one-minute intervals** for active signal testing.

---


## How It Works

1. **Fetch historical data**  
   Retrieves the last `N` bars for the specified symbol and timeframe using `api.get_bars()`.

2. **Generate trading signal**  
   - `BUY` e.g, if short-term MA crosses above long-term MA  
   - `SELL` e.g, if short-term MA crosses below long-term MA  
   - `HOLD` e.g, otherwise

3. **Place orders**  
   Checks account balance, calculates order cost, and submits market orders through Alpaca.

4. **Loop every minute**  
   The bot continuously fetches new data, recalculates signals, and executes orders.

5. **Example console output**
  Latest signal: BUY
  Placed BUY order for AAPL
  Latest signal: HOLD
  HOLD signal, no order placed
  Latest signal: SELL
  Placed SELL order for AAPL

---

## Notes

- This bot uses **paper trading** only. No real money is involved.
- Minimum timeframe for historical bars is **1 minute**. Sub-minute bars (e.g., 30s) are not supported.
- Customize strategy logic by modifying the `generate_signal()` function.
- Monitor orders and account balance when testing.
- You can adjust `short_window` and `long_window` for more frequent trading signals.
