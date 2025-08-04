import pandas as pd
import io
import os
import requests
import time
from datetime import datetime

API_Key = "" # Replace with your Alpha Vantage API key

def get_OHLCV(symbol, start, end):
    save_file = f"{symbol}.csv"

    # Return already saved data if it exists
    if os.path.exists(save_file):
        data = pd.read_csv(save_file, index_col='date', parse_dates=True)
        return data
    
    # Alpha Vantage OHLCV data request url
    request_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={API_Key}&datatype=csv"

    try:
        response = requests.get(request_url, timeout=10)
        response.raise_for_status()
        
        # Parse returned CSV data
        data = pd.read_csv(io.StringIO(response.text), index_col='timestamp', parse_dates=True)
        data = data.sort_index()
        data = data.loc[start:end]

        # Save parsed data
        data.to_csv(save_file)
        return data
    
    except Exception as e:
        print (f"Error fetching data for {symbol}: ", e)
        return pd.DataFrame()
