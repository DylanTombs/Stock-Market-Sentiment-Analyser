import pandas as pd
import requests
import time
from tqdm import tqdm
import pickle

# Constants
API_KEY = ""  # Replace with your actual API key
PRICE_BUFFER_DAYS = 5
SAVE_INTERVAL = 50
MIN_DATE = pd.to_datetime("2010-04-08", utc=True)
MAX_DATE = pd.to_datetime("2020-06-11", utc=True)

INPUT_FILE = "data_collection/Data/smaller_training_set.csv"
OUTPUT_FILE = "data_collection/Data/smaller_training_set_with_prices.csv"
CACHE_FILE = "price_cache.pkl"

# Load dataset
df = pd.read_csv(INPUT_FILE)
df['date'] = pd.to_datetime(df['date'], utc=True)

tickers = df['stock'].unique()
print(f"Downloading historical data for {len(tickers)} tickers...")

# Load cache if exists
try:
    with open(CACHE_FILE, 'rb') as f:
        all_prices = pickle.load(f)
    print(f"Loaded {len(all_prices)} cached tickers.")
except FileNotFoundError:
    all_prices = {}

# Keep track of failed tickers
failed_tickers = set()

# Download prices
request_count = 0
DAILY_LIMIT = 25  # Alpha Vantage free tier daily limit

for i, ticker in enumerate(tqdm(tickers, desc="Fetching prices"), start=1):
    if ticker in all_prices:
        continue  # Already cached
    
    if ticker in failed_tickers:
        continue  # Skip previously failed tickers
    
    # Check if we've hit the daily limit
    if request_count >= DAILY_LIMIT:
        print(f"\nReached daily limit of {DAILY_LIMIT} requests!")
        print("Saving progress and pausing...")
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(all_prices, f)
        print(f"Progress saved. Resume tomorrow by running the script again.")
        print(f"Progress: {len(all_prices)}/{len(tickers)} tickers completed ({len(all_prices)/len(tickers)*100:.1f}%)")
        break

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",  # Free endpoint
        "symbol": ticker,
        "outputsize": "full",
        "apikey": API_KEY
    }

    max_retries = 3
    retry_count = 0
    success = False
    
    while retry_count < max_retries and not success:
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()  # Raises an HTTPError for bad responses
            data = r.json()

            # Check for API errors
            if "Error Message" in data:
                print(f"⚠️ API Error for {ticker}: {data['Error Message']}")
                failed_tickers.add(ticker)
                break
            
            if "Note" in data:
                print(f"⚠️ Rate limit hit for {ticker}: {data['Note']}")
                print("Waiting 60 seconds...")
                time.sleep(60)
                retry_count += 1
                continue
            
            if "Time Series (Daily)" not in data:
                print(f"⚠️ No time series data for {ticker}. Response keys: {list(data.keys())}")
                # Print first few characters of response for debugging
                print(f"Response preview: {str(data)[:200]}...")
                failed_tickers.add(ticker)
                break

            hist_df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
            
            if hist_df.empty:
                print(f"⚠️ Empty dataframe for {ticker}")
                failed_tickers.add(ticker)
                break
            
            hist_df.index = pd.to_datetime(hist_df.index, utc=True)
            hist_df = hist_df.rename(columns={"4. close": "Close"})
            hist_df["Close"] = pd.to_numeric(hist_df["Close"], errors='coerce')
            hist_df = hist_df.reset_index().rename(columns={"index": "Date"})

            # Filter date range + buffer
            start_date = MIN_DATE - pd.Timedelta(days=PRICE_BUFFER_DAYS)
            end_date = MAX_DATE + pd.Timedelta(days=PRICE_BUFFER_DAYS)
            hist_df = hist_df[(hist_df['Date'] >= start_date) & (hist_df['Date'] <= end_date)]

            if hist_df.empty:
                print(f"⚠️ No data in date range for {ticker}")
                failed_tickers.add(ticker)
                break

            all_prices[ticker] = hist_df
            success = True
            request_count += 1  # Increment request counter
            print(f"Successfully fetched {len(hist_df)} records for {ticker} (Request {request_count}/{DAILY_LIMIT})")

        except requests.exceptions.RequestException as e:
            print(f"Request error for {ticker}: {e}")
            retry_count += 1
            if retry_count < max_retries:
                print(f"Retrying {ticker} ({retry_count}/{max_retries})...")
                time.sleep(30)
        except Exception as e:
            print(f"Unexpected error for {ticker}: {e}")
            failed_tickers.add(ticker)
            break

    # Sleep between requests to respect rate limits
    if success or retry_count >= max_retries:
        time.sleep(15)  # 12-15 seconds between requests

    # Save progress periodically
    if i % SAVE_INTERVAL == 0:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(all_prices, f)
        print(f"💾 Saved partial cache at ticker {i}")

# Final save
with open(CACHE_FILE, 'wb') as f:
    pickle.dump(all_prices, f)

print(f"Finished downloading prices for {len(all_prices)} tickers.")
print(f"Failed to fetch {len(failed_tickers)} tickers: {list(failed_tickers)[:10]}...")

# ===== Assign before/after prices & auto-label =====
before_prices, after_prices, labels = [], [], []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Assigning prices"):
    ticker = row['stock']
    news_date = row['date']
    price_data = all_prices.get(ticker)

    if price_data is None or price_data.empty:
        before_prices.append(None)
        after_prices.append(None)
        labels.append(None)
        continue

    # Use more robust date filtering
    before_date = news_date - pd.Timedelta(days=1)
    after_date = news_date + pd.Timedelta(days=1)
    
    before_rows = price_data[price_data['Date'] <= before_date]
    after_rows = price_data[price_data['Date'] >= after_date]
    
    # Get the closest available dates
    before_row = before_rows.tail(1) if not before_rows.empty else pd.DataFrame()
    after_row = after_rows.head(1) if not after_rows.empty else pd.DataFrame()

    before_close = before_row['Close'].iloc[0] if not before_row.empty else None
    after_close = after_row['Close'].iloc[0] if not after_row.empty else None

    before_prices.append(before_close)
    after_prices.append(after_close)

    # Label assignment
    if before_close is None or after_close is None or pd.isna(before_close) or pd.isna(after_close):
        labels.append(None)
    else:
        price_change = (after_close - before_close) / before_close
        if price_change > 0.001:  # > 0.1% increase
            labels.append(2)  # Positive
        elif price_change < -0.001:  # > 0.1% decrease
            labels.append(0)  # Negative
        else:
            labels.append(1)  # Neutral

df['before_close'] = before_prices
df['after_close'] = after_prices
df['label'] = labels

# Remove rows with missing data if desired
# df = df.dropna(subset=['before_close', 'after_close', 'label'])

df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved labeled dataset to {OUTPUT_FILE}")

# Print summary statistics
total_rows = len(df)
labeled_rows = df['label'].notna().sum()
print(f"\nSummary:")
print(f"Total rows: {total_rows}")
print(f"Successfully labeled: {labeled_rows} ({labeled_rows/total_rows*100:.1f}%)")
print(f"Missing labels: {total_rows - labeled_rows}")

label_counts = df['label'].value_counts().sort_index()
print(f"Label distribution: {dict(label_counts)}")