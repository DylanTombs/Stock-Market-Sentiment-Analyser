import pandas as pd
import time
from datetime import timedelta, date
import requests
import os
import json
from pathlib import Path
from pymongo import MongoClient

# Database connection 

mongodb_username = ""
mongodb_password = ""

client = MongoClient(f"mongodb+srv://{mongodb_username}:{mongodb_password}@database.pohgsdb.mongodb.net/")
db = client['dataset']
collection = db['training_dataset']

# Configuration
ALPHA_VANTAGE_API_KEY = 'ZXM9BM6AJ6KOFYBN'  # Replace with your actual API key
MAX_CALLS_PER_MINUTE = 5  # Free tier limit
REQUEST_INTERVAL = 60 / MAX_CALLS_PER_MINUTE  # Seconds between requests

# File paths
CSV_DATA_PATH = "data_collection/Data/smaller_training_set.csv"
OUTPUT_PATH = 'data_collection/Data/processed_training.csv'
PROGRESS_PATH = 'data_collection/Data/stock_progress.json'
STOCK_DATA_CACHE = 'data_collection/Data/stock_data_cache.json'
BACKUP_DIR = 'data_collection/Data/backups'

def load_progress():
    """Load processing progress from file"""
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, 'r') as f:
            return json.load(f)
    return {'processed_stocks': [], 'completed_stocks': 0}

def save_progress(progress):
    """Save processing progress to file"""
    os.makedirs(os.path.dirname(PROGRESS_PATH), exist_ok=True)
    with open(PROGRESS_PATH, 'w') as f:
        json.dump(progress, f, indent=2)

def load_stock_cache():
    """Load cached stock data"""
    if os.path.exists(STOCK_DATA_CACHE):
        with open(STOCK_DATA_CACHE, 'r') as f:
            cache = json.load(f)
            # Convert date strings back to dates in the data
            for stock, data in cache.items():
                if 'price_data' in data:
                    data['price_data'] = {pd.to_datetime(k).date(): v for k, v in data['price_data'].items()}
            return cache
    return {}

def save_stock_cache(cache):
    """Save stock data cache"""
    os.makedirs(os.path.dirname(STOCK_DATA_CACHE), exist_ok=True)
    # Convert dates to strings for JSON serialization
    serializable_cache = {}
    for stock, data in cache.items():
        serializable_cache[stock] = {
            'date_range': data['date_range'],
            'price_data': {str(k): v for k, v in data['price_data'].items()}
        }
    
    with open(STOCK_DATA_CACHE, 'w') as f:
        json.dump(serializable_cache, f, indent=2)

def analyze_stock_requirements(news_df):
    """Analyze what stocks need to be processed and their date ranges"""
    stock_info = {}
    
    for _, row in news_df.iterrows():
        stock = row['stock']
        article_date = pd.to_datetime(row['date']).date()
        
        if stock not in stock_info:
            stock_info[stock] = {
                'min_date': article_date,
                'max_date': article_date,
                'news_count': 0
            }
        
        stock_info[stock]['min_date'] = min(stock_info[stock]['min_date'], article_date)
        stock_info[stock]['max_date'] = max(stock_info[stock]['max_date'], article_date)
        stock_info[stock]['news_count'] += 1
    
    # Add buffer days for before/after analysis
    for stock in stock_info:
        stock_info[stock]['fetch_start'] = stock_info[stock]['min_date'] - timedelta(days=2)
        stock_info[stock]['fetch_end'] = stock_info[stock]['max_date'] + timedelta(days=2)
    
    return stock_info

def get_stock_data_full_range(stock, start_date, end_date):
    """Get stock data for full date range using direct CSV API"""
    try:
        print(f"  Fetching data from {start_date} to {end_date}")
        
        # Direct Alpha Vantage CSV API call
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock}&apikey={ALPHA_VANTAGE_API_KEY}&outputsize=full&datatype=csv"
        
        print(f"  Downloading {stock} from Alpha Vantage...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Check if we got valid CSV data (not error message)
        content = response.text
        if content.startswith('Error Message') or content.startswith('Note') or content.startswith('Information'):
            print(f"  API Error for {stock}: {content[:100]}...")
            return {}
        
        # Parse CSV from response text
        from io import StringIO
        data = pd.read_csv(StringIO(content), index_col='timestamp', parse_dates=True)
        data = data.sort_index()
        
        # Filter for our date range
        data = data.loc[start_date:end_date]
        
        if data.empty:
            print(f"  No data found for {stock} in date range")
            return {}
        
        # Convert to dict with date as key
        price_data = {}
        for date_idx, row_data in data.iterrows():
            price_data[date_idx.date()] = {
                'open': float(row_data['open']),
                'high': float(row_data['high']), 
                'low': float(row_data['low']),
                'close': float(row_data['close']),
                'volume': int(row_data['volume'])
            }
        
        print(f"  Retrieved {len(price_data)} trading days")
        return price_data
        
    except requests.exceptions.RequestException as e:
        print(f"  Network error for {stock}: {str(e)}")
        return {}
    except Exception as e:
        print(f"  Processing error for {stock}: {str(e)}")
        return {}

def get_price_for_sentiment(stock_cache, stock, article_date, days_before=1, days_after=1):
    """Get price data for a specific sentiment entry"""
    if stock not in stock_cache or 'price_data' not in stock_cache[stock]:
        return None, None
    
    price_data = stock_cache[stock]['price_data']
    
    # Find price before news (1-2 days before)
    start_price = None
    for i in range(days_before, days_before + 7):  # Try 1-3 days before
        check_date = article_date - timedelta(days=i)
        if check_date in price_data:
            start_price = price_data[check_date]['close']
            break
    
    # Find price after news (1-2 days after)
    end_price = None
    for i in range(days_after, days_after + 7):  # Try 1-3 days after
        check_date = article_date + timedelta(days=i)
        if check_date in price_data:
            end_price = price_data[check_date]['close']
            break
    
    return start_price, end_price

def main():
    # Validate API key
    if not ALPHA_VANTAGE_API_KEY:
        print("ERROR: Please set your ALPHA_VANTAGE_API_KEY")
        return
    
    if not (mongodb_username and mongodb_password):
        print("ERROR: Please set your mongodb credentials")
        return
    
    # Load sentiment data
    print("Loading sentiment data...")
    news_df = pd.read_csv(CSV_DATA_PATH)
    total_rows = len(news_df)
    
    # Analyze stock requirements
    print("Analyzing stock requirements...")
    stock_info = analyze_stock_requirements(news_df)
    unique_stocks = list(stock_info.keys())
    total_stocks = len(unique_stocks)
    
    print(f"Total news articles: {total_rows}")
    print(f"Unique stocks to fetch: {total_stocks}")
    print(f"Average news per stock: {total_rows/total_stocks:.1f}")
    
    # Load existing progress and cache
    progress = load_progress()
    stock_cache = load_stock_cache()
    processed_stocks = set(progress['processed_stocks'])
    
    print(f"Already processed stocks: {len(processed_stocks)}")
    
    # Process each unique stock
    api_calls_made = 0
    session_start_time = time.time()
    DAILY_LIMIT = 25
    
    for i, stock in enumerate(unique_stocks):
        # Skip if already processed
        if stock in processed_stocks:
            print(f"Skipping {stock} (already processed)")
            continue
        
        # Check daily limit
        if api_calls_made >= DAILY_LIMIT:
            print(f"\n=== DAILY LIMIT REACHED ===")
            print(f"Made {api_calls_made} API calls (limit: {DAILY_LIMIT})")
            print(f"Processed {len(processed_stocks)}/{total_stocks} stocks")
            print("Save your progress and run again tomorrow or let a teammate continue!")
            break
        
        info = stock_info[stock]
        print(f"\nProcessing stock {i+1}/{total_stocks}: {stock} (API call {api_calls_made + 1}/{DAILY_LIMIT})")
        print(f"  News articles: {info['news_count']}")
        print(f"  Date range: {info['min_date']} to {info['max_date']}")
        
        # Rate limiting
        if api_calls_made > 0 and api_calls_made % MAX_CALLS_PER_MINUTE == 0:
            elapsed = time.time() - session_start_time
            if elapsed < 60:
                sleep_time = 60 - elapsed
                print(f"  Rate limit: sleeping for {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
            session_start_time = time.time()
        
        # Get stock data for full range
        price_data = get_stock_data_full_range(
            stock, info['fetch_start'], info['fetch_end']
        )
        api_calls_made += 1
        
        # Cache the results
        stock_cache[stock] = {
            'date_range': [str(info['fetch_start']), str(info['fetch_end'])],
            'price_data': price_data,
            'news_count': info['news_count']
        }
        
        # Update progress
        processed_stocks.add(stock)
        progress['processed_stocks'] = list(processed_stocks)
        progress['completed_stocks'] = len(processed_stocks)
        
        # Save progress and cache every stock
        save_progress(progress)
        save_stock_cache(stock_cache)
        
        print(f"  Cached {len(price_data)} price points for {stock}")
        print(f"  Progress: {len(processed_stocks)}/{total_stocks} stocks completed")
        
        # Generate and save processed CSV every 5 stocks
        if len(processed_stocks) % 5 == 0:
            print(f"  Generating interim CSV file...")
            generate_processed_csv(news_df, stock_cache, interim=True)
        
        # Add delay between requests
        time.sleep(REQUEST_INTERVAL)
    
def generate_processed_csv(news_df, stock_cache, interim=False):
    """Generate the processed CSV with all sentiment entries and price data"""
    results = []
    
    for _, row in news_df.iterrows():
        stock = row['stock']
        article_date = pd.to_datetime(row['date']).date()
        
        # Get prices for this sentiment entry
        start_close, end_close = get_price_for_sentiment(stock_cache, stock, article_date)
        
        # Calculate additional metrics if we have both prices
        price_change = None
        price_change_pct = None
        if start_close is not None and end_close is not None:
            price_change = end_close - start_close
            price_change_pct = (price_change / start_close) * 100
        
        results.append({
            'stock': stock,
            'title': row['title'],
            'date': row['date'],
            'article_date': article_date,
            'start_close': start_close,
            'end_close': end_close,
            'price_change': price_change,
            'price_change_pct': price_change_pct
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)

    #Create threshold-based label
    def compute_label(start_close, end_close, threshold=0.01):
        """Return 1 (buy), -1 (sell), or 0 (hold) based on percentage return."""
        if start_close is None or end_close is None:
            return None  # Skip if missing data
        ret = (end_close - start_close) / start_close
        if ret > threshold:
            return 1
        elif ret < -threshold:
            return -1
        else:
            return 0
        
    label = compute_label(start_close, end_close)

    results_df['price_change_label'] = label
    
    
    # Save to main output path
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)
    
    if interim:
        print(f"  Interim CSV saved: {len(results)} entries")
    else:
        # Create final backup
        os.makedirs(BACKUP_DIR, exist_ok=True)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{BACKUP_DIR}/processed_training_final_{timestamp}.csv"
        results_df.to_csv(backup_path, index=False)
        print(f"Final backup saved to: {backup_path}")

     # Insert updated DataFrame into MongoDB
    try:
        records = results_df.to_dict('records')
        collection.insert_many(records)
        print(f"Inserted {len(records)} records into MongoDB collection '{collection.name}'")
    except Exception as e:
        print(f"Error inserting into MongoDB: {e}")
    
    return results_df
    

    
        
    # Generate final processed CSV with all available data
    print(f"\n=== GENERATING FINAL CSV ===")
    final_df = generate_processed_csv(news_df, stock_cache, interim=False)
    
    # Summary statistics
    successful_matches = final_df['start_close'].notna().sum()
    stocks_with_data = final_df[final_df['start_close'].notna()]['stock'].nunique()
    
    print(f"\n=== SESSION COMPLETE ===")
    print(f"API calls made this session: {api_calls_made}/{DAILY_LIMIT}")
    print(f"Stocks processed this session: {len(processed_stocks) - len(set(progress['processed_stocks']))}")
    print(f"Total stocks completed: {len(processed_stocks)}/{total_stocks}")
    print(f"Total sentiment entries: {len(final_df)}")
    print(f"Successful price matches: {successful_matches}/{len(final_df)} ({successful_matches/len(final_df)*100:.1f}%)")
    print(f"Stocks with price data: {stocks_with_data}/{total_stocks}")
    print(f"Results saved to: {OUTPUT_PATH}")
    
    # Show next steps
    if len(processed_stocks) < total_stocks:
        remaining = total_stocks - len(processed_stocks)
        print(f"\n=== NEXT STEPS ===")
        print(f"Still need to process {remaining} stocks")
        print("Run this script again tomorrow or let a teammate continue!")
    else:
        print("\n🎉 ALL STOCKS PROCESSED! 🎉")
        # Clean up progress file if completely done
        if os.path.exists(PROGRESS_PATH):
            os.remove(PROGRESS_PATH)
        print("Progress file cleaned up (all stocks processed)")

def check_status():
    """Check current processing status"""
    if not os.path.exists(CSV_DATA_PATH):
        print(f"Input file not found: {CSV_DATA_PATH}")
        return
    
    news_df = pd.read_csv(CSV_DATA_PATH)
    stock_info = analyze_stock_requirements(news_df)
    
    progress = load_progress()
    processed_stocks = set(progress['processed_stocks'])
    
    total_stocks = len(stock_info)
    total_articles = len(news_df)
    
    print(f"=== STATUS CHECK ===")
    print(f"Total news articles: {total_articles}")
    print(f"Unique stocks: {total_stocks}")
    print(f"Processed stocks: {len(processed_stocks)}")
    print(f"Remaining stocks: {total_stocks - len(processed_stocks)}")
    print(f"Progress: {len(processed_stocks)/total_stocks*100:.1f}%")
    
    if os.path.exists(STOCK_DATA_CACHE):
        cache = load_stock_cache()
        cached_stocks = len(cache)
        print(f"Stocks in cache: {cached_stocks}")
    
    if os.path.exists(OUTPUT_PATH):
        results_df = pd.read_csv(OUTPUT_PATH)
        successful = results_df['start_close'].notna().sum()
        print(f"Successful price matches in output: {successful}/{len(results_df)}")

def reset_progress():
    """Utility function to reset progress and start fresh"""
    files_to_remove = [PROGRESS_PATH, STOCK_DATA_CACHE, OUTPUT_PATH]
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed: {file_path}")
    
    print("All progress and cache files reset")

def list_remaining_stocks():
    """Show which stocks still need to be processed"""
    if not os.path.exists(CSV_DATA_PATH):
        print(f"Input file not found: {CSV_DATA_PATH}")
        return
    
    news_df = pd.read_csv(CSV_DATA_PATH)
    stock_info = analyze_stock_requirements(news_df)
    progress = load_progress()
    processed_stocks = set(progress['processed_stocks'])
    
    remaining = [stock for stock in stock_info.keys() if stock not in processed_stocks]
    
    print(f"=== REMAINING STOCKS ({len(remaining)}) ===")
    for stock in sorted(remaining)[:20]:  # Show first 20
        info = stock_info[stock]
        print(f"{stock}: {info['news_count']} articles ({info['min_date']} to {info['max_date']})")
    
    if len(remaining) > 20:
        print(f"... and {len(remaining) - 20} more")

if __name__ == "__main__":
    # Uncomment one of these based on what you want to do:
    
    # Normal processing (resume from where left off)
    main()
    
    # Check status without processing
    # check_status()
    
    # List remaining stocks to process
    # list_remaining_stocks()
    
    # Reset everything and start fresh (use with caution!)
    # reset_progress()