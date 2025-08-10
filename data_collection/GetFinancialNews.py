import pandas as pd
from datetime import timedelta, datetime
from alpha_vantage.timeseries import TimeSeries
import time
import os

# Configuration
ALPHA_VANTAGE_API_KEY = ''  # Replace with your actual API key
MAX_CALLS_PER_MINUTE = 5  # Free tier limit
REQUEST_INTERVAL = 60 / MAX_CALLS_PER_MINUTE  # Seconds between requests

# Load your sentiment data
csv_data = "data_collection/Data/smaller_training_Set.csv"
news_df = pd.read_csv(csv_data)

# Initialize Alpha Vantage
ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')

# Create storage for results
results = []

# Process each row with rate limiting
for i, row in news_df.iterrows():
    stock = row['stock']
    article_date = pd.to_datetime(row['date']).date()
    
    # Calculate date windows
    start_date = article_date - timedelta(days=1)
    end_date = article_date + timedelta(days=1)  # Reduced from 3 to 2 days total
    
    try:
        # Get stock data with rate limiting
        if i % MAX_CALLS_PER_MINUTE == 0 and i > 0:
            time.sleep(REQUEST_INTERVAL)
        
        # Get daily data (compact returns ~100 days, full returns 20+ years)
        data, _ = ts.get_daily(symbol=stock, outputsize='compact')
        
        # Filter for our date range
        mask = (data.index.date >= start_date) & (data.index.date <= end_date)
        filtered_data = data[mask]
        
        # Extract closes if available
        if not filtered_data.empty:
            start_close = filtered_data.iloc[0]['4. close'] if len(filtered_data) > 0 else None
            end_close = filtered_data.iloc[-1]['4. close'] if len(filtered_data) > 1 else None
        else:
            start_close = end_close = None
        
        # Store results
        results.append({
            'stock': stock,
            'title': row['title'],
            'date': row['date'],
            'start_date': start_date,
            'end_date': end_date,
            'start_close': start_close,
            'end_close': end_close
        })
        
        print(f"Processed {i+1}/{len(news_df)}: {stock} - {article_date}")
    
    except Exception as e:
        print(f"Error processing {stock} on {article_date}: {str(e)}")
        results.append({
            'stock': stock,
            'title': row['title'],
            'date': row['date'],
            'start_date': start_date,
            'end_date': end_date,
            'start_close': None,
            'end_close': None
        })

# Create final dataframe
final_df = pd.DataFrame(results)

# Save processed data
output_path = 'data_collection/Data/processed_training.csv'
final_df.to_csv(output_path, index=False)

print(f"Processing complete. Results saved to {output_path}")
print(f"API calls made: {len(news_df)}")
print(f"Successful price retrievals: {final_df['start_close'].notna().sum()}")
