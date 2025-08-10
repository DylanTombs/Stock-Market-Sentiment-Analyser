import pandas as pd

df = pd.read_csv('data_collection/Data/analyst_ratings_processed.csv')


sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp500_df = pd.read_html(sp500_url)[0]  # first table is the tickers list
sp500_tickers = set(sp500_df['Symbol'].str.upper())


df['stock'] = df['stock'].str.upper().str.strip()
df = df[df['stock'].isin(sp500_tickers)]

df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
cutoff_date = pd.to_datetime('2010-04-08').tz_localize('UTC')
filtered_df = df[df['date'] > cutoff_date]

# Remove rows with bad date parsing
filtered_df = filtered_df.dropna(subset=['date'])

filtered_df.to_csv('data_collection/Data/smaller_training_set.csv', index=False)

print(f"Original size: {len(df)} rows | Filtered size: {len(filtered_df)} rows")
print(f"Unique tickers in filtered set: {filtered_df['stock'].nunique()}")
