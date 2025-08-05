import pandas as pd
from datetime import datetime, timedelta

csv_data = "analyst_ratings_processed.csv"

df = pd.read_csv("analyst_ratings_processed.csv")
print(df.columns)

# Read CSV file into pandas DataFrame and parse dates to remove time
news_dataframe = pd.read_csv(csv_data)
news_dataframe['date'] = news_dataframe['date'].str[:10]
news_dataframe['date'] = pd.to_datetime(news_dataframe['date'], format = '%Y-%m-%d', errors='coerce')

# Calculate suitable star and end dates for each article for stock price analysis
news_dataframe['start_date'] = news_dataframe['date'] - timedelta(days=7)
news_dataframe['end_date'] = news_dataframe['date'] + timedelta(days=7)

# Rearrange columns to create final formatted DataFrame
final_news_dataframe = news_dataframe[['stock', 'title', 'date', 'start_date', 'end_date']]

print(final_news_dataframe.head())