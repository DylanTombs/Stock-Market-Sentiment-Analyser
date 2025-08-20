import pandas as pd
from pymongo import MongoClient

# Paths
OUTPUT_CSV_PATH = 'data_collection/Data/processed_training.csv'

# --- Step 1: Load the processed CSV ---
df = pd.read_csv(OUTPUT_CSV_PATH)

# --- Step 2: Create price_change_label based on price_change ---
def label_price_change(change):
    if pd.isna(change):
        return None
    if change > 0:
        return 1
    elif change < 0:
        return -1
    else:
        return 0

df['price_change_label'] = df['price_change'].apply(label_price_change)

# --- Step 3: Save updated CSV ---
df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"CSV file updated with new column 'price_change_label'.")

# Step 3: Connect to MongoDB Atlas
client = MongoClient("mongodb+srv://USERNAME:PASSWORD@database.pohgsdb.mongodb.net/")
db = client['dataset']          # Replace with your database name
collection = db['training_dataset']    # Replace with your collection name

# Step 4: Insert the updated dataframe into MongoDB
collection.drop() # removes existing data - used to override if thresholds for assigning labels change.
collection.insert_many(df.to_dict('records'))
