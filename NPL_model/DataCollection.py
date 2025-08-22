from pymongo import MongoClient
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.model_selection import train_test_split
import numpy as np

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def loadDataSet(tokenizer, batch_size=32):
    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:27017/')
    db = client['your_database_name']
    collection = db['your_collection_name']
    
    # Fetch data from MongoDB
    cursor = collection.find({})
    
    texts = []
    labels = []
    
    for doc in cursor:
        if 'title' in doc and doc['title']:
            texts.append(doc['title'])
            
            # Get the price_change_label (already numerical: -1, 0, 1)
            price_label = doc.get('price_change_label', 0)
            
            # Convert -1, 0, 1 to 0, 1, 2 for PyTorch (must be non-negative integers)
            label = price_label + 1  # This maps -1->0, 0->1, 1->2
            
            labels.append(label)
    
    client.close()
    
    print(f"Loaded {len(texts)} samples from MongoDB")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets and loaders
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader