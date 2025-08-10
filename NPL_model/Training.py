
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from data_collection.DataSet import loadDataSet

def set_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def evaluate(model, data_loader, device):
    model.eval()
    preds, labels_all = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    acc = accuracy_score(labels_all, preds)
    f1 = f1_score(labels_all, preds, average="weighted")
    return acc, f1

def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    return total_loss / len(data_loader)

def train_model():

    set_seed(42)

    model_name = "ProsusAI/finbert"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 3 classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    train_loader, val_loader = loadDataSet(tokenizer)

    epochs = 3
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
        acc, f1 = evaluate(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}")

    model.save_pretrained("./finbert_finetuned")
    tokenizer.save_pretrained("./finbert_finetuned")

def get_training_config():
    return {
        "train_data": "train_dataset",
        "val_data": "val_dataset",
        "model_name": "trial",
        "output_dir": "./models",
        "epochs": 3,
        "batch_size": 32
    }

if __name__ == "__main__":
    
    train_model()

    print("Model fine-tuned and saved!")


