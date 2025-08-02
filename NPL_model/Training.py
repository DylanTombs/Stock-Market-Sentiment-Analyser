# train_sentiment_model.py

import os
import torch
import numpy as np
import random
import argparse
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

def set_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def evaluate(model, dataloader, device):
    model.eval()
    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader:
        batch = tuple(b.to(device) for b in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2],
        }

        with torch.no_grad():
            outputs = model(**inputs)

        loss, logits = outputs.loss, outputs.logits
        loss_val_total += loss.item()

        predictions.append(logits.detach().cpu().numpy())
        true_vals.append(inputs["labels"].cpu().numpy())

    avg_loss = loss_val_total / len(dataloader)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return avg_loss, predictions, true_vals

def train_model(
    model_name,
    train_dataset,
    val_dataset,
    output_dir="./models",
    epochs=3,
    batch_size=32,
    lr=5e-5,
    seed=42
):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.to(device)

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=len(train_dataloader) * epochs
    )

    for epoch in range(1, epochs + 1):
        model.train()
        loss_train_total = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}", leave=False)
        for batch in progress_bar:
            model.zero_grad()
            batch = tuple(b.to(device) for b in batch)

            outputs = model(
                input_ids=batch[0],
                attention_mask=batch[1],
                labels=batch[2]
            )

            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            loss_train_total += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.3f}"})

        print(f"\nEpoch {epoch} finished.")
        print(f"Training loss: {loss_train_total / len(train_dataloader):.4f}")

        val_loss, val_preds, val_labels = evaluate(model, val_dataloader, device)
        val_f1 = f1_score(val_labels, np.argmax(val_preds, axis=1), average="weighted")
        val_acc = accuracy_score(val_labels, np.argmax(val_preds, axis=1))
        print(f"Validation loss: {val_loss:.4f}")
        print(f"F1 Score: {val_f1:.4f}")
        print(f"Accuracy: {val_acc:.4f}")

        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_dir, f"finbert_epoch_{epoch}.pt"))
        print(f"Model saved: finbert_epoch_{epoch}.pt\n")

    return model

def get_training_config():
    return {
        "train_data": "train_dataset.pt",
        "val_data": "val_dataset.pt",
        "model_name": "ProsusAI/finbert",
        "output_dir": "./models",
        "epochs": 3,
        "batch_size": 32
    }

if __name__ == "__main__":
    config = get_training_config()

    # Load datasets
    train = torch.load(config["train_data"])
    val = torch.load(config["val_data"])

    # Run training
    train_model(
        model_name=config["model_name"],
        train_dataset=train,
        val_dataset=val,
        output_dir=config["output_dir"],
        epochs=config["epochs"],
        batch_size=config["batch_size"]
    )


