import os
import torch
import numpy as np
import pandas as pd

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

print(os.getcwd())

# --- 1️⃣ Load and preprocess dataset ---
data = pd.read_csv(
    "BertModel/FinancialPhraseBank-v1.0/Sentences_50Agree.txt",
    delimiter=".@",
    encoding="latin-1",
    names=["text", "sentiment"],
    engine="python",
)

# Strip whitespace
data["text"] = data["text"].str.strip()
data["sentiment"] = data["sentiment"].str.strip()

# Convert sentiment labels to integers
label2id = {label: idx for idx, label in enumerate(data["sentiment"].unique())}
id2label = {v: k for k, v in label2id.items()}
data["labels"] = data["sentiment"].map(label2id)

print("Label mapping:", label2id)

# --- 2️⃣ Split dataset ---
train_df, test_df = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data["labels"]
)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# --- 3️⃣ Tokenizer and model ---
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)

# --- 4️⃣ Tokenization function ---
def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=128
    )

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# --- 5️⃣ Format for PyTorch ---
tokenized_train.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)
tokenized_test.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)

# --- 6️⃣ Compute metrics ---
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": f1}

# --- 7️⃣ Training arguments ---
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=64,  # reduce from 64 if memory issues
    per_device_eval_batch_size=64,
    warmup_ratio=0.1,
    weight_decay=0.01,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
    fp16=torch.cuda.is_available(),
    report_to="none",
)

# --- 8️⃣ Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# --- 9️⃣ Train & save ---
trainer.train()

model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_tokenizer")

print("✅ Model trained and saved successfully!")
