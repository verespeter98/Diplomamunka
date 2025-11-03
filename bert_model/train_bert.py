import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

import pandas as pd

data = pd.read_csv('bert_model/FinancialPhraseBank-v1.0/Sentences_50Agree.txt', delimiter=".@" ,encoding='latin-1', names=['text', 'sentiment'], engine="python")

sentiment_map = {"positive" : 0, "negative": 1, "neutral" : 2}
data['labels'] = data['sentiment'].str.strip().map(sentiment_map)

train_df, test_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data["labels"])

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = 3)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

tokenized_train.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)
tokenized_test.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": f1}

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=16,
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
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained("./test_saved_model")
tokenizer.save_pretrained("./test_saved_tokenizer")

print("Model trained and saved successfully!")