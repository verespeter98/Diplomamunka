import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch


data = pd.read_csv('bert_model/FinancialPhraseBank-v1.0/Sentences_50Agree.txt', delimiter=".@", encoding='latin-1', names=['text', 'sentiment'], engine="python")

sentiment_map = {"positive" : 0, "negative": 1, "neutral" : 2}
data['labels'] = data['sentiment'].str.strip().map(sentiment_map)


train_df, test_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['labels'])

train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)

from datasets import Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)


tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
tokenized_test.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Define compute_metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {'accuracy': acc, 'macro_f1': f1}

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=int(0.21 * len(tokenized_train) / 64 * 10),
    weight_decay=0.01,
    learning_rate=2e-5,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    logging_dir='./logs',
    fp16=True if torch.cuda.is_available() else False,
    report_to='none',
    dataloader_pin_memory=False
)

# Custom MLflow callback to log metrics during training
class MLflowCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=state.epoch)

with mlflow.start_run() as run:
    mlflow.log_params(vars(training_args))
    mlflow.log_param("model_type", "bert-base-uncased")
    mlflow.log_param("num_labels", 3)
    mlflow.log_param("max_length", 512)
    mlflow.log_param("train_samples", len(train_df))
    mlflow.log_param("test_samples", len(test_df))
    mlflow.log_param("sentiment_map", str(sentiment_map))


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
        callbacks=[MLflowCallback()]
    )

    trainer.train()

    eval_results = trainer.evaluate()
    mlflow.log_metrics(eval_results)

    model.save_pretrained('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_tokenizer')
    mlflow.log_artifacts('./saved_model', artifact_path='model')
    mlflow.log_artifacts('./saved_tokenizer', artifact_path='tokenizer')

    mlflow.pytorch.log_model(model, "saved_model")
    mlflow.pytorch.log_model(tokenizer, "saved_tokenizer")

print("Model trained, saved, and logged with MLflow!")