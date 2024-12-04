import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import BertTokenizer, BertModel, Trainer, TrainingArguments, EarlyStoppingCallback
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 데이터셋 로드 및 전처리
dataset = load_dataset("nsmc")
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

def generate_time_series(num_samples):
    """시계열 데이터 생성"""
    base_time = datetime(2022, 1, 1)
    time_series_data = []
    for _ in range(num_samples):
        random_offset = timedelta(minutes=np.random.randint(0, 1440))
        current_time = base_time + random_offset
        time_series_data.append([current_time.hour, current_time.minute])  # (hour, minute)
    return time_series_data

def preprocess_function(examples):
    """코멘트 데이터 전처리 및 시계열 데이터 추가"""
    tokenized_inputs = tokenizer(examples["document"], truncation=True, padding="max_length", max_length=128)
    time_series_data = generate_time_series(len(examples["document"]))
    tokenized_inputs["time_series"] = torch.tensor(time_series_data, dtype=torch.float32)  # 시간 데이터
    tokenized_inputs["labels"] = torch.tensor(examples["label"], dtype=torch.long)
    return tokenized_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "time_series", "labels"])

# Baseline 모델
baseline_model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)

# Time Series 모델
class BertWithTimeSeries(nn.Module):
    def __init__(self):
        super(BertWithTimeSeries, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.classifier = nn.Linear(self.bert.config.hidden_size + 2, 2)  # BERT hidden size + 2 for time series
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, time_series, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        combined_input = torch.cat((pooled_output, time_series), dim=1)
        logits = self.classifier(combined_input)
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

time_series_model = BertWithTimeSeries()
earlystop_model = BertWithTimeSeries()

# DataCollator
class CustomDataCollator:
    def __call__(self, features):
        input_ids = torch.stack([f["input_ids"] for f in features])
        attention_mask = torch.stack([f["attention_mask"] for f in features])
        time_series = torch.stack([f["time_series"] for f in features])
        labels = torch.tensor([f["labels"] for f in features])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "time_series": time_series,
            "labels": labels,
        }

data_collator = CustomDataCollator()

# 평가 함수
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Baseline 훈련
baseline_training_args = TrainingArguments(
    output_dir="./results_baseline",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
)

baseline_trainer = Trainer(
    model=baseline_model,
    args=baseline_training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("Training Baseline Model...")
baseline_trainer.train()
baseline_results = baseline_trainer.evaluate()

# Time Series 훈련
time_series_training_args = TrainingArguments(
    output_dir="./results_time_series",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
)

time_series_trainer = Trainer(
    model=time_series_model,
    args=time_series_training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("\nTraining Time Series Model...")
time_series_trainer.train()
time_series_results = time_series_trainer.evaluate()

# Time Series + EarlyStopping 훈련
earlystop_training_args = TrainingArguments(
    output_dir="./results_earlystop",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

earlystop_trainer = Trainer(
    model=earlystop_model,
    args=earlystop_training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    compute_metrics=compute_metrics
)

print("\nTraining Time Series Model with EarlyStopping...")
earlystop_trainer.train()
earlystop_results = earlystop_trainer.evaluate()

# 결과 출력
print("\nBaseline Results:", baseline_results)
print("\nTime Series Results:", time_series_results)
print("\nTime Series + EarlyStopping Results:", earlystop_results)

# 시각화
metrics = ["accuracy", "precision", "recall", "f1"]
baseline_scores = [baseline_results[f"eval_{metric}"] for metric in metrics]
time_series_scores = [time_series_results[f"eval_{metric}"] for metric in metrics]
earlystop_scores = [earlystop_results[f"eval_{metric}"] for metric in metrics]

x = np.arange(len(metrics))
width = 0.2

plt.figure(figsize=(12, 6))
plt.bar(x - width, baseline_scores, width, label="Baseline Model", alpha=0.7)
plt.bar(x, time_series_scores, width, label="Time Series Model", alpha=0.7)
plt.bar(x + width, earlystop_scores, width, label="Time Series + EarlyStopping", alpha=0.7)
plt.xticks(x, metrics)
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.grid(True)
plt.show()
