import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import BertTokenizer, BertModel, Trainer, TrainingArguments
import numpy as np
from datetime import datetime, timedelta

# 데이터셋 로드 및 전처리
dataset = load_dataset("nsmc")
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

def generate_time_series(num_samples):
    """시계열 데이터 생성"""
    base_time = datetime(2022, 1, 1)
    time_series_data = []
    for i in range(num_samples):
        random_offset = timedelta(minutes=np.random.randint(0, 1440))
        current_time = base_time + random_offset
        time_series_data.append([current_time.hour, current_time.minute])  # (hour, minute)
    return time_series_data

def preprocess_function(examples):
    """코멘트 데이터 전처리 및 시계열 데이터 추가"""
    tokenized_inputs = tokenizer(examples['document'], truncation=True, padding='max_length', max_length=128)
    time_series_data = generate_time_series(len(examples['document']))
    tokenized_inputs['time_series'] = torch.tensor(time_series_data, dtype=torch.float32).unsqueeze(1)  # (batch_size, seq_len=1, input_dim=2)
    tokenized_inputs['labels'] = torch.tensor(examples['label'], dtype=torch.long)
    return tokenized_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'time_series', 'labels'])

# 모델 정의
class BertWithTimeSeries(nn.Module):
    def __init__(self):
        super(BertWithTimeSeries, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.classifier = nn.Linear(self.bert.config.hidden_size + 2, 2)  
        self.loss_fn = nn.CrossEntropyLoss()  # Loss function for classification

    def forward(self, input_ids, attention_mask, time_series, labels=None):
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        pooled_output = outputs.pooler_output
       
        combined_input = torch.cat((pooled_output, time_series.squeeze(1)), dim=1)
        logits = self.classifier(combined_input)

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits  
        return logits  

model = BertWithTimeSeries()

# Trainer 설정
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],  # 이미 정의된 데이터셋 사용
    eval_dataset=tokenized_dataset['test'],  # 평가 데이터셋 설정
    data_collator=None, 
    compute_metrics=None,  
)

# 학습 시작
trainer.train()

# 평가
eval_results = trainer.evaluate()
print(f"Validation Results: {eval_results}")

# 모델 훈련 후 저장
torch.save(model, "trained_timestamp_model.pth")
print("Model saved successfully!")