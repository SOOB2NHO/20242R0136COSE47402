from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, PreTrainedTokenizer
from tqdm import tqdm
from torch.optim import AdamW 
from torch.nn import CrossEntropyLoss

# 데이터 로드 (NSMC 데이터셋)
dataset = load_dataset("e9t/nsmc")  # Naver Sentiment Movie Corpus

# 데이터프레임 변환
data = dataset['train'].to_pandas()
data = data.rename(columns={"document": "text", "label": "label"})

# 데이터 중복 제거
data = data.drop_duplicates(subset=['text'])

# 사용자 ID 생성 (1,000명의 사용자 임의 생성)
data['user_id'] = np.random.randint(1, 1001, size=len(data))

# 데이터 분할
train_data, val_data = train_test_split(data, test_size=0.2, stratify=data['label'], random_state=42)

# 데이터 크기 확인
print(f"Train Data Size: {len(train_data)}, Validation Data Size: {len(val_data)}")
print("Train Class Distribution:", train_data['label'].value_counts())
print("Validation Class Distribution:", val_data['label'].value_counts())

# 모델 준비
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
model = BertForSequenceClassification.from_pretrained("monologg/kobert", num_labels=2)

# PyTorch Dataset
class EmotionDataset(Dataset):
    def __init__(self, data, tokenizer: PreTrainedTokenizer, max_len=128):
        self.texts = data['text'].tolist()
        self.labels = data['label'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# DataLoader
train_dataset = EmotionDataset(train_data, tokenizer)
val_dataset = EmotionDataset(val_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Optimizer 및 손실 함수 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = CrossEntropyLoss()

# training 함수 정의
def train_model(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    # TQDM으로 진행률 표시 및 업데이트
    progress_bar = tqdm(dataloader, desc="Training", leave=True)
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        progress_bar.set_postfix({"Batch Loss": loss.item()})

    return total_loss / len(dataloader)

# training
for epoch in range(3):
    print(f"Epoch {epoch + 1} Starting...")
    train_loss = train_model(model, train_loader, optimizer, loss_fn, device)
    print(f"Epoch {epoch + 1} Completed. Average Loss: {train_loss:.4f}")
