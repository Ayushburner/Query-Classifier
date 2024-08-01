import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Load dataset
data = pd.read_csv('query_dataset.csv')

# Preprocess dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['query'].tolist(), padding=True, truncation=True, max_length=128)

train_encodings = tokenize(data)

# Custom Dataset Class
class QueryDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create Dataset object for classification
train_dataset = QueryDataset(train_encodings, data['classification'].tolist())

# Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train model
trainer.train()

# Save the model
model.save_pretrained('./query_classification_model')
tokenizer.save_pretrained('./query_classification_model')

# Sub-classification data
from datasets import Dataset as HFDataset

sub_data = data[data['classification'] == 1]
sub_train_encodings = tokenize(sub_data)
sub_train_dataset = QueryDataset(sub_train_encodings, sub_data['sub_classification'].tolist())

# Convert PyTorch Dataset to Hugging Face Dataset
train_data_dict = {'input_ids': [], 'attention_mask': [], 'labels': []}
for item in sub_train_dataset:
    train_data_dict['input_ids'].append(item['input_ids'])
    train_data_dict['attention_mask'].append(item['attention_mask'])
    train_data_dict['labels'].append(item['labels'])

train_dataset_hf = HFDataset.from_dict(train_data_dict)
train_dataset_hf.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Load sub-classification model
model_sub = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Trainer for sub-classification
trainer_sub = Trainer(
    model=model_sub,
    args=training_args,
    train_dataset=train_dataset_hf,
)

# Train sub-classification model
trainer_sub.train()

# Save the model
model_sub.save_pretrained('./sub_classification_model')
tokenizer.save_pretrained('./sub_classification_model')


