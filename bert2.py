import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd 


def get_labels(cornell_df, duke_df):
    res = []
    for row in duke_df['Cornell Class Code']:
      res += [cornell_df[cornell_df['title'] == row]['ID'].values]
    return res

def prep():
    # Filtered dataset
    filtered = pd.read_csv('./stringcourse.csv',  sep=',',  header=0)
    # Add an id column
    filtered['ID'] = range(0, len(filtered))
    filtered.columns = ['text', 'ID']
    filtered['title'] = filtered['text'].apply(lambda x: str(
        x).split()[1] + str(x).split()[2] if len(str(x).split()) > 1 else None)
    filtered.columns = ['text', 'ID', 'title']
    return filtered
    

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
num_classes = 1954
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
filtered = prep()
duke_df = pd.read_csv('Duke Roster.csv')
labels = get_labels(filtered, duke_df)
tokenized_train = tokenizer(filtered['text'].tolist(), return_tensors='pt', padding=True, truncation=True)
tokenized_val = tokenizer(duke_df['Duke Description'].tolist(), return_tensors='pt', padding=True, truncation=True)
train_labels = torch.tensor(range(0, len(filtered)))
val_labels = torch.tensor(labels)

train_dataset = TensorDataset(tokenized_train['input_ids'], tokenized_train['attention_mask'], train_labels)
val_dataset = TensorDataset(tokenized_val['input_ids'], tokenized_val['attention_mask'], val_labels)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataloader = DataLoader(val_dataset,  batch_size=1, shuffle=True)

num_epochs = 5
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

# Training
for epoch in range(num_epochs):
    for x_batch, y_batch in train_dataloader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
preds = []
x_test , y_test = val_dataset
with torch.no_grad():
  y_pred = model(x_test)
    
probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
ranking = torch.argsort(probabilities, descending=True)

data = {'course' : labels,'ranking' : ranking, 'probabilities': probabilities }

# Create a new DataFrame
res = pd.DataFrame(data)

csv_file_path = '/bert2.csv'
res.to_csv(csv_file_path, index=False)