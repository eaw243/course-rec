import pandas as pd
import csv
import torch
import random
# import torch.autocast
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertForSequenceClassification

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
        x).split()[1].upper() +" "+ str(x).split()[2] if len(str(x).split()) > 1 else None)
    filtered.columns = ['text', 'ID', 'title']
    return filtered

def get_batch(trainset, batch_size, total):
  res = []    
  indices = random.sample(range(0, total), batch_size)
  return [trainset[i] for i in indices]


cornell_df = prep()
texts = cornell_df['text'].tolist()
duke_df = pd.read_csv('Duke Roster.csv')
new_texts = duke_df['Duke Description'].tolist()
labels = get_labels(cornell_df, duke_df)
num_classes = 1954
learning_rate = 0.01

# Preprocessing
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=64)
# Model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes)
# Fine-tuning
num_epochs = 1
batch_size = 16
size = len(encoded_inputs['input_ids'][:1])
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss()
model.train()
for epoch in range(num_epochs):
    for b in range(size // batch_size):
      batch = get_batch(encoded_inputs['input_ids'][:1], batch_size, size)
      
      optimizer.zero_grad()
       # with autocast(device_type='cuda', dtype=torch.float16):
            # output = model(input)
            # loss = loss_fn(output, target)
      outputs = model(batch) #labels = label)
      loss = outputs.loss
      loss.backward()
      optimizer.step()
# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(**encoded_inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
# Inference
new_inputs = tokenizer(new_texts, padding=True, truncation=True, return_tensors='pt')
model.eval()
with torch.no_grad():
    outputs = model(**new_inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
all_preds = torch.argsort(outputs.logits, descending=True)

cornell_titles = cornell_df['title'].tolist()
duke_titles = duke_df['Cornell Class Code']
title_preds = []
for i in range(len(all_preds)):
  class_pred = []
  for j in range(len(all_preds[0])):
    class_pred += [cornell_titles[all_preds[i][j]]]
  title_preds += [class_pred]

duke_df['Ranked Results'] = title_preds
duke_df['Similarities'] = outputs
# Creating a DataFrame from the dictionary
csv_file_path = '/content/output.csv'

duke_df.to_csv(csv_file_path, index=False)

from google.colab import files
files.download(csv_file_path)