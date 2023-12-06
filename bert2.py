import pandas as pd
import csv
import torch
import random
from datasets import Dataset
# import torch.autocast
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
# Creates a list of labels for the Duke testing data 
def get_labels(cornell_df, duke_df):
    res = []
    for row in duke_df['Cornell Class Code']:
      res += [cornell_df[cornell_df['title'] == row]['ID'].values]
    return res

# Creates a dataframe for Cornell classes including a column for ID (0,..,1953),
# text (the course description) and title (e.g. CS4701)
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

# Creates the Cornell Dataframe
cornell_df = prep()
# Gets a list of input train descriptions
texts = cornell_df['text'].tolist()
# Creates the Duke Dataframe
duke_df = pd.read_csv('Duke Roster.csv')
# Gets a list of input test descriptions
new_texts = duke_df['Duke Description'].tolist()
# Creates a list of labels for the Duke Classes (y_test)
labels = get_labels(cornell_df, duke_df)
# We have 1954 different courses in our train set which we 
# direct users to 
num_classes = 1954
# Instantiate a tokenizer object
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# Tokenize our training set
tokenized_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=64)
# Make our labels accessible attached to the dataset
dataset = list(zip(tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'], labels))


# Fine-tuning
def train(num_epochs, batch_size, learning_rate):
  # Construct a model
  model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes)
  # Create a dataloader that shuffles data into our batches
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  # We will use SGD to make this more efficient, take momentum 0.9 by convention
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)
  # We will use a Cross Entropy Loss Function
  loss_fn = torch.nn.CrossEntropyLoss()
  # Begin Training
  model.train()
  # for each epoch
  for epoch in range(num_epochs):
      # Get a dataloader that shuffles our data
      dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
      # For each batch in the dataloader
      for batch in dataloader:
        # Get our inputs
        input_ids, attention_mask,batch_labels = batch
        # Make the batch label 1d
        batch_labels = batch_labels.view(-1)
        # Zero our gradient
        optimizer.zero_grad()
         # with autocast(device_type='cuda', dtype=torch.float16):
              # output = model(input)
              # loss = loss_fn(output, target)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask) #labels = label)
        print(outputs.logits)
        print(batch_labels)
        print(outputs.logits.shape)
        print(batch_labels.shape)
        # We are minimizing the cross entropy
        loss = torch.nn.functional.cross_entropy(outputs.logits, torch.tensor(batch_labels))
        print(loss)
        # Backpropagation
        loss.backward()
        # Next step of SGD
        optimizer.step()
  # Inference
  # Tokenize our Duke Descriptions
  new_inputs = tokenizer(new_texts, padding=True, truncation=True, return_tensors='pt')
  # Begin evaluation
  model.eval()
  with torch.no_grad():
      outputs = model(**new_inputs)
  # Rank the predictions
  all_preds = torch.argsort(outputs.logits, descending=True)
  # List of all Cornell class codes
  cornell_titles = cornell_df['title'].tolist()
  # List of title predictions
  title_preds = []
  # For each Duke class
  for i in range(len(all_preds)):
    class_pred = []
    #For each Cornell class
    for j in range(len(all_preds[0])):
      # Add the class title
      class_pred += [cornell_titles[all_preds[i][j]]]
    # This is our list of Cornell classes for each Duke class
    title_preds += [class_pred]
  # Add results to our dataframe
  duke_df['Ranked Results'] = title_preds
  # Add our output
  duke_df['Similarities'] = outputs
  # Creating a DataFrame from the dictionary
  return duke_df

# Hyperparameter optimization
# Grid search on batches, epochs, and learning rates
batches = [16, 32, 64, 128]
epochs = [1,2,3,4]
lrs = [0.1, 0.01, 0.001, 0.0001]
for batch in batches:
  for epoch in epochs:
    for lr in lrs:
      # unique instance of batch, epoch, lr
      # train our model
      duke_df=train(epoch, batch, lr)
      from google.colab import files
      print(batch)
      # Create a csv path specifies the parameters
      csv_file_path = '/content/bert2-' + str(batch)  + ','+ str(epoch) +',' + str(lr) + '.csv'
      # Make this a csv file
      duke_df.to_csv(csv_file_path, index=False)
      # Download csv
      files.download(csv_file_path)
      
## This is code for a colab notebook
