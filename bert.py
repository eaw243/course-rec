import pandas as pd
import csv
import torch
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from transformers import DistilBertModel, DistilBertTokenizer

# Prepare a dataframe for the Cornell Data
def prep():
    # Read Cornell Data csv
    filtered = pd.read_csv('./stringcourse.csv',  sep=',',  header=0)
    # Add an id column with values 0,...,1953
    filtered['ID'] = range(0, len(filtered))
    # Add a column for text (description string)
    filtered.columns = ['text', 'ID']
    # Add the title (e.g. CS 4701)
    filtered['title'] = filtered['text'].apply(lambda x: str(
        x).split()[1] + str(x).split()[2] if len(str(x).split()) > 1 else None)
    filtered.columns = ['text', 'ID', 'title']
    return filtered

bert_top_n = []
knn_top_n = []

# Initialize model
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# Create Cornell dataframe
filtered = prep()
# Get a list of training descriptions
course_descriptions = filtered['text'].tolist()

description_embeddings = []
# Tokenize each description (we use the last layer as our embedding)
for description in course_descriptions:
    inputs = tokenizer(description, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(
        dim=1).squeeze().detach().numpy()
    description_embeddings.append(embeddings)

# Find the best matches for an input description
def get_top_matches(input_description, reference_descriptions, k=5):
    #  Embed our input_description
    input_embedding = get_embedding(input_description)
    # Create similarity matrix
    similarities = cosine_similarity(
        [input_embedding], reference_descriptions)[0]
    # Return most similar descriptions from train set
    top_indices = similarities.argsort()[:][::-1]

    top_matches = [(course_descriptions[i], similarities[i])
                   for i in top_indices]
    return top_matches

# Embed description using our BERT model
def get_embedding(description):
    inputs = tokenizer(description, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(
        dim=1).squeeze().detach().numpy()
    return embedding

# Upload Duke Roster Data
res = pd.read_csv('./Duke Roster.csv',  sep=',',  header=0)
res.columns = ['Duke Class Name', 'Duke Class Code',
               'Duke Description', 'Cornell Class Name', 'Cornell Class Code']

# Initialize list for our top matches, titles, and similarities
top_matches = []
titles = []
sims = []

# Going through each row of our Duke Data
for index, row in res.iterrows():
    top_5 = []
    similarities = []

    # Find the top matches (these are Cornell descriptions)
    top_matches = get_top_matches(
        row['Duke Description'], description_embeddings)
    
    # For each match, find the title of the Cornell Class with that description
    for m, similarity in top_matches:
        title = filtered.loc[filtered['text'] == m]['title'].values[0]
        top_5.append(title)
        similarities.append(similarity)

    # Append these titles and similarities
    titles.append(top_5)
    sims.append(similarities)

# Add these to our dataframe
res['Top 5'] = titles
res['Similarities'] = sims

# Datapath
csv_file_path = '/bert.csv'

# Convert Data frame to CSV
res.to_csv(csv_file_path, index=False)

# Data in bert.csv : this is identical to the knn results
# This returned the exact same as KNN 
