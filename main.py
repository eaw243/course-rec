import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

# Filtered dataset
filtered = pd.read_csv('./stringcourse.csv',  sep=',',  header=0)
# Add an id column
filtered['ID'] = range(0, len(filtered))
filtered.columns = ['text', 'ID']
filtered['title'] = filtered['text'].apply(lambda x: str(
    x).split()[1] + str(x).split()[2] if len(str(x).split()) > 1 else None)
filtered.columns = ['text', 'ID', 'title']
tfidf_vectorizer = TfidfVectorizer()
tfidf_desc = tfidf_vectorizer.fit_transform((filtered['text']))

# tfidf
user_tfidf = tfidf_vectorizer.transform(
    filtered[filtered['title'] == 'cs4700'])
print(filtered['title'])
print(filtered[filtered['title'] == 'cs4700'])
cos_similarity_tfidf = map(
    lambda x: cosine_similarity(user_tfidf, x), tfidf_desc)
n_neighbors = 10
print(user_tfidf)
KNN = NearestNeighbors(n_neighbors=n_neighbors, p=2)
KNN.fit(tfidf_desc)
NNs = KNN.kneighbors(user_tfidf, return_distance=True)
print(NNs)
