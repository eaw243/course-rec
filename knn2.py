import numpy as np
import pandas as pd
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors

# Prepare our Cornell dataframe with column text (description), ID, title(e.g. CS4701)
def prep():
    filtered = pd.read_csv('./stringcourse.csv',  sep=',',  header=0)
    filtered['ID'] = range(0, len(filtered))
    filtered.columns = ['text', 'ID']
    filtered['title'] = filtered['text'].apply(lambda x: str(
        x).split()[1] + str(x).split()[2] if len(str(x).split()) > 1 else None)
    filtered.columns = ['text', 'ID', 'title']
    return filtered

# Separate our Duke data into descriptions and labels
def validate():
  xTe = pd.DataFrame([], columns=['text'])
  yTe = []
  with open('Duke Roster.csv', newline='') as csvfile:

    # read in course data
    csvreader = csv.reader(csvfile)
    header = next(csvreader)

    for row in csvreader:
      if row[0] != '':
        xTe.loc[len(xTe.index)] = [row[2]] 
        yTe.append(row[4])

    return xTe, yTe

# Res is our Duke dataframe
res = pd.read_csv('./Duke Roster.csv',  sep=',',  header=0)
res.columns = ['Duke Class Name', 'Duke Class Code',
               'Duke Description', 'Cornell Class Name', 'Cornell Class Code']
# Initialize necessary lists
top_matches = []
titles = []
sims = []
# Make Cornell df
filtered = prep()
# Separate Duke data into descriptions and labels
xTe, yTe = validate()
# Create a Tfidf vectorizer
vectorizer = TfidfVectorizer()
# Vectorize our training data
trainset_tfidf = vectorizer.fit_transform((filtered['text']))
n_neighbors = 1954

# Returns the top k matches based on cosine similarity
def knn_top_matches(filtered, test):
    input_vector = vectorizer.transform([test])

    # Using NearestNeighbors for k-NN search
    KNN = NearestNeighbors(n_neighbors, metric='euclidean')
    KNN.fit(trainset_tfidf)

    # Finding k-nearest neighbors
    _, top_indices = KNN.kneighbors(input_vector)

    titles = filtered['title'].values 

    similarities = euclidean_distances(input_vector, trainset_tfidf[top_indices.flatten()]).flatten()

    top_matches = [(titles[i], similarities[j]) for j, i in enumerate(top_indices.flatten())]
    top_matches.sort(key=lambda x: x[1], reverse=True)

    return top_matches

# Initialize lists
top_matches = []
titles = []
sims = []
# Gather top 5 similarities and top 5 titles of courses
for index, row in res.iterrows():
    top_results = []
    similarities = []

    top_matches = knn_top_matches(filtered, row['Duke Description'])

    # print(top_matches)

    for m, similarity in top_matches:
        top_results.append(m)
        similarities.append(similarity)

    titles.append(top_results)
    sims.append(similarities)

res['Ranked Results'] = titles
res['Similarities'] = sims

# Create csv file path
csv_file_path = './knn2.csv'
# Make this a csv
res.to_csv(csv_file_path, index=False)


