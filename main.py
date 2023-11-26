import numpy as np
import pandas as pd
import csv
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier


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
# user_tfidf = tfidf_vectorizer.transform(
#     filtered[filtered['title'] == 'aas2043']['text'])
xTe, yTe = validate()
# dataframe = pd.DataFrame(['The structures and reactions of the compounds of carbon and the impact of selected organic compounds on society. Laboratory: techniques of separation, organic reactions and preparations, and systematic identification of compounds by their spectral and chemical properties.'], columns=['text'])
user_tfidf = tfidf_vectorizer.transform(xTe['text'])


cos_similarity_tfidf = map(
    lambda x: cosine_similarity(user_tfidf, x), tfidf_desc)
n_neighbors = 5

# print(user_tfidf)
KNN = NearestNeighbors(n_neighbors=n_neighbors, p=2)
KNN.fit(tfidf_desc)
NNs = KNN.kneighbors(user_tfidf, return_distance=True)
# print(NNs)

# print(NNs)
score = 0
total = 0
for i in range(len(yTe)):
    try:
        id = filtered[filtered['title'] == yTe[i].lower().replace(' ', '')]['ID'].iloc[0]

        found = False
        for course in NNs[1][i]:
            if course == int(id):
                score += 1
                found = True
                break

        if not found:
           print(yTe[i])
           print([filtered['title'][NNs[1][i][0]], filtered['title'][NNs[1][i][1]], filtered['title'][NNs[1][i][2]], filtered['title'][NNs[1][i][3]], filtered['title'][NNs[1][i][4]]])
           print()
           
    except:
       total -= 1
       pass
    
    total += 1
   

print(score)
print(total)
    
      
    


# print(NNs[1][0][0])
# print(filtered['title'][NNs[1][0][0]])
# print(filtered['title'][NNs[1][0][1]])
# print(filtered['title'][NNs[1][0][2]])
# print(filtered['title'][NNs[1][0][3]])
# print(filtered['title'][NNs[1][0][4]])



      




# I'm confused if this should be for one example or for all of them???
