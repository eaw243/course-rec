import pandas as pd
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def prep():
    filtered = pd.read_csv('./stringcourse.csv',  sep=',',  header=0)
    filtered['ID'] = range(0, len(filtered))
    filtered.columns = ['text', 'ID']
    filtered['title'] = filtered['text'].apply(lambda x: str(
        x).split()[1] + str(x).split()[2] if len(str(x).split()) > 1 else None)
    filtered.columns = ['text', 'ID', 'title']
    return filtered


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


def hit_rate(filtered, yTe, NNs):
    score = 0
    total = 0
    for i in range(len(yTe)):
        try:
            id = filtered[filtered['title'] ==
                          yTe[i].lower().replace(' ', '')]['ID'].iloc[0]

            found = False
            for course in NNs[1][i]:
                if course == int(id):
                    score += 1
                    found = True
                    break

            if not found:
                # print(yTe[i])
                print([filtered['title'][NNs[1][i][0]], filtered['title'][NNs[1][i][1]], filtered['title']
                      [NNs[1][i][2]], filtered['title'][NNs[1][i][3]], filtered['title'][NNs[1][i][4]]])

        except:
            total -= 1
            pass

        total += 1

    return (str(score) + " out of " + str(total))


res = pd.read_csv('./Duke Roster.csv',  sep=',',  header=0)
res.columns = ['Duke Class Name', 'Duke Class Code',
               'Duke Description', 'Cornell Class Name', 'Cornell Class Code']

top_matches = []
titles = []
sims = []
filtered = prep()
xTe, yTe = validate()
vectorizer = TfidfVectorizer()
trainset_tfidf = vectorizer.fit_transform((filtered['text']))


def knn_top_matches(filtered, k, test):
    input_vector = vectorizer.transform([test])
    titles = filtered['title']
    similarities = cosine_similarity(input_vector, trainset_tfidf).flatten()
    # top_indices = similarities.argsort()[-k:][::-1]
    top_indices = similarities.argsort()[:][::-1]
    top_matches = [(titles[i], similarities[i])
                   for i in top_indices]
    return top_matches


top_matches = []
titles = []
sims = []

for index, row in res.iterrows():
    top_5 = []
    similarities = []

    top_matches = knn_top_matches(filtered, 5,
                                  row['Duke Description'])

    for m, similarity in top_matches:
        top_5.append(m)
        similarities.append(similarity)

    titles.append(top_5)
    sims.append(similarities)

res['Top 5'] = titles
res['Similarities'] = sims


csv_file_path = './knn.csv'

res.to_csv(csv_file_path, index=False)
