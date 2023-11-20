import pandas as pd
import csv
import torch
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from transformers import BertModel, BertTokenizer


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
                print(yTe[i])
                print([filtered['title'][NNs[1][i][0]], filtered['title'][NNs[1][i][1]], filtered['title']
                      [NNs[1][i][2]], filtered['title'][NNs[1][i][3]], filtered['title'][NNs[1][i][4]]])

        except:
            total -= 1
            pass

        total += 1

    return (str(score) + " out of " + str(total))


def cos_KNN(filtered, n_neighbors, xTe):

    vectorizer = TfidfVectorizer()
    trainset_tfidf = vectorizer.fit_transform((filtered['text']))

    query_tfidf = vectorizer.transform(xTe['text'])
    # before had p=2
    KNN = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    KNN.fit(trainset_tfidf)
    NNs = KNN.kneighbors(query_tfidf, return_distance=True)
    return NNs


def p_KNN(filtered, n_neighbors, xTe):

    vectorizer = TfidfVectorizer()
    trainset_tfidf = vectorizer.fit_transform((filtered['text']))

    query_tfidf = vectorizer.transform(xTe['text'])
    # before had p=2
    KNN = NearestNeighbors(n_neighbors=n_neighbors, p=2)
    KNN.fit(trainset_tfidf)
    NNs = KNN.kneighbors(query_tfidf, return_distance=True)
    return NNs


def bert(filtered, n, test):
    titles = filtered['title'].tolist()
    texts = filtered['text'].tolist()
    tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    model = BertModel.from_pretrained("bert-large-cased")
    texts = filtered['text'].tolist()
    encoded_inputs = tokenizer(
        texts, return_tensors='pt', padding=True, truncation=True, max_length=2000)
    outputs = model(**encoded_inputs)
    embeddings = outputs.last_hidden_state
    encoded_test_input = tokenizer(
        test, return_tensors='pt', padding=True, truncation=True, max_length=128)
    test_output = model(**encoded_test_input)
    test_embedding = test_output.last_hidden_state.mean(dim=1)
    # Calculate cosine similarity between the test point and all other data points
    similarity_matrix = cosine_similarity(
        test_embedding.detach().numpy(), embeddings.detach().numpy())
    # indices of the nearest descriptions to our description
    n_nearest = similarity_matrix.argsort()[0][-n:][::-1]
    top_n = [titles[i] for i in n_nearest]
    return top_n


if __name__ == "__main__":
    filtered = prep()
    xTe, yTe = validate()
    bert_top_n = []
    knn_top_n = []
    # BERT MODEL
    # https://huggingface.co/bert-large-cased
    # https://arxiv.org/abs/1810.04805


# print(NNs[1][0][0])
# print(filtered['title'][NNs[1][0][0]])
# print(filtered['title'][NNs[1][0][1]])
# print(filtered['title'][NNs[1][0][2]])
# print(filtered['title'][NNs[1][0][3]])
# print(filtered['title'][NNs[1][0][4]])
