import pandas as pd
import csv
import torch
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from transformers import DistilBertModel, DistilBertTokenizer


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


def bert(filtered, n, xTe):
    top_n = []
    for test in xTe:
        tokenizer = DistilBertTokenizer.from_pretrained(
            'distilbert-base-uncased')

        texts = filtered['text'].tolist()
        encoded_inputs = tokenizer(
            texts, return_tensors='pt', padding=True, truncation=True, max_length=2000)
        encoded_test_input = tokenizer(
            test, return_tensors='pt', padding=True, truncation=True, max_length=128)

        model = DistilBertModel.from_pretrained("distilbert-base-uncased")

        with torch.no_grad():
            outputs = model(**encoded_inputs)
            embeddings = outputs.last_hidden_state

        with torch.no_grad():
            test_output = model(**encoded_test_input)
            test_embedding = test_output.last_hidden_state.mean(dim=1)

        similarity_matrix = cosine_similarity(
            test_embedding.detach().numpy(), embeddings.detach().numpy())

        n_nearest = similarity_matrix.argsort()[0][-n:][::-1]

        titles = filtered['title'].tolist()
        top_n += [titles[i] for i in n_nearest]
    return top_n


bert_top_n = []
knn_top_n = []

model = DistilBertModel.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
filtered = prep()
course_descriptions = filtered['text'].tolist()

description_embeddings = []
for description in course_descriptions:
    inputs = tokenizer(description, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(
        dim=1).squeeze().detach().numpy()
    description_embeddings.append(embeddings)


def get_top_matches(input_description, reference_descriptions, k=5):
    input_embedding = get_embedding(input_description)

    similarities = cosine_similarity(
        [input_embedding], reference_descriptions)[0]

    top_indices = similarities.argsort()[-k:][::-1]

    top_matches = [(course_descriptions[i], similarities[i])
                   for i in top_indices]
    return top_matches


def get_embedding(description):
    inputs = tokenizer(description, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(
        dim=1).squeeze().detach().numpy()
    return embedding


res = pd.read_csv('./Duke Roster.csv',  sep=',',  header=0)
res.columns = ['Duke Class Name', 'Duke Class Code',
               'Duke Description', 'Cornell Class Name', 'Cornell Class Code']

top_matches = []
titles = []
sims = []

for index, row in res.iterrows():
    top_5 = []
    similarities = []

    top_matches = get_top_matches(
        row['Duke Description'], description_embeddings)

    for m, similarity in top_matches:
        title = filtered.loc[filtered['text'] == m]['title'].values[0]
        top_5.append(title)
        similarities.append(similarity)

    titles.append(top_5)
    sims.append(similarities)

res['Top 5'] = titles
res['Similarities'] = sims
