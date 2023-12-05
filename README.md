# course-rec

Welcome to our course recommender GitHub!

This artificial intelligence project was created with the goal of helping
Cornell students explore the roster with the help of AI.

Our course recommendation system takes in a user's description of a class
and outputs 5 recommended courses. We used Duke class descriptions as our valiation
set by manually determining what Cornell class corresponds to a given Duke class.

In this repo, we have 5 models.

(1) knn.py contains a KNN classifier which returns the top 5 courses
(2) knn2.py contains a KNN classifier which returns a ranked order of recommended
courses (all 1954)

Both of these KNN models tokenize our inputs using TF-IDF. We then apply
cosine similarity as our distance metric to determine which descriptions
are nearest. Data from these runs are in knn.csv and knn2.csv respectively.

(3) bert.py. This uses BERT to tokenize our inputs, then applies cosine_similarity
and KNN to find best course recommendations. On our testing data, this model
outputted the same results as our KNN clasifier!

(4) bert2.py. This uses DistilBERT to tokenize our input. We then train a
DistilBERT model to complete our task. An original bug let to the training loop
not beginning. Data from this is in bert2 no training folder. Data after the bug was
fixed is in bert2 training folder. Data was reported with different hyperparameters in
in order to optimize our hyperparameters for SGD (epochs, learning rate, and batch size).

(4) bert3.py. This uses DistilBERT to tokenize our input. We then train a
DistilBERT model to complete our task. This utilized the Trainer class from
transformers. However, this code is still buggy. Output for the metrics
from this data is in the trainer model folder.
