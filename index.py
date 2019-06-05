# see https://www.analyticsvidhya.com/blog/2019/04/predicting-movie-genres-nlp-multi-label-classification/

from __future__ import absolute_import, division, print_function

import pathlib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# formatting text to binary data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# training model stuff
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier  # Binary Relevance
from sklearn.metrics import f1_score  # Performance metric

from tqdm import tqdm

# local helper funcs
from utils.funcs import infer_tags, remove_stopwords, clean_text


# raw data and list of genres
raw_data = 'data_raw.csv'
data_cols = ['imdb_is', 'title', 'year', 'score', 'genres', 'image']

# load csv file as raw dataset
dataset = pd.read_csv(
    raw_data,
    names=data_cols,
    na_values="?",
    comment='\t',
    sep=",",
    skipinitialspace=True
)

dataset = dataset.dropna()


# each record has a comma delimited string in it's Genres col
# replace each of these with an array of values
genres = []
for i in dataset['genres']:
    genres.append(i.split(','))

dataset['genres'] = genres

print(dataset.shape)


# convert text data (titles, genres, etc) into "features" (multidim arrays of binary data)
# so that we can feed them into a neural network for processing

# binarize the genres using MultiLabelBinarizer
#  --> becuase multiple genres in each row
multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(dataset['genres'])
y = multilabel_binarizer.transform(dataset['genres'])


# prepare vectorizer object to convert titles into binary data
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)

# split dataset into training and validation set
# note that the only data passed into these sets is the
# titles and genres (via the y value from multilabel_binarizer)
# for other data, update this.
xtrain, xval, ytrain, yval = train_test_split(
    dataset['title'], y, test_size=0.2, random_state=9)

# create TF-IDF features for training and validation dataset
xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xval_tfidf = tfidf_vectorizer.transform(xval)

# setup model to use sk-learn's OneVsRestClassifier
# solve the problem as a 'binary relevance' or 'one-vs-all' problem
lr = LogisticRegression()
clf = OneVsRestClassifier(lr)
clf.fit(xtrain_tfidf, ytrain)  # fit model on train data

#  ----- clean text in the content if needed -------
# dataset['title'] = dataset['title'].apply(lambda x: clean_text(x))
# dataset['title'] = dataset['title'].apply(lambda x: remove_stopwords(x))


# ----------- Predictions
# # make predictions for validation set
# predict probabilities
# y_pred = clf.predict(xval_tfidf)

y_pred_prob = clf.predict_proba(xval_tfidf)
t = 0.4  # threshold value
y_pred = (y_pred_prob >= t).astype(int)
performance = f1_score(yval, y_pred, average="micro")


def lists_have_shared_items(a, b):
    return not set(a).isdisjoint(b)

# run predictions on random data from the dataset


def test_model(count=5):
    total_correct = 0
    for i in range(count):
        k = xval.sample(1).index[0]
        title = dataset['title'][k]
        genre_actual = dataset['genres'][k]
        prediction = infer_tags(
            xval[k],
            clf,
            tfidf_vectorizer,
            multilabel_binarizer
        )

        genre_prediction = []
        for item in prediction:
            for sub in item:
                genre_prediction.append(sub)

        correct = lists_have_shared_items(genre_prediction, genre_actual)
        total_correct = total_correct + 1 if correct else total_correct

        print(f'--> {title} <--')
        print(f'Prediction: {genre_prediction} (raw: {prediction})')
        print(f'Actual: {genre_actual}')
        print(f'Is correct: {correct} | Total correct: {total_correct}\n')

    print(f'Total score: {total_correct}/{count} -- {total_correct/count}%')
    print(f'Model Performance: {performance}')


test_model(100)


# not_found = []
# key_error = []
# train_image = []
# for i in tqdm(range(dataset.shape[0])):
#     try:
#         fName = 'posters/' + dataset['image'][i]
#         img = image.load_img(fName, target_size=(400, 400, 3))
#         img = image.img_to_array(img)
#         img = img/255
#         train_image.append(img)
#     except FileNotFoundError:
#         not_found.append(fName)
#     except KeyError:
#         key_error.append(fName)

# print(f'not_found: {len(not_found)}')
# print(f'key_error: {len(key_error)}')

# x = np.array(train_image)
# x.shape()
