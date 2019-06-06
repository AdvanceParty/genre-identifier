# see https://www.analyticsvidhya.com/blog/2019/04/predicting-movie-genres-nlp-multi-label-classification/

from __future__ import absolute_import, division, print_function

import pathlib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


import tensorflow as tf
from tensorflow import keras

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image


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

# config/settings
fpath = 'data_test.csv'
columns = ['id', 'title', 'year', 'score', 'genres', 'image']
poster_w = 182
poster_h = 268

# load csv file as raw dataset


def load_data(fpath, columns):
    dataset = pd.read_csv(
        fpath,
        names=columns,
        na_values="?",
        comment='\t',
        sep=",",
        skipinitialspace=True
    )

    return dataset.dropna()

# Remove comma-delimited string of genres in each record
# convert each records genres string into an array of genre names
# and then build an array of genre arrays : [dataset-records][record-genres]


def extract_genres(dataframe):
    genres = []
    dataset = dataframe[0:].copy()
    for i in dataset['genres']:
        genres.append(i.split(','))

    dataset['genres'] = genres

    # array of each unique genre name found across all records
    all_genres = sum(genres, [])

    # Insert one column for each genre into the dataset
    # And then for each record, add a value of 1 for genres the records belongs to
    # and 0 for genres it does not belong t0)
    for i in tqdm(range(dataset.shape[0])):
        for genre in set(all_genres):
            try:
                dataset.at[i, genre] = 1 if genre in dataset['genres'][i] else 0
            except Exception as e:
                #  errors here indicate problems with data
                #  try loooking at the rows immediately before
                #  and after row[i] oin the data source
                print('ERROR ' + str(i))
    return dataset

# load the images for each record


def load_images(dataframe):
    dataset = dataframe[0:].copy()
    images = []
    for i in tqdm(range(dataset.shape[0])):
        try:
            fName = 'posters/' + dataset['image'][i]
            img = image.load_img(fName, target_size=(poster_h, poster_w, 3))
            img = image.img_to_array(img)
            img = img/255
            images.append(img)
        except:
            dataset.at[i, 'image'] = ''
    dataset = dataset[~(dataset['image'].str.len() == 0)]
    dataset = dataset.reset_index(drop=True)
    return dataset, images


def build_model(input_shape, output_shape):
    # Create the model
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(5, 5),
                     activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation='sigmoid'))

    return model


def train(X_train, y_train, validation_data, model, num_epochs=10, batch_size=64):
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10,
              validation_data=validation_data, batch_size=64)


dataset = load_data(fpath, columns)
dataset = extract_genres(dataset)
dataset, images = load_images(dataset)


X = np.array(images)
y = np.array(dataset.drop(
    ['id', 'genres', 'title', 'year', 'score', 'image'], axis=1))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.1)


input_shape = (poster_h, poster_w, 3)
output_shape = y.shape[1]

model = build_model(input_shape, output_shape)
print(model.summary())


#  compile and train the model
# train_x = X_train
# train_y = y_train
# validation_data = (X-test, y_test)

# train(X_train, y_train, validation_data, model)
