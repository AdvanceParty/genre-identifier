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


# def train(X_train, y_train, xtest, ytest, model, num_epochs=10, batch_size=64):
#     model.compile(optimizer='adam', loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     model.fit(X_train, y_train, epochs=10,
#               validation_data=(xtest, ytest), batch_size=64)


def train(model, dataset, images, num_epochs=10, batch_size=64):

    X = np.array(images)
    # y = np.array(dataset.drop(['id', 'genres', 'title', 'year', 'score', 'image'], axis = 1))
    y = np.array(dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.1)

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=num_epochs,
              validation_data=(X_test, y_test), batch_size=batch_size)
