
# see https://www.analyticsvidhya.com/blog/2019/04/predicting-movie-genres-nlp-multi-label-classification/
# and https://www.analyticsvidhya.com/blog/2019/04/build-first-multi-label-image-classification-model-python/

import os
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

# local helper funcs
# from utils.funcs import infer_tags, remove_stopwords, clean_text
from funcs.data import load_data, extract_genres, load_images
from funcs.model import build_model, train

# config/settings

# os.path.abspath("mydir/myfile.txt")
csv_path = os.path.abspath('data_test.csv')
poster_path = os.path.abspath('posters/')
columns = ['id', 'title', 'year', 'score', 'genres', 'image']
poster_w = 182
poster_h = 268


dataset = load_data(csv_path, columns)

print(f"raw data shape: {dataset.shape}")

dataset = extract_genres(dataset)
dataset, images = load_images(dataset, poster_path, poster_w, poster_h)
print(dataset.columns)
dataset = dataset.drop(
    columns=['id', 'genres', 'title', 'year', 'score', 'image'])
print(f"cleaned data shape: {dataset.shape}")
print(f"image count: {len(images)}")

X = np.array(images)


# print(dataset['title'][903])
# plt.imshow(X[903])
# plt.show()
# print(images[780])

# y = np.array(dataset.drop(
#     ['id', 'genres', 'title', 'year', 'score', 'image'], axis=1))
y = np.array(dataset)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.1)


input_shape = (poster_h, poster_w, 3)
output_shape = y.shape[1]

print('---')
print(dataset.shape)
print(y.shape)
print(output_shape)
print('----')

model = build_model(input_shape, dataset.shape[1])
print(model.summary())


#  compile and train the model
train_x = X_train
train_y = y_train

validation_data = (X_test, y_test)
# train(X_train, y_train, X_test, y_test, model)

train(model, dataset, images)

#  comile and train the model
# model.compile(optimizer='adam', loss='binary_crossentropy',
#               metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=10,
#           validation_data=(X_test, y_test), batch_size=64)
