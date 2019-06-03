from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import nltk

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# raw data and list of genres
raw_data = 'training_data.csv'
data_cols = ['IMDB', 'Title', 'Year', 'Score', 'Poster_URL']
genres = ['Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Doc', 'Drama', 'Family', 'Fantasy', 'Noir',
          'History', 'Horror', 'Musical', 'Mystery', 'Romance', 'SciFi', 'Thriller', 'War', 'Western', 'Short', 'News', 'Sport', 'Talkshow', 'Music']

col_names = data_cols + genres

# load csv file as raw dataset
dataset = pd.read_csv(raw_data, names=col_names,
                      na_values="?", comment='\t',
                      sep=",", skipinitialspace=True)


# print(dataset.head(20))

# train_dataset = dataset.sample(frac=0.8, random_state=0)
# test_dataset = dataset.drop(train_dataset.index)

# train_stats = train_dataset.describe()
# train_stats.pop("Score")
# train_stats = train_stats.transpose()
# print(train_stats)
# plt.show()
