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
raw_data = 'data_raw.csv'
data_cols = ['IMDB', 'Title', 'Year', 'Score', 'Genres', 'Poster_URL']

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

# genres_str = dataset['Genres']
# each record has a comma delimited string in it's Genres col
# replace each of these with an array of values

genres = []
for i in dataset['Genres']:
    genres.append(list(i.split(',')))

dataset['Genres'] = genres

print(dataset.head())


#  Visualise list of all genres, and distribution
# all_genres = sum(genres_lists, [])
# all_genres = nltk.FreqDist(all_genres)
# all_genres_df = pd.DataFrame({'Genre': list(all_genres.keys()),
#                               'Count': list(all_genres.values())})
# g = all_genres_df.nlargest(columns="Count", n=50)
# plt.figure(figsize=(12, 15))
# ax = sns.barplot(data=g, x="Count", y="Genre")
# ax.set(ylabel='Count')
# plt.show()

# print(len(set(genre_total)))

# train_dataset = dataset.sample(frac=0.8, random_state=0)
# test_dataset = dataset.drop(train_dataset.index)

# train_stats = train_dataset.describe()
# train_stats.pop("Score")
# train_stats = train_stats.transpose()
# print(train_stats)
# plt.show()
