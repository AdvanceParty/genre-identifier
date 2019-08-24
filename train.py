
# see https://www.analyticsvidhya.com/blog/2019/04/predicting-movie-genres-nlp-multi-label-classification/
# and https://www.analyticsvidhya.com/blog/2019/04/build-first-multi-label-image-classification-model-python/

import os
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from funcs.data import load_data, extract_genres, load_images, get_genres,  check_genres
from funcs.model import build_model, train

#  Settings & Config
poster_path = os.path.abspath('posters/')
checkpoint_path = os.path.abspath('checkpoints/cp.ckpt')
dataframe_pre_training_path = os.path.abspath("dataframes/pre_training.pkl")
dataframe_post_training_path = os.path.abspath("dataframes/post_training.pkl")

columns = ['id', 'title', 'year', 'score', 'genres', 'image']
poster_w = 182
poster_h = 268
input_shape = (poster_h, poster_w, 3)
genre_sample_size = 1000

# Load csv data
dataset = pd.read_pickle(dataframe_pre_training_path)

shorts = dataset[(dataset['Short'] > 0)].index
dataset.drop(shorts, inplace=True)

sampled = pd.DataFrame(columns=dataset.columns)
genres = get_genres(dataset)


for genre in set(genres):

    if genre != "Drama" and genre != "Comedy" and genre != "Thriller" and genre != "Crime" and genre != "Action" and genre != "Romance" and genre != "Horrow":
        genre_records = dataset.loc[dataset[genre] == 1]
        sample = genre_records.sample(
            min(genre_sample_size, len(genre_records)))
        print(f'Sample size for {genre}: {genre_sample_size}')
        print(f'Sampled {len(sample)} records from {genre}')
        sampled = sampled.append(sample, ignore_index=True)

hasDrama = sampled[(sampled['Drama'] > 0)].index
sampled.drop(hasDrama, inplace=True)

genre_records = dataset.loc[dataset['genres'] == 'Drama']
sample = genre_records.sample(
    min(genre_sample_size, len(genre_records)))
sampled = sampled.append(sample, ignore_index=True)

# print(f'Total records sampled: {len(sampled)}')
# print(sampled.head(20))
# print(sampled.columns)
# check_genres(get_genres(sampled))


# Load images
dataset, images = load_images(sampled, poster_path, poster_w, poster_h)

# remove unneccessary columns from the data
dataset = dataset.drop(columns=columns)

# save modified dataframe for the prediction script to use
dataset.to_pickle(dataframe_post_training_path)

# build and train the model
model = build_model(input_shape, dataset.shape[1])
train(model, dataset, images, checkpoint_path)
