
# see https://www.analyticsvidhya.com/blog/2019/04/predicting-movie-genres-nlp-multi-label-classification/
# and https://www.analyticsvidhya.com/blog/2019/04/build-first-multi-label-image-classification-model-python/

import os
import numpy as np
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns

from funcs.data import load_data, extract_genres, load_images, check_genres
from funcs.model import build_model, train

#  Settings & Config
csv_path = os.path.abspath('raw_data/full.csv')
poster_path = os.path.abspath('posters/')
dataframe_pre_training_path = os.path.abspath("dataframes/pre_training.pkl")
columns = ['id', 'title', 'year', 'score', 'genres', 'image']


# Load csv data and images
dataset = load_data(csv_path, columns)
dataset, all_genres = extract_genres(dataset)

check_genres(all_genres)

dataset.to_pickle(dataframe_pre_training_path)
