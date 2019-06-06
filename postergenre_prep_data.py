
# see https://www.analyticsvidhya.com/blog/2019/04/predicting-movie-genres-nlp-multi-label-classification/
# and https://www.analyticsvidhya.com/blog/2019/04/build-first-multi-label-image-classification-model-python/

import os
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

from funcs.data import load_data, extract_genres, load_images
from funcs.model import build_model, train

#  Settings & Config
csv_path = os.path.abspath('data_test.csv')
poster_path = os.path.abspath('posters/')
cp_path = os.path.abspath('checkpoints/cp.ckpt')
columns = ['id', 'title', 'year', 'score', 'genres', 'image']
poster_w = 182
poster_h = 268
input_shape = (poster_h, poster_w, 3)

# Load csv data and images
dataset = load_data(csv_path, columns)
dataset = extract_genres(dataset)
dataset.to_pickle('prepped_dataframe.pkl')
