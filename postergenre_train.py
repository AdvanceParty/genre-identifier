
# see https://www.analyticsvidhya.com/blog/2019/04/predicting-movie-genres-nlp-multi-label-classification/
# and https://www.analyticsvidhya.com/blog/2019/04/build-first-multi-label-image-classification-model-python/

import os
import pickle
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
dataset, images = load_images(dataset, poster_path, poster_w, poster_h)

# remove unneccessary columns from the data
dataset = dataset.drop(
    columns=['id', 'genres', 'title', 'year', 'score', 'image'])

# build and train the model
model = build_model(input_shape, dataset.shape[1])
train(model, dataset, images, cp_path)
