
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
poster_path = os.path.abspath('posters/')
checkpoint_path = os.path.abspath('checkpoints/cp.ckpt')
dataframe_pre_training_path = os.path.abspath("dataframes/pre_training.pkl")
dataframe_post_training_path = os.path.abspath("dataframes/post_training.pkl")

columns = ['id', 'title', 'year', 'score', 'genres', 'image']
poster_w = 182
poster_h = 268
input_shape = (poster_h, poster_w, 3)

# Load csv data and images
dataset = pd.read_pickle(dataframe_pre_training_path)
dataset, images = load_images(dataset, poster_path, poster_w, poster_h)

# remove unneccessary columns from the data
dataset = dataset.drop(columns=columns)

# save modified dataframe for the prediction script to use
dataset.to_pickle(dataframe_post_training_path)

# build and train the model
model = build_model(input_shape, dataset.shape[1])
train(model, dataset, images, checkpoint_path)
