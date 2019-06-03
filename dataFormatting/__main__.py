# takes raw csv data from kaggle
# removes/cleans junk rows
# normalises genre titles (uniform spelling and labels etc)
# inserts a new column for each genre found in the data
# and replaces the array of genre names in each record with a true/false value
# --> Saves result to cleaned_data.csv

from __future__ import absolute_import, division, print_function

import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import genre

fPath = os.path.dirname(os.path.abspath(__file__))
inFile = f'{fPath}/data_raw.csv'
outFile = f'{fPath}/data_cleaned.csv'

# location of raw data, and column info
column_names = ['IMDB', 'Title', 'Year', 'Score',
                'genres', 'Poster_URL']

# load csv file as raw dataset
raw_dataset = pd.read_csv(inFile, names=column_names,
                          na_values="?", comment='\t',
                          sep=",", skipinitialspace=True)

# create a copy to clean up
dataset = raw_dataset.copy()

# clean up inknown values
dataset = dataset.dropna()

# The data has each movie's genres in one column
# as a comma-separated list. We need to break this apart
# so that the training data has one column for every genre,
# and each movie in the training data has a value of 0 or 1 in each genre column
# eg shift from this:
#     "title:string, genres:string"
#  to this:
#     "title:string, action:bool, comedy:bool, drama:bool...etc"

# pop raw genres column out of the dataset.
# we don't need it in the final data, but we will use it to build the
# columns and values for each specific genre

# Movies can have more than one genre -- the data uses commas as delimiters
# So we need to split the raw data in each row to create an array of genres for each row
# ALSO: Some genres are listed in multiple ways in the raw data
#   eg: 'Film-Noir' and 'Noir'
# so we using the genre module's getId() function to map the known permutations
# into standardised genre labels.

gmap = []
genre_names = genre.get_names()  # get the name of each genre -- to use as columns
data_genres = dataset.pop('genres')

# normalise each genre item in the data so tp remove spelling variants
for record in data_genres:
    record_genres = record.split(',')
    normalised = list(map(lambda x: genre.normalise(x), record_genres))
    gmap.append(normalised)

#  ------
for genre_name in genre_names:
    values = []
    for row in gmap:
        values.append(genre_name in row)

    dataset[genre_name] = values

exported = dataset.to_csv(r''+outFile, index=None, header=False)
print(exported)
