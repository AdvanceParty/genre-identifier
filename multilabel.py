# see https://www.analyticsvidhya.com/blog/2019/04/predicting-movie-genres-nlp-multi-label-classification/

import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# %matplotlib inline
# pd.set_option('display.max_colwidth', 300)
meta = pd.read_csv("cmu_corpus/movie.metadata.tsv", sep='\t', header=None)

# rename columns
meta.columns = ["movie_id", 1, "movie_name", 3, 4, 5, 6, 7, "genre"]
print(meta.head())

plots = []

with open("cmu_corpus/plot_summaries.txt", 'r') as f:
    reader = csv.reader(f, dialect='excel-tab')
    for row in tqdm(reader):
        plots.append(row)


# extract movie Ids and plot summaries
movie_id = []
plot = []

for i in tqdm(plots):
    movie_id.append(i[0])
    plot.append(i[1])

# create dataframe
movies = pd.DataFrame({'movie_id': movie_id, 'plot': plot})


# change datatype of 'movie_id' and merge meta with movies dataframe
meta['movie_id'] = meta['movie_id'].astype(str)
movies = pd.merge(
    movies, meta[['movie_id', 'movie_name', 'genre']], on='movie_id')

print(movies.head())
