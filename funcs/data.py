
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from keras.preprocessing import image
import matplotlib.pyplot as plt
import nltk


def load_data(fpath, columns):
    dataset = pd.read_csv(
        fpath,
        names=columns,
        na_values="?",
        comment='\t',
        sep=",",
        skipinitialspace=True
    )

    return dataset.dropna()


def check_genres(all_genres):
    all_genres = nltk.FreqDist(all_genres)
    all_genres_df = pd.DataFrame(
        {'Genre': list(all_genres.keys()), 'Count': list(all_genres.values())})

    g = all_genres_df.nlargest(columns="Count", n=50)
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(data=g, x="Count", y="Genre")
    ax.set(ylabel='Count')
    plt.show()


# Remove comma-delimited string of genres in each record
# convert each records genres string into an array of genre names
# and then build an array of genre arrays : [dataset-records][record-genres]
def get_genres(dataframe):
    genres = []
    for i in dataframe['genres']:
        if isinstance(i, str):
            genres.append(i.split(','))
        else:
            genres.append(i)

    return sum(genres, [])


def extract_genres(dataframe):
    dataset = dataframe[0:].copy()

    # genres = []
    # for i in dataset['genres']:
    #     genres.append(i.split(','))

    # dataset['genres'] = genres
    # # array of each unique genre name found across all records
    # all_genres = sum(genres, [])
    all_genres = get_genres(dataset)

    # Insert one column for each genre into the dataset
    # And then for each record, add a value of 1 for genres the records belongs to
    # and 0 for genres it does not belong t0)
    for i in tqdm(range(dataset.shape[0])):
        for genre in set(all_genres):
            try:
                dataset.at[i, genre] = 1 if genre in dataset['genres'][i] else 0
            except Exception as e:
                #  errors here indicate problems with data
                #  try loooking at the rows immediately before
                #  and after row[i] oin the data source
                print('ERROR ' + str(i))
    return dataset, all_genres

# load the images for each record


def load_images(dataframe, path, w, h):
    dataset = dataframe[0:].copy()
    images = []
    for i in tqdm(range(dataset.shape[0])):
        try:
            fName = path + '/' + dataset['image'][i]
            img = image.load_img(fName, target_size=(h, w, 3))
            img = image.img_to_array(img)
            img = img/255
            images.append(img)
        except Exception as e:
            dataset.at[i, 'image'] = ''
    dataset = dataset[~(dataset['image'].str.len() == 0)]
    dataset = dataset.reset_index(drop=True)
    return dataset, images
