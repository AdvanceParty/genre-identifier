import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image

from funcs.model import build_model

poster_w = 182
poster_h = 268
input_shape = (poster_h, poster_w, 3)

cp_path = os.path.abspath('checkpoints/cp.ckpt')
dataframe_path = os.path.abspath("model_dataframe.pkl")

dataset = pd.read_pickle(dataframe_path)

model = build_model(input_shape, dataset.shape[1])
model.load_weights(cp_path)


def predict(image_path):
    img = image.load_img(image_path, target_size=(poster_h, poster_w, 3))
    img = image.img_to_array(img)
    img = img/255

    classes = np.array(dataset.columns[0:])
    proba = model.predict(img.reshape(1, poster_h, poster_w, 3))
    top_3 = np.argsort(proba[0])[:-4:-1]

    print(f'-- Genres for {image_path} --')
    for i in range(3):
        print("{}".format(classes[top_3[i]]) +
              " ({:.3})".format(proba[0][top_3[i]]))
    # plt.imshow(img)
    # plt.show()


predict('prediction_tests/ep1.jpg')
predict('prediction_tests/got.jpg')
predict('prediction_tests/rugrats.jpg')
