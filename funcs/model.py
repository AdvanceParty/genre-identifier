import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


import tensorflow as tf
from tensorflow import keras

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image


# formatting text to binary data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# training model stuff
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier  # Binary Relevance
from sklearn.metrics import f1_score  # Performance metric


def build_model(input_shape, output_shape):
    # Create the model
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(5, 5),
                     activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation='sigmoid'))

    return model


def train(model, dataset, images, output='model.ckpt', num_epochs=10, batch_size=64):

    # checkpoint callback function for saving model during training
    cp_callback = tf.keras.callbacks.ModelCheckpoint(output,
                                                     save_weights_only=True,
                                                     verbose=1)
    np_images = np.array(images)
    np_dataset = np.array(dataset)

    images_training, images_testing, labels_training, labels_testing = train_test_split(
        np_images, np_dataset, random_state=42, test_size=0.1)

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(
        images_training,
        labels_training,
        epochs=num_epochs,
        validation_data=(images_testing, labels_testing),
        batch_size=batch_size,
        callbacks=[cp_callback]
    )
