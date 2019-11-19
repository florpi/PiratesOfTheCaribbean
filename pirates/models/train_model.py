
from comet_ml import Experiment

import logging

import argparse
import collections
from datetime import datetime
import os.path
import random
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import keras
import tensorflow_hub as hub

from pirates.visualization import visualize

LABELS = [
    "healthy_metal", 
    "irregular_metal", 
    "concrete_cement", 
    "incomplete",
    "other",
]

class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []

  def on_train_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])
    self.model.reset_metrics()


def transfer_train(train_generator, validation_generator, 
                    test_generator,
                    train_all=False, BATCH_SIZE=100,
                    N_EPOCHS=10):

    experiment = Experiment(api_key="VNQSdbR1pw33EkuHbUsGUSZWr",
                        project_name="piratesofthecaribbean", workspace="florpi")

    IMAGE_SHAPE = train_generator._img_shape + (3,)
    N_SAMPLES=train_generator._num_examples
    NUM_CLASSES = len(LABELS) 

    feature_extractor_url="https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/4"

    feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                            trainable=True,
                                            input_shape=IMAGE_SHAPE)

    # Freeze feature extrcator, train only new classifier layer
    if train_all == False:
        feature_extractor_layer.trainable = False

    model = tf.keras.Sequential([
      feature_extractor_layer,
      layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.summary()


    model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss='binary_crossentropy',
      metrics=['acc'])


    batch_stats_callback = CollectBatchStats()

    history = model.fit_generator(train_generator, epochs=N_EPOCHS,
                                  validation_data=validation_generator)
    #                              callbacks = [batch_stats_callback])

    probabilities = []
    y_val_all = []
    for  X_val, y_val in validation_generator:
        y_val_all += y_val.tolist()
        probabilities += model.predict(X_val).tolist()
    
    visualize.plot_confusion_matrix(np.argmax(y_val, axis=-1), 
            np.argmax(probabilities, axis=-1),
            classes=LABELS,
            normalize=True,
            experiment=experiment)

    visualize.plot_confusion_matrix(np.argmax(y_val, axis=-1), 
            np.argmax(probabilities, axis=-1),
            classes=LABELS,
            normalize=False,
            experiment=experiment)

    probabilities = []
    test_IDs = []
    for  test_ID, X_test, y_test in test_generator:
        test_IDs.append(test_ID)
        probabilities += model.predict(X_test).tolist()
    
    print(probabilities)


if __name__=='__main__':
    RATIO=1.33
    WIDTH=300
    IMAGE_SHAPE=(int(WIDTH/RATIO), WIDTH, 3)
    IMAGE_SHAPE = (224, 224, 3)

    X_train = np.random.random(size=(100,)+IMAGE_SHAPE)
    y_train = keras.utils.to_categorical(np.random.randint(2, size=100, dtype=np.int32))
    X_test = np.random.random(size=(10,)+IMAGE_SHAPE)
    y_test = keras.utils.to_categorical(np.random.randint(2, size=10, dtype=np.int32))

    transfer_train(X_train, y_train, X_test, y_test)
