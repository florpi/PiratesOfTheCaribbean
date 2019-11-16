
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

experiment = Experiment(api_key="VNQSdbR1pw33EkuHbUsGUSZWr",
                        project_name="piratesofthecaribbean", workspace="florpi")
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


def transfer_train(train_generator, validation_generator):


    IMAGE_SHAPE = X_train.shape[1:]
    N_SAMPLES=X_train.shape[0]
    BATCH_SIZE=100
    NUM_CLASSES = len(LABELS) 

    feature_extractor_url="https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/4"

    feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                            trainable=True,
                                            input_shape=IMAGE_SHAPE)

    feature_batch = feature_extractor_layer(X_train)

    # Freeze feature extrcator, train only new classifier layer
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

    #steps_per_epoch = np.ceil(N_SAMPLES/BATCH_SIZE)

    batch_stats_callback = CollectBatchStats()

    history = model.fit_generator(train_generator, epochs=2,
                                  batch_size=BATCH_SIZE,
                                  validation_data=validation_generator,
                                  callbacks = [batch_stats_callback])
    '''
    history = model.fit(X_train, y_train, epochs=10,
                                  callbacks = [batch_stats_callback])
    '''
    probabilities = model.predict_generator(validation_generator, 2000)
    
    visualize.plot_confusion_matrix(np.argmax(y_test, axis=-1), np.argmax(y_pred, axis=-1),
            classes=LABELS,
            normalize=True,
            experiment=experiment)

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
