from comet_ml import Experiment

import logging

import argparse
import collections
from datetime import datetime
import os.path
import random
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow_hub as hub
from sklearn.utils import class_weight
from tqdm import tqdm_notebook as tqdm
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch

from pirates.visualization import visualize

LABELS = ["concrete_cement", "healthy_metal", "incomplete", "irregular_metal", "other"]


class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs["loss"])
        self.batch_acc.append(logs["acc"])
        self.model.reset_metrics()


def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    Credit:
        https://github.com/umbertogriffo/focal-loss-keras/blob/master/losses.py
    """

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed


class ConvertImage(layers.Layer):
    def call(self, inputs):
        return tf.image.convert_image_dtype(inputs, dtype=tf.float32, saturate=True)


class CaribbeanModel:
    """
    """

    def __init__(self, input_shape, directory):
        self.input_shape = input_shape
        self.num_classes = len(LABELS)
        self.directory = directory

    def build(self):
        """
        """
        feature_extractor_url = (
            "https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/4"
        )
        # Define keras model
        images_uint8 = layers.Input(shape=self.input_shape)
        images_float32 = ConvertImage()(images_uint8)
        features = hub.KerasLayer(feature_extractor_url, trainable=True)(images_float32)
        outputs = layers.Dense(self.num_classes, activation="softmax")(features)
        model = Model(inputs=images_uint8, outputs=outputs)
        # Print summary
        model.summary()
        # Loss layer
        loss = categorical_focal_loss(alpha=0.25, gamma=2.0)
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss, metrics=["acc"])
        return model

    def train_and_evaluate(self, train_gen, val_gen, epochs):
        """
        """
        experiment = Experiment(
            api_key="VNQSdbR1pw33EkuHbUsGUSZWr",
            project_name="piratesofthecaribbean",
            workspace="florpi",
        )
        model = self.build()
        with experiment.train():
            model.fit_generator(train_gen, epochs=epochs, validation_data=val_gen)
        model.save(os.path.join(self.directory, "pirates_cnn_{idx}.h5"))
        # Run validation
        with experiment.test():
            probabilities = []
            y_val_all = []
            for X_val, y_val in tqdm(val_gen, desc="valset"):
                y_val_all += y_val.tolist()
                probs = model.predict(X_val)
                probabilities += probs.tolist()

            visualize.plot_confusion_matrix(
                np.argmax(y_val_all, axis=-1),
                np.argmax(probabilities, axis=-1),
                classes=LABELS,
                normalize=True,
                experiment=experiment,
            )

            visualize.plot_confusion_matrix(
                np.argmax(y_val_all, axis=-1),
                np.argmax(probabilities, axis=-1),
                classes=LABELS,
                normalize=False,
                experiment=experiment,
            )
        return model


def transfer_train(
    train_generator,
    validation_generator,
    test_generator,
    train_all=False,
    n_epochs=10,
    directory="/content/drive/My Drive/pirates/cnn_model/",
):

    caribbean = CaribbeanModel(input_shape=(224, 224, 3), directory=directory)
    model = caribbean.train_and_evaluate(
        train_generator, validation_generator, n_epochs
    )
    # batch_stats_callback = CollectBatchStats()
    print("Finished training!")
    probabilities = []
    test_IDs = []
    for i in tqdm(range(len(test_generator)), desc="testset"):
        test_ID, X_test = next(test_generator)
        test_IDs.append(test_ID)
        probabilities.append(model.predict(X_test).tolist())

    probabilities = np.reshape(np.asarray(probabilities), [-1, 5])
    test_IDs = np.reshape(np.asarray(test_IDs), [-1, 1])
    submission = np.concatenate([test_IDs, probabilities], axis=1)
    submission = pd.DataFrame(
        submission,
        columns=[
            "id",
            "concrete_cement",
            "healthy_metal",
            "incomplete",
            "irregular_metal",
            "other",
        ],
    )
    # Save test csv file for submission
    submission.to_csv(directory + "submission.csv", index=False)


if __name__ == "__main__":
    RATIO = 1.33
    WIDTH = 300
    IMAGE_SHAPE = (int(WIDTH / RATIO), WIDTH, 3)
    IMAGE_SHAPE = (224, 224, 3)

    X_train = np.random.random(size=(100,) + IMAGE_SHAPE)
    y_train = tf.keras.utils.to_categorical(
        np.random.randint(2, size=100, dtype=np.int32)
    )
    X_test = np.random.random(size=(10,) + IMAGE_SHAPE)
    y_test = tf.keras.utils.to_categorical(
        np.random.randint(2, size=10, dtype=np.int32)
    )

    transfer_train(X_train, y_train, X_test, y_test)
