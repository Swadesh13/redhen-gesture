import os
from typing import List
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from config import TRAIN_COLS
import time

NO_COLS = len(TRAIN_COLS)


def load_model(filepath: str):
    return tf.keras.models.load_model(filepath)


def gen_model(WINDOW_SIZE: int, MAX_PERSONS: int, CHANNELS: int = 1):
    inp = layers.Input(shape=(CHANNELS, WINDOW_SIZE, MAX_PERSONS, NO_COLS))
    x = layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding="same", return_sequences=True,
                          activation="relu", data_format="channels_first", go_backwards=True)(inp)
    x = layers.BatchNormalization()(x)
    # vgg-like Conv3D layers
    x = layers.Conv3D(filters=32, kernel_size=(3, 3, 3),
                      padding="same", activation="relu", data_format="channels_first",)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=32, kernel_size=(3, 3, 3),
                      padding="same", activation="relu", data_format="channels_first",)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=(2, 2, 2), strides=(1, 1, 1),
                         data_format="channels_first", padding="same")(x)
    x = layers.Conv3D(filters=64, kernel_size=(3, 3, 3),
                      padding="same", activation="relu", data_format="channels_first",)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=64, kernel_size=(3, 3, 3),
                      padding="same", activation="relu", data_format="channels_first",)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         data_format="channels_first")(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(units=128, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2())(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(units=128, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2())(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(units=1, activation="sigmoid",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)

    model = tf.keras.models.Model(inp, x)
    return model


def train_model(model, x_train, y_train, x_val, y_val, batch_size, epochs, output_dir):
    train_dir = os.path.join(output_dir, f"training_{int(time.time())}")
    os.makedirs(train_dir)
    if len(x_val):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, verbose=1)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=10, verbose=1, factor=0.5)
        save_model = tf.keras.callbacks.ModelCheckpoint(f"{train_dir}/best_model.h5",
                                                        monitor="val_precision", mode="max", verbose=1, save_best_only=True)
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=f"{train_dir}/logs/")

        model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=[early_stopping, reduce_lr, save_model, tensorboard],
        )
    else:
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=5, verbose=1)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", patience=10, verbose=1, factor=0.5)
        save_model = tf.keras.callbacks.ModelCheckpoint(f"{train_dir}/best_model.h5",
                                                        monitor="precision", mode="max", verbose=1, save_best_only=True)
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=f"{train_dir}/logs/")
        model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stopping, reduce_lr, save_model, tensorboard],
        )


def train(MODEL_PATH, x_train, y_train, x_val, y_val, learning_rate, batch_size, epochs, output_dir):
    window_size = x_train.shape[2]
    max_persons = x_train.shape[3]
    if MODEL_PATH:
        model = load_model(MODEL_PATH)
    else:
        model = gen_model(WINDOW_SIZE=window_size, MAX_PERSONS=max_persons)
    model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    assert model.input_shape[2:] == x_train.shape[2:], \
        f"model input shape ({model.input_shape[2:]}) and data shape ({x_train.shape[2:]}) do not match!"
    train_model(model, x_train, y_train, x_val, y_val,
                batch_size, epochs, output_dir)
