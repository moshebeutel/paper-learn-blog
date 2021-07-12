
import tensorflow as tf
from tensorflow.keras import layers

def create_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(96, 11, activation="relu")(inputs)
    x = layers.MaxPooling2D()
    x = layers.Conv2D(256, 5, activation="relu")(x)
    x = layers.MaxPooling2D(3)(x)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.Conv2D(16, 3, activation="relu")(x)
    encoder_output = layers.GlobalMaxPooling2D()(x)

    # A layer instance is callable on a tensor, and returns a tensor.
    x = layers.Dense(64, activation=’relu’)(inputs)
    x = layers.Dense(64, activation=’relu’)(x)
    predictions = layers.Dense(10, activation=’softmax’)(x)
    # Instantiate the model given inputs and outputs.
    model = tf.keras.Model(inputs=inputs, outputs=predictions)