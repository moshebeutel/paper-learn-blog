
import tensorflow as tf
from tensorflow.keras import layers

inputs = tf.keras.Input(shape=(32,))
# A layer instance is callable on a tensor, and returns a tensor.
x = layers.Dense(64, activation=’relu’)(inputs)
x = layers.Dense(64, activation=’relu’)(x)
predictions = layers.Dense(10, activation=’softmax’)(x)
# Instantiate the model given inputs and outputs.
model = tf.keras.Model(inputs=inputs, outputs=predictions)