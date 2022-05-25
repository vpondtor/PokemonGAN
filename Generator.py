import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten

class Generator(tf.keras.Model):

    def __init__(self, z_dim, out_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.flatten = Flatten()
        self.dense1 = Dense(512, tf.nn.leaky_relu)
        self.dense2 = Dense(1024, tf.nn.leaky_relu)
        self.dense3 = Dense(out_dim, tf.nn.leaky_relu)

    def call(self, inputs):
        logits = self.flatten(inputs)
        logits = self.dense1(logits)
        logits = self.dense2(logits)
        logits = self.dense3(logits)
        return logits

    # This could be better
    def sample_z(self, n):
        return np.random.random_sample((n, self.z_dim))

    def loss(self, logits_fake):
        return tf.reduce_mean((tf.nn.sigmoid_cross_entropy_with_logits(tf.ones((len(logits_fake), 1)), logits_fake)))
