import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten

class Discriminator(tf.keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.flatten = Flatten()
        self.dense1 = Dense(1024, tf.nn.leaky_relu)
        self.dense2 = Dense(256, tf.nn.leaky_relu)
        self.dense3 = Dense(1, tf.nn.leaky_relu)

    # Return a vector of (0-1) predictions for all n inputs
    def call(self, inputs):
        logits = self.flatten(inputs)
        logits = self.dense1(logits)
        logits = self.dense2(logits)
        logits = self.dense3(logits)
        return logits

    # Return the (not log) loss on the data, it's not really loss it's more like avg correct
    def loss(self, logits_real, logits_fake):
        total = tf.reduce_mean((tf.nn.sigmoid_cross_entropy_with_logits(tf.ones((len(logits_real), 1)), logits_real)))
        total += tf.reduce_mean((tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros((len(logits_fake), 1)), logits_fake)))
        return total
