import tensorflow as tf
import pickle
from tensorflow.keras.optimizers import Adam
from Discriminator import Discriminator
from Generator import Generator
import matplotlib.pyplot as plt

LATENT_DIM = 256
INPUT_DIM = 784
EPOCHS = 5
BATCH_SIZE = 10000
LEARNING_RATE = 0.001
NUM_EXAMPLES = 5
CYCLE = 24

with open("data/mnist.pkl", "rb") as file:
    (x_train, _), (x_test, _) = pickle.load(file, encoding="latin")
batches = tf.data.Dataset.from_tensor_slices(x_train)
batches = batches.batch(BATCH_SIZE)

gen = Generator(LATENT_DIM, INPUT_DIM)
discrim = Discriminator()
optimizer = Adam(LEARNING_RATE)

def visualize(n):
    noise = gen.sample_z(n)
    images = gen(noise)
    plt.imshow(images)
    plt.show()

def optimize(tape, loss, model):
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def main():
    for i in range(EPOCHS):
        for batch in batches:
            with tf.GradientTape(persistent=True) as tape:
                fakes = gen(gen.sample_z(BATCH_SIZE))
                logits_fake = discrim(fakes)
                logits_real = discrim(batch)
                gen_loss = gen.loss(logits_fake)
                discrim_loss = discrim.loss(logits_real, logits_fake)
            optimize(tape, gen_loss, gen)
            optimize(tape, discrim_loss, discrim)

    visualize(1)
    

if __name__ == "__main__":
    main()