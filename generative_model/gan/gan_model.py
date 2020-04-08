from deep_learning.generative_model.gan import generator as gen
from deep_learning.generative_model.gan import discriminator as dis
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

class GanModel:
    def __init__(self, batch_size):
        self.net = None
        self.generator = None
        self.discriminator = None
        self.batch_size = batch_size
        self.noise_dim = 100
        self.epoch = 2
        self.discriminator_loss = []
        self.generator_loss = []
        self.real_images = []

    def network(self):
        self.generator = gen.Generator(self.noise_dim)
        self.discriminator = dis.Discriminator()

        self.generator.network()
        self.generator.compile()

        self.discriminator.network()
        self.discriminator.compile()

        self.net = Sequential([self.generator.net, self.discriminator.net])
        self.net.compile(optimizer=Adam(lr=0.001, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, epoch, real_images, y_real, y_fake):
        num_images = len(real_images)
        noise = np.random.normal(0, 1, size=(num_images, self.noise_dim))

        # Train Discriminator
        fake_images = self.generator.generate(noise)
        self.discriminator.trainable(True)
        d_loss = self.discriminator.train(real_images, fake_images, y_real, y_fake)

        # Train Generator
        self.discriminator.trainable(False)
        g_loss = self.net.train_on_batch(noise, y_real)

        print("Training epoch {}: discriminator loss: {} generator loss: {}".format(epoch, d_loss, g_loss))

    def evaluate(self, epoch, real_images, y_real, y_fake):
        self.discriminator.trainable(False)
        num_images = len(real_images)
        noise = np.random.normal(0, 1, size=(num_images, self.noise_dim))
        fake_images = self.generator.generate(noise)
        discriminator_loss  = self.discriminator.test(real_images, fake_images, y_real, y_fake)
        generator_loss = self.net.test_on_batch(noise, y_real)
        self.generator_loss.append(generator_loss)
        self.discriminator_loss.append(discriminator_loss)
        print("Evaluation epoch {}: discriminator loss: {} generator loss: {}".format(epoch, discriminator_loss, generator_loss))

    def save_model(self):
        self.generator.save_model()

    def get_image(self, num_imgs=1):
        noise = np.random.normal(0, 1, size=(num_imgs, self.noise_dim))
        fake_images = self.generator.generate(noise)
        return fake_images

    def show_losses(self):
        self.discriminator_loss = np.array(self.discriminator_loss)
        self.generator_loss = np.array(self.generator_loss)
        print(self.generator_loss)
        print(self.discriminator_loss)
        plt.plot(self.discriminator_loss, label='Discriminator')
        plt.plot(self.generator_loss.T[0], label='Generator')
        plt.title("Validation Losses")
        plt.legend()
        plt.show()
        plt.savefig("loss.jpg")