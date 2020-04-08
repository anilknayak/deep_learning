from keras.models import Sequential, Model
from keras.layers import Dense,  Activation, Flatten, BatchNormalization,  Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal
from keras.optimizers import Adam

class Discriminator:
    def __init__(self):
        self.net = None

    def network(self, leaky_alpha=0.2, init_stddev=0.02):
        self.net = Sequential()
        self.net.add(Conv2D(64, kernel_size=5, strides=2, padding='same',
                     kernel_initializer=RandomNormal(stddev=init_stddev),
                     input_shape=(32, 32, 3)))
        self.net.add(Activation(LeakyReLU(alpha=leaky_alpha)))
        self.net.add(Conv2D(128, kernel_size=5, strides=2, padding='same',
                     kernel_initializer=RandomNormal(stddev=init_stddev)))
        self.net.add(BatchNormalization())
        self.net.add(Activation(LeakyReLU(alpha=leaky_alpha)))
        self.net.add(Conv2D(256, kernel_size=5, strides=2, padding='same',
                     kernel_initializer=RandomNormal(stddev=init_stddev)))
        self.net.add(BatchNormalization())
        self.net.add(Activation(LeakyReLU(alpha=leaky_alpha)))
        self.net.add(Flatten())
        self.net.add(Dense(1, kernel_initializer=RandomNormal(stddev=init_stddev)))
        self.net.add(Activation('sigmoid'))
        print("Discriminator model")
        self.net.summary()

    def compile(self):
        self.net.compile(optimizer=Adam(lr=0.001, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, real_image_batch, fake_images, y_train_real, y_train_fake):
        real_image_loss = self.net.train_on_batch(real_image_batch, y_train_real)
        fake_image_loss = self.net.train_on_batch(fake_images, y_train_fake)
        return real_image_loss+fake_image_loss

    def test(self, real_image_batch, fake_images, y_train_real, y_train_fake):
        loss1 = self.net.test_on_batch(real_image_batch, y_train_real)
        loss2 = self.net.test_on_batch(fake_images, y_train_fake)
        return loss1[0]+loss2[0]

    def trainable(self, val=False):
        self.net.trainable = val

