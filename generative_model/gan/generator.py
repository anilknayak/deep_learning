from keras.models import Sequential, Model
from keras.layers import Dense, Activation, LeakyReLU, BatchNormalization, Conv2DTranspose, Reshape
from keras.initializers import RandomNormal
from keras.optimizers import Adam

class Generator:
    def __init__(self, noise_dim):
        self.net = None
        self.noise_dim = noise_dim

    def network(self, leaky_alpha=0.2, init_stddev=0.02):
        self.net = Sequential()
        self.net.add(Dense(4*4*512, input_shape=(self.noise_dim,),
                           kernel_initializer=RandomNormal(stddev=init_stddev)))
        self.net.add(Reshape(target_shape=(4, 4, 512)))
        self.net.add(BatchNormalization())
        self.net.add(Activation(LeakyReLU(alpha=leaky_alpha)))
        self.net.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding='same',
                                     kernel_initializer=RandomNormal(stddev=init_stddev)))
        self.net.add(BatchNormalization())
        self.net.add(Activation(LeakyReLU(alpha=leaky_alpha)))
        self.net.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same',
                                     kernel_initializer=RandomNormal(stddev=init_stddev)))
        self.net.add(BatchNormalization())
        self.net.add(Activation(LeakyReLU(alpha=leaky_alpha)))
        self.net.add(Conv2DTranspose(3, kernel_size=4, strides=2, padding='same',
                                     kernel_initializer=RandomNormal(stddev=init_stddev)))
        self.net.add(Activation('tanh'))
        print("Generator model")
        self.net.summary()

    def generate(self, noise):
        return self.net.predict_on_batch(noise)

    def compile(self):
        self.net.compile(optimizer=Adam(lr=0.001, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])

    def trainable(self, val=False):
        self.net.trainable = val

    def save_model(self):
        model_json = self.net.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.net.save_weights("model.h5")
