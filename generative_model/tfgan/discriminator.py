import tensorflow as tf
from deep_learning.generative_model.gan.common import Layer as layer
class Discriminator:
    def __init__(self):
        print("Creating discriminator")
        self.loss = None
        self.loss_func = None
        self.discriminator_lr = 0.0002

    def network(self, mode, img, weight_decay):
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        net = layer.conv2d(img, 64, 4, 2, weight_decay)
        net = layer.leaky_relu(net)

        net = layer.conv2d(net, 128, 4, 2, weight_decay)
        net = layer.leaky_relu(net)

        net = tf.layers.flatten(net)

        net = layer.dense(net, 1024, weight_decay)
        net = layer.batch_norm(net, is_training)
        net = layer.leaky_relu(net)

        net = layer.dense(net, 1, weight_decay)

        return net

    def optimizer(self):
        return tf.train.AdamOptimizer(self.discriminator_lr, 0.5)

    def loss(self):
        ''

    def compile(self):
        ''

    def train(self):
        ''

    def test(self):
        ''

    def evaluate(self):
        ''





