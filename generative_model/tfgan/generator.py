import tensorflow as tf
from deep_learning.generative_model.gan.common import Layer as layer
class Generator:
    def __init__(self):
        print("Creating generator")
        self.loss = None
        self.loss_func = None
        self.generator_lr = 0.001

    def network(self, noise, mode, weight_decay):
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        net = layer.dense(noise, 1024, weight_decay)
        net = layer.batch_norm(net, is_training)
        net = tf.nn.relu(net)

        net = layer.dense(net, 7 * 7 * 256, weight_decay)
        net = layer.batch_norm(net, is_training)
        net = tf.nn.relu(net)

        net = tf.reshape(net, [-1, 7, 7, 256])
        net = layer.deconv2d(net, 64, 4, 2, weight_decay)
        net = layer.deconv2d(net, 64, 4, 2, weight_decay)

        net = layer.conv2d(net, 1, 4, 1, 0.0)
        net = tf.tanh(net)
        return net

    def optimizer(self):
        gstep = tf.train.get_or_create_global_step()
        base_lr = self.generator_lr
        lr = tf.cond(gstep < 1000, lambda: base_lr, lambda: base_lr / 2.0)
        return tf.train.AdamOptimizer(lr, 0.5)

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


