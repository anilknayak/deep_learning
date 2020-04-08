import tensorflow as tf
class Layer:
    def __init__(self):
        ''

    def leaky_relu(self, net):
        return  lambda net: tf.nn.leaky_relu(net, alpha=0.01)

    def dense(self, inputs, units, l2_weight):
        return tf.layers.dense(
            inputs, units, None,
            kernel_initializer=tf.keras.initializers.glorot_uniform,
            kernel_regularizer=tf.keras.regularizers.l2(l=l2_weight),
            bias_regularizer=tf.keras.regularizers.l2(l=l2_weight))

    def batch_norm(self, inputs, is_training):
        return tf.layers.batch_normalization(
            inputs, momentum=0.999, epsilon=0.001, training=is_training)

    def deconv2d(self, inputs, filters, kernel_size, stride, l2_weight):
        return tf.layers.conv2d_transpose(
            inputs, filters, [kernel_size, kernel_size], strides=[stride, stride],
            activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.keras.initializers.glorot_uniform,
            kernel_regularizer=tf.keras.regularizers.l2(l=l2_weight),
            bias_regularizer=tf.keras.regularizers.l2(l=l2_weight))

    def conv2d(self, inputs, filters, kernel_size, stride, l2_weight):
        return tf.layers.conv2d(
            inputs, filters, [kernel_size, kernel_size], strides=[stride, stride],
            activation=None, padding='same',
            kernel_initializer=tf.keras.initializers.glorot_uniform,
            kernel_regularizer=tf.keras.regularizers.l2(l=l2_weight),
            bias_regularizer=tf.keras.regularizers.l2(l=l2_weight))