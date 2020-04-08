import tensorflow as tf
import tensorflow_gan as tfgan
from deep_learning.generative_model.gan.generator import Generator as generator
from deep_learning.generative_model.gan.discriminator import Discriminator as discriminator
from tensorflow_gan.examples.mnist import util as eval_util
import os

print(tf.__version__)

train_batch_size = 32
noise_dimensions = 64

def get_eval_metric_ops_fn(gan_model):
    real_data_logits = tf.reduce_mean(gan_model.discriminator_real_outputs)
    gen_data_logits = tf.reduce_mean(gan_model.discriminator_gen_outputs)
    real_mnist_score = eval_util.mnist_score(gan_model.real_data)
    generated_mnist_score = eval_util.mnist_score(gan_model.generated_data)
    frechet_distance = eval_util.mnist_frechet_distance(
        gan_model.real_data, gan_model.generated_data)
    return {
        'real_data_logits': tf.metrics.mean(real_data_logits),
        'gen_data_logits': tf.metrics.mean(gen_data_logits),
        'real_mnist_score': tf.metrics.mean(real_mnist_score),
        'mnist_score': tf.metrics.mean(generated_mnist_score),
        'frechet_distance': tf.metrics.mean(frechet_distance),
    }

gan_estimator = tfgan.estimator.GANEstimator(
    generator_fn=generator.network,
    discriminator_fn=discriminator.network,
    generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
    params={'batch_size': train_batch_size, 'noise_dims': noise_dimensions},
    generator_optimizer=generator.optimizer,
    discriminator_optimizer=discriminator.optimizer,
    get_eval_metric_ops_fn=get_eval_metric_ops_fn)

