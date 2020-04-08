import tensorflow as tf
import tensorflow_gan as tfgan
from deep_learning.generative_model.tfgan.generator import Generator as generator
from deep_learning.generative_model.tfgan.discriminator import Discriminator as discriminator
from tensorflow_gan.examples.mnist import util as eval_util
from deep_learning.generative_model.tfgan import input
import numpy as np
import matplotlib.pyplot as plt

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

real_logits, fake_logits = [], []
real_mnist_scores, mnist_scores, frechet_distances = [], [], []
eval_batch = 2
eval_after_epoch = 2
epochs = 10
epoch = 0
num_epochs = []
while epoch < epochs:
    # Train GAN
    next_epoch = min(epoch+eval_after_epoch, epochs)
    gan_estimator.train(input.input_fn, max_steps=next_epoch)
    epoch = next_epoch
    num_epochs.append(epoch)

    # Evaluate GAN
    metrics = gan_estimator.evaluate(input.input_fn, steps=eval_batch)
    real_logits.append(metrics['real_data_logits'])
    fake_logits.append(metrics['gen_data_logits'])
    real_mnist_scores.append(metrics['real_mnist_score'])
    mnist_scores.append(metrics['mnist_score'])
    frechet_distances.append(metrics['frechet_distance'])
    print('Average discriminator output on Real: %.2f  Fake: %.2f' % (real_logits[-1], fake_logits[-1]))
    print('Inception Score: %.2f / %.2f  Frechet Distance: %.2f' % (mnist_scores[-1], real_mnist_scores[-1], frechet_distances[-1]))

    iterator = gan_estimator.predict(input.input_fn, hooks=[tf.train.StopAtStepHook(num_steps=3)])
    try:
        imgs = np.array([next(iterator) for _ in range(20)])
    except StopIteration:
        pass

    tiled = tfgan.eval.python_image_grid(imgs, grid_shape=(2, 10))
    plt.imshow(np.squeeze(tiled))

plt.title('MNIST Frechet distance per step')
plt.plot(num_epochs, frechet_distances)
plt.figure()
plt.title('MNIST Score per step')
plt.plot(num_epochs, mnist_scores)
plt.plot(num_epochs, real_mnist_scores)