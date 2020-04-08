import tensorflow_datasets as tfds
import tensorflow as tf

def input_fn(mode, params):
    assert 'batch_size' in params
    assert 'noise_dims' in params
    bs = params['batch_size']
    nd = params['noise_dims']
    split = 'train' if mode == tf.estimator.ModeKeys.TRAIN else 'test'
    shuffle = (mode == tf.estimator.ModeKeys.TRAIN)
    just_noise = (mode == tf.estimator.ModeKeys.PREDICT)
    noise_ds = (tf.data.Dataset.from_tensors(0).repeat()
                .map(lambda _: tf.random.normal([bs, nd])))
    if just_noise:
        return noise_ds

    def _preprocess(element):
        # Map [0, 255] to [-1, 1].
        images = (tf.cast(element['image'], tf.float32) - 127.5) / 127.5
        return images

    images_ds = (tfds.load('mnist', split=split) .map(_preprocess).cache().repeat())
    if shuffle:
        images_ds = images_ds.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
    images_ds = (images_ds.batch(bs, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))

    return tf.data.Dataset.zip((noise_ds, images_ds))