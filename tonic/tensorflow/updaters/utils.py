import tensorflow as tf


def tile(x, n):
    return tf.tile(x[None], [n] + [1] * len(x.shape))


def merge_first_two_dims(x):
    return tf.reshape(x, [x.shape[0] * x.shape[1]] + x.shape[2:])
