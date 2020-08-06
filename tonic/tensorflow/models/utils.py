import tensorflow as tf


def default_dense_kwargs():
    return dict(
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=1 / 3, mode='fan_in', distribution='uniform'),
        bias_initializer=tf.keras.initializers.VarianceScaling(
            scale=1 / 3, mode='fan_in', distribution='uniform'))


def mlp(units, activation, dense_kwargs=None):
    if dense_kwargs is None:
        dense_kwargs = default_dense_kwargs()
    layers = [tf.keras.layers.Dense(u, activation, **dense_kwargs)
              for u in units]
    return tf.keras.Sequential(layers)


MLP = mlp
