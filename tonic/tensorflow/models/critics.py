import tensorflow as tf

from tonic.tensorflow import models


class ValueHead(tf.keras.Model):
    def __init__(self, dense_kwargs=None):
        super().__init__()
        if dense_kwargs is None:
            dense_kwargs = models.default_dense_kwargs()
        self.v_layer = tf.keras.layers.Dense(1, **dense_kwargs)

    def initialize(self, return_normalizer=None):
        self.return_normalizer = return_normalizer

    def call(self, inputs):
        out = self.v_layer(inputs)
        out = tf.squeeze(out, -1)
        if self.return_normalizer:
            out = self.return_normalizer(out)
        return out


class CategoricalWithSupport:
    def __init__(self, values, logits):
        self.values = values
        self.logits = logits
        self.probabilities = tf.nn.softmax(logits)

    def mean(self):
        return tf.reduce_sum(self.probabilities * self.values, axis=-1)

    def project(self, returns):
        vmin, vmax = self.values[0], self.values[-1]
        d_pos = tf.concat([self.values, vmin[None]], 0)[1:]
        d_pos = (d_pos - self.values)[None, :, None]
        d_neg = tf.concat([vmax[None], self.values], 0)[:-1]
        d_neg = (self.values - d_neg)[None, :, None]

        clipped_returns = tf.clip_by_value(returns, vmin, vmax)
        delta_values = clipped_returns[:, None] - self.values[None, :, None]
        delta_sign = tf.cast(delta_values >= 0, tf.float32)
        delta_hat = ((delta_sign * delta_values / d_pos) -
                     ((1 - delta_sign) * delta_values / d_neg))
        delta_clipped = tf.clip_by_value(1 - delta_hat, 0, 1)

        return tf.reduce_sum(delta_clipped * self.probabilities[:, None], 2)


class DistributionalValueHead(tf.keras.Model):
    def __init__(self, vmin, vmax, num_atoms, dense_kwargs=None):
        super().__init__()
        if dense_kwargs is None:
            dense_kwargs = models.default_dense_kwargs()
        self.distributional_layer = tf.keras.layers.Dense(
            num_atoms, **dense_kwargs)
        self.values = tf.cast(tf.linspace(vmin, vmax, num_atoms), tf.float32)

    def initialize(self, return_normalizer=None):
        if return_normalizer:
            raise ValueError(
                'Return normalizers cannot be used with distributional value'
                'heads.')

    def call(self, inputs):
        logits = self.distributional_layer(inputs)
        return CategoricalWithSupport(values=self.values, logits=logits)


class Critic(tf.keras.Model):
    def __init__(self, encoder, torso, head):
        super().__init__()
        self.encoder = encoder
        self.torso = torso
        self.head = head

    def initialize(
        self, observation_space, action_space, observation_normalizer=None,
        return_normalizer=None
    ):
        self.encoder.initialize(observation_normalizer)
        self.head.initialize(return_normalizer)

    def call(self, *inputs):
        out = self.encoder(*inputs)
        out = self.torso(out)
        return self.head(out)
