import tensorflow as tf
import tensorflow_probability as tfp

from tonic.tensorflow import models


FLOAT_EPSILON = 1e-8


class SquashedMultivariateNormalDiag:
    def __init__(self, loc, scale):
        self._distribution = tfp.distributions.MultivariateNormalDiag(
            loc, scale)

    def sample_with_log_prob(self, shape=()):
        samples = self._distribution.sample(shape)
        squashed_samples = tf.tanh(samples)
        log_probs = self._distribution.log_prob(samples)
        log_probs -= tf.reduce_sum(
            tf.math.log(1 - squashed_samples ** 2 + 1e-6), axis=-1)
        return squashed_samples, log_probs

    def sample(self, shape=()):
        samples = self._distribution.sample(shape)
        return tf.tanh(samples)

    def log_prob(self, samples):
        '''Required unsquashed samples cannot be accurately recovered.'''
        raise NotImplementedError(
            'Not implemented to avoid approximation errors. '
            'Use sample_with_log_prob directly.')

    def mode(self):
        return tf.tanh(self._distribution.mode())


class DetachedScaleGaussianPolicyHead(tf.keras.Model):
    def __init__(
        self, loc_activation='tanh', dense_loc_kwargs=None, log_scale_init=0.,
        scale_min=1e-4, scale_max=1.,
        distribution=tfp.distributions.MultivariateNormalDiag
    ):
        super().__init__()
        self.loc_activation = loc_activation
        if dense_loc_kwargs is None:
            dense_loc_kwargs = models.default_dense_kwargs()
        self.dense_loc_kwargs = dense_loc_kwargs
        self.log_scale_init = log_scale_init
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.distribution = distribution

    def initialize(self, action_size):
        self.loc_layer = tf.keras.layers.Dense(
            action_size, self.loc_activation, **self.dense_loc_kwargs)
        log_scale = [[self.log_scale_init] * action_size]
        self.log_scale = tf.Variable(log_scale, dtype=tf.float32)

    def call(self, inputs):
        loc = self.loc_layer(inputs)
        batch_size = tf.shape(inputs)[0]
        scale = tf.math.softplus(self.log_scale) + FLOAT_EPSILON
        scale = tf.clip_by_value(scale, self.scale_min, self.scale_max)
        scale = tf.tile(scale, (batch_size, 1))
        return self.distribution(loc, scale)


class GaussianPolicyHead(tf.keras.Model):
    def __init__(
        self, loc_activation='tanh', dense_loc_kwargs=None,
        scale_activation='softplus', scale_min=1e-4, scale_max=1,
        dense_scale_kwargs=None,
        distribution=tfp.distributions.MultivariateNormalDiag
    ):
        super().__init__()
        self.loc_activation = loc_activation
        if dense_loc_kwargs is None:
            dense_loc_kwargs = models.default_dense_kwargs()
        self.dense_loc_kwargs = dense_loc_kwargs
        self.scale_activation = scale_activation
        self.scale_min = scale_min
        self.scale_max = scale_max
        if dense_scale_kwargs is None:
            dense_scale_kwargs = models.default_dense_kwargs()
        self.dense_scale_kwargs = dense_scale_kwargs
        self.distribution = distribution

    def initialize(self, action_size):
        self.loc_layer = tf.keras.layers.Dense(
            action_size, self.loc_activation, **self.dense_loc_kwargs)
        self.scale_layer = tf.keras.layers.Dense(
            action_size, self.scale_activation, **self.dense_scale_kwargs)

    def call(self, inputs):
        loc = self.loc_layer(inputs)
        scale = self.scale_layer(inputs)
        scale = tf.clip_by_value(scale, self.scale_min, self.scale_max)
        return self.distribution(loc, scale)


class DeterministicPolicyHead(tf.keras.Model):
    def __init__(self, activation='tanh', dense_kwargs=None):
        super().__init__()
        self.activation = activation
        if dense_kwargs is None:
            dense_kwargs = models.default_dense_kwargs()
        self.dense_kwargs = dense_kwargs

    def initialize(self, action_size):
        self.action_layer = tf.keras.layers.Dense(
            action_size, self.activation, **self.dense_kwargs)

    def call(self, inputs):
        return self.action_layer(inputs)


class Actor(tf.keras.Model):
    def __init__(self, encoder, torso, head):
        super().__init__()
        self.encoder = encoder
        self.torso = torso
        self.head = head

    def initialize(
        self, observation_space, action_space, observation_normalizer=None
    ):
        self.encoder.initialize(observation_normalizer)
        self.head.initialize(action_space.shape[0])

    def call(self, *inputs):
        out = self.encoder(*inputs)
        out = self.torso(out)
        return self.head(out)
