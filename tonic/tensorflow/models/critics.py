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
