import tensorflow as tf


class ObservationEncoder(tf.keras.Model):
    def initialize(self, observation_normalizer=None):
        self.observation_normalizer = observation_normalizer

    def call(self, observations):
        if self.observation_normalizer:
            observations = self.observation_normalizer(observations)
        return observations


class ObservationActionEncoder(tf.keras.Model):
    def initialize(self, observation_normalizer=None):
        self.observation_normalizer = observation_normalizer

    def call(self, observations, actions):
        if self.observation_normalizer:
            observations = self.observation_normalizer(observations)
        return tf.concat([observations, actions], axis=-1)
