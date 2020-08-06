import numpy as np
import tensorflow as tf


class Return(tf.keras.Model):
    def __init__(self, discount_factor):
        super().__init__(name='reward_normalizer')
        assert 0 <= discount_factor < 1
        self.coefficient = 1 / (1 - discount_factor)
        self.min_reward = np.float32(-1)
        self.max_reward = np.float32(1)
        self._low = tf.Variable(
            self.coefficient * self.min_reward, dtype=tf.float32,
            trainable=False, name='low')
        self._high = tf.Variable(
            self.coefficient * self.max_reward, dtype=np.float32,
            trainable=False, name='high')

    def call(self, val):
        val = tf.sigmoid(val)
        return self._low + val * (self._high - self._low)

    def record(self, values):
        for val in values:
            if val < self.min_reward:
                self.min_reward = np.float32(val)
            elif val > self.max_reward:
                self.max_reward = np.float32(val)

    # Careful: do not use in @tf.function
    def update(self):
        self._update(self.min_reward, self.max_reward)

    @tf.function
    def _update(self, min_reward, max_reward):
        self._low.assign(self.coefficient * min_reward)
        self._high.assign(self.coefficient * max_reward)
