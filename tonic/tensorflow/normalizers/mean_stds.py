import numpy as np
import tensorflow as tf


class MeanStd(tf.keras.Model):
    def __init__(self, mean=0, std=1, clip=None, shape=None):
        super().__init__(name='global_mean_std')
        self.mean = mean
        self.std = std
        self.clip = clip
        self.count = 0
        self.new_sum = 0
        self.new_sum_sq = 0
        self.new_count = 0
        self.eps = 1e-2
        if shape:
            self.initialize(shape)

    def initialize(self, shape):
        if isinstance(self.mean, (int, float)):
            self.mean = np.full(shape, self.mean, np.float32)
        else:
            self.mean = np.array(self.mean, np.float32)
        if isinstance(self.std, (int, float)):
            self.std = np.full(shape, self.std, np.float32)
        else:
            self.std = np.array(self.std, np.float32)
        self.mean_sq = np.square(self.mean)
        self._mean = tf.Variable(self.mean, trainable=False, name='mean')
        self._std = tf.Variable(self.std, trainable=False, name='std')

    def call(self, val):
        val = (val - self._mean) / self._std
        if self.clip is not None:
            val = tf.clip_by_value(val, -self.clip, self.clip)
        return val

    def unnormalize(self, val):
        return val * self._std + self._mean

    def record(self, values):
        for val in values:
            self.new_sum += val
            self.new_sum_sq += np.square(val)
            self.new_count += 1

    # Careful: do not use in @tf.function
    def update(self):
        new_count = self.count + self.new_count
        new_mean = self.new_sum / self.new_count
        new_mean_sq = self.new_sum_sq / self.new_count
        w_old = self.count / new_count
        w_new = self.new_count / new_count
        self.mean = w_old * self.mean + w_new * new_mean
        self.mean_sq = w_old * self.mean_sq + w_new * new_mean_sq
        self.std = self._compute_std(self.mean, self.mean_sq)
        self.count = new_count
        self.new_count = 0
        self.new_sum = 0
        self.new_sum_sq = 0
        self._update(self.mean.astype(np.float32), self.std.astype(np.float32))

    def _compute_std(self, mean, mean_sq):
        var = mean_sq - np.square(mean)
        var = np.maximum(var, 0)
        std = np.sqrt(var)
        std = np.maximum(std, self.eps)
        return std

    @tf.function
    def _update(self, mean, std):
        self._mean.assign(mean)
        self._std.assign(std)
