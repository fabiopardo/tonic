import numpy as np
import tensorflow as tf

from tonic.tensorflow import updaters


class StochasticPolicyGradient:
    def __init__(self, optimizer=None, entropy_coeff=0, gradient_clip=0):
        self.optimizer = optimizer or tf.keras.optimizers.Adam(
            lr=3e-4, epsilon=1e-8)
        self.entropy_coeff = entropy_coeff
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = self.model.actor.trainable_variables

    @tf.function
    def __call__(self, observations, actions, advantages, log_probs):
        if tf.math.count_nonzero(advantages) == 0:
            loss, kl = 0., 0.
            distributions = self.model.actor(observations)
            entropy = tf.reduce_mean(distributions.entropy())
            std = tf.reduce_mean(distributions.stddev())

        else:
            with tf.GradientTape() as tape:
                distributions = self.model.actor(observations)
                new_log_probs = distributions.log_prob(actions)
                loss = -tf.reduce_mean(advantages * new_log_probs)
                entropy = tf.reduce_mean(distributions.entropy())
                if self.entropy_coeff != 0:
                    loss -= self.entropy_coeff * entropy

            gradients = tape.gradient(loss, self.variables)
            if self.gradient_clip > 0:
                gradients = tf.clip_by_global_norm(
                    gradients, self.gradient_clip)[0]
            self.optimizer.apply_gradients(zip(gradients, self.variables))

            kl = tf.reduce_mean(log_probs - new_log_probs)
            std = tf.reduce_mean(distributions.stddev())

        return dict(loss=loss, kl=kl, entropy=entropy, std=std)


class ClippedRatio:
    def __init__(
        self, optimizer=None, ratio_clip=0.2, kl_threshold=0.015,
        entropy_coeff=0, gradient_clip=0
    ):
        self.optimizer = optimizer or tf.keras.optimizers.Adam(
            lr=3e-4, epsilon=1e-8)
        self.ratio_clip = ratio_clip
        self.kl_threshold = kl_threshold
        self.entropy_coeff = entropy_coeff
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = self.model.actor.trainable_variables

    @tf.function
    def __call__(self, observations, actions, advantages, log_probs):
        if tf.math.count_nonzero(advantages) == 0:
            loss, kl, clip_fraction = 0., 0., 0.
            distributions = self.model.actor(observations)
            entropy = tf.reduce_mean(distributions.entropy())
            std = tf.reduce_mean(distributions.stddev())

        else:
            with tf.GradientTape() as tape:
                distributions = self.model.actor(observations)
                new_log_probs = distributions.log_prob(actions)
                ratios_1 = tf.exp(new_log_probs - log_probs)
                surrogates_1 = advantages * ratios_1
                ratio_low = 1 - self.ratio_clip
                ratio_high = 1 + self.ratio_clip
                ratios_2 = tf.clip_by_value(ratios_1, ratio_low, ratio_high)
                surrogates_2 = advantages * ratios_2
                loss = -tf.reduce_mean(tf.minimum(surrogates_1, surrogates_2))
                entropy = tf.reduce_mean(distributions.entropy())
                if self.entropy_coeff != 0:
                    loss -= self.entropy_coeff * entropy

            gradients = tape.gradient(loss, self.variables)
            if self.gradient_clip > 0:
                gradients = tf.clip_by_global_norm(
                    gradients, self.gradient_clip)[0]
            self.optimizer.apply_gradients(zip(gradients, self.variables))

            kl = tf.reduce_mean(log_probs - new_log_probs)
            clipped = tf.logical_or(
                ratios_1 > ratio_high, ratios_1 < ratio_low)
            clip_fraction = tf.reduce_mean(tf.cast(clipped, tf.float32))
            std = tf.reduce_mean(distributions.stddev())

        return dict(
            loss=loss, kl=kl, entropy=entropy, clip_fraction=clip_fraction,
            std=std, stop=kl > self.kl_threshold)


class TrustRegionPolicyGradient:
    def __init__(self, optimizer=None, entropy_coeff=0):
        self.optimizer = optimizer or updaters.ConjugateGradient(
            constraint_threshold=0.01,
            damping_coefficient=0.1,
            conjugate_gradient_steps=10,
            backtrack_steps=10,
            backtrack_coefficient=0.8)
        self.entropy_coeff = entropy_coeff

    def initialize(self, model):
        self.model = model
        self.variables = self.model.actor.trainable_variables

    # Careful: do not use in @tf.function
    def __call__(
        self, observations, actions, log_probs, locs, scales, advantages
    ):
        if np.all(advantages == 0.):
            kl = tf.convert_to_tensor(0.)
            loss = tf.convert_to_tensor(0.)
            steps = tf.convert_to_tensor(0)

        else:
            kl, loss, steps = self.optimizer.optimize(
                loss_function=lambda: self._loss(
                    observations, actions, log_probs, advantages),
                constraint_function=lambda: self._kl(
                    observations, locs, scales),
                variables=self.variables)

        return dict(loss=loss, kl=kl, backtrack_steps=steps)

    @tf.function
    def _loss(self, observations, actions, old_log_probs, advantages):
        distributions = self.model.actor(observations)
        log_probs = distributions.log_prob(actions)
        ratios = tf.exp(log_probs - old_log_probs)
        loss = -tf.reduce_mean(ratios * advantages)
        if self.entropy_coeff != 0:
            entropy = tf.reduce_mean(distributions.entropy())
            loss -= self.entropy_coeff * entropy
        return loss

    @tf.function
    def _kl(self, observations, locs, scales):
        distributions = self.model.actor(observations)
        old_distributions = type(distributions)(locs, scales)
        return tf.reduce_mean(distributions.kl_divergence(old_distributions))


class DeterministicPolicyGradient:
    def __init__(self, optimizer=None, gradient_clip=0):
        self.optimizer = optimizer or tf.keras.optimizers.Adam(
            lr=1e-3, epsilon=1e-8)
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = self.model.actor.trainable_variables

    @tf.function
    def __call__(self, observations):
        with tf.GradientTape() as tape:
            actions = self.model.actor(observations)
            values = self.model.critic(observations, actions)
            loss = -tf.reduce_mean(values)

        gradients = tape.gradient(loss, self.variables)
        if self.gradient_clip > 0:
            gradients = tf.clip_by_global_norm(
                gradients, self.gradient_clip)[0]
        self.optimizer.apply_gradients(zip(gradients, self.variables))

        return dict(loss=loss)


class TwinCriticSoftDeterministicPolicyGradient:
    def __init__(self, optimizer=None, entropy_coeff=0.2, gradient_clip=0):
        self.optimizer = optimizer or tf.keras.optimizers.Adam(
            lr=1e-3, epsilon=1e-8)
        self.entropy_coeff = entropy_coeff
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = self.model.actor.trainable_variables

    @tf.function
    def __call__(self, observations):
        with tf.GradientTape() as tape:
            distributions = self.model.actor(observations)
            if hasattr(distributions, 'sample_with_log_prob'):
                actions, log_probs = distributions.sample_with_log_prob()
            else:
                actions = distributions.sample()
                log_probs = distributions.log_prob(actions)
            values_1 = self.model.critic_1(observations, actions)
            values_2 = self.model.critic_2(observations, actions)
            values = tf.minimum(values_1, values_2)
            loss = tf.reduce_mean(self.entropy_coeff * log_probs - values)

        gradients = tape.gradient(loss, self.variables)
        if self.gradient_clip > 0:
            gradients = tf.clip_by_global_norm(
                gradients, self.gradient_clip)[0]
        self.optimizer.apply_gradients(zip(gradients, self.variables))

        return dict(loss=loss)
