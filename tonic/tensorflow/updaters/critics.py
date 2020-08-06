import tensorflow as tf


class VRegression:
    def __init__(self, loss=None, optimizer=None, gradient_clip=0):
        self.loss = loss or tf.keras.losses.MeanSquaredError()
        self.optimizer = optimizer or tf.keras.optimizers.Adam(
            lr=1e-3, epsilon=1e-8)
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = self.model.critic.trainable_variables

    @tf.function
    def __call__(self, observations, returns):
        with tf.GradientTape() as tape:
            values = self.model.critic(observations)
            loss = self.loss(returns, values)

        gradients = tape.gradient(loss, self.variables)
        if self.gradient_clip > 0:
            gradients = tf.clip_by_global_norm(
                gradients, self.gradient_clip)[0]
        self.optimizer.apply_gradients(zip(gradients, self.variables))

        return dict(loss=loss, v=values)


class QRegression:
    def __init__(self, loss=None, optimizer=None, gradient_clip=0):
        self.loss = loss or tf.keras.losses.MeanSquaredError()
        self.optimizer = optimizer or tf.keras.optimizers.Adam(
            lr=1e-3, epsilon=1e-8)
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = self.model.critic.trainable_variables

    @tf.function
    def __call__(self, observations, actions, returns):
        with tf.GradientTape() as tape:
            values = self.model.critic(observations, actions)
            loss = self.loss(returns, values)

        gradients = tape.gradient(loss, self.variables)
        if self.gradient_clip > 0:
            gradients = tf.clip_by_global_norm(
                gradients, self.gradient_clip)[0]
        self.optimizer.apply_gradients(zip(gradients, self.variables))

        return dict(loss=loss, q=values)


class DeterministicQLearning:
    def __init__(self, loss=None, optimizer=None, gradient_clip=0):
        self.loss = loss or tf.keras.losses.MeanSquaredError()
        self.optimizer = optimizer or tf.keras.optimizers.Adam(
            lr=1e-3, epsilon=1e-8)
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = self.model.critic.trainable_variables

    @tf.function
    def __call__(
        self, observations, actions, next_observations, rewards, discounts
    ):
        next_actions = self.model.target_actor(next_observations)
        next_values = self.model.target_critic(next_observations, next_actions)
        returns = rewards + discounts * next_values

        with tf.GradientTape() as tape:
            values = self.model.critic(observations, actions)
            loss = self.loss(returns, values)

        gradients = tape.gradient(loss, self.variables)
        if self.gradient_clip > 0:
            gradients = tf.clip_by_global_norm(
                gradients, self.gradient_clip)[0]
        self.optimizer.apply_gradients(zip(gradients, self.variables))

        return dict(loss=loss, q=values)


class TargetActionNoise:
    def __init__(self, scale=0.2, clip=0.5):
        self.scale = scale
        self.clip = clip

    def __call__(self, actions):
        noises = self.scale * tf.random.normal(actions.shape)
        noises = tf.clip_by_value(noises, -self.clip, self.clip)
        actions = actions + noises
        return tf.clip_by_value(actions, -1, 1)


class TwinCriticDeterministicQLearning:
    def __init__(
        self, loss=None, optimizer=None, target_action_noise=None,
        gradient_clip=0
    ):
        self.loss = loss or tf.keras.losses.MeanSquaredError()
        self.optimizer = optimizer or tf.keras.optimizers.Adam(
            lr=1e-3, epsilon=1e-8)
        self.target_action_noise = target_action_noise or TargetActionNoise(
            scale=0.2, clip=0.5)
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        variables_1 = self.model.critic_1.trainable_variables
        variables_2 = self.model.critic_2.trainable_variables
        self.variables = variables_1 + variables_2

    @tf.function
    def __call__(
        self, observations, actions, next_observations, rewards, discounts
    ):
        next_actions = self.model.target_actor(next_observations)
        next_actions = self.target_action_noise(next_actions)
        next_values_1 = self.model.target_critic_1(
            next_observations, next_actions)
        next_values_2 = self.model.target_critic_2(
            next_observations, next_actions)
        next_values = tf.minimum(next_values_1, next_values_2)
        returns = rewards + discounts * next_values

        with tf.GradientTape() as tape:
            values_1 = self.model.critic_1(observations, actions)
            values_2 = self.model.critic_2(observations, actions)
            loss_1 = self.loss(returns, values_1)
            loss_2 = self.loss(returns, values_2)
            loss = loss_1 + loss_2

        gradients = tape.gradient(loss, self.variables)
        if self.gradient_clip > 0:
            gradients = tf.clip_by_global_norm(
                gradients, self.gradient_clip)[0]
        self.optimizer.apply_gradients(zip(gradients, self.variables))

        return dict(loss=loss, q1=values_1, q2=values_2)


class TwinCriticSoftQLearning:
    def __init__(
        self, loss=None, optimizer=None, entropy_coeff=0.2, gradient_clip=0
    ):
        self.loss = loss or tf.keras.losses.MeanSquaredError()
        self.optimizer = optimizer or tf.keras.optimizers.Adam(
            lr=1e-3, epsilon=1e-8)
        self.entropy_coeff = entropy_coeff
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        variables_1 = self.model.critic_1.trainable_variables
        variables_2 = self.model.critic_2.trainable_variables
        self.variables = variables_1 + variables_2

    @tf.function
    def __call__(
        self, observations, actions, next_observations, rewards, discounts
    ):
        next_distributions = self.model.actor(next_observations)
        if hasattr(next_distributions, 'sample_with_log_prob'):
            outs = next_distributions.sample_with_log_prob()
            next_actions, next_log_probs = outs
        else:
            next_actions = next_distributions.sample()
            next_log_probs = next_distributions.log_prob(next_actions)
        next_values_1 = self.model.target_critic_1(
            next_observations, next_actions)
        next_values_2 = self.model.target_critic_2(
            next_observations, next_actions)
        next_values = tf.minimum(next_values_1, next_values_2)
        returns = rewards + discounts * (
            next_values - self.entropy_coeff * next_log_probs)

        with tf.GradientTape() as tape:
            values_1 = self.model.critic_1(observations, actions)
            values_2 = self.model.critic_2(observations, actions)
            loss_1 = self.loss(returns, values_1)
            loss_2 = self.loss(returns, values_2)
            loss = loss_1 + loss_2

        gradients = tape.gradient(loss, self.variables)
        if self.gradient_clip > 0:
            gradients = tf.clip_by_global_norm(
                gradients, self.gradient_clip)[0]
        self.optimizer.apply_gradients(zip(gradients, self.variables))

        return dict(loss=loss, q1=values_1, q2=values_2)
