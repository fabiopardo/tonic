import tensorflow as tf

from tonic.tensorflow import updaters


class VRegression:
    def __init__(self, loss=None, optimizer=None, gradient_clip=0):
        self.loss = loss or tf.keras.losses.MeanSquaredError()
        self.optimizer = optimizer or \
            tf.keras.optimizers.Adam(lr=1e-3, epsilon=1e-8)
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
        self.optimizer = optimizer or \
            tf.keras.optimizers.Adam(lr=1e-3, epsilon=1e-8)
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
        self.optimizer = optimizer or \
            tf.keras.optimizers.Adam(lr=1e-3, epsilon=1e-8)
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


class DistributionalDeterministicQLearning:
    def __init__(self, optimizer=None, gradient_clip=0):
        self.optimizer = optimizer or \
            tf.keras.optimizers.Adam(lr=1e-3, epsilon=1e-8)
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = self.model.critic.trainable_variables

    @tf.function
    def __call__(
        self, observations, actions, next_observations, rewards, discounts
    ):
        next_actions = self.model.target_actor(next_observations)
        next_value_distributions = self.model.target_critic(
            next_observations, next_actions)

        values = next_value_distributions.values
        returns = rewards[:, None] + discounts[:, None] * values
        targets = next_value_distributions.project(returns)

        with tf.GradientTape() as tape:
            value_distributions = self.model.critic(observations, actions)
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=value_distributions.logits, labels=targets)
            loss = tf.reduce_mean(losses)

        gradients = tape.gradient(loss, self.variables)
        if self.gradient_clip > 0:
            gradients = tf.clip_by_global_norm(
                gradients, self.gradient_clip)[0]
        self.optimizer.apply_gradients(zip(gradients, self.variables))

        return dict(loss=loss)


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
        self.optimizer = optimizer or \
            tf.keras.optimizers.Adam(lr=1e-3, epsilon=1e-8)
        self.target_action_noise = target_action_noise or \
            TargetActionNoise(scale=0.2, clip=0.5)
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
        self.optimizer = optimizer or \
            tf.keras.optimizers.Adam(lr=1e-3, epsilon=1e-8)
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


class ExpectedSARSA:
    def __init__(
        self, num_samples=20, loss=None, optimizer=None, gradient_clip=0
    ):
        self.num_samples = num_samples
        self.loss = loss or tf.keras.losses.MeanSquaredError()
        self.optimizer = optimizer or \
            tf.keras.optimizers.Adam(lr=1e-3, epsilon=1e-8)
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = self.model.critic.trainable_variables

    @tf.function
    def __call__(
        self, observations, actions, next_observations, rewards, discounts
    ):
        # Approximate the expected next values.
        next_target_distributions = self.model.target_actor(next_observations)
        next_actions = next_target_distributions.sample(self.num_samples)
        next_actions = updaters.merge_first_two_dims(next_actions)
        next_observations = updaters.tile(next_observations, self.num_samples)
        next_observations = updaters.merge_first_two_dims(next_observations)
        next_values = self.model.target_critic(next_observations, next_actions)
        next_values = tf.reshape(next_values, (self.num_samples, -1))
        next_values = tf.reduce_mean(next_values, axis=0)
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


class TwinCriticDistributionalDeterministicQLearning:
    def __init__(
        self, optimizer=None, target_action_noise=None, gradient_clip=0
    ):
        self.optimizer = optimizer or \
            tf.keras.optimizers.Adam(lr=1e-3, epsilon=1e-8)
        self.target_action_noise = target_action_noise or \
            TargetActionNoise(scale=0.2, clip=0.5)
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
        next_value_distributions_1 = self.model.target_critic_1(
            next_observations, next_actions)
        next_value_distributions_2 = self.model.target_critic_2(
            next_observations, next_actions)

        values = next_value_distributions_1.values
        returns = rewards[:, None] + discounts[:, None] * values
        targets_1 = next_value_distributions_1.project(returns)
        targets_2 = next_value_distributions_2.project(returns)
        next_values_1 = next_value_distributions_1.mean()
        next_values_2 = next_value_distributions_2.mean()
        twin_next_values = tf.concat(
            [next_values_1[None], next_values_2[None]], axis=0)
        indices = tf.argmin(twin_next_values, axis=0, output_type=tf.int32)
        twin_targets = tf.concat([targets_1[None], targets_2[None]], axis=0)
        batch_size = tf.shape(observations)[0]
        indices = tf.stack([indices, tf.range(batch_size)], axis=-1)
        targets = tf.gather_nd(twin_targets, indices)

        with tf.GradientTape() as tape:
            value_distributions_1 = self.model.critic_1(observations, actions)
            losses_1 = tf.nn.softmax_cross_entropy_with_logits(
                logits=value_distributions_1.logits, labels=targets)
            value_distributions_2 = self.model.critic_2(observations, actions)
            losses_2 = tf.nn.softmax_cross_entropy_with_logits(
                logits=value_distributions_2.logits, labels=targets)
            loss = tf.reduce_mean(losses_1) + tf.reduce_mean(losses_2)

        gradients = tape.gradient(loss, self.variables)
        if self.gradient_clip > 0:
            gradients = tf.clip_by_global_norm(
                gradients, self.gradient_clip)[0]
        self.optimizer.apply_gradients(zip(gradients, self.variables))

        return dict(loss=loss, q1=value_distributions_1.mean(),
                    q2=value_distributions_2.mean())
