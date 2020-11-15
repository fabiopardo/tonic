import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tonic.tensorflow import updaters


FLOAT_EPSILON = 1e-8


class StochasticPolicyGradient:
    def __init__(self, optimizer=None, entropy_coeff=0, gradient_clip=0):
        self.optimizer = optimizer or \
            tf.keras.optimizers.Adam(lr=3e-4, epsilon=1e-8)
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
        self.optimizer = optimizer or \
            tf.keras.optimizers.Adam(lr=3e-4, epsilon=1e-8)
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
        self.optimizer = optimizer or updaters.ConjugateGradient()
        self.entropy_coeff = entropy_coeff

    def initialize(self, model):
        self.model = model
        self.variables = self.model.actor.trainable_variables

    # Careful: do not use in @tf.function
    def __call__(
        self, observations, actions, log_probs, locs, scales, advantages
    ):
        if np.all(advantages == 0.):
            kl = tf.convert_to_tensor(0., dtype=tf.float32)
            loss = tf.convert_to_tensor(0., dtype=tf.float32)
            steps = tf.convert_to_tensor(0, dtype=tf.float32)

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
        self.optimizer = optimizer or \
            tf.keras.optimizers.Adam(lr=1e-3, epsilon=1e-8)
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


class DistributionalDeterministicPolicyGradient:
    def __init__(self, optimizer=None, gradient_clip=0):
        self.optimizer = optimizer or \
            tf.keras.optimizers.Adam(lr=1e-3, epsilon=1e-8)
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = self.model.actor.trainable_variables

    @tf.function
    def __call__(self, observations):
        with tf.GradientTape() as tape:
            actions = self.model.actor(observations)
            value_distributions = self.model.critic(observations, actions)
            values = value_distributions.mean()
            loss = -tf.reduce_mean(values)

        gradients = tape.gradient(loss, self.variables)
        if self.gradient_clip > 0:
            gradients = tf.clip_by_global_norm(
                gradients, self.gradient_clip)[0]
        self.optimizer.apply_gradients(zip(gradients, self.variables))

        return dict(loss=loss)


class TwinCriticSoftDeterministicPolicyGradient:
    def __init__(self, optimizer=None, entropy_coeff=0.2, gradient_clip=0):
        self.optimizer = optimizer or \
            tf.keras.optimizers.Adam(lr=1e-3, epsilon=1e-8)
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


class MaximumAPosterioriPolicyOptimization:
    def __init__(
        self, num_samples=20, epsilon=1e-1, epsilon_penalty=1e-3,
        epsilon_mean=1e-3, epsilon_std=1e-6, initial_log_temperature=1.,
        initial_log_alpha_mean=1., initial_log_alpha_std=10.,
        min_log_dual=-18., per_dim_constraining=True, action_penalization=True,
        actor_optimizer=None, dual_optimizer=None, gradient_clip=0
    ):
        self.num_samples = num_samples
        self.epsilon = epsilon
        self.epsilon_mean = epsilon_mean
        self.epsilon_std = epsilon_std
        self.initial_log_temperature = initial_log_temperature
        self.initial_log_alpha_mean = initial_log_alpha_mean
        self.initial_log_alpha_std = initial_log_alpha_std
        self.min_log_dual = min_log_dual
        self.action_penalization = action_penalization
        self.epsilon_penalty = epsilon_penalty
        self.per_dim_constraining = per_dim_constraining
        self.actor_optimizer = actor_optimizer or \
            tf.keras.optimizers.Adam(lr=3e-4, epsilon=1e-8)
        self.dual_optimizer = actor_optimizer or \
            tf.keras.optimizers.Adam(lr=1e-2, epsilon=1e-8)
        self.gradient_clip = gradient_clip

    def initialize(self, model, action_space):
        self.model = model
        self.actor_variables = model.actor.trainable_variables

        # Dual variables.
        self.dual_variables = []
        self.log_temperature = tf.Variable(
            [self.initial_log_temperature], dtype=tf.float32)
        self.dual_variables.append(self.log_temperature)
        shape = [action_space.shape[0]] if self.per_dim_constraining else [1]
        self.log_alpha_mean = tf.Variable(
            tf.fill(shape, self.initial_log_alpha_mean), dtype=tf.float32)
        self.dual_variables.append(self.log_alpha_mean)
        self.log_alpha_std = tf.Variable(
            tf.fill(shape, self.initial_log_alpha_std), dtype=tf.float32)
        self.dual_variables.append(self.log_alpha_std)
        if self.action_penalization:
            self.log_penalty_temperature = tf.Variable(
                [self.initial_log_temperature], dtype=tf.float32)
            self.dual_variables.append(self.log_penalty_temperature)

    @tf.function
    def __call__(self, observations):
        def parametric_kl_and_dual_losses(kl, alpha, epsilon):
            kl_mean = tf.reduce_mean(kl, axis=0)
            kl_loss = tf.reduce_sum(tf.stop_gradient(alpha) * kl_mean)
            alpha_loss = tf.reduce_sum(
                alpha * (epsilon - tf.stop_gradient(kl_mean)))
            return kl_loss, alpha_loss

        def weights_and_temperature_loss(q_values, epsilon, temperature):
            tempered_q_values = tf.stop_gradient(q_values) / temperature
            weights = tf.nn.softmax(tempered_q_values, axis=0)
            weights = tf.stop_gradient(weights)

            # Temperature loss (dual of the E-step).
            q_log_sum_exp = tf.reduce_logsumexp(tempered_q_values, axis=0)
            num_actions = tf.cast(q_values.shape[0], tf.float32)
            log_num_actions = tf.math.log(num_actions)
            loss = epsilon + tf.reduce_mean(q_log_sum_exp) - log_num_actions
            loss = temperature * loss

            return weights, loss

        # Use independent normals to satisfy KL constraints per-dimension.
        def independent_normals(distribution_1, distribution_2=None):
            distribution_2 = distribution_2 or distribution_1
            return tfp.distributions.Independent(tfp.distributions.Normal(
                distribution_1.mean(), distribution_2.stddev()))

        self.log_temperature.assign(
            tf.maximum(self.min_log_dual, self.log_temperature))
        self.log_alpha_mean.assign(
            tf.maximum(self.min_log_dual, self.log_alpha_mean))
        self.log_alpha_std.assign(
            tf.maximum(self.min_log_dual, self.log_alpha_std))
        if self.action_penalization:
            self.log_penalty_temperature.assign(tf.maximum(
                self.min_log_dual, self.log_penalty_temperature))

        target_distributions = self.model.target_actor(observations)
        actions = target_distributions.sample(self.num_samples)

        tiled_observations = updaters.tile(observations, self.num_samples)
        flat_observations = updaters.merge_first_two_dims(tiled_observations)
        flat_actions = updaters.merge_first_two_dims(actions)
        values = self.model.target_critic(flat_observations, flat_actions)
        values = tf.reshape(values, (self.num_samples, -1))

        assert isinstance(
            target_distributions, tfp.distributions.MultivariateNormalDiag)
        target_distributions = independent_normals(target_distributions)

        with tf.GradientTape() as tape:
            distributions = self.model.actor(observations)
            distributions = independent_normals(distributions)

            temperature = tf.math.softplus(
                self.log_temperature) + FLOAT_EPSILON
            alpha_mean = tf.math.softplus(self.log_alpha_mean) + FLOAT_EPSILON
            alpha_std = tf.math.softplus(self.log_alpha_std) + FLOAT_EPSILON
            weights, temperature_loss = weights_and_temperature_loss(
                values, self.epsilon, temperature)

            # Action penalization is quadratic beyond [-1, 1].
            if self.action_penalization:
                penalty_temperature = tf.math.softplus(
                    self.log_penalty_temperature) + FLOAT_EPSILON
                diff_bounds = actions - tf.clip_by_value(actions, -1, 1)
                action_bound_costs = -tf.norm(diff_bounds, axis=-1)
                penalty_weights, penalty_temperature_loss = \
                    weights_and_temperature_loss(
                        action_bound_costs,
                        self.epsilon_penalty, penalty_temperature)
                weights += penalty_weights
                temperature_loss += penalty_temperature_loss

            # Decompose the policy into fixed-mean and fixed-std distributions.
            fixed_std_distribution = independent_normals(
                distributions.distribution, target_distributions.distribution)
            fixed_mean_distribution = independent_normals(
                target_distributions.distribution, distributions.distribution)

            # Compute the decomposed policy losses.
            policy_mean_losses = tf.reduce_sum(
                fixed_std_distribution.log_prob(actions) * weights, axis=0)
            policy_mean_loss = -tf.reduce_mean(policy_mean_losses)
            policy_std_losses = tf.reduce_sum(
                fixed_mean_distribution.log_prob(actions) * weights, axis=0)
            policy_std_loss = -tf.reduce_mean(policy_std_losses)

            # Compute the decomposed KL between the target and online policies.
            if self.per_dim_constraining:
                kl_mean = target_distributions.distribution.kl_divergence(
                    fixed_std_distribution.distribution)
                kl_std = target_distributions.distribution.kl_divergence(
                    fixed_mean_distribution.distribution)
            else:
                kl_mean = target_distributions.kl_divergence(
                    fixed_std_distribution)
                kl_std = target_distributions.kl_divergence(
                    fixed_mean_distribution)

            # Compute the alpha-weighted KL-penalty and dual losses.
            kl_mean_loss, alpha_mean_loss = parametric_kl_and_dual_losses(
                kl_mean, alpha_mean, self.epsilon_mean)
            kl_std_loss, alpha_std_loss = parametric_kl_and_dual_losses(
                kl_std, alpha_std, self.epsilon_std)

            # Combine losses.
            policy_loss = policy_mean_loss + policy_std_loss
            kl_loss = kl_mean_loss + kl_std_loss
            dual_loss = alpha_mean_loss + alpha_std_loss + temperature_loss
            loss = policy_loss + kl_loss + dual_loss

        actor_gradients, dual_gradients = tape.gradient(
            loss, (self.actor_variables, self.dual_variables))

        if self.gradient_clip > 0:
            actor_gradients = tf.clip_by_global_norm(
                actor_gradients, self.gradient_clip)[0]
            dual_gradients = tf.clip_by_global_norm(
                dual_gradients, self.gradient_clip)[0]

        self.actor_optimizer.apply_gradients(
            zip(actor_gradients, self.actor_variables))
        self.dual_optimizer.apply_gradients(
            zip(dual_gradients, self.dual_variables))

        dual_variables = dict(
            temperature=temperature, alpha_mean=alpha_mean,
            alpha_std=alpha_std)
        if self.action_penalization:
            dual_variables['penalty_temperature'] = penalty_temperature

        return dict(
            policy_mean_loss=policy_mean_loss, policy_std_loss=policy_std_loss,
            kl_mean_loss=kl_mean_loss, kl_std_loss=kl_std_loss,
            alpha_mean_loss=alpha_mean_loss, alpha_std_loss=alpha_std_loss,
            temperature_loss=temperature_loss, **dual_variables)
