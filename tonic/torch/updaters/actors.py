import torch

from tonic.torch import models, updaters  # noqa


FLOAT_EPSILON = 1e-8


class StochasticPolicyGradient:
    def __init__(self, optimizer=None, entropy_coeff=0, gradient_clip=0):
        self.optimizer = optimizer or (
            lambda params: torch.optim.Adam(params, lr=3e-4))
        self.entropy_coeff = entropy_coeff
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = models.trainable_variables(self.model.actor)
        self.optimizer = self.optimizer(self.variables)

    def __call__(self, observations, actions, advantages, log_probs):
        if (advantages == 0.).all():
            loss = torch.as_tensor(0., dtype=torch.float32)
            kl = torch.as_tensor(0., dtype=torch.float32)
            with torch.no_grad():
                distributions = self.model.actor(observations)
                entropy = distributions.entropy().mean()
                std = distributions.stddev.mean()

        else:
            self.optimizer.zero_grad()
            distributions = self.model.actor(observations)
            new_log_probs = distributions.log_prob(actions).sum(dim=-1)
            loss = -(advantages * new_log_probs).mean()
            entropy = distributions.entropy().mean()
            if self.entropy_coeff != 0:
                loss -= self.entropy_coeff * entropy

            loss.backward()
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.variables, self.gradient_clip)
            self.optimizer.step()

            loss = loss.detach()
            kl = (log_probs - new_log_probs).mean().detach()
            entropy = entropy.detach()
            std = distributions.stddev.mean().detach()

        return dict(loss=loss, kl=kl, entropy=entropy, std=std)


class ClippedRatio:
    def __init__(
        self, optimizer=None, ratio_clip=0.2, kl_threshold=0.015,
        entropy_coeff=0, gradient_clip=0
    ):
        self.optimizer = optimizer or (
            lambda params: torch.optim.Adam(params, lr=3e-4))
        self.ratio_clip = ratio_clip
        self.kl_threshold = kl_threshold
        self.entropy_coeff = entropy_coeff
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = models.trainable_variables(self.model.actor)
        self.optimizer = self.optimizer(self.variables)

    def __call__(self, observations, actions, advantages, log_probs):
        if (advantages == 0.).all():
            loss = torch.as_tensor(0., dtype=torch.float32)
            kl = torch.as_tensor(0., dtype=torch.float32)
            clip_fraction = torch.as_tensor(0., dtype=torch.float32)
            with torch.no_grad():
                distributions = self.model.actor(observations)
                entropy = distributions.entropy().mean()
                std = distributions.stddev.mean()

        else:
            self.optimizer.zero_grad()
            distributions = self.model.actor(observations)
            new_log_probs = distributions.log_prob(actions).sum(dim=-1)
            ratios_1 = torch.exp(new_log_probs - log_probs)
            surrogates_1 = advantages * ratios_1
            ratio_low = 1 - self.ratio_clip
            ratio_high = 1 + self.ratio_clip
            ratios_2 = torch.clamp(ratios_1, ratio_low, ratio_high)
            surrogates_2 = advantages * ratios_2
            loss = -(torch.min(surrogates_1, surrogates_2)).mean()
            entropy = distributions.entropy().mean()
            if self.entropy_coeff != 0:
                loss -= self.entropy_coeff * entropy

            loss.backward()
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.variables, self.gradient_clip)
            self.optimizer.step()

            loss = loss.detach()
            with torch.no_grad():
                kl = (log_probs - new_log_probs).mean()
            entropy = entropy.detach()
            clipped = ratios_1.gt(ratio_high) | ratios_1.lt(ratio_low)
            clip_fraction = torch.as_tensor(
                clipped, dtype=torch.float32).mean()
            std = distributions.stddev.mean().detach()

        return dict(
            loss=loss, kl=kl, entropy=entropy, clip_fraction=clip_fraction,
            std=std, stop=kl > self.kl_threshold)


class TrustRegionPolicyGradient:
    def __init__(self, optimizer=None, entropy_coeff=0):
        self.optimizer = optimizer or updaters.ConjugateGradient()
        self.entropy_coeff = entropy_coeff

    def initialize(self, model):
        self.model = model
        self.variables = models.trainable_variables(self.model.actor)

    def __call__(
        self, observations, actions, log_probs, locs, scales, advantages
    ):
        if (advantages == 0.).all():
            kl = torch.as_tensor(0., dtype=torch.float32)
            loss = torch.as_tensor(0., dtype=torch.float32)
            steps = torch.as_tensor(0, dtype=torch.int32)

        else:
            kl, loss, steps = self.optimizer.optimize(
                loss_function=lambda: self._loss(
                    observations, actions, log_probs, advantages),
                constraint_function=lambda: self._kl(
                    observations, locs, scales),
                variables=self.variables)

        return dict(loss=loss, kl=kl, backtrack_steps=steps)

    def _loss(self, observations, actions, old_log_probs, advantages):
        distributions = self.model.actor(observations)
        log_probs = distributions.log_prob(actions).sum(dim=-1)
        ratios = torch.exp(log_probs - old_log_probs)
        loss = -(ratios * advantages).mean()
        if self.entropy_coeff != 0:
            entropy = distributions.entropy().mean()
            loss -= self.entropy_coeff * entropy
        return loss

    def _kl(self, observations, locs, scales):
        distributions = self.model.actor(observations)
        old_distributions = type(distributions)(locs, scales)
        return torch.distributions.kl.kl_divergence(
            distributions, old_distributions).mean()


class DeterministicPolicyGradient:
    def __init__(self, optimizer=None, gradient_clip=0):
        self.optimizer = optimizer or (
            lambda params: torch.optim.Adam(params, lr=1e-3))
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = models.trainable_variables(self.model.actor)
        self.optimizer = self.optimizer(self.variables)

    def __call__(self, observations):
        critic_variables = models.trainable_variables(self.model.critic)

        for var in critic_variables:
            var.requires_grad = False

        self.optimizer.zero_grad()
        actions = self.model.actor(observations)
        values = self.model.critic(observations, actions)
        loss = -values.mean()

        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        self.optimizer.step()

        for var in critic_variables:
            var.requires_grad = True

        return dict(loss=loss.detach())


class DistributionalDeterministicPolicyGradient:
    def __init__(self, optimizer=None, gradient_clip=0):
        self.optimizer = optimizer or (
            lambda params: torch.optim.Adam(params, lr=1e-3))
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = models.trainable_variables(self.model.actor)
        self.optimizer = self.optimizer(self.variables)

    def __call__(self, observations):
        critic_variables = models.trainable_variables(self.model.critic)

        for var in critic_variables:
            var.requires_grad = False

        self.optimizer.zero_grad()
        actions = self.model.actor(observations)
        value_distributions = self.model.critic(observations, actions)
        values = value_distributions.mean()
        loss = -values.mean()

        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        self.optimizer.step()

        for var in critic_variables:
            var.requires_grad = True

        return dict(loss=loss.detach())


class TwinCriticSoftDeterministicPolicyGradient:
    def __init__(self, optimizer=None, entropy_coeff=0.2, gradient_clip=0):
        self.optimizer = optimizer or (
            lambda params: torch.optim.Adam(params, lr=3e-4))
        self.entropy_coeff = entropy_coeff
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = models.trainable_variables(self.model.actor)
        self.optimizer = self.optimizer(self.variables)

    def __call__(self, observations):
        critic_1_variables = models.trainable_variables(self.model.critic_1)
        critic_2_variables = models.trainable_variables(self.model.critic_2)
        critic_variables = critic_1_variables + critic_2_variables

        for var in critic_variables:
            var.requires_grad = False

        self.optimizer.zero_grad()
        distributions = self.model.actor(observations)
        if hasattr(distributions, 'rsample_with_log_prob'):
            actions, log_probs = distributions.rsample_with_log_prob()
        else:
            actions = distributions.rsample()
            log_probs = distributions.log_prob(actions)
        log_probs = log_probs.sum(dim=-1)
        values_1 = self.model.critic_1(observations, actions)
        values_2 = self.model.critic_2(observations, actions)
        values = torch.min(values_1, values_2)
        loss = (self.entropy_coeff * log_probs - values).mean()

        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        self.optimizer.step()

        for var in critic_variables:
            var.requires_grad = True

        return dict(loss=loss.detach())


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
        self.min_log_dual = torch.as_tensor(min_log_dual, dtype=torch.float32)
        self.action_penalization = action_penalization
        self.epsilon_penalty = epsilon_penalty
        self.per_dim_constraining = per_dim_constraining
        self.actor_optimizer = actor_optimizer or (
            lambda params: torch.optim.Adam(params, lr=3e-4))
        self.dual_optimizer = actor_optimizer or (
            lambda params: torch.optim.Adam(params, lr=1e-2))
        self.gradient_clip = gradient_clip

    def initialize(self, model, action_space):
        self.model = model
        self.actor_variables = models.trainable_variables(self.model.actor)
        self.actor_optimizer = self.actor_optimizer(self.actor_variables)

        # Dual variables.
        self.dual_variables = []
        self.log_temperature = torch.nn.Parameter(torch.as_tensor(
            [self.initial_log_temperature], dtype=torch.float32))
        self.dual_variables.append(self.log_temperature)
        shape = [action_space.shape[0]] if self.per_dim_constraining else [1]
        self.log_alpha_mean = torch.nn.Parameter(torch.full(
            shape, self.initial_log_alpha_mean, dtype=torch.float32))
        self.dual_variables.append(self.log_alpha_mean)
        self.log_alpha_std = torch.nn.Parameter(torch.full(
            shape, self.initial_log_alpha_std, dtype=torch.float32))
        self.dual_variables.append(self.log_alpha_std)
        if self.action_penalization:
            self.log_penalty_temperature = torch.nn.Parameter(torch.as_tensor(
                [self.initial_log_temperature], dtype=torch.float32))
            self.dual_variables.append(self.log_penalty_temperature)
        self.dual_optimizer = self.dual_optimizer(self.dual_variables)

    def __call__(self, observations):
        def parametric_kl_and_dual_losses(kl, alpha, epsilon):
            kl_mean = kl.mean(dim=0)
            kl_loss = (alpha.detach() * kl_mean).sum()
            alpha_loss = (alpha * (epsilon - kl_mean.detach())).sum()
            return kl_loss, alpha_loss

        def weights_and_temperature_loss(q_values, epsilon, temperature):
            tempered_q_values = q_values.detach() / temperature
            weights = torch.nn.functional.softmax(tempered_q_values, dim=0)
            weights = weights.detach()

            # Temperature loss (dual of the E-step).
            q_log_sum_exp = torch.logsumexp(tempered_q_values, dim=0)
            num_actions = torch.as_tensor(
                q_values.shape[0], dtype=torch.float32)
            log_num_actions = torch.log(num_actions)
            loss = epsilon + (q_log_sum_exp).mean() - log_num_actions
            loss = temperature * loss

            return weights, loss

        # Use independent normals to satisfy KL constraints per-dimension.
        def independent_normals(distribution_1, distribution_2=None):
            distribution_2 = distribution_2 or distribution_1
            return torch.distributions.independent.Independent(
                torch.distributions.normal.Normal(
                    distribution_1.mean, distribution_2.stddev), -1)

        with torch.no_grad():
            self.log_temperature.data.copy_(
                torch.maximum(self.min_log_dual, self.log_temperature))
            self.log_alpha_mean.data.copy_(
                torch.maximum(self.min_log_dual, self.log_alpha_mean))
            self.log_alpha_std.data.copy_(
                torch.maximum(self.min_log_dual, self.log_alpha_std))
            if self.action_penalization:
                self.log_penalty_temperature.data.copy_(torch.maximum(
                    self.min_log_dual, self.log_penalty_temperature))

            target_distributions = self.model.target_actor(observations)
            actions = target_distributions.sample((self.num_samples,))

            tiled_observations = updaters.tile(observations, self.num_samples)
            flat_observations = updaters.merge_first_two_dims(
                tiled_observations)
            flat_actions = updaters.merge_first_two_dims(actions)
            values = self.model.target_critic(flat_observations, flat_actions)
            values = values.view(self.num_samples, -1)

            assert isinstance(
                target_distributions, torch.distributions.normal.Normal)
            target_distributions = independent_normals(target_distributions)

        self.actor_optimizer.zero_grad()
        self.dual_optimizer.zero_grad()

        distributions = self.model.actor(observations)
        distributions = independent_normals(distributions)

        temperature = torch.nn.functional.softplus(
            self.log_temperature) + FLOAT_EPSILON
        alpha_mean = torch.nn.functional.softplus(
            self.log_alpha_mean) + FLOAT_EPSILON
        alpha_std = torch.nn.functional.softplus(
            self.log_alpha_std) + FLOAT_EPSILON
        weights, temperature_loss = weights_and_temperature_loss(
            values, self.epsilon, temperature)

        # Action penalization is quadratic beyond [-1, 1].
        if self.action_penalization:
            penalty_temperature = torch.nn.functional.softplus(
                self.log_penalty_temperature) + FLOAT_EPSILON
            diff_bounds = actions - torch.clamp(actions, -1, 1)
            action_bound_costs = -torch.norm(diff_bounds, dim=-1)
            penalty_weights, penalty_temperature_loss = \
                weights_and_temperature_loss(
                    action_bound_costs,
                    self.epsilon_penalty, penalty_temperature)
            weights += penalty_weights
            temperature_loss += penalty_temperature_loss

        # Decompose the policy into fixed-mean and fixed-std distributions.
        fixed_std_distribution = independent_normals(
            distributions.base_dist, target_distributions.base_dist)
        fixed_mean_distribution = independent_normals(
            target_distributions.base_dist, distributions.base_dist)

        # Compute the decomposed policy losses.
        policy_mean_losses = (fixed_std_distribution.base_dist.log_prob(
            actions).sum(dim=-1) * weights).sum(dim=0)
        policy_mean_loss = -(policy_mean_losses).mean()
        policy_std_losses = (fixed_mean_distribution.base_dist.log_prob(
            actions).sum(dim=-1) * weights).sum(dim=0)
        policy_std_loss = -policy_std_losses.mean()

        # Compute the decomposed KL between the target and online policies.
        if self.per_dim_constraining:
            kl_mean = torch.distributions.kl.kl_divergence(
                target_distributions.base_dist,
                fixed_std_distribution.base_dist)
            kl_std = torch.distributions.kl.kl_divergence(
                target_distributions.base_dist,
                fixed_mean_distribution.base_dist)
        else:
            kl_mean = torch.distributions.kl.kl_divergence(
                target_distributions, fixed_std_distribution)
            kl_std = torch.distributions.kl.kl_divergence(
                target_distributions, fixed_mean_distribution)

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

        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.actor_variables, self.gradient_clip)
            torch.nn.utils.clip_grad_norm_(
                self.dual_variables, self.gradient_clip)
        self.actor_optimizer.step()
        self.dual_optimizer.step()

        dual_variables = dict(
            temperature=temperature.detach(), alpha_mean=alpha_mean.detach(),
            alpha_std=alpha_std.detach())
        if self.action_penalization:
            dual_variables['penalty_temperature'] = \
                penalty_temperature.detach()

        return dict(
            policy_mean_loss=policy_mean_loss.detach(),
            policy_std_loss=policy_std_loss.detach(),
            kl_mean_loss=kl_mean_loss.detach(),
            kl_std_loss=kl_std_loss.detach(),
            alpha_mean_loss=alpha_mean_loss.detach(),
            alpha_std_loss=alpha_std_loss.detach(),
            temperature_loss=temperature_loss.detach(),
            **dual_variables)
