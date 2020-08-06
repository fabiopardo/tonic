import numpy as np
import torch

from tonic.torch import models, updaters  # noqa


class StochasticPolicyGradient:
    def __init__(self, optimizer=None, entropy_coeff=0, gradient_clip=0):
        self.optimizer = optimizer or (lambda params: torch.optim.Adam(
            params, lr=3e-4))
        self.entropy_coeff = entropy_coeff
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = models.trainable_variables(self.model.actor)
        self.optimizer = self.optimizer(self.variables)

    def __call__(self, observations, actions, advantages, log_probs):
        observations = torch.as_tensor(observations)

        if np.all(advantages == 0.):
            loss = torch.as_tensor(0.)
            kl = torch.as_tensor(0.)
            with torch.no_grad():
                distributions = self.model.actor(observations)
                entropy = distributions.entropy().mean()
                std = distributions.stddev.mean()

        else:
            actions = torch.as_tensor(actions)
            advantages = torch.as_tensor(advantages)
            log_probs = torch.as_tensor(log_probs)

            self.optimizer.zero_grad()
            distributions = self.model.actor(observations)
            new_log_probs = distributions.log_prob(actions).sum(axis=-1)
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
        self.optimizer = optimizer or (lambda params: torch.optim.Adam(
            params, lr=3e-4))
        self.ratio_clip = ratio_clip
        self.kl_threshold = kl_threshold
        self.entropy_coeff = entropy_coeff
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = models.trainable_variables(self.model.actor)
        self.optimizer = self.optimizer(self.variables)

    def __call__(self, observations, actions, advantages, log_probs):
        observations = torch.as_tensor(observations)

        if np.all(advantages == 0.):
            loss = torch.as_tensor(0.)
            kl = torch.as_tensor(0.)
            clip_fraction = torch.as_tensor(0.)
            with torch.no_grad():
                distributions = self.model.actor(observations)
                entropy = distributions.entropy().mean()
                std = distributions.stddev.mean()

        else:
            actions = torch.as_tensor(actions)
            advantages = torch.as_tensor(advantages)
            log_probs = torch.as_tensor(log_probs)

            self.optimizer.zero_grad()
            distributions = self.model.actor(observations)
            new_log_probs = distributions.log_prob(actions).sum(axis=-1)
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
        self.optimizer = optimizer or updaters.ConjugateGradient(
            constraint_threshold=0.01,
            damping_coefficient=0.1,
            conjugate_gradient_steps=10,
            backtrack_steps=10,
            backtrack_coefficient=0.8)
        self.entropy_coeff = entropy_coeff

    def initialize(self, model):
        self.model = model
        self.variables = models.trainable_variables(self.model.actor)

    def __call__(
        self, observations, actions, log_probs, locs, scales, advantages
    ):
        if np.all(advantages == 0.):
            kl = torch.as_tensor(0.)
            loss = torch.as_tensor(0.)
            steps = torch.as_tensor(0)

        else:
            observations = torch.as_tensor(observations)
            actions = torch.as_tensor(actions)
            log_probs = torch.as_tensor(log_probs)
            locs = torch.as_tensor(locs)
            scales = torch.as_tensor(scales)
            advantages = torch.as_tensor(advantages)

            kl, loss, steps = self.optimizer.optimize(
                loss_function=lambda: self._loss(
                    observations, actions, log_probs, advantages),
                constraint_function=lambda: self._kl(
                    observations, locs, scales),
                variables=self.variables)

        return dict(loss=loss, kl=kl, backtrack_steps=steps)

    def _loss(self, observations, actions, old_log_probs, advantages):
        distributions = self.model.actor(observations)
        log_probs = distributions.log_prob(actions).sum(axis=-1)
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
        self.optimizer = optimizer or (lambda params: torch.optim.Adam(
            params, lr=1e-3))
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = models.trainable_variables(self.model.actor)
        self.optimizer = self.optimizer(self.variables)

    def __call__(self, observations):
        observations = torch.as_tensor(observations)
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


class TwinCriticSoftDeterministicPolicyGradient:
    def __init__(self, optimizer=None, entropy_coeff=0.2, gradient_clip=0):
        self.optimizer = optimizer or (lambda params: torch.optim.Adam(
            params, lr=3e-4))
        self.entropy_coeff = entropy_coeff
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = models.trainable_variables(self.model.actor)
        self.optimizer = self.optimizer(self.variables)

    def __call__(self, observations):
        observations = torch.as_tensor(observations)
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
        log_probs = log_probs.sum(axis=-1)
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
