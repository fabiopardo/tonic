import torch

from tonic.torch import models, updaters  # noqa


class VRegression:
    def __init__(self, loss=None, optimizer=None, gradient_clip=0):
        self.loss = loss or torch.nn.MSELoss()
        self.optimizer = optimizer or (
            lambda params: torch.optim.Adam(params, lr=1e-3))
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = models.trainable_variables(self.model.critic)
        self.optimizer = self.optimizer(self.variables)

    def __call__(self, observations, returns):
        self.optimizer.zero_grad()
        values = self.model.critic(observations)
        loss = self.loss(values, returns)

        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        self.optimizer.step()

        return dict(loss=loss.detach(), v=values.detach())


class QRegression:
    def __init__(self, loss=None, optimizer=None, gradient_clip=0):
        self.loss = loss or torch.nn.MSELoss()
        self.optimizer = optimizer or (
            lambda params: torch.optim.Adam(params, lr=1e-3))
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = models.trainable_variables(self.model.critic)
        self.optimizer = self.optimizer(self.variables)

    def __call__(self, observations, actions, returns):
        self.optimizer.zero_grad()
        values = self.model.critic(observations, actions)
        loss = self.loss(values, returns)

        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        self.optimizer.step()

        return dict(loss=loss.detach(), q=values.detach())


class DeterministicQLearning:
    def __init__(self, loss=None, optimizer=None, gradient_clip=0):
        self.loss = loss or torch.nn.MSELoss()
        self.optimizer = optimizer or (
            lambda params: torch.optim.Adam(params, lr=1e-3))
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = models.trainable_variables(self.model.critic)
        self.optimizer = self.optimizer(self.variables)

    def __call__(
        self, observations, actions, next_observations, rewards, discounts
    ):
        with torch.no_grad():
            next_actions = self.model.target_actor(next_observations)
            next_values = self.model.target_critic(
                next_observations, next_actions)
            returns = rewards + discounts * next_values

        self.optimizer.zero_grad()
        values = self.model.critic(observations, actions)
        loss = self.loss(values, returns)

        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        self.optimizer.step()

        return dict(loss=loss.detach(), q=values.detach())


class DistributionalDeterministicQLearning:
    def __init__(self, optimizer=None, gradient_clip=0):
        self.optimizer = optimizer or (
            lambda params: torch.optim.Adam(params, lr=1e-3))
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = models.trainable_variables(self.model.critic)
        self.optimizer = self.optimizer(self.variables)

    def __call__(
        self, observations, actions, next_observations, rewards, discounts
    ):
        with torch.no_grad():
            next_actions = self.model.target_actor(next_observations)
            next_value_distributions = self.model.target_critic(
                next_observations, next_actions)
            values = next_value_distributions.values
            returns = rewards[:, None] + discounts[:, None] * values
            targets = next_value_distributions.project(returns)

        self.optimizer.zero_grad()
        value_distributions = self.model.critic(observations, actions)
        log_probabilities = torch.nn.functional.log_softmax(
            value_distributions.logits, dim=-1)
        loss = -(targets * log_probabilities).sum(dim=-1).mean()

        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        self.optimizer.step()

        return dict(loss=loss.detach())


class TargetActionNoise:
    def __init__(self, scale=0.2, clip=0.5):
        self.scale = scale
        self.clip = clip

    def __call__(self, actions):
        noises = self.scale * torch.randn_like(actions)
        noises = torch.clamp(noises, -self.clip, self.clip)
        actions = actions + noises
        return torch.clamp(actions, -1, 1)


class TwinCriticDeterministicQLearning:
    def __init__(
        self, loss=None, optimizer=None, target_action_noise=None,
        gradient_clip=0
    ):
        self.loss = loss or torch.nn.MSELoss()
        self.optimizer = optimizer or (
            lambda params: torch.optim.Adam(params, lr=1e-3))
        self.target_action_noise = target_action_noise or \
            TargetActionNoise(scale=0.2, clip=0.5)
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        variables_1 = models.trainable_variables(self.model.critic_1)
        variables_2 = models.trainable_variables(self.model.critic_2)
        self.variables = variables_1 + variables_2
        self.optimizer = self.optimizer(self.variables)

    def __call__(
        self, observations, actions, next_observations, rewards, discounts
    ):
        with torch.no_grad():
            next_actions = self.model.target_actor(next_observations)
            next_actions = self.target_action_noise(next_actions)
            next_values_1 = self.model.target_critic_1(
                next_observations, next_actions)
            next_values_2 = self.model.target_critic_2(
                next_observations, next_actions)
            next_values = torch.min(next_values_1, next_values_2)
            returns = rewards + discounts * next_values

        self.optimizer.zero_grad()
        values_1 = self.model.critic_1(observations, actions)
        values_2 = self.model.critic_2(observations, actions)
        loss_1 = self.loss(values_1, returns)
        loss_2 = self.loss(values_2, returns)
        loss = loss_1 + loss_2

        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        self.optimizer.step()

        return dict(
            loss=loss.detach(), q1=values_1.detach(), q2=values_2.detach())


class TwinCriticSoftQLearning:
    def __init__(
        self, loss=None, optimizer=None, entropy_coeff=0.2, gradient_clip=0
    ):
        self.loss = loss or torch.nn.MSELoss()
        self.optimizer = optimizer or (
            lambda params: torch.optim.Adam(params, lr=3e-4))
        self.entropy_coeff = entropy_coeff
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        variables_1 = models.trainable_variables(self.model.critic_1)
        variables_2 = models.trainable_variables(self.model.critic_2)
        self.variables = variables_1 + variables_2
        self.optimizer = self.optimizer(self.variables)

    def __call__(
        self, observations, actions, next_observations, rewards, discounts
    ):
        with torch.no_grad():
            next_distributions = self.model.actor(next_observations)
            if hasattr(next_distributions, 'rsample_with_log_prob'):
                outs = next_distributions.rsample_with_log_prob()
                next_actions, next_log_probs = outs
            else:
                next_actions = next_distributions.rsample()
                next_log_probs = next_distributions.log_prob(next_actions)
            next_log_probs = next_log_probs.sum(dim=-1)
            next_values_1 = self.model.target_critic_1(
                next_observations, next_actions)
            next_values_2 = self.model.target_critic_2(
                next_observations, next_actions)
            next_values = torch.min(next_values_1, next_values_2)
            returns = rewards + discounts * (
                next_values - self.entropy_coeff * next_log_probs)

        self.optimizer.zero_grad()
        values_1 = self.model.critic_1(observations, actions)
        values_2 = self.model.critic_2(observations, actions)
        loss_1 = self.loss(values_1, returns)
        loss_2 = self.loss(values_2, returns)
        loss = loss_1 + loss_2

        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        self.optimizer.step()

        return dict(
            loss=loss.detach(), q1=values_1.detach(), q2=values_2.detach())


class ExpectedSARSA:
    def __init__(
        self, num_samples=20, loss=None, optimizer=None, gradient_clip=0
    ):
        self.num_samples = num_samples
        self.loss = loss or torch.nn.MSELoss()
        self.optimizer = optimizer or (
            lambda params: torch.optim.Adam(params, lr=3e-4))
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = models.trainable_variables(self.model.critic)
        self.optimizer = self.optimizer(self.variables)

    def __call__(
        self, observations, actions, next_observations, rewards, discounts
    ):
        # Approximate the expected next values.
        with torch.no_grad():
            next_target_distributions = self.model.target_actor(
                next_observations)
            next_actions = next_target_distributions.rsample(
                (self.num_samples,))
            next_actions = updaters.merge_first_two_dims(next_actions)
            next_observations = updaters.tile(
                next_observations, self.num_samples)
            next_observations = updaters.merge_first_two_dims(
                next_observations)
            next_values = self.model.target_critic(
                next_observations, next_actions)
            next_values = next_values.view(self.num_samples, -1)
            next_values = next_values.mean(dim=0)
            returns = rewards + discounts * next_values

        self.optimizer.zero_grad()
        values = self.model.critic(observations, actions)
        loss = self.loss(returns, values)

        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        self.optimizer.step()

        return dict(loss=loss.detach(), q=values.detach())
