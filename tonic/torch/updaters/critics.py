import torch

from tonic.torch import models  # noqa


class VRegression:
    def __init__(self, loss=None, optimizer=None, gradient_clip=0):
        self.loss = loss or torch.nn.MSELoss()
        self.optimizer = optimizer or (lambda params: torch.optim.Adam(
            params, lr=1e-3))
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = models.trainable_variables(self.model.critic)
        self.optimizer = self.optimizer(self.variables)

    def __call__(self, observations, returns):
        observations = torch.as_tensor(observations)
        returns = torch.as_tensor(returns)

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
        self.optimizer = optimizer or (lambda params: torch.optim.Adam(
            params, lr=1e-3))
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = models.trainable_variables(self.model.critic)
        self.optimizer = self.optimizer(self.variables)

    def __call__(self, observations, actions, returns):
        observations = torch.as_tensor(observations)
        actions = torch.as_tensor(observations)
        returns = torch.as_tensor(returns)

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
        self.optimizer = optimizer or (lambda params: torch.optim.Adam(
            params, lr=1e-3))
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = models.trainable_variables(self.model.critic)
        self.optimizer = self.optimizer(self.variables)

    def __call__(
        self, observations, actions, next_observations, rewards, discounts
    ):
        observations = torch.as_tensor(observations)
        actions = torch.as_tensor(actions)
        next_observations = torch.as_tensor(next_observations)
        rewards = torch.as_tensor(rewards)
        discounts = torch.as_tensor(discounts)

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
        self.optimizer = optimizer or (lambda params: torch.optim.Adam(
            params, lr=1e-3))
        self.target_action_noise = target_action_noise or TargetActionNoise(
            scale=0.2, clip=0.5)
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
        observations = torch.as_tensor(observations)
        actions = torch.as_tensor(actions)
        next_observations = torch.as_tensor(next_observations)
        rewards = torch.as_tensor(rewards)
        discounts = torch.as_tensor(discounts)

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
        self.optimizer = optimizer or (lambda params: torch.optim.Adam(
            params, lr=3e-4))
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
        observations = torch.as_tensor(observations)
        actions = torch.as_tensor(actions)
        next_observations = torch.as_tensor(next_observations)
        rewards = torch.as_tensor(rewards)
        discounts = torch.as_tensor(discounts)

        with torch.no_grad():
            next_distributions = self.model.actor(next_observations)
            if hasattr(next_distributions, 'rsample_with_log_prob'):
                outs = next_distributions.rsample_with_log_prob()
                next_actions, next_log_probs = outs
            else:
                next_actions = next_distributions.rsample()
                next_log_probs = next_distributions.log_prob(next_actions)
            next_log_probs = next_log_probs.sum(axis=-1)
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
