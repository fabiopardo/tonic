import copy

import torch

from tonic.torch import models  # noqa


class ActorCritic(torch.nn.Module):
    def __init__(
        self, actor, critic, observation_normalizer=None,
        return_normalizer=None
    ):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.observation_normalizer = observation_normalizer
        self.return_normalizer = return_normalizer

    def initialize(self, observation_space, action_space):
        if self.observation_normalizer:
            self.observation_normalizer.initialize(observation_space.shape)
        self.actor.initialize(
            observation_space, action_space, self.observation_normalizer)
        self.critic.initialize(
            observation_space, action_space, self.observation_normalizer,
            self.return_normalizer)


class ActorCriticWithTargets(torch.nn.Module):
    def __init__(
        self, actor, critic, observation_normalizer=None,
        return_normalizer=None, target_coeff=0.005
    ):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.target_actor = copy.deepcopy(actor)
        self.target_critic = copy.deepcopy(critic)
        self.observation_normalizer = observation_normalizer
        self.return_normalizer = return_normalizer
        self.target_coeff = target_coeff

    def initialize(self, observation_space, action_space):
        if self.observation_normalizer:
            self.observation_normalizer.initialize(observation_space.shape)
        self.actor.initialize(
            observation_space, action_space, self.observation_normalizer)
        self.critic.initialize(
            observation_space, action_space, self.observation_normalizer,
            self.return_normalizer)
        self.target_actor.initialize(
            observation_space, action_space, self.observation_normalizer)
        self.target_critic.initialize(
            observation_space, action_space, self.observation_normalizer,
            self.return_normalizer)
        self.online_variables = models.trainable_variables(self.actor)
        self.online_variables += models.trainable_variables(self.critic)
        self.target_variables = models.trainable_variables(self.target_actor)
        self.target_variables += models.trainable_variables(self.target_critic)
        for target in self.target_variables:
            target.requires_grad = False
        self.assign_targets()

    def assign_targets(self):
        for o, t in zip(self.online_variables, self.target_variables):
            t.data.copy_(o.data)

    def update_targets(self):
        with torch.no_grad():
            for o, t in zip(self.online_variables, self.target_variables):
                t.data.mul_(1 - self.target_coeff)
                t.data.add_(self.target_coeff * o.data)


class ActorTwinCriticWithTargets(torch.nn.Module):
    def __init__(
        self, actor, critic, observation_normalizer=None,
        return_normalizer=None, target_coeff=0.005
    ):
        super().__init__()
        self.actor = actor
        self.critic_1 = critic
        self.critic_2 = copy.deepcopy(critic)
        self.target_actor = copy.deepcopy(actor)
        self.target_critic_1 = copy.deepcopy(critic)
        self.target_critic_2 = copy.deepcopy(critic)
        self.observation_normalizer = observation_normalizer
        self.return_normalizer = return_normalizer
        self.target_coeff = target_coeff

    def initialize(self, observation_space, action_space):
        if self.observation_normalizer:
            self.observation_normalizer.initialize(observation_space.shape)
        self.actor.initialize(
            observation_space, action_space, self.observation_normalizer)
        self.critic_1.initialize(
            observation_space, action_space, self.observation_normalizer,
            self.return_normalizer)
        self.critic_2.initialize(
            observation_space, action_space, self.observation_normalizer,
            self.return_normalizer)
        self.target_actor.initialize(
            observation_space, action_space, self.observation_normalizer)
        self.target_critic_1.initialize(
            observation_space, action_space, self.observation_normalizer,
            self.return_normalizer)
        self.target_critic_2.initialize(
            observation_space, action_space, self.observation_normalizer,
            self.return_normalizer)
        self.online_variables = models.trainable_variables(self.actor)
        self.online_variables += models.trainable_variables(self.critic_1)
        self.online_variables += models.trainable_variables(self.critic_2)
        self.target_variables = models.trainable_variables(self.target_actor)
        self.target_variables += models.trainable_variables(
            self.target_critic_1)
        self.target_variables += models.trainable_variables(
            self.target_critic_2)
        for target in self.target_variables:
            target.requires_grad = False
        self.assign_targets()

    def assign_targets(self):
        for o, t in zip(self.online_variables, self.target_variables):
            t.data.copy_(o.data)

    def update_targets(self):
        with torch.no_grad():
            for o, t in zip(self.online_variables, self.target_variables):
                t.data.mul_(1 - self.target_coeff)
                t.data.add_(self.target_coeff * o.data)
