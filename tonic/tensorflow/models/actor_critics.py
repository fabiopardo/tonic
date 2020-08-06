import copy

import tensorflow as tf


class ActorCritic(tf.keras.Model):
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
        dummy_observations = tf.zeros((1,) + observation_space.shape)
        self.actor(dummy_observations)
        self.critic(dummy_observations)


class ActorCriticWithTargets(tf.keras.Model):
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
        dummy_observations = tf.zeros((1,) + observation_space.shape)
        dummy_actions = tf.zeros((1,) + action_space.shape)
        self.actor(dummy_observations)
        self.critic(dummy_observations, dummy_actions)
        self.target_actor(dummy_observations)
        self.target_critic(dummy_observations, dummy_actions)
        self.online_variables = (
            self.actor.trainable_variables +
            self.critic.trainable_variables)
        self.target_variables = (
            self.target_actor.trainable_variables +
            self.target_critic.trainable_variables)
        self.assign_targets()

    def assign_targets(self):
        for o, t in zip(self.online_variables, self.target_variables):
            t.assign(o)

    def update_targets(self):
        for o, t in zip(self.online_variables, self.target_variables):
            t.assign((1 - self.target_coeff) * t + self.target_coeff * o)


class ActorTwinCriticWithTargets(tf.keras.Model):
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
        dummy_observations = tf.zeros((1,) + observation_space.shape)
        dummy_actions = tf.zeros((1,) + action_space.shape)
        self.actor(dummy_observations)
        self.critic_1(dummy_observations, dummy_actions)
        self.critic_2(dummy_observations, dummy_actions)
        self.target_actor(dummy_observations)
        self.target_critic_1(dummy_observations, dummy_actions)
        self.target_critic_2(dummy_observations, dummy_actions)
        self.online_variables = (
            self.actor.trainable_variables +
            self.critic_1.trainable_variables +
            self.critic_2.trainable_variables)
        self.target_variables = (
            self.target_actor.trainable_variables +
            self.target_critic_1.trainable_variables +
            self.target_critic_2.trainable_variables)
        self.assign_targets()

    def assign_targets(self):
        for o, t in zip(self.online_variables, self.target_variables):
            t.assign(o)

    def update_targets(self):
        for o, t in zip(self.online_variables, self.target_variables):
            t.assign((1 - self.target_coeff) * t + self.target_coeff * o)
