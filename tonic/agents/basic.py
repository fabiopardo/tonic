'''Some basic non-learning agents used for example for debugging.'''

import numpy as np

from tonic import agents


class NormalRandom(agents.Agent):
    '''Random agent producing actions from normal distributions.'''

    def __init__(self, loc=0, scale=1):
        self.loc = loc
        self.scale = scale

    def initialize(self, observation_space, action_space, seed=None):
        self.action_size = action_space.shape[0]
        self.np_random = np.random.RandomState(seed)

    def step(self, observations, steps):
        return self._policy(observations)

    def test_step(self, observations, steps):
        return self._policy(observations)

    def _policy(self, observations):
        batch_size = len(observations)
        shape = (batch_size, self.action_size)
        return self.np_random.normal(self.loc, self.scale, shape)


class UniformRandom(agents.Agent):
    '''Random agent producing actions from uniform distributions.'''

    def initialize(self, observation_space, action_space, seed=None):
        self.action_size = action_space.shape[0]
        self.np_random = np.random.RandomState(seed)

    def step(self, observations, steps):
        return self._policy(observations)

    def test_step(self, observations, steps):
        return self._policy(observations)

    def _policy(self, observations):
        batch_size = len(observations)
        shape = (batch_size, self.action_size)
        return self.np_random.uniform(-1, 1, shape)


class OrnsteinUhlenbeck(agents.Agent):
    '''Random agent producing correlated actions from an OU process.'''

    def __init__(self, scale=0.2, clip=2, theta=.15, dt=1e-2):
        self.scale = scale
        self.clip = clip
        self.theta = theta
        self.dt = dt

    def initialize(self, observation_space, action_space, seed=None):
        self.action_size = action_space.shape[0]
        self.np_random = np.random.RandomState(seed)
        self.train_actions = None
        self.test_actions = None

    def step(self, observations, steps):
        return self._train_policy(observations)

    def test_step(self, observations, steps):
        return self._test_policy(observations)

    def _train_policy(self, observations):
        if self.train_actions is None:
            shape = (len(observations), self.action_size)
            self.train_actions = np.zeros(shape)
        self.train_actions = self._next_actions(self.train_actions)
        return self.train_actions

    def _test_policy(self, observations):
        if self.test_actions is None:
            shape = (len(observations), self.action_size)
            self.test_actions = np.zeros(shape)
        self.test_actions = self._next_actions(self.test_actions)
        return self.test_actions

    def _next_actions(self, actions):
        noises = self.np_random.normal(size=actions.shape)
        noises = np.clip(noises, -self.clip, self.clip)
        next_actions = (1 - self.theta * self.dt) * actions
        next_actions += self.scale * np.sqrt(self.dt) * noises
        next_actions = np.clip(next_actions, -1, 1)
        return next_actions

    def update(self, observations, rewards, resets, terminations, steps):
        self.train_actions *= (1. - resets)[:, None]

    def test_update(self, observations, rewards, resets, terminations, steps):
        self.test_actions *= (1. - resets)[:, None]


class Constant(agents.Agent):
    '''Agent producing a unique constant action.'''

    def __init__(self, constant=0):
        self.constant = constant

    def initialize(self, observation_space, action_space, seed=None):
        self.action_size = action_space.shape[0]

    def step(self, observations, steps):
        return self._policy(observations)

    def test_step(self, observations, steps):
        return self._policy(observations)

    def _policy(self, observations):
        shape = (len(observations), self.action_size)
        return np.full(shape, self.constant)
