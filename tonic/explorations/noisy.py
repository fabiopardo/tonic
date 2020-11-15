'''Non-differentiable noisy exploration methods.'''

import numpy as np


class NoActionNoise:
    def __init__(self, start_steps=20000):
        self.start_steps = start_steps

    def initialize(self, policy, action_space, seed=None):
        self.policy = policy
        self.action_size = action_space.shape[0]
        self.np_random = np.random.RandomState(seed)

    def __call__(self, observations, steps):
        if steps > self.start_steps:
            actions = self.policy(observations)
            actions = np.clip(actions, -1, 1)
        else:
            shape = (len(observations), self.action_size)
            actions = self.np_random.uniform(-1, 1, shape)
        return actions

    def update(self, resets):
        pass


class NormalActionNoise:
    def __init__(self, scale=0.1, start_steps=20000):
        self.scale = scale
        self.start_steps = start_steps

    def initialize(self, policy, action_space, seed=None):
        self.policy = policy
        self.action_size = action_space.shape[0]
        self.np_random = np.random.RandomState(seed)

    def __call__(self, observations, steps):
        if steps > self.start_steps:
            actions = self.policy(observations)
            noises = self.scale * self.np_random.normal(size=actions.shape)
            actions = (actions + noises).astype(np.float32)
            actions = np.clip(actions, -1, 1)
        else:
            shape = (len(observations), self.action_size)
            actions = self.np_random.uniform(-1, 1, shape)
        return actions

    def update(self, resets):
        pass


class OrnsteinUhlenbeckActionNoise:
    def __init__(
        self, scale=0.1, clip=2, theta=.15, dt=1e-2, start_steps=20000
    ):
        self.scale = scale
        self.clip = clip
        self.theta = theta
        self.dt = dt
        self.start_steps = start_steps

    def initialize(self, policy, action_space, seed=None):
        self.policy = policy
        self.action_size = action_space.shape[0]
        self.np_random = np.random.RandomState(seed)
        self.noises = None

    def __call__(self, observations, steps):
        if steps > self.start_steps:
            actions = self.policy(observations)

            if self.noises is None:
                self.noises = np.zeros_like(actions)
            noises = self.np_random.normal(size=actions.shape)
            noises = np.clip(noises, -self.clip, self.clip)
            self.noises -= self.theta * self.noises * self.dt
            self.noises += self.scale * np.sqrt(self.dt) * noises
            actions = (actions + self.noises).astype(np.float32)
            actions = np.clip(actions, -1, 1)
        else:
            shape = (len(observations), self.action_size)
            actions = self.np_random.uniform(-1, 1, shape)
        return actions

    def update(self, resets):
        if self.noises is not None:
            self.noises *= (1. - resets)[:, None]
