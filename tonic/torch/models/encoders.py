import torch


class ObservationEncoder(torch.nn.Module):
    def initialize(
        self, observation_space, action_space=None,
        observation_normalizer=None,
    ):
        self.observation_normalizer = observation_normalizer
        observation_size = observation_space.shape[0]
        return observation_size

    def forward(self, observations):
        if self.observation_normalizer:
            observations = self.observation_normalizer(observations)
        return observations


class ObservationActionEncoder(torch.nn.Module):
    def initialize(
        self, observation_space, action_space, observation_normalizer=None
    ):
        self.observation_normalizer = observation_normalizer
        observation_size = observation_space.shape[0]
        action_size = action_space.shape[0]
        return observation_size + action_size

    def forward(self, observations, actions):
        if self.observation_normalizer:
            observations = self.observation_normalizer(observations)
        return torch.cat([observations, actions], dim=-1)
