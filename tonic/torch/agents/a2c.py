import torch

from tonic import logger, replays  # noqa
from tonic.torch import agents, models, normalizers, updaters


def default_model():
    return models.ActorCritic(
        actor=models.Actor(
            encoder=models.ObservationEncoder(),
            torso=models.MLP((64, 64), torch.nn.Tanh),
            head=models.DetachedScaleGaussianPolicyHead()),
        critic=models.Critic(
            encoder=models.ObservationEncoder(),
            torso=models.MLP((64, 64), torch.nn.Tanh),
            head=models.ValueHead()),
        observation_normalizer=normalizers.MeanStd())


class A2C(agents.Agent):
    '''Advantage Actor Critic (aka Vanilla Policy Gradient).
    A3C: https://arxiv.org/pdf/1602.01783.pdf
    '''

    def __init__(
        self, model=None, replay=None, actor_updater=None, critic_updater=None
    ):
        self.model = model or default_model()
        self.replay = replay or replays.Segment()
        self.actor_updater = actor_updater or \
            updaters.StochasticPolicyGradient()
        self.critic_updater = critic_updater or updaters.VRegression()

    def initialize(self, observation_space, action_space, seed=None):
        super().initialize(seed=seed)
        self.model.initialize(observation_space, action_space)
        self.replay.initialize(seed)
        self.actor_updater.initialize(self.model)
        self.critic_updater.initialize(self.model)

    def step(self, observations, steps):
        # Sample actions and get their log-probabilities for training.
        actions, log_probs = self._step(observations)
        actions = actions.numpy()
        log_probs = log_probs.numpy()

        # Keep some values for the next update.
        self.last_observations = observations.copy()
        self.last_actions = actions.copy()
        self.last_log_probs = log_probs.copy()

        return actions

    def test_step(self, observations, steps):
        # Sample actions for testing.
        return self._test_step(observations).numpy()

    def update(self, observations, rewards, resets, terminations, steps):
        # Store the last transitions in the replay.
        self.replay.store(
            observations=self.last_observations, actions=self.last_actions,
            next_observations=observations, rewards=rewards, resets=resets,
            terminations=terminations, log_probs=self.last_log_probs)

        # Prepare to update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.record(self.last_observations)
        if self.model.return_normalizer:
            self.model.return_normalizer.record(rewards)

        # Update the model if the replay is ready.
        if self.replay.ready():
            self._update()

    def _step(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        with torch.no_grad():
            distributions = self.model.actor(observations)
            if hasattr(distributions, 'sample_with_log_prob'):
                actions, log_probs = distributions.sample_with_log_prob()
            else:
                actions = distributions.sample()
                log_probs = distributions.log_prob(actions)
            log_probs = log_probs.sum(dim=-1)
        return actions, log_probs

    def _test_step(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        with torch.no_grad():
            return self.model.actor(observations).sample()

    def _evaluate(self, observations, next_observations):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        next_observations = torch.as_tensor(
            next_observations, dtype=torch.float32)
        with torch.no_grad():
            values = self.model.critic(observations)
            next_values = self.model.critic(next_observations)
        return values, next_values

    def _update(self):
        # Compute the lambda-returns.
        batch = self.replay.get_full('observations', 'next_observations')
        values, next_values = self._evaluate(**batch)
        values, next_values = values.numpy(), next_values.numpy()
        self.replay.compute_returns(values, next_values)

        # Update the actor once.
        keys = 'observations', 'actions', 'advantages', 'log_probs'
        batch = self.replay.get_full(*keys)
        batch = {k: torch.as_tensor(v) for k, v in batch.items()}
        infos = self.actor_updater(**batch)
        for k, v in infos.items():
            logger.store('actor/' + k, v.numpy())

        # Update the critic multiple times.
        for batch in self.replay.get('observations', 'returns'):
            batch = {k: torch.as_tensor(v) for k, v in batch.items()}
            infos = self.critic_updater(**batch)
            for k, v in infos.items():
                logger.store('critic/' + k, v.numpy())

        # Update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.update()
        if self.model.return_normalizer:
            self.model.return_normalizer.update()
