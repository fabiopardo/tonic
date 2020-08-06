import torch

from tonic import explorations, logger, replays  # noqa
from tonic.torch import agents, models, normalizers, updaters


def default_model():
    return models.ActorCriticWithTargets(
        actor=models.Actor(
            encoder=models.ObservationEncoder(),
            torso=models.MLP((256, 256), torch.nn.ReLU),
            head=models.DeterministicPolicyHead()),
        critic=models.Critic(
            encoder=models.ObservationActionEncoder(),
            torso=models.MLP((256, 256), torch.nn.ReLU),
            head=models.ValueHead()),
        observation_normalizer=normalizers.MeanStd(),
        target_coeff=0.005)


def default_replay():
    return replays.Buffer(
        size=int(1e6), batch_iterations=50, batch_size=100,
        discount_factor=0.98, steps_before_batches=5000,
        steps_between_batches=50)


def default_exploration():
    return explorations.NormalActionNoise(scale=0.1, start_steps=10000)


def default_actor_updater():
    return updaters.DeterministicPolicyGradient(
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3))


def default_critic_updater():
    return updaters.DeterministicQLearning(
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3))


class DDPG(agents.TorchAgent):
    '''Deep Deterministic Policy Gradient.
    DDPG: https://arxiv.org/pdf/1509.02971.pdf
    '''

    def __init__(
        self, model=None, replay=None, exploration=None, actor_updater=None,
        critic_updater=None
    ):
        self.model = model or default_model()
        self.replay = replay or default_replay()
        self.exploration = exploration or default_exploration()
        self.actor_updater = actor_updater or default_actor_updater()
        self.critic_updater = critic_updater or default_critic_updater()

    def initialize(self, observation_space, action_space, seed=None):
        super().initialize(seed=seed)
        self.model.initialize(observation_space, action_space)
        self.replay.initialize(seed)
        self.exploration.initialize(self._policy, action_space, seed)
        self.actor_updater.initialize(self.model)
        self.critic_updater.initialize(self.model)
        self.steps = 0

    def step(self, observations):
        # Get actions from the actor and exploration method.
        actions = self.exploration(observations, self.steps)

        # Keep some values for the next update.
        self.last_observations = observations.copy()
        self.last_actions = actions.copy()
        self.steps += len(observations)

        return actions

    def test_step(self, observations):
        # Greedy actions for testing.
        return self._greedy_actions(observations).numpy()

    def update(self, observations, rewards, resets, terminations):
        # Store the last transitions in the replay.
        self.replay.store(
            observations=self.last_observations, actions=self.last_actions,
            next_observations=observations, rewards=rewards,
            terminations=terminations)

        # Prepare to update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.record(self.last_observations)
        if self.model.return_normalizer:
            self.model.return_normalizer.record(rewards)

        # Update the model if the replay is ready.
        if self.replay.ready():
            self._update()

        self.exploration.update(resets)

    def _greedy_actions(self, observations):
        observations = torch.as_tensor(observations)
        with torch.no_grad():
            return self.model.actor(observations)

    def _policy(self, observations):
        return self._greedy_actions(observations).numpy()

    def _update(self):
        keys = ('observations', 'actions', 'next_observations', 'rewards',
                'discounts')

        # Update both the actor and the critic multiple times.
        for batch in self.replay.get(*keys):
            infos = self._update_actor_critic(**batch)

            for key in infos:
                for k, v in infos[key].items():
                    logger.store(key + '/' + k, v.numpy())

        # Update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.update()
        if self.model.return_normalizer:
            self.model.return_normalizer.update()

    def _update_actor_critic(
        self, observations, actions, next_observations, rewards, discounts
    ):
        critic_infos = self.critic_updater(
            observations, actions, next_observations, rewards, discounts)
        actor_infos = self.actor_updater(observations)
        self.model.update_targets()
        return dict(critic=critic_infos, actor=actor_infos)
