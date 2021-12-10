import tensorflow as tf

from tonic import logger
from tonic.tensorflow import agents, updaters


class TRPO(agents.A2C):
    '''Trust Region Policy Optimization.
    TRPO: https://arxiv.org/pdf/1502.05477.pdf
    '''

    def __init__(
        self, model=None, replay=None, actor_updater=None, critic_updater=None
    ):
        actor_updater = actor_updater or updaters.TrustRegionPolicyGradient()
        super().__init__(
            model=model, replay=replay, actor_updater=actor_updater,
            critic_updater=critic_updater)

    def step(self, observations, steps):
        # Sample actions and get their log-probabilities for training.
        actions, log_probs, locs, scales = self._step(observations)
        actions = actions.numpy()
        log_probs = log_probs.numpy()
        locs = locs.numpy()
        scales = scales.numpy()

        # Keep some values for the next update.
        self.last_observations = observations.copy()
        self.last_actions = actions.copy()
        self.last_log_probs = log_probs.copy()
        self.last_locs = locs.copy()
        self.last_scales = scales.copy()

        return actions

    def update(self, observations, rewards, resets, terminations, steps):
        # Store the last transitions in the replay.
        self.replay.store(
            observations=self.last_observations, actions=self.last_actions,
            next_observations=observations, rewards=rewards, resets=resets,
            terminations=terminations, log_probs=self.last_log_probs,
            locs=self.last_locs, scales=self.last_scales)

        # Prepare to update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.record(self.last_observations)
        if self.model.return_normalizer:
            self.model.return_normalizer.record(rewards)

        # Update the model if the replay is ready.
        if self.replay.ready():
            self._update()

    @tf.function
    def _step(self, observations):
        distributions = self.model.actor(observations)
        if hasattr(distributions, 'sample_with_log_prob'):
            actions, log_probs = distributions.sample_with_log_prob()
        else:
            actions = distributions.sample()
            log_probs = distributions.log_prob(actions)
        locs = distributions.loc
        scales = distributions.stddev()
        return actions, log_probs, locs, scales

    def _update(self):
        # Compute the lambda-returns.
        batch = self.replay.get_full('observations', 'next_observations')
        values, next_values = self._evaluate(**batch)
        values, next_values = values.numpy(), next_values.numpy()
        self.replay.compute_returns(values, next_values)

        keys = ('observations', 'actions', 'log_probs', 'locs', 'scales',
                'advantages')
        batch = self.replay.get_full(*keys)
        infos = self.actor_updater(**batch)
        for k, v in infos.items():
            logger.store('actor/' + k, v.numpy())

        critic_iterations = 0
        for batch in self.replay.get('observations', 'returns'):
            infos = self.critic_updater(**batch)
            critic_iterations += 1
            for k, v in infos.items():
                logger.store('critic/' + k, v.numpy())
        logger.store('critic/iterations', critic_iterations)

        # Update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.update()
        if self.model.return_normalizer:
            self.model.return_normalizer.update()
