import tensorflow as tf

from tonic import explorations
from tonic.tensorflow import agents, models, normalizers, updaters


def default_model():
    return models.ActorTwinCriticWithTargets(
        actor=models.Actor(
            encoder=models.ObservationEncoder(),
            torso=models.MLP((256, 256), 'relu'),
            head=models.GaussianPolicyHead(
                loc_activation=None,
                distribution=models.SquashedMultivariateNormalDiag)),
        critic=models.Critic(
            encoder=models.ObservationActionEncoder(),
            torso=models.MLP((256, 256), 'relu'),
            head=models.ValueHead()),
        observation_normalizer=normalizers.MeanStd())


class SAC(agents.DDPG):
    '''Soft Actor-Critic.
    SAC: https://arxiv.org/pdf/1801.01290.pdf
    '''

    def __init__(
        self, model=None, replay=None, exploration=None, actor_updater=None,
        critic_updater=None
    ):
        model = model or default_model()
        exploration = exploration or explorations.NoActionNoise()
        actor_updater = actor_updater or \
            updaters.TwinCriticSoftDeterministicPolicyGradient()
        critic_updater = critic_updater or updaters.TwinCriticSoftQLearning()
        super().__init__(
            model=model, replay=replay, exploration=exploration,
            actor_updater=actor_updater, critic_updater=critic_updater)

    @tf.function
    def _stochastic_actions(self, observations):
        return self.model.actor(observations).sample()

    def _policy(self, observations):
        return self._stochastic_actions(observations).numpy()

    @tf.function
    def _greedy_actions(self, observations):
        return self.model.actor(observations).mode()
