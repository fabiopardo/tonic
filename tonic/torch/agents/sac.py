import torch

from tonic import explorations  # noqa
from tonic.torch import agents, models, normalizers, updaters


def default_model():
    return models.ActorTwinCriticWithTargets(
        actor=models.Actor(
            encoder=models.ObservationEncoder(),
            torso=models.MLP((256, 256), torch.nn.ReLU),
            head=models.GaussianPolicyHead(
                loc_activation=torch.nn.Identity,
                distribution=models.SquashedMultivariateNormalDiag)),
        critic=models.Critic(
            encoder=models.ObservationActionEncoder(),
            torso=models.MLP((256, 256), torch.nn.ReLU),
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

    def _stochastic_actions(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        with torch.no_grad():
            return self.model.actor(observations).sample()

    def _policy(self, observations):
        return self._stochastic_actions(observations).numpy()

    def _greedy_actions(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        with torch.no_grad():
            return self.model.actor(observations).loc
