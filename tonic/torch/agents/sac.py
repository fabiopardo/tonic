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
        observation_normalizer=normalizers.MeanStd(),
        target_coeff=0.005)


def default_exploration():
    return explorations.NoActionNoise(start_steps=10000)


def default_actor_updater():
    return updaters.TwinCriticSoftDeterministicPolicyGradient(
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
        entropy_coeff=0.2)


def default_critic_updater():
    return updaters.TwinCriticSoftQLearning(
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
        entropy_coeff=0.2)


class SAC(agents.DDPG):
    '''Soft Actor-Critic.
    SAC: https://arxiv.org/pdf/1801.01290.pdf
    '''

    def __init__(
        self, model=None, replay=None, exploration=None, actor_updater=None,
        critic_updater=None
    ):
        model = model or default_model()
        exploration = exploration or default_exploration()
        actor_updater = actor_updater or default_actor_updater()
        critic_updater = critic_updater or default_critic_updater()
        super().__init__(
            model=model, replay=replay, exploration=exploration,
            actor_updater=actor_updater, critic_updater=critic_updater)

    def _stochastic_actions(self, observations):
        observations = torch.as_tensor(observations)
        with torch.no_grad():
            return self.model.actor(observations).sample()

    def _policy(self, observations):
        return self._stochastic_actions(observations).numpy()

    def _greedy_actions(self, observations):
        observations = torch.as_tensor(observations)
        with torch.no_grad():
            return self.model.actor(observations).loc
