from tonic import replays
from tonic.tensorflow import agents, models, normalizers, updaters


def default_model():
    return models.ActorCriticWithTargets(
        actor=models.Actor(
            encoder=models.ObservationEncoder(),
            torso=models.MLP((256, 256), 'relu'),
            head=models.DeterministicPolicyHead()),
        critic=models.Critic(
            encoder=models.ObservationActionEncoder(),
            torso=models.MLP((256, 256), 'relu'),
            # These values are for the control suite with 0.99 discount.
            head=models.DistributionalValueHead(-150., 150., 51)),
        observation_normalizer=normalizers.MeanStd())


class D4PG(agents.DDPG):
    '''Distributed Distributional Deterministic Policy Gradients.
    D4PG: https://arxiv.org/pdf/1804.08617.pdf
    '''

    def __init__(
        self, model=None, replay=None, exploration=None, actor_updater=None,
        critic_updater=None
    ):
        model = model or default_model()
        replay = replay or replays.Buffer(return_steps=5)
        actor_updater = actor_updater or \
            updaters.DistributionalDeterministicPolicyGradient()
        critic_updater = critic_updater or \
            updaters.DistributionalDeterministicQLearning()
        super().__init__(
            model, replay, exploration, actor_updater, critic_updater)
