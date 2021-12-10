from tonic import replays
from tonic.tensorflow import agents, models, normalizers, updaters


def default_model():
    return models.ActorTwinCriticWithTargets(
        actor=models.Actor(
            encoder=models.ObservationEncoder(),
            torso=models.MLP((256, 256), 'relu'),
            head=models.DeterministicPolicyHead()),
        critic=models.Critic(
            encoder=models.ObservationActionEncoder(),
            torso=models.MLP((256, 256), 'relu'),
            head=models.DistributionalValueHead(-150., 150., 51)),
        observation_normalizer=normalizers.MeanStd())


class TD4(agents.TD3):
    def __init__(
        self, model=None, replay=None, exploration=None, actor_updater=None,
        critic_updater=None, delay_steps=2
    ):
        model = model or default_model()
        replay = replay or replays.Buffer(return_steps=5)
        actor_updater = actor_updater or \
            updaters.DistributionalDeterministicPolicyGradient()
        critic_updater = critic_updater or \
            updaters.TwinCriticDistributionalDeterministicQLearning()
        super().__init__(
            model=model, replay=replay, exploration=exploration,
            actor_updater=actor_updater, critic_updater=critic_updater,
            delay_steps=delay_steps)
