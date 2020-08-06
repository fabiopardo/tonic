python -m tonic.train \
    --header "import tonic.tensorflow" \
    --agent "tonic.tensorflow.agents.PPO()" \
    --environment "tonic.environments.Gym('BipedalWalker-v3')" \
    --trainer "tonic.Trainer(steps=100000, save_steps=100000)" \
    --seed 0

python -m tonic.plot --path BipedalWalker-v3 --show True --baselines PPO

python -m tonic.play --path BipedalWalker-v3/PPO/0
