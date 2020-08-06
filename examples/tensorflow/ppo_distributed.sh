replay="tonic.replays.Segment(size=10000, batch_size=2000, batch_iterations=30)"

python -m tonic.train \
    --header "import tonic.tensorflow" \
    --agent "tonic.tensorflow.agents.PPO(replay=$replay)" \
    --environment "tonic.environments.Gym('HalfCheetah-v3')" \
    --trainer "tonic.Trainer(epoch_steps=100000)" \
    --parallel 10 \
    --sequential 100 \
    --name PPO-10x100 \
    --seed 0
