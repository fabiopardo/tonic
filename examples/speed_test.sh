trainer="tonic.Trainer(steps=int(1e6), epoch_steps=int(1e6), save_steps=None, test_episodes=0)"

python -m tonic.train \
    --agent "tonic.agents.Constant()" \
    --environment "tonic.environments.Gym('Pendulum-v0')" \
    --trainer "$trainer" \
    --parallel 10 \
    --sequential 100 \
    --name SpeedTest \
    --seed 0
