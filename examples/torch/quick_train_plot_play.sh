#!/bin/bash

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# Quick distributed training.
python3 -m tonic.train \
    --header "import tonic.torch" \
    --agent "tonic.torch.agents.PPO(replay=tonic.replays.Segment(size=10, batch_size=2000, batch_iterations=30))" \
    --environment "tonic.environments.Gym('LunarLanderContinuous-v2')" \
    --trainer "tonic.Trainer(epoch_steps=100, steps=500000, save_steps=500000)" \
    --parallel 10 \
    --sequential 100 \
    --name "PPO-torch-demo" \
    --seed 0

# Plot and reload.
python3 -m tonic.plot --path LunarLanderContinuous-v2 --baselines all &
python3 -m tonic.play --path LunarLanderContinuous-v2/PPO-torch-demo/0 &
wait
