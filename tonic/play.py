'''Script used to play with trained agents.'''

import argparse
import os

import numpy as np
import yaml

import tonic  # noqa


def play_gym(agent, environment):
    '''Launches an agent in a Gym-based environment.'''

    environment = tonic.environments.distribute(lambda: environment)

    observations = environment.start()
    environment.render()

    score = 0
    length = 0
    min_reward = float('inf')
    max_reward = -float('inf')
    episodes = 0

    while True:
        actions = agent.test_step(observations)
        observations, infos = environment.step(actions)
        agent.test_update(**infos)
        environment.render()

        reward = infos['rewards'][0]
        score += reward
        min_reward = min(min_reward, reward)
        max_reward = max(max_reward, reward)
        length += 1

        if infos['resets'][0]:
            episodes += 1

            print()
            print('Episodes:', episodes)
            print('Score:', score)
            print('Length:', length)
            print('Min reward:', min_reward)
            print('Max reward:', max_reward)

            score = 0
            length = 0


def play_control_suite(agent, environment):
    '''Launches an agent in a DeepMind Control Suite-based environment.'''

    from dm_control import viewer

    class Wrapper:
        '''Wrapper used to plug a Tonic environment in a dm_control viewer.'''

        def __init__(self, environment):
            self.environment = environment
            self.unwrapped = environment.unwrapped
            self.action_spec = self.unwrapped.environment.action_spec
            self.physics = self.unwrapped.environment.physics
            self.infos = None
            self.episodes = 0

        def reset(self):
            '''Mimics a dm_control reset for the viewer.'''

            self.observations = self.environment.reset()[None]

            self.score = 0
            self.length = 0

            return self.unwrapped.last_time_step

        def step(self, actions):
            '''Mimics a dm_control step for the viewer.'''

            ob, rew, term, _ = self.environment.step(actions)
            self.score += rew
            self.length += 1
            timeout = self.length == self.environment.max_episode_steps
            done = term or timeout

            if done:
                print()
                self.episodes += 1
                print('Episodes:', self.episodes)
                print('Score:', self.score)
                print('Length:', self.length)

            self.observations = ob[None]
            self.infos = dict(
                observations=ob[None], rewards=np.array([rew]),
                resets=np.array([done]), terminations=[term])

            return self.unwrapped.last_time_step

    # Wrap the environment for the viewer.
    environment = Wrapper(environment)

    def policy(timestep):
        '''Mimics a dm_control policy for the viewer.'''

        if environment.infos is not None:
            agent.test_update(**environment.infos)
        return agent.test_step(environment.observations)

    # Launch the viewer with the wrapped environment and policy.
    viewer.launch(environment, policy)


def play(path, checkpoint, seed):
    '''Reloads an agent and an environment from a previous experiment.'''

    tonic.logger.log(f'Loading experiment from {path}')

    # Use no checkpoint, the agent is freshly created.
    if checkpoint == 'none':
        checkpoint_path = None
        tonic.logger.log('Not loading any weights')

    else:
        checkpoint_path = os.path.join(path, 'checkpoints')
        if not os.path.isdir(checkpoint_path):
            tonic.logger.error(f'{checkpoint_path} is not a directory')
            checkpoint_path = None

        # List all the checkpoints.
        checkpoint_ids = []
        for file in os.listdir(checkpoint_path):
            if file[:5] == 'step_':
                checkpoint_id = file.split('.')[0]
                checkpoint_ids.append(int(checkpoint_id[5:]))

        if checkpoint_ids:
            # Use the last checkpoint.
            if checkpoint == 'last':
                checkpoint_id = max(checkpoint_ids)
                checkpoint_path = os.path.join(
                    checkpoint_path, f'step_{checkpoint_id}')

            # Use the specified checkpoint.
            else:
                checkpoint_id = int(checkpoint)
                if checkpoint_id in checkpoint_ids:
                    checkpoint_path = os.path.join(
                        checkpoint_path, f'step_{checkpoint_id}')
                else:
                    tonic.logger.error(f'Checkpoint {checkpoint_id} '
                                       f'not found in {checkpoint_path}')
                    checkpoint_path = None

        else:
            tonic.logger.error(f'No checkpoint found in {checkpoint_path}')
            checkpoint_path = None

    # Load the experiment configuration.
    arguments_path = os.path.join(path, 'config.yaml')
    with open(arguments_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config = argparse.Namespace(**config)

    # Run the header first, e.g. to load an ML framework.
    if config.header:
        exec(config.header)

    # Build the agent.
    agent = eval(config.agent)

    # Build the environment.
    environment = eval(config.environment)
    environment.seed(seed)

    # Initialize the agent.
    agent.initialize(
        observation_space=environment.observation_space,
        action_space=environment.action_space)

    # Load the weights of the agent form a checkpoint.
    if checkpoint_path:
        agent.load(checkpoint_path)

    # Play with the agent in the environment.
    if 'ControlSuite' in config.environment:
        play_control_suite(agent, environment)
    else:
        if 'Bullet' in config.environment:
            environment.render()
        play_gym(agent, environment)


if __name__ == '__main__':
    # Argument parsing.
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='.')
    parser.add_argument('--checkpoint', default='last')
    parser.add_argument('--seed', type=int, default=0)
    args = vars(parser.parse_args())
    play(**args)
