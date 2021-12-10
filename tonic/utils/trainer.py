import os
import time

import numpy as np

from tonic import logger


class Trainer:
    '''Trainer used to train and evaluate an agent on an environment.'''

    def __init__(
        self, steps=int(1e7), epoch_steps=int(2e4), save_steps=int(5e5),
        test_episodes=5, show_progress=True, replace_checkpoint=False,
    ):
        self.max_steps = steps
        self.epoch_steps = epoch_steps
        self.save_steps = save_steps
        self.test_episodes = test_episodes
        self.show_progress = show_progress
        self.replace_checkpoint = replace_checkpoint

    def initialize(self, agent, environment, test_environment=None):
        self.agent = agent
        self.environment = environment
        self.test_environment = test_environment

    def run(self):
        '''Runs the main training loop.'''

        start_time = last_epoch_time = time.time()

        # Start the environments.
        observations = self.environment.start()

        num_workers = len(observations)
        scores = np.zeros(num_workers)
        lengths = np.zeros(num_workers, int)
        self.steps, epoch_steps, epochs, episodes = 0, 0, 0, 0
        steps_since_save = 0

        while True:
            # Select actions.
            actions = self.agent.step(observations, self.steps)
            assert not np.isnan(actions.sum())
            logger.store('train/action', actions, stats=True)

            # Take a step in the environments.
            observations, infos = self.environment.step(actions)
            self.agent.update(**infos, steps=self.steps)

            scores += infos['rewards']
            lengths += 1
            self.steps += num_workers
            epoch_steps += num_workers
            steps_since_save += num_workers

            # Show the progress bar.
            if self.show_progress:
                logger.show_progress(
                    self.steps, self.epoch_steps, self.max_steps)

            # Check the finished episodes.
            for i in range(num_workers):
                if infos['resets'][i]:
                    logger.store('train/episode_score', scores[i], stats=True)
                    logger.store(
                        'train/episode_length', lengths[i], stats=True)
                    scores[i] = 0
                    lengths[i] = 0
                    episodes += 1

            # End of the epoch.
            if epoch_steps >= self.epoch_steps:
                # Evaluate the agent on the test environment.
                if self.test_environment:
                    self._test()

                # Log the data.
                epochs += 1
                current_time = time.time()
                epoch_time = current_time - last_epoch_time
                sps = epoch_steps / epoch_time
                logger.store('train/episodes', episodes)
                logger.store('train/epochs', epochs)
                logger.store('train/seconds', current_time - start_time)
                logger.store('train/epoch_seconds', epoch_time)
                logger.store('train/epoch_steps', epoch_steps)
                logger.store('train/steps', self.steps)
                logger.store('train/worker_steps', self.steps // num_workers)
                logger.store('train/steps_per_second', sps)
                logger.dump()
                last_epoch_time = time.time()
                epoch_steps = 0

            # End of training.
            stop_training = self.steps >= self.max_steps

            # Save a checkpoint.
            if stop_training or steps_since_save >= self.save_steps:
                path = os.path.join(logger.get_path(), 'checkpoints')
                if os.path.isdir(path) and self.replace_checkpoint:
                    for file in os.listdir(path):
                        if file.startswith('step_'):
                            os.remove(os.path.join(path, file))
                checkpoint_name = f'step_{self.steps}'
                save_path = os.path.join(path, checkpoint_name)
                self.agent.save(save_path)
                steps_since_save = self.steps % self.save_steps

            if stop_training:
                break

    def _test(self):
        '''Tests the agent on the test environment.'''

        # Start the environment.
        if not hasattr(self, 'test_observations'):
            self.test_observations = self.test_environment.start()
            assert len(self.test_observations) == 1

        # Test loop.
        for _ in range(self.test_episodes):
            score, length = 0, 0

            while True:
                # Select an action.
                actions = self.agent.test_step(
                    self.test_observations, self.steps)
                assert not np.isnan(actions.sum())
                logger.store('test/action', actions, stats=True)

                # Take a step in the environment.
                self.test_observations, infos = self.test_environment.step(
                    actions)
                self.agent.test_update(**infos, steps=self.steps)

                score += infos['rewards'][0]
                length += 1

                if infos['resets'][0]:
                    break

            # Log the data.
            logger.store('test/episode_score', score, stats=True)
            logger.store('test/episode_length', length, stats=True)
