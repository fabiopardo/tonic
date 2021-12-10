import abc


class Agent(abc.ABC):
    '''Abstract class used to build agents.'''

    def initialize(self, observation_space, action_space, seed=None):
        pass

    @abc.abstractmethod
    def step(self, observations, steps):
        '''Returns actions during training.'''
        pass

    def update(self, observations, rewards, resets, terminations, steps):
        '''Informs the agent of the latest transitions during training.'''
        pass

    @abc.abstractmethod
    def test_step(self, observations, steps):
        '''Returns actions during testing.'''
        pass

    def test_update(self, observations, rewards, resets, terminations, steps):
        '''Informs the agent of the latest transitions during testing.'''
        pass

    def save(self, path):
        '''Saves the agent weights during training.'''
        pass

    def load(self, path):
        '''Reloads the agent weights from a checkpoint.'''
        pass
