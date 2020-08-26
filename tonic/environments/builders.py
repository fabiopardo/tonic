'''Environment builders for popular domains.'''

import gym.wrappers
import numpy as np
from dm_control import rl

from tonic import environments

def gym_environment(*args, **kwargs):
    '''Returns a wrapped Gym environment.'''

    def _builder(*args, **kwargs):
        return gym.make(*args, **kwargs)

    return build_environment(_builder, *args, **kwargs)


def bullet_environment(*args, **kwargs):
    '''Returns a wrapped PyBullet environment.'''

    def _builder(*args, **kwargs):
        import pybullet_envs  # noqa
        return gym.make(*args, **kwargs)

    return build_environment(_builder, *args, **kwargs)


def control_suite_environment(*args, **kwargs):
    '''Returns a wrapped Control Suite environment.'''

    def _builder(name, *args, **kwargs):
        domain, task = name.split('-')
        environment = ControlSuiteEnvironment(
            domain_name=domain, task_name=task, *args, **kwargs)
        return gym.wrappers.TimeLimit(environment, 1000)

    return build_environment(_builder, *args, **kwargs)

def custom_environment(*args, **kwargs):
    def _builder(task, *args, **kwargs):

        task_path = 'environments.custom_environments.' + task

        prefix = eval(task_path)
        _WALK_SPEED = 1

        import inspect
        for name, obj in inspect.getmembers(prefix):
            if inspect.isclass(obj):
                if obj.__name__ != 'Physics':
                    task = eval('prefix.' + obj.__name__ + '(_WALK_SPEED)')

        physics = prefix.Physics.from_xml_string(*prefix.get_model_and_assets())
        env = rl.control.Environment(physics, task)

        environment = ControlSuiteEnvironment(domain_name="", task_name="", custom=True, custom_env=env)

        return gym.wrappers.TimeLimit(environment, 2000)

    return build_environment(_builder, *args, **kwargs)


def build_environment(
    builder, name, terminal_timeouts=False, time_feature=False,
    max_episode_steps='default', scaled_actions=True, *args, **kwargs
):
    '''Builds and wrap an environment.
    Time limits can be properly handled with terminal_timeouts=False or
    time_feature=True, see https://arxiv.org/pdf/1712.00378.pdf for more
    details.
    '''

    # Build the environment.
    environment = builder(name, *args, **kwargs)

    # Get the default time limit.
    if max_episode_steps == 'default':
        max_episode_steps = environment._max_episode_steps

    # Remove the TimeLimit wrapper if needed.
    if not terminal_timeouts:
        assert type(environment) == gym.wrappers.TimeLimit, environment
        environment = environment.env

    # Add time as a feature if needed.
    if time_feature:
        environment = environments.wrappers.TimeFeature(
            environment, max_episode_steps)

    # Scale actions from [-1, 1]^n to the true action space if needed.
    if scaled_actions:
        environment = environments.wrappers.ActionRescaler(environment)

    environment.name = name
    environment.max_episode_steps = max_episode_steps

    return environment


def _flatten_observation(observation):
    '''Turns OrderedDict observations into vectors.'''
    observation = [np.array([o]) if np.isscalar(o) else o.ravel()
                   for o in observation.values()]
    return np.concatenate(observation, axis=0)


class ControlSuiteEnvironment(gym.core.Env):
    '''Turns a Control Suite environment into a Gym environment.'''

    def __init__(
        self, domain_name, task_name, task_kwargs=None, visualize_reward=True,
        environment_kwargs=None, custom=False, custom_env=None
    ):
        if custom:
            self.environment = custom_env
        else:
            from dm_control import suite
            self.environment = suite.load(
                domain_name=domain_name, task_name=task_name,
                task_kwargs=task_kwargs, visualize_reward=visualize_reward,
                environment_kwargs=environment_kwargs)

        # Create the observation space.
        observation_spec = self.environment.observation_spec()
        dim = sum([np.int(np.prod(spec.shape))
                   for spec in observation_spec.values()])
        high = np.full(dim, np.inf, np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # Create the action space.
        action_spec = self.environment.action_spec()
        self.action_space = gym.spaces.Box(
            action_spec.minimum, action_spec.maximum, dtype=np.float32)

    def seed(self, seed):
        self.environment.task._random = np.random.RandomState(seed)

    def step(self, action):
        time_step = self.environment.step(action)
        observation = _flatten_observation(time_step.observation)
        reward = time_step.reward

        # Remove terminations from timeouts.
        done = time_step.last()
        if done:
            done = self.environment.task.get_termination(
                self.environment.physics)
            done = done is not None

        self.last_time_step = time_step
        return observation, reward, done, {}

    def reset(self):
        time_step = self.environment.reset()
        self.last_time_step = time_step
        return _flatten_observation(time_step.observation)

    def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
        '''Returns RGB frames from a camera.'''
        assert mode == 'rgb_array'
        return self.environment.physics.render(
            height=height, width=width, camera_id=camera_id)


# Aliases
Gym = gym_environment
Bullet = bullet_environment
ControlSuite = control_suite_environment
Custom = custom_environment