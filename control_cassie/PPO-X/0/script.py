'''Script used to train agents.'''

import argparse
import os

import tonic



'''
Note: header, agent, environment and trainer are all strings with inside code that need to be evaluated
header       --> specifies which framework to use either Pytorch or TF2
agent        --> specifies which agent to use DDPG, PPO, etc
environment  --> specifies which environment to import
'''
def train(
    header, agent, environment, trainer, parallel, sequential, seed, name
):
    '''Trains an agent on an environment.'''

    # Capture the arguments to save them, e.g. to play with the trained agent.
    args = dict(locals())

    # Run the header first, e.g. to load an ML framework.
    if header:
        exec(header)

    # Build the agent.
    agent = eval(agent)

    # Build the train and test environments.
    _environment = environment
    environment = tonic.environments.distribute(
        lambda: eval(_environment), parallel, sequential)
    test_environment = tonic.environments.distribute(
        lambda: eval(_environment))

    # Build the trainer.
    trainer = eval(trainer)

    # Choose a name for the experiment.
    if hasattr(test_environment, 'name'):
        environment_name = test_environment.name
    else:
        environment_name = test_environment.__class__.__name__
    if not name:
        if hasattr(agent, 'name'):
            name = agent.name
        else:
            name = agent.__class__.__name__

    # Initialize the logger to save data to the path environment/name/seed.
    path = os.path.join(environment_name, name, str(seed))
    tonic.logger.initialize(path, script_path=__file__, config=args)

    # Train.
    trainer.initialize(
        agent=agent, environment=environment,
        test_environment=test_environment, seed=seed)
    trainer.run()


if __name__ == '__main__':
    # Argument parsing.
    parser = argparse.ArgumentParser()
    parser.add_argument('--header')
    parser.add_argument('--agent', required=True)
    parser.add_argument('--environment', '--env', required=True)
    parser.add_argument('--trainer', default='tonic.Trainer()')
    parser.add_argument('--parallel', type=int, default=1)
    parser.add_argument('--sequential', type=int, default=1)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--name')
    args = vars(parser.parse_args())
    train(**args)
