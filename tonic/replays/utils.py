import numpy as np


def lambda_returns(
    values, next_values, rewards, resets, terminations, discount_factor,
    trace_decay
):
    '''Function used to calculate lambda-returns on parallel buffers.'''

    returns = np.zeros_like(values)
    last_returns = next_values[-1]
    for t in reversed(range(len(rewards))):
        bootstrap = (
            (1 - trace_decay) * next_values[t] + trace_decay * last_returns)
        bootstrap *= (1 - resets[t])
        bootstrap += resets[t] * next_values[t]
        bootstrap *= (1 - terminations[t])
        returns[t] = last_returns = rewards[t] + discount_factor * bootstrap
    return returns


def flatten_batch(values):
    shape = values.shape
    new_shape = (np.prod(shape[:2], dtype=int),) + shape[2:]
    return values.reshape(new_shape)
