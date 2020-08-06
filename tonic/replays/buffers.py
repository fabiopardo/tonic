import numpy as np

from tonic import replays


class Buffer:
    '''Replay storing a large number of transitions for off-policy learning.'''

    def __init__(
        self, size=int(1e6), batch_iterations=50, batch_size=100,
        discount_factor=0.98, trace_decay=0.97, steps_before_batches=1000,
        steps_between_batches=50
    ):
        self.max_size = size
        self.batch_iterations = batch_iterations
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.trace_decay = trace_decay
        self.steps_before_batches = steps_before_batches
        self.steps_between_batches = steps_between_batches

    def initialize(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        self.buffers = None
        self.index = 0
        self.size = 0
        self.steps = 0

    def ready(self):
        if self.steps < self.steps_before_batches:
            return False
        return self.steps % self.steps_between_batches == 0

    def store(self, **kwargs):
        if self.buffers is None:
            self.num_workers = len(list(kwargs.values())[0])
            assert self.max_size % self.num_workers == 0
            self.max_size //= self.num_workers
            assert self.steps_before_batches % self.num_workers == 0
            self.steps_before_batches /= self.num_workers
            assert self.steps_between_batches % self.num_workers == 0
            self.steps_between_batches /= self.num_workers
            self.buffers = {}
            for key, val in kwargs.items():
                shape = (self.max_size,) + np.array(val).shape
                self.buffers[key] = np.zeros(shape, 'float32')
        for key, val in kwargs.items():
            self.buffers[key][self.index] = val
        self.index = (self.index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.steps += 1

    def get_full(self, *keys):
        if 'discounts' in keys:
            self.buffers['discounts'] = (
                (1 - self.buffers['terminations']) * self.discount_factor)

        if 'advantages' in keys:
            advs = self.buffers['returns'] - self.buffers['values']
            std = advs.std()
            if std != 0:
                advs = (advs - advs.mean()) / std
            self.buffers['advantages'] = advs

        return {k: replays.flatten_batch(self.buffers[k][:self.size])
                for k in keys}

    def get(self, *keys):
        '''Get mini-batches from named buffers.'''

        batch = self.get_full(*keys)
        size = self.size * self.num_workers

        if self.batch_size is None:
            for _ in range(self.batch_iterations):
                yield batch
        else:
            for _ in range(self.batch_iterations):
                indices = self.np_random.randint(size, size=self.batch_size)
                yield {k: v[indices] for k, v in batch.items()}

    def compute_returns(self, values, next_values):
        shape = np.prod(values.shape, dtype=int)
        self.buffers['values'] = values.reshape(shape)
        self.buffers['next_values'] = next_values.reshape(shape)
        self.buffers['returns'] = replays.lambda_returns(
            values=self.buffers['values'],
            next_values=self.buffers['next_values'],
            rewards=self.buffers['rewards'][:self.size],
            resets=self.buffers['resets'][:self.size],
            terminations=self.buffers['terminations'][:self.size],
            discount_factor=self.discount_factor,
            trace_decay=self.trace_decay)
