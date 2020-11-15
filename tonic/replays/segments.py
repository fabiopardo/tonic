import numpy as np

from tonic import replays


class Segment:
    '''Replay storing recent transitions for on-policy learning.'''

    def __init__(
        self, size=4096, batch_iterations=80, batch_size=None,
        discount_factor=0.99, trace_decay=0.97
    ):
        self.max_size = size
        self.batch_iterations = batch_iterations
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.trace_decay = trace_decay

    def initialize(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        self.buffers = None
        self.index = 0

    def ready(self):
        return self.index == self.max_size

    def store(self, **kwargs):
        if self.buffers is None:
            self.num_workers = len(list(kwargs.values())[0])
            self.buffers = {}
            for key, val in kwargs.items():
                shape = (self.max_size,) + np.array(val).shape
                self.buffers[key] = np.zeros(shape, np.float32)
        for key, val in kwargs.items():
            self.buffers[key][self.index] = val
        self.index += 1

    def get_full(self, *keys):
        self.index = 0

        if 'advantages' in keys:
            advs = self.buffers['returns'] - self.buffers['values']
            std = advs.std()
            if std != 0:
                advs = (advs - advs.mean()) / std
            self.buffers['advantages'] = advs

        return {k: replays.flatten_batch(self.buffers[k]) for k in keys}

    def get(self, *keys):
        '''Get mini-batches from named buffers.'''

        batch = self.get_full(*keys)

        if self.batch_size is None:
            for _ in range(self.batch_iterations):
                yield batch
        else:
            size = self.max_size * self.num_workers
            all_indices = np.arange(size)
            for _ in range(self.batch_iterations):
                self.np_random.shuffle(all_indices)
                for i in range(0, size, self.batch_size):
                    indices = all_indices[i:i + self.batch_size]
                    yield {k: v[indices] for k, v in batch.items()}

    def compute_returns(self, values, next_values):
        shape = self.buffers['rewards'].shape
        self.buffers['values'] = values.reshape(shape)
        self.buffers['next_values'] = next_values.reshape(shape)
        self.buffers['returns'] = replays.lambda_returns(
            values=self.buffers['values'],
            next_values=self.buffers['next_values'],
            rewards=self.buffers['rewards'],
            resets=self.buffers['resets'],
            terminations=self.buffers['terminations'],
            discount_factor=self.discount_factor,
            trace_decay=self.trace_decay)
