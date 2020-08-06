import random

import numpy as np
import tensorflow as tf

from tonic import agents, logger


class TensorFlowAgent(agents.Agent):
    def initialize(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            tf.random.set_seed(seed)

    def save(self, path):
        logger.log('Saving weights to {}'.format(path))
        self.model.save_weights(path)

    def load(self, path):
        logger.log('Loading weights from {}'.format(path))
        self.model.load_weights(path)
