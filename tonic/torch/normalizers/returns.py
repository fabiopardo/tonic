import numpy as np
import torch


class Return(torch.nn.Module):
    def __init__(self, discount_factor):
        super().__init__()
        assert 0 <= discount_factor < 1
        self.coefficient = 1 / (1 - discount_factor)
        self.min_reward = np.float32(-1)
        self.max_reward = np.float32(1)
        self._low = torch.nn.Parameter(torch.as_tensor(
            self.coefficient * self.min_reward, dtype=torch.float32),
            requires_grad=False)
        self._high = torch.nn.Parameter(torch.as_tensor(
            self.coefficient * self.max_reward, dtype=torch.float32),
            requires_grad=False)

    def forward(self, val):
        val = torch.sigmoid(val)
        return self._low + val * (self._high - self._low)

    def record(self, values):
        for val in values:
            if val < self.min_reward:
                self.min_reward = np.float32(val)
            elif val > self.max_reward:
                self.max_reward = np.float32(val)

    def update(self):
        self._update(self.min_reward, self.max_reward)

    def _update(self, min_reward, max_reward):
        self._low.data.copy_(torch.as_tensor(
            self.coefficient * min_reward, dtype=torch.float32))
        self._high.data.copy_(torch.as_tensor(
            self.coefficient * max_reward, dtype=torch.float32))
