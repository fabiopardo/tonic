import torch


class ValueHead(torch.nn.Module):
    def __init__(self, fn=None):
        super().__init__()
        self.fn = fn

    def initialize(self, input_size, return_normalizer=None):
        self.return_normalizer = return_normalizer
        self.v_layer = torch.nn.Linear(input_size, 1)
        if self.fn:
            self.v_layer.apply(self.fn)

    def forward(self, inputs):
        out = self.v_layer(inputs)
        out = torch.squeeze(out, -1)
        if self.return_normalizer:
            out = self.return_normalizer(out)
        return out


class Critic(torch.nn.Module):
    def __init__(self, encoder, torso, head):
        super().__init__()
        self.encoder = encoder
        self.torso = torso
        self.head = head

    def initialize(
        self, observation_space, action_space, observation_normalizer=None,
        return_normalizer=None
    ):
        size = self.encoder.initialize(
            observation_space=observation_space, action_space=action_space,
            observation_normalizer=observation_normalizer)
        size = self.torso.initialize(size)
        self.head.initialize(size, return_normalizer)

    def forward(self, *inputs):
        out = self.encoder(*inputs)
        out = self.torso(out)
        return self.head(out)
