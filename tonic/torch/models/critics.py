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


class CategoricalWithSupport:
    def __init__(self, values, logits):
        self.values = values
        self.logits = logits
        self.probabilities = torch.nn.functional.softmax(logits, dim=-1)

    def mean(self):
        return (self.probabilities * self.values).sum(dim=-1)

    def project(self, returns):
        vmin, vmax = self.values[0], self.values[-1]
        d_pos = torch.cat([self.values, vmin[None]], 0)[1:]
        d_pos = (d_pos - self.values)[None, :, None]
        d_neg = torch.cat([vmax[None], self.values], 0)[:-1]
        d_neg = (self.values - d_neg)[None, :, None]

        clipped_returns = torch.clamp(returns, vmin, vmax)
        delta_values = clipped_returns[:, None] - self.values[None, :, None]
        delta_sign = (delta_values >= 0).float()
        delta_hat = ((delta_sign * delta_values / d_pos) -
                     ((1 - delta_sign) * delta_values / d_neg))
        delta_clipped = torch.clamp(1 - delta_hat, 0, 1)

        return (delta_clipped * self.probabilities[:, None]).sum(dim=2)


class DistributionalValueHead(torch.nn.Module):
    def __init__(self, vmin, vmax, num_atoms, fn=None):
        super().__init__()
        self.num_atoms = num_atoms
        self.fn = fn
        self.values = torch.linspace(vmin, vmax, num_atoms).float()

    def initialize(self, input_size, return_normalizer=None):
        if return_normalizer:
            raise ValueError(
                'Return normalizers cannot be used with distributional value'
                'heads.')
        self.distributional_layer = torch.nn.Linear(input_size, self.num_atoms)
        if self.fn:
            self.distributional_layer.apply(self.fn)

    def forward(self, inputs):
        logits = self.distributional_layer(inputs)
        return CategoricalWithSupport(values=self.values, logits=logits)


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
