import torch


FLOAT_EPSILON = 1e-8


class SquashedMultivariateNormalDiag:
    def __init__(self, loc, scale):
        self._distribution = torch.distributions.normal.Normal(loc, scale)

    def rsample_with_log_prob(self, shape=()):
        samples = self._distribution.rsample(shape)
        squashed_samples = torch.tanh(samples)
        log_probs = self._distribution.log_prob(samples)
        log_probs -= torch.log(1 - squashed_samples ** 2 + 1e-6)
        return squashed_samples, log_probs

    def rsample(self, shape=()):
        samples = self._distribution.rsample(shape)
        return torch.tanh(samples)

    def sample(self, shape=()):
        samples = self._distribution.sample(shape)
        return torch.tanh(samples)

    def log_prob(self, samples):
        '''Required unsquashed samples cannot be accurately recovered.'''
        raise NotImplementedError(
            'Not implemented to avoid approximation errors. '
            'Use sample_with_log_prob directly.')

    @property
    def loc(self):
        return torch.tanh(self._distribution.mean)


class DetachedScaleGaussianPolicyHead(torch.nn.Module):
    def __init__(
        self, loc_activation=torch.nn.Tanh, loc_fn=None, log_scale_init=0.,
        scale_min=1e-4, scale_max=1.,
        distribution=torch.distributions.normal.Normal
    ):
        super().__init__()
        self.loc_activation = loc_activation
        self.loc_fn = loc_fn
        self.log_scale_init = log_scale_init
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.distribution = distribution

    def initialize(self, input_size, action_size):
        self.loc_layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, action_size), self.loc_activation())
        if self.loc_fn:
            self.loc_layer.apply(self.loc_fn)
        log_scale = [[self.log_scale_init] * action_size]
        self.log_scale = torch.nn.Parameter(
            torch.as_tensor(log_scale, dtype=torch.float32))

    def forward(self, inputs):
        loc = self.loc_layer(inputs)
        batch_size = inputs.shape[0]
        scale = torch.nn.functional.softplus(self.log_scale) + FLOAT_EPSILON
        scale = torch.clamp(scale, self.scale_min, self.scale_max)
        scale = scale.repeat(batch_size, 1)
        return self.distribution(loc, scale)


class GaussianPolicyHead(torch.nn.Module):
    def __init__(
        self, loc_activation=torch.nn.Tanh, loc_fn=None,
        scale_activation=torch.nn.Softplus, scale_min=1e-4, scale_max=1,
        scale_fn=None, distribution=torch.distributions.normal.Normal
    ):
        super().__init__()
        self.loc_activation = loc_activation
        self.loc_fn = loc_fn
        self.scale_activation = scale_activation
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.scale_fn = scale_fn
        self.distribution = distribution

    def initialize(self, input_size, action_size):
        self.loc_layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, action_size), self.loc_activation())
        if self.loc_fn:
            self.loc_layer.apply(self.loc_fn)
        self.scale_layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, action_size), self.scale_activation())
        if self.scale_fn:
            self.scale_layer.apply(self.scale_fn)

    def forward(self, inputs):
        loc = self.loc_layer(inputs)
        scale = self.scale_layer(inputs)
        scale = torch.clamp(scale, self.scale_min, self.scale_max)
        return self.distribution(loc, scale)


class DeterministicPolicyHead(torch.nn.Module):
    def __init__(self, activation=torch.nn.Tanh, fn=None):
        super().__init__()
        self.activation = activation
        self.fn = fn

    def initialize(self, input_size, action_size):
        self.action_layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, action_size),
            self.activation())
        if self.fn is not None:
            self.action_layer.apply(self.fn)

    def forward(self, inputs):
        return self.action_layer(inputs)


class Actor(torch.nn.Module):
    def __init__(self, encoder, torso, head):
        super().__init__()
        self.encoder = encoder
        self.torso = torso
        self.head = head

    def initialize(
        self, observation_space, action_space, observation_normalizer=None
    ):
        size = self.encoder.initialize(
            observation_space, observation_normalizer)
        size = self.torso.initialize(size)
        action_size = action_space.shape[0]
        self.head.initialize(size, action_size)

    def forward(self, *inputs):
        out = self.encoder(*inputs)
        out = self.torso(out)
        return self.head(out)
