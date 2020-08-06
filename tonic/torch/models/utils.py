import torch


class MLP(torch.nn.Module):
    def __init__(self, sizes, activation, fn=None):
        super().__init__()
        self.sizes = sizes
        self.activation = activation
        self.fn = fn

    def initialize(self, input_size):
        sizes = [input_size] + list(self.sizes)
        layers = []
        for i in range(len(sizes) - 1):
            layers += [torch.nn.Linear(sizes[i], sizes[i + 1]),
                       self.activation()]
        self.model = torch.nn.Sequential(*layers)
        if self.fn is not None:
            self.model.apply(self.fn)
        return sizes[-1]

    def forward(self, inputs):
        return self.model(inputs)


def trainable_variables(model):
    return [p for p in model.parameters() if p.requires_grad]
