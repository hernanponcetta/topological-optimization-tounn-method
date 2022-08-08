import torch
import torch.nn as nn


class TopOptimizerNN(nn.Sequential):

    def __init__(self, input_dim, neurons_per_layer=20, numbers_of_layers=5, use_softmax=False):
        super().__init__()

        # Input layer
        self.add_module("Linear-0", nn.Linear(input_dim, neurons_per_layer))
        self.add_module("BatchNorm1d-0", nn.BatchNorm1d(neurons_per_layer))
        self.add_module("ReLU6-0", nn.ReLU6())

        # Hidden layers
        for layer in range(numbers_of_layers - 1):
            self.add_module("Linear-{n}".format(n=layer + 1),
                            nn.Linear(neurons_per_layer, neurons_per_layer))
            self.add_module("BatchNorm1d-{n}".format(n=layer + 1),
                            nn.BatchNorm1d(neurons_per_layer))
            self.add_module("ReLU6-{n}".format(n=layer + 1), nn.ReLU6())

        # Output layer
        self.add_module("Linear-{n}".format(n=numbers_of_layers),
                        nn.Linear(neurons_per_layer, neurons_per_layer))
        if (use_softmax):
            self.add_module("Softmax", nn.Softmax(dim=1))
            return

        self.add_module("Sigmoid", nn.Sigmoid())

    def forward(self, mid_points):
        return (0.01 + super().forward(mid_points))[:, 0].view(-1)
