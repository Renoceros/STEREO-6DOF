import torch.nn as nn

def make_head(input_dim, output_dim, layer_sizes):
    """
    Creates a sequential neural network head with ReLU activations.
    """
    layers = []
    prev_dim = input_dim
    for size in layer_sizes:
        layers.append(nn.Linear(prev_dim, size))
        layers.append(nn.ReLU())
        prev_dim = size
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)
