# Example task for local harness integration checks.
# Replace or extend these examples with your own task set.
import torch
import torch.nn as nn


class Model(nn.Module):
    """Mean-pool over the last dim, add a bias, then ReLU."""

    def __init__(self, channels):
        super(Model, self).__init__()
        self.bias = nn.Parameter(torch.randn(channels) * 0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x (torch.Tensor): (batch_size, channels, length)

        Returns:
            torch.Tensor: (batch_size, channels)
        """
        pooled = x.mean(dim=2)
        return torch.relu(pooled + self.bias)


batch_size = 256
channels = 512
length = 4096


def get_inputs():
    x = torch.randn(batch_size, channels, length)
    return [x]


def get_init_inputs():
    return [channels]
