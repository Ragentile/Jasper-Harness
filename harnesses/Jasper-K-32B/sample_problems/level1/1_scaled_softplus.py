# Example task for local harness integration checks.
# Replace or extend these examples with your own task set.
import torch
import torch.nn as nn


class Model(nn.Module):
    """Scaled softplus: y = alpha * log1p(exp(beta * x)), elementwise."""

    def __init__(self, alpha, beta):
        super(Model, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.log1p(torch.exp(self.beta * x))


batch_size = 16
dim = 1048576
alpha = 1.5
beta = 0.75


def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]


def get_init_inputs():
    return [alpha, beta]
