import torch
from torch import nn

class Time2VecSineActivation(nn.Module):
    def __init__(self, output_dim):
        super(Time2VecSineActivation, self).__init__()

        # Parameter for the linear part of Time2Vec
        self.w0 = nn.Parameter(torch.randn(1, 1))
        self.b0 = nn.Parameter(torch.randn(1))

        # Parameters for the periodic part of Time2Vec
        self.w = nn.Parameter(torch.randn(1, output_dim - 1))
        self.b = nn.Parameter(torch.randn(output_dim - 1))

    def forward(self, tau):
        # tau should have shape [batch_size, num_elements], e.g., [8, 15]
        tau = tau.unsqueeze(-1)  # Shape becomes [8, 15, 1] for multiplication

        # Linear transformation
        linear_part = tau * self.w0 + self.b0  # Broadcasting applies the transformation to each element

        # Periodic transformation using sinusoidal functions
        periodic_part = torch.sin(tau * self.w + self.b)  # Again broadcasting

        # Concatenate the linear and periodic parts
        return torch.cat((linear_part, periodic_part), dim=-1)
