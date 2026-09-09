import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    SimCLR projection head:
    h -> hidden layer -> ReLU -> z
    """

    def __init__(
        self,
        input_dim=2048,
        hidden_dim=2048,
        output_dim=128,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, h):
        return self.net(h)   