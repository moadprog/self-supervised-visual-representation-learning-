import torch.nn as nn

from src.models.encoder import ResNet50Encoder
from src.models.projection_head import ProjectionHead


class SimCLR(nn.Module):

    def __init__(
        self,
        projection_dim=128,
        projection_hidden_dim=2048,
    ):
        super().__init__()

        self.encoder = ResNet50Encoder()

        self.projector = ProjectionHead(
            input_dim=self.encoder.feature_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_dim,
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)

        return h, z