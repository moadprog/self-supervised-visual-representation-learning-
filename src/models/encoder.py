import torch.nn as nn
from torchvision.models import resnet50


class ResNet50Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        backbone = resnet50(weights=None)

        # ResNet-50 representation dimension before the classifier
        self.feature_dim = backbone.fc.in_features  # 2048

        # Remove the supervised classification layer
        backbone.fc = nn.Identity()

        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x) 