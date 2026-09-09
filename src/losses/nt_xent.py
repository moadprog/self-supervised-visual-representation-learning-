import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        z1: [N, D] projections of first views
        z2: [N, D] projections of second views
        """

        batch_size = z1.shape[0]

        # Normalize so dot product becomes cosine similarity
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # [2N, D]
        z = torch.cat([z1, z2], dim=0)

        # Pairwise cosine similarities
        # [2N, 2N]
        similarity_matrix = torch.matmul(z, z.T)

        similarity_matrix = similarity_matrix / self.temperature

        # An embedding must not be compared with itself
        mask = torch.eye(
            2 * batch_size,
            dtype=torch.bool,
            device=z.device,
        )

        similarity_matrix = similarity_matrix.masked_fill(
            mask,
            float("-inf"),
        )

        # Positive pair:
        # i       <-> i + N
        # i + N   <-> i
        labels = torch.arange(
            2 * batch_size,
            device=z.device,
        )

        labels = (labels + batch_size) % (2 * batch_size)

        loss = F.cross_entropy(
            similarity_matrix,
            labels,
        )

        return loss

    