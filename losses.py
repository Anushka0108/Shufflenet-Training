import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, features, labels):
        distances = torch.cdist(features, features, p=2)

        labels = labels.view(-1, 1)
        same_identity = labels.eq(labels.T)
        different_identity = ~same_identity

        diagonal = torch.eye(
            labels.size(0), dtype=torch.bool, device=labels.device
        )
        positive_mask = same_identity & ~diagonal

        hardest_positive = distances.masked_fill(
            ~positive_mask, float("-inf")
        ).max(dim=1).values

        hardest_negative = distances.masked_fill(
            ~different_identity, float("inf")
        ).min(dim=1).values

        valid = positive_mask.any(dim=1) & different_identity.any(dim=1)
        if not valid.any():
            return features.sum() * 0.0

        return F.relu(
            hardest_positive[valid] - hardest_negative[valid] + self.margin
        ).mean()


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))

    def forward(self, features, labels):
        selected_centers = self.centers.index_select(0, labels)
        return 0.5 * (features - selected_centers).pow(2).sum(dim=1).mean()
