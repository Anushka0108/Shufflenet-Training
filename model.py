import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x1_0


class LastStrideOne(nn.Module):
    def __init__(self, original_block):
        super().__init__()
        self.block = original_block
        self.block.branch1[0].stride = (1, 1)
        self.block.branch2[3].stride = (1, 1)

    def forward(self, x):
        branch1 = self.block.branch1(x)
        branch2 = self.block.branch2(x)
        out = torch.cat((branch1, branch2), dim=1)

        batch, channels, height, width = out.shape
        out = out.reshape(batch, 2, channels // 2, height, width)
        out = out.transpose(1, 2).contiguous()
        return out.reshape(batch, channels, height, width)


class ShuffleNetV2BNNeck(nn.Module):
    def __init__(self, num_classes, use_last_stride_one=True):
        super().__init__()

        backbone = shufflenet_v2_x1_0(weights=None)

        if use_last_stride_one:
            backbone.stage4[0] = LastStrideOne(backbone.stage4[0])

        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.bottleneck = nn.BatchNorm1d(self.feature_dim)
        self.bottleneck.bias.requires_grad_(False)

        self.classifier = nn.Linear(self.feature_dim, num_classes, bias=False)

        nn.init.kaiming_normal_(
            self.classifier.weight, mode="fan_out"
        )
        nn.init.ones_(self.bottleneck.weight)
        nn.init.zeros_(self.bottleneck.bias)

    def forward(self, images):
        raw_features = self.backbone(images)
        bn_features = self.bottleneck(raw_features)
        logits = self.classifier(bn_features)
        return raw_features, bn_features, logits

    def forward_embeddings(self, images):
        raw_features = self.backbone(images)
        bn_features = self.bottleneck(raw_features)
        return bn_features
