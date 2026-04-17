import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights


class ChannelAttention(nn.Module):
    """채널 어텐션: 어떤 특징 채널이 중요한지 학습합니다."""

    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(in_planes // ratio, 8)
        self.fc1 = nn.Conv2d(in_planes, mid, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(mid, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """공간 어텐션: 이미지의 어느 위치가 중요한지 학습합니다."""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    """CBAM: 채널 어텐션 → 공간 어텐션 순서로 적용합니다."""

    def __init__(self, planes):
        super().__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling
    다양한 dilation rate(1, 6, 12, 18)로 다중 스케일 특징을 추출합니다.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3,
                               padding=6, dilation=6, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3,
                               padding=12, dilation=12, bias=False)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3,
                               padding=18, dilation=18, bias=False)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:],
                           mode='bilinear', align_corners=True)
        return self.out_conv(torch.cat((x1, x2, x3, x4, x5), dim=1))


class FractographyNet(nn.Module):
    """
    EfficientNet-B2 + ASPP + CBAM
    ──────────────────────────────
    backbone:   EfficientNet-B2 (.features) → 1408ch 특징맵
    aspp:       1408 → 256 (다중 스케일 특징 융합)
    attention:  CBAM 256ch (채널 + 공간 어텐션)
    classifier: 256 → 128 → num_classes
    """

    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        if pretrained:
            weights = EfficientNet_B2_Weights.DEFAULT
        else:
            weights = None
        self.backbone = efficientnet_b2(weights=weights).features

        self.aspp = ASPP(in_channels=1408, out_channels=256)
        self.attention = CBAM(planes=256)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.attention(x)
        return self.classifier(x)
