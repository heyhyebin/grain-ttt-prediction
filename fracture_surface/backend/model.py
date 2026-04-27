import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# 백본 로드 함수
# ================================================================

def load_convnext_backbone(version="auto", pretrained=True):
    """
    ConvNeXt 백본을 로드합니다.

    Args:
        version: "v2" (timm), "v1" (torchvision), "auto" (v2 시도 후 v1 대체)
        pretrained: True면 사전학습 가중치 로드, False면 구조만 생성

    Returns:
        backbone (nn.Module), out_channels (int), backbone_name (str)
    """
    # ── ConvNeXt-V2-Tiny (timm) ──
    if version in ("v2", "auto"):
        try:
            import timm
            model = timm.create_model(
                "convnextv2_tiny.fcmae_ft_in22k_in1k",
                pretrained=pretrained,
                features_only=True,
                out_indices=(3,),
            )
            name = "ConvNeXt-V2-Tiny"
            print(f"✅ {name} 로드 완료 (timm, pretrained={pretrained})")
            return model, 768, name
        except (ImportError, Exception) as e:
            if version == "v2":
                raise RuntimeError(
                    f"timm 라이브러리가 필요합니다: pip install timm\n오류: {e}"
                )
            print(f"⚠️ timm 로드 실패 ({e}), ConvNeXt-V1으로 대체합니다.")

    # ── ConvNeXt-V1-Tiny (torchvision) ──
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

    if pretrained:
        weights = ConvNeXt_Tiny_Weights.DEFAULT
    else:
        weights = None

    model = convnext_tiny(weights=weights)
    backbone = model.features
    name = "ConvNeXt-V1-Tiny"
    print(f"✅ {name} 로드 완료 (torchvision, pretrained={pretrained})")
    return backbone, 768, name


# ================================================================
# CBAM (Channel + Spatial Attention)
# ================================================================

class ChannelAttention(nn.Module):
    """채널 어텐션: 어떤 특징 채널이 중요한지 학습"""

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
    """공간 어텐션: 이미지의 어느 위치가 중요한지 학습"""

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
    """CBAM: 채널 어텐션 → 공간 어텐션 순서로 적용"""

    def __init__(self, planes):
        super().__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


# ================================================================
# ASPP (Atrous Spatial Pyramid Pooling)
# ================================================================

class ASPP(nn.Module):
    """다양한 dilation rate로 다중 스케일 특징 추출"""

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


# ================================================================
# FractographyNet (메인 모델)
# ================================================================

class FractographyNet(nn.Module):
    """
    ConvNeXt + ASPP + CBAM
    ──────────────────────────
    backbone:   ConvNeXt-V2-Tiny → 768ch (또는 V1)
    aspp:       768 → 256 (다중 스케일 특징 융합)
    attention:  CBAM 256ch (채널 + 공간 어텐션)
    classifier: 256 → 128 → num_classes
    """

    def __init__(self, num_classes=4, pretrained=True, backbone_version="auto"):
        super().__init__()

        self.backbone, backbone_ch, self.backbone_name = load_convnext_backbone(
            version=backbone_version,
            pretrained=pretrained,
        )
        self._is_timm = hasattr(self.backbone, 'feature_info')

        self.aspp = ASPP(in_channels=backbone_ch, out_channels=256)
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
        if self._is_timm:
            features = self.backbone(x)
            x = features[-1] if isinstance(features, (list, tuple)) else features
        else:
            x = self.backbone(x)

        x = self.aspp(x)
        x = self.attention(x)
        x = self.classifier(x)
        return x

    def freeze_backbone(self):
        """백본 파라미터 고정 (학습 초기 단계용)"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """백본 파라미터 해제 (파인튜닝용)"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_param_groups(self, lr_backbone, lr_head):
        """백본과 헤드에 서로 다른 학습률 적용"""
        backbone_params = list(self.backbone.parameters())
        head_params = (
            list(self.aspp.parameters()) +
            list(self.attention.parameters()) +
            list(self.classifier.parameters())
        )
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ]
