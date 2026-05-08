import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_small


class ASPP(nn.Module):
    def __init__(self, in_channels: int = 768, out_channels: int = 256):
        super().__init__()

        def _branch(dilation: int):
            if dilation == 1:
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                )

            return nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    3,
                    padding=dilation,
                    dilation=dilation,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )

        self.b1 = _branch(1)
        self.b6 = _branch(6)
        self.b12 = _branch(12)
        self.b18 = _branch(18)

        self.gap_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(0.3),
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]

        gap = self.gap_branch(x)
        gap = F.interpolate(
            gap,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat(
            [
                self.b1(x),
                self.b6(x),
                self.b12(x),
                self.b18(x),
                gap,
            ],
            dim=1,
        )

        return self.project(x)


class FractographyNet(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()

        base = convnext_small(weights=None)

        self.backbone = base.features
        self.aspp = ASPP(in_channels=768, out_channels=256)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def load_model(model_path: str, device, num_classes: int = 4):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"pth 파일 없음: {model_path}")

    model = FractographyNet(num_classes).to(device)

    state = torch.load(model_path, map_location=device)

    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    clean_state = {}
    for k, v in state.items():
        clean_state[k.replace("module.", "")] = v

    model.load_state_dict(clean_state, strict=True)
    model.eval()

    print("✅ pth 로드 완료")
    print(f"✅ 경로: {model_path}")
    print("✅ 모델: ConvNeXt-Small + ASPP")
    print(f"✅ device: {device}")

    return model
