import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_small, ConvNeXt_Small_Weights

# =========================
# 설정
# =========================
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH  = "checkpoints/fractography_best.pth"
CLASS_NAMES = ["Cleavage", "Ductile", "Fatigue", "Intergranular"]
IMG_SIZE    = 224

# =========================
# 모델 정의 (train.py와 동일)
# =========================
class ASPP(nn.Module):
    def __init__(self, in_channels=768, out_channels=256):
        super().__init__()
        def _branch(d):
            if d == 1:
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels), nn.GELU())
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3,
                          padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_channels), nn.GELU())
        self.b1  = _branch(1);  self.b6  = _branch(6)
        self.b12 = _branch(12); self.b18 = _branch(18)
        self.gap_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.GELU())
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.GELU(),
            nn.Dropout2d(0.3))

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        gap  = F.interpolate(self.gap_branch(x), (h, w),
                             mode="bilinear", align_corners=False)
        return self.project(
            torch.cat([self.b1(x), self.b6(x),
                       self.b12(x), self.b18(x), gap], dim=1))


class FractographyNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base           = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
        self.backbone  = base.features
        self.aspp      = ASPP(768, 256)
        self.gap       = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(256, 128),
            nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# =========================
# 모델 로드
# =========================
model = FractographyNet(num_classes=4).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("모델 로드 완료:", MODEL_PATH)

# =========================
# 이미지 전처리
# =========================
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =========================
# 예측 + 시각화
# =========================
def predict_and_visualize(image_path):
    pil_img = Image.open(image_path).convert("RGB")

    # 추론
    inp = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(inp)
        probs  = torch.softmax(output, dim=1)[0].cpu().numpy()

    pred_idx   = probs.argmax()
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx]

    # 터미널 출력
    print(f"\n예측 클래스: {pred_class}")
    print(f"신뢰도: {confidence:.4f}")
    print("클래스별 확률:")
    for cls, p in zip(CLASS_NAMES, probs):
        marker = " ◀" if cls == pred_class else ""
        print(f"  {cls:15s}: {p:.4f}{marker}")

    # =========================
    # 시각화: 사진 + 확률 막대
    # =========================
    fig = plt.figure(figsize=(11, 5))

    # 왼쪽: 원본 사진
    ax_img = fig.add_axes([0.02, 0.05, 0.52, 0.90])
    ax_img.imshow(pil_img)
    ax_img.axis("off")
    ax_img.text(
        0.5, 0.97,
        f"{pred_class}  ({confidence:.1%})",
        transform=ax_img.transAxes,
        fontsize=14, fontweight="bold",
        color="white", ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.55)
    )

    # 오른쪽: 확률 막대 그래프
    ax_bar = fig.add_axes([0.60, 0.12, 0.36, 0.75])
    colors = ["#aaaaaa"] * len(CLASS_NAMES)
    colors[pred_idx] = "#FF6B35"

    bars = ax_bar.barh(CLASS_NAMES, probs * 100, color=colors, height=0.5)
    for bar, p in zip(bars, probs):
        ax_bar.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{p:.1%}", va="center", ha="left", fontsize=10
        )

    ax_bar.set_xlim(0, 110)
    ax_bar.set_xlabel("Probability (%)", fontsize=10)
    ax_bar.set_title("Class Probabilities", fontsize=11)
    ax_bar.invert_yaxis()
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # 저장
    base, _ = os.path.splitext(image_path)
    save_path = f"{base}_predict.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n결과 저장: {save_path}")

    return pred_class, confidence, probs


# =========================
# 실행
# =========================
image_path = r"test.jpg"   # ← 테스트할 이미지 경로로 변경
predict_and_visualize(image_path)
