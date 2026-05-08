"""
predict.py — 단일 이미지 GradCAM++ 오버레이 시각화
====================================================
사용법:
    python predict.py                        <- test.jpg 자동 사용
    python predict.py --image 파일명.jpg
    python predict.py --image 파일명.jpg --cam_thresh 0.4
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as colormap_module
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_small, ConvNeXt_Small_Weights

# ================================================================
# 1. 모델 정의 (train.py와 동일)
# ================================================================

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


# ================================================================
# 2. GradCAM++
# ================================================================

class GradCAMPlusPlus:
    def __init__(self, model):
        self.model       = model
        self.activations = None
        self.gradients   = None
        model.aspp.register_forward_hook(self._save_activation)
        model.aspp.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, x, class_idx=None):
        self.model.eval()
        out   = self.model(x)
        probs = torch.softmax(out, dim=1)[0]

        if class_idx is None:
            class_idx = probs.argmax().item()

        self.model.zero_grad()
        out[0, class_idx].backward()

        grads    = self.gradients
        acts     = self.activations
        grads_sq = grads ** 2
        grads_cu = grads ** 3
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)
        denom    = 2 * grads_sq + sum_acts * grads_cu
        denom    = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha    = grads_sq / denom
        weights  = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, probs.detach().cpu().numpy()


# ================================================================
# 3. 설정
# ================================================================

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH  = "checkpoints/fractography_best.pth"
CLASS_NAMES = ["Cleavage", "Ductile", "Fatigue", "Intergranular"]
IMG_SIZE    = 224

CLASS_COLORS_MPL = {
    "Cleavage":      (1.0, 0.55, 0.0),
    "Ductile":       (0.7, 0.24, 0.86),
    "Fatigue":       (0.0, 0.78, 0.31),
    "Intergranular": (1.0, 0.71, 0.0),
}

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ================================================================
# 4. 처리 함수
# ================================================================

def predict_and_visualize(image_path, model, gradcam, cam_alpha=0.5):
    pil_img = Image.open(image_path).convert("RGB")
    img_np  = np.array(pil_img)

    # 추론
    x = transform(pil_img).unsqueeze(0).to(DEVICE)
    x.requires_grad_(True)
    cam, pred_idx, probs = gradcam.generate(x)

    pred_name = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx]

    # 터미널 출력
    print(f"\n예측 클래스: {pred_name}")
    print(f"신뢰도: {confidence:.4f}")
    print("클래스별 확률:")
    for cls, p in zip(CLASS_NAMES, probs):
        marker = " <" if cls == pred_name else ""
        print(f"  {cls:15s}: {p:.4f}{marker}")

    # CAM → 원본 크기로 부드럽게 업스케일
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (pil_img.width, pil_img.height), Image.BILINEAR)
    ) / 255.0

    # 히트맵 오버레이
    heatmap = (colormap_module.jet(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    overlay = (cam_alpha * heatmap + (1 - cam_alpha) * img_np).astype(np.uint8)

    # ── 시각화 ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # 왼쪽: 원본
    axes[0].imshow(pil_img)
    axes[0].set_title("원본 이미지", fontsize=13)
    axes[0].axis("off")

    # 오른쪽: GradCAM++ 오버레이
    axes[1].imshow(overlay)
    axes[1].set_title(f"GradCAM++  |  예측: {pred_name} ({confidence:.1%})",
                      fontsize=13)
    axes[1].axis("off")

    # 범례 (확률 목록)
    legend = [
        mpatches.Patch(color=CLASS_COLORS_MPL[n],
                       label=f"{n}: {probs[i]:.1%}")
        for i, n in enumerate(CLASS_NAMES)
    ]
    axes[1].legend(handles=legend, loc="lower right",
                   fontsize=10, framealpha=0.8)

    plt.tight_layout()

    # 저장
    base, _ = os.path.splitext(image_path)
    plot_path = f"{base}_gradcam_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n결과 저장 -> {plot_path}")

    return pred_name, confidence, probs


# ================================================================
# 5. 메인
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",     type=str,   default="test.jpg",
                        help="이미지 경로 (기본: test.jpg)")
    parser.add_argument("--cam_alpha", type=float, default=0.5,
                        help="히트맵 투명도 0~1 (기본 0.5, 높을수록 히트맵 강조)")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"이미지를 찾을 수 없습니다: {args.image}")
        return

    model = FractographyNet(num_classes=4).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("모델 로드 완료:", MODEL_PATH)

    gradcam = GradCAMPlusPlus(model)
    predict_and_visualize(args.image, model, gradcam, cam_alpha=args.cam_alpha)


if __name__ == "__main__":
    main()
