"""
visualize.py
GradCAM 기반으로 파면 예측 영역을 윤곽선으로 표시합니다.

사용법:
    python visualize.py               ← test.jpg 자동 사용
    python visualize.py --image 파일명.jpg
    python visualize.py --image 파일명.jpg --cam_thresh 0.4
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms, models

# ──────────────────────────────────────
# 모델 정의 (train.py와 동일)
# ──────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(in_channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(),
            nn.Linear(mid, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx  = self.fc(self.max_pool(x).view(b, c))
        return self.sigmoid(avg + mx).view(b, c, 1, 1) * x


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1))) * x


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.conv1  = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                    nn.BatchNorm2d(out_channels), nn.ReLU())
        self.conv6  = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,
                                              padding=6, dilation=6, bias=False),
                                    nn.BatchNorm2d(out_channels), nn.ReLU())
        self.conv12 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,
                                              padding=12, dilation=12, bias=False),
                                    nn.BatchNorm2d(out_channels), nn.ReLU())
        self.conv18 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,
                                              padding=18, dilation=18, bias=False),
                                    nn.BatchNorm2d(out_channels), nn.ReLU())
        self.gap    = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                    nn.BatchNorm2d(out_channels), nn.ReLU())
        self.project = nn.Sequential(nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels), nn.ReLU(),
                                     nn.Dropout(0.3))

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        gap  = F.interpolate(self.gap(x), size=(h, w),
                             mode='bilinear', align_corners=False)
        return self.project(torch.cat([self.conv1(x), self.conv6(x),
                                       self.conv12(x), self.conv18(x), gap], dim=1))


class FractographyModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base           = models.efficientnet_b2(weights=None)
        self.backbone  = base.features
        self.aspp      = ASPP(1408, 256)
        self.cbam      = CBAM(256, 16)
        self.gap       = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.aspp(feat)
        feat = self.cbam(feat)
        feat = self.gap(feat)
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat)
# ──────────────────────────────────────


# ──────────────────────────────────────
# GradCAM
# ──────────────────────────────────────
class GradCAM:
    """CBAM 출력에 GradCAM 훅을 걸어 공간적 활성화 맵을 생성합니다."""

    def __init__(self, model):
        self.model      = model
        self.gradients  = None
        self.activations = None
        # ASPP → CBAM 출력이 가장 의미있는 feature map
        model.cbam.register_forward_hook(self._save_activation)
        model.cbam.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, x, class_idx=None):
        """
        Returns:
            cam        : (H, W) numpy array, 0~1 정규화
            class_idx  : 예측(또는 지정) 클래스 인덱스
            probs      : (num_classes,) softmax 확률
        """
        self.model.eval()
        out   = self.model(x)
        probs = torch.softmax(out, dim=1)[0]

        if class_idx is None:
            class_idx = probs.argmax().item()

        self.model.zero_grad()
        out[0, class_idx].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam     = cam.squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, probs.detach().cpu().numpy()
# ──────────────────────────────────────


DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH  = "fractography_cnn_best.pth"
CLASS_NAMES = ["Cleavage", "Ductile", "Fatigue", "Intergranular"]
IMG_SIZE    = 224

# 클래스별 색상 (BGR)
CLASS_COLORS_BGR = {
    "Cleavage":      (0,   140, 255),
    "Ductile":       (180,  60, 220),
    "Fatigue":       (0,   200,  80),
    "Intergranular": (255, 180,   0),
}
# matplotlib용 (RGB 0~1)
CLASS_COLORS_MPL = {
    k: tuple(v / 255 for v in reversed(bgr))
    for k, bgr in CLASS_COLORS_BGR.items()
}

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# =========================
# 메인 처리 함수
# =========================
def process_image(image_path, model, gradcam, cam_thresh=0.4):
    pil_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil_img.size
    orig_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 추론
    x = transform(pil_img).unsqueeze(0).to(DEVICE)
    x.requires_grad_(True)
    cam, pred_idx, probs = gradcam.generate(x)

    pred_name = CLASS_NAMES[pred_idx]
    color_bgr = CLASS_COLORS_BGR[pred_name]

    # CAM → 원본 크기 업스케일
    cam_resized = cv2.resize(cam, (orig_w, orig_h))

    # 이진화 → 모폴로지 정리 → 윤곽선
    cam_uint8 = (cam_resized * 255).astype(np.uint8)
    _, thresh = cv2.threshold(cam_uint8, int(255 * cam_thresh), 255, cv2.THRESH_BINARY)
    kernel_c  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    kernel_o  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    thresh    = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_c)
    thresh    = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel_o)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours  = [c for c in contours if cv2.contourArea(c) >= 1000]

    # 결과 이미지 합성
    result  = orig_bgr.copy()
    overlay = orig_bgr.copy()
    cv2.fillPoly(overlay, contours, color_bgr)
    cv2.addWeighted(overlay, 0.25, result, 0.75, 0, result)

    for cnt in contours:
        cv2.drawContours(result, [cnt], -1, color_bgr, 2)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            label = f"{pred_name} {probs[pred_idx]:.1%}"
            # 배경 박스
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(result, (cx - 4, cy - th - bl - 2), (cx + tw + 4, cy + bl), (0, 0, 0), -1)
            cv2.putText(result, label, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2, cv2.LINE_AA)

    # 확률 목록 (좌상단) — 배경 박스
    line_h = 28
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
        c   = CLASS_COLORS_BGR[name]
        txt = f"{name}: {prob:.1%}"
        y   = 28 + i * line_h
        (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(result, (6, y - th - 2), (6 + tw + 4, y + bl + 2), (0, 0, 0), -1)
        cv2.putText(result, txt, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)

    # 저장
    base, _ = os.path.splitext(image_path)
    save_path = f"{base}_gradcam.png"
    cv2.imwrite(save_path, result)
    print(f"저장 완료 → {save_path}")

    # 화면 출력
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(pil_img)
    axes[0].set_title("원본 이미지", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"GradCAM 예측: {pred_name} ({probs[pred_idx]:.1%})", fontsize=13)
    axes[1].axis("off")

    legend = [
        mpatches.Patch(color=CLASS_COLORS_MPL[n], label=f"{n}: {probs[i]:.1%}")
        for i, n in enumerate(CLASS_NAMES)
    ]
    axes[1].legend(handles=legend, loc="lower right", fontsize=9,
                   framealpha=0.8)

    plt.tight_layout()
    plt.savefig(f"{base}_gradcam_plot.png", dpi=150)
    plt.show()
    print(f"플롯 저장 → {base}_gradcam_plot.png")


# =========================
# 메인
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      type=str,   default="test.jpg",
                        help="이미지 경로 (기본: test.jpg)")
    parser.add_argument("--cam_thresh", type=float, default=0.4,
                        help="GradCAM 이진화 임계값 0~1 (기본 0.4, 낮을수록 영역 넓어짐)")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"이미지를 찾을 수 없습니다: {args.image}")
        return

    model = FractographyModel(num_classes=4).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("모델 로드 완료:", MODEL_PATH)

    gradcam = GradCAM(model)
    process_image(args.image, model, gradcam, cam_thresh=args.cam_thresh)


if __name__ == "__main__":
    main()
