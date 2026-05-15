"""
visualize_interactive.py — GradCAM++ 인터랙티브 뷰어
=====================================================
기능:
    - 마우스 호버 → 원형 돋보기 (배율 조절 가능)
    - 마우스 호버 → 해당 클래스 영역 하이라이트 + 툴팁 (클래스명, 확률, 단독/겹침)
    - 클릭 → 우측 패널에 클래스 상세 정보 + CAM 강도 표시
    - 우측 패널 → 클래스별 확률 바 (실시간)
    - 키보드: +/- 배율, ESC 선택 해제, S 저장

사용법:
    python visualize.py
    python visualize.py --image 파일명.jpg
    python visualize.py --image 파일명.jpg --conf_thresh 0.05 --cam_percentile 80
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
from matplotlib.patches import FancyArrowPatch
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_small, ConvNeXt_Small_Weights

matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']

# ═══════════════════════════════════════════════════════
# 1. 모델 (train.py와 동일)
# ═══════════════════════════════════════════════════════

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
            nn.Dropout(0.2), nn.Linear(128, 4))

    def forward(self, x):
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ═══════════════════════════════════════════════════════
# 2. GradCAM++
# ═══════════════════════════════════════════════════════

class GradCAMPlusPlus:
    def __init__(self, model):
        self.model       = model
        self.activations = None
        self.gradients   = None
        model.aspp.register_forward_hook(self._save_activation)
        model.aspp.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate_for_class(self, x, class_idx):
        self.model.eval()
        self.model.zero_grad()
        out   = self.model(x)
        probs = torch.softmax(out, dim=1)[0]
        score = out[0, class_idx]
        score.backward(retain_graph=False)
        grads    = self.gradients
        acts     = self.activations.detach()
        grads_sq = grads ** 2
        grads_cu = grads ** 3
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)
        denom    = 2 * grads_sq + sum_acts * grads_cu
        denom    = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha    = grads_sq / denom
        weights  = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        cam_max = cam.max()
        if cam_max > 1e-8:
            cam = (cam - cam.min()) / (cam_max - cam.min() + 1e-8)
        else:
            cam = np.zeros_like(cam)
        return cam, probs.detach().cpu().numpy()

    def generate_all_classes(self, x, num_classes=4):
        cams = {}
        probs_final = None
        for cls_idx in range(num_classes):
            x_input = x.clone().detach().requires_grad_(True)
            cam, probs = self.generate_for_class(x_input, cls_idx)
            cams[cls_idx] = cam
            probs_final = probs
        return cams, probs_final


# ═══════════════════════════════════════════════════════
# 3. 설정
# ═══════════════════════════════════════════════════════

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH  = "fractography_best5.pth"
CLASS_NAMES = ["Cleavage", "Ductile", "Fatigue", "Intergranular"]

CLASS_COLORS = {
    "Cleavage":      (0.23, 0.51, 0.96),   # 파랑
    "Ductile":       (0.13, 0.77, 0.37),   # 초록
    "Fatigue":       (0.98, 0.80, 0.08),   # 노랑
    "Intergranular": (0.94, 0.27, 0.27),   # 빨강
}
CLASS_COLORS_BGR = {
    "Cleavage":      (245, 130, 59),
    "Ductile":       (94,  197, 34),
    "Fatigue":       (21,  204, 250),
    "Intergranular": (68,  68,  239),
}
CLASS_DESC = {
    "Cleavage":      "취성 파괴. 결정면을 따라 직선적으로 쪼개짐.\n저온 또는 고변형속도 환경에서 발생.",
    "Ductile":       "연성 파괴. 딤플(dimple) 패턴 특징.\n소성 변형 후 최종 파단하는 정상 모드.",
    "Fatigue":       "피로 파괴. 반복 하중에 의한 균열 전파.\n비치 마크(beach mark)가 관찰됨.",
    "Intergranular": "입계 파괴. 결정립 경계를 따라 전파.\n수소 취성·응력 부식 균열과 관련.",
}

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ═══════════════════════════════════════════════════════
# 4. CAM → 마스크
# ═══════════════════════════════════════════════════════

def cam_to_mask(cam, target_size, cam_percentile=80, min_area_ratio=0.005):
    if cam.max() < 1e-6:
        return np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
    cam_resized = cv2.resize(cam, target_size, interpolation=cv2.INTER_LINEAR)
    thresh_val  = np.percentile(cam_resized, cam_percentile)
    mask = (cam_resized >= thresh_val).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = mask.shape[0] * mask.shape[1] * min_area_ratio
    cleaned  = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def split_solo_overlap(masks_dict):
    count_map = np.zeros_like(next(iter(masks_dict.values())), dtype=np.int32)
    for m in masks_dict.values():
        count_map += (m > 0).astype(np.int32)
    solo, overlap = {}, {}
    for name, m in masks_dict.items():
        active = (m > 0)
        solo[name]    = (active & (count_map == 1)).astype(np.uint8) * 255
        overlap[name] = (active & (count_map >= 2)).astype(np.uint8) * 255
    return solo, overlap


def draw_dashed_contour(img, contour, color_bgr, thickness=2, dash_length=10):
    pts = contour.reshape(-1, 2)
    n = len(pts)
    if n < 2:
        return
    i, draw_on = 0, True
    while i < n - 1:
        end_i = min(i + dash_length, n - 1)
        if draw_on:
            for j in range(i, end_i):
                cv2.line(img, tuple(pts[j]), tuple(pts[j+1]),
                         color_bgr, thickness, lineType=cv2.LINE_AA)
        i = end_i
        draw_on = not draw_on


def build_contour_image(img_rgb, solo_masks, overlap_masks):
    """윤곽선이 그려진 RGB 이미지 반환"""
    canvas = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    for name, mask in solo_masks.items():
        if mask.sum() == 0:
            continue
        color = CLASS_COLORS_BGR[name]
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            cv2.drawContours(canvas, [cnt], -1, color, thickness=3, lineType=cv2.LINE_AA)
    for name, mask in overlap_masks.items():
        if mask.sum() == 0:
            continue
        color = CLASS_COLORS_BGR[name]
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            draw_dashed_contour(canvas, cnt, color, thickness=3, dash_length=12)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


# ═══════════════════════════════════════════════════════
# 5. 인터랙티브 뷰어
# ═══════════════════════════════════════════════════════

class InteractiveFractureViewer:
    """
    마우스 호버 → 돋보기 + 영역 하이라이트 + 툴팁
    마우스 클릭 → 우측 패널에 클래스 상세 정보
    키보드 +/- → 배율 조절
    키보드 S   → 현재 화면 저장
    키보드 ESC → 선택 해제
    """

    ZOOM_LEVELS = [2, 3, 4, 5, 6, 8]
    LENS_RADIUS = 70   # px (display)

    def __init__(self, img_rgb, contour_rgb, masks_dict,
                 solo_masks, overlap_masks,
                 cams_dict, probs, image_path,
                 cam_percentile=80):

        self.img_rgb      = img_rgb           # 원본
        self.contour_rgb  = contour_rgb       # 윤곽선 합성
        self.masks_dict   = masks_dict        # {name: binary mask}
        self.solo_masks   = solo_masks
        self.overlap_masks= overlap_masks
        self.cams_dict    = cams_dict         # {idx: cam float}
        self.probs        = probs
        self.image_path   = image_path
        self.cam_percentile = cam_percentile

        self.H, self.W    = img_rgb.shape[:2]
        self.zoom_idx     = 1                 # 기본 ×3
        self.selected_cls = None              # 클릭으로 선택한 클래스명

        # CAM 원본 크기로 리사이즈 (hover용)
        self.cams_full = {}
        for i, name in enumerate(CLASS_NAMES):
            c = cv2.resize(cams_dict[i], (self.W, self.H),
                           interpolation=cv2.INTER_LINEAR)
            c_max = c.max()
            if c_max > 1e-8:
                c = (c - c.min()) / (c_max - c.min() + 1e-8)
            self.cams_full[name] = c

        # 하이라이트 오버레이 (클래스별 반투명 컬러 마스크)
        self.highlight_overlays = {}
        for name in CLASS_NAMES:
            if name not in masks_dict:
                continue
            overlay = np.zeros((self.H, self.W, 4), dtype=np.float32)
            r, g, b = CLASS_COLORS[name]
            m = masks_dict[name].astype(bool)
            overlay[m] = [r, g, b, 0.38]
            self.highlight_overlays[name] = overlay

        self._build_figure()
        self._connect_events()

    # ── 레이아웃 ──────────────────────────────────────

    def _build_figure(self):
        self.fig = plt.figure(figsize=(16, 8), facecolor='#0d0f11')
        self.fig.canvas.manager.set_window_title("Fractography Interactive Viewer")

        # 그리드: [main viewer | info panel]
        gs = self.fig.add_gridspec(
            1, 2, width_ratios=[2.8, 1],
            left=0.01, right=0.99, top=0.97, bottom=0.03,
            wspace=0.02
        )

        # ── 메인 이미지 축 ──
        self.ax_main = self.fig.add_subplot(gs[0])
        self.ax_main.set_facecolor('#080a0b')
        self.ax_main.imshow(self.contour_rgb)
        self.ax_main.axis('off')
        pred_idx   = int(self.probs.argmax())
        pred_name  = CLASS_NAMES[pred_idx]
        pred_color = CLASS_COLORS[pred_name]
        self.ax_main.set_title(
            f"GradCAM++ Viewer  ·  Primary: {pred_name}  ({self.probs[pred_idx]:.1%})  "
            f"│  Hover: magnify + highlight  │  Click: details  │  +/-: zoom  │  S: save",
            fontsize=9, color='#8b9096', pad=6,
            fontfamily='DejaVu Sans'
        )

        # 돋보기용 인셋 축 (원형 clip)
        self.ax_lens = self.fig.add_axes([0, 0, 0.001, 0.001])  # 초기엔 숨김
        self.ax_lens.set_facecolor('#080a0b')
        self.ax_lens.axis('off')
        self.lens_img_obj = self.ax_lens.imshow(
            np.zeros((10, 10, 3), dtype=np.uint8))

        # 십자선
        self.ch_h = self.ax_main.axhline(
            y=-1, color='white', lw=0.6, alpha=0.3, visible=False)
        self.ch_v = self.ax_main.axvline(
            x=-1, color='white', lw=0.6, alpha=0.3, visible=False)

        # 툴팁 텍스트 — edgecolor를 튜플로 수정 (rgba 문자열 불가)
        self.tooltip_box = self.ax_main.text(
            0, 0, '', fontsize=8.5, color='white',
            fontfamily='DejaVu Sans',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#1c1f23',
                      edgecolor=(1.0, 1.0, 1.0, 0.2), alpha=0.92),
            visible=False, zorder=20
        )

        # 하이라이트 오버레이 (imshow 위에 alpha image)
        self.highlight_obj = self.ax_main.imshow(
            np.zeros((self.H, self.W, 4), dtype=np.float32),
            extent=[0, self.W, self.H, 0],
            zorder=5, interpolation='bilinear'
        )

        # ── 우측 정보 패널 ──
        self.ax_info = self.fig.add_subplot(gs[1])
        self.ax_info.set_facecolor('#141619')
        self.ax_info.axis('off')
        self._draw_info_panel(init=True)

        # ── 범례 — edgecolor를 튜플로 수정 ──
        legend_patches = []
        for name in CLASS_NAMES:
            shown = name in self.masks_dict
            lbl   = f"{name}: {self.probs[CLASS_NAMES.index(name)]:.1%}"
            if not shown:
                lbl += " (hidden)"
            legend_patches.append(
                mpatches.Patch(color=CLASS_COLORS[name], label=lbl, alpha=0.85)
            )
        self.ax_main.legend(
            handles=legend_patches,
            loc='lower left', fontsize=8,
            framealpha=0.85,
            facecolor='#1c1f23',
            edgecolor=(1.0, 1.0, 1.0, 0.15),
            labelcolor='white'
        )

    # ── 정보 패널 ──────────────────────────────────────

    def _draw_info_panel(self, init=False, hover_name=None,
                         hover_prob=None, hover_type=None,
                         click_name=None, cam_val=None,
                         mx=None, my=None):
        ax = self.ax_info
        ax.cla()
        ax.set_facecolor('#141619')
        ax.axis('off')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        y = 0.97

        def txt(x, yy, s, **kw):
            ax.text(x, yy, s, transform=ax.transAxes,
                    va='top', **kw)

        # ── 패널 제목 ──
        txt(0.05, y, "CLASS PROBABILITIES",
            fontsize=7.5, color='#555c63',
            fontfamily='DejaVu Sans', fontweight='bold')
        y -= 0.06

        # ── 확률 바 ──
        bar_h = 0.028
        bar_gap = 0.062
        for i, name in enumerate(CLASS_NAMES):
            prob  = self.probs[i]
            color = CLASS_COLORS[name]
            is_hover  = (name == hover_name)
            is_select = (name == self.selected_cls)

            # 배경 하이라이트
            if is_hover or is_select:
                ax.add_patch(plt.Rectangle(
                    (0.02, y - 0.005), 0.96, bar_gap * 0.92,
                    transform=ax.transAxes,
                    facecolor='#1c1f23', edgecolor='none', zorder=0
                ))

            # 클래스 색 점
            ax.add_patch(plt.Circle(
                (0.07, y - 0.008), 0.018,
                transform=ax.transAxes,
                facecolor=color, zorder=2
            ))

            # 클래스명
            txt(0.13, y, name,
                fontsize=8.5, color='white' if (is_hover or is_select) else '#8b9096',
                fontfamily='DejaVu Sans',
                fontweight='bold' if (is_hover or is_select) else 'normal')

            # 확률값
            txt(0.88, y, f"{prob:.1%}",
                fontsize=8, color=color if (is_hover or is_select) else '#555c63',
                fontfamily='DejaVu Mono', ha='right')

            # 바 배경
            ax.add_patch(plt.Rectangle(
                (0.13, y - bar_h - 0.01), 0.74, bar_h,
                transform=ax.transAxes,
                facecolor='#1c1f23', zorder=1
            ))
            # 바 채우기
            ax.add_patch(plt.Rectangle(
                (0.13, y - bar_h - 0.01), 0.74 * prob, bar_h,
                transform=ax.transAxes,
                facecolor=color,
                alpha=0.9 if (is_hover or is_select) else 0.5,
                zorder=2
            ))

            y -= bar_gap

        # ── 구분선 ──
        ax.axhline(y - 0.005, color='#2a2d31', lw=0.8, xmin=0.02, xmax=0.98)
        y -= 0.04

        # ── Hover 정보 ──
        if hover_name:
            txt(0.05, y, "HOVER",
                fontsize=7.5, color='#555c63', fontfamily='DejaVu Sans', fontweight='bold')
            y -= 0.06
            color = CLASS_COLORS[hover_name]
            txt(0.05, y, hover_name,
                fontsize=11, color=color, fontfamily='DejaVu Sans', fontweight='bold')
            y -= 0.055
            type_lbl = "단독 영역" if hover_type == 'solo' else "겹침 영역" if hover_type == 'overlap' else "—"
            txt(0.05, y, f"유형  {type_lbl}",
                fontsize=8.5, color='#8b9096', fontfamily='DejaVu Sans')
            y -= 0.045
            if hover_prob is not None:
                txt(0.05, y, f"확률  {hover_prob:.3f}",
                    fontsize=8.5, color='#8b9096', fontfamily='DejaVu Mono')
            y -= 0.045
            if cam_val is not None:
                txt(0.05, y, f"CAM 강도  {cam_val:.3f}",
                    fontsize=8.5, color='#8b9096', fontfamily='DejaVu Mono')
            y -= 0.045
            ax.axhline(y - 0.005, color='#2a2d31', lw=0.8, xmin=0.02, xmax=0.98)
            y -= 0.04

        # ── 클릭 선택 상세 ──
        sel = self.selected_cls
        if sel:
            sel_idx = CLASS_NAMES.index(sel)
            color   = CLASS_COLORS[sel]
            txt(0.05, y, "SELECTED",
                fontsize=7.5, color='#555c63', fontfamily='DejaVu Sans', fontweight='bold')
            y -= 0.06
            txt(0.05, y, sel,
                fontsize=12, color=color, fontfamily='DejaVu Sans', fontweight='bold')
            y -= 0.06

            # 확률 강조
            prob = self.probs[sel_idx]
            txt(0.05, y, f"{prob:.2%}",
                fontsize=14, color=color, fontfamily='DejaVu Mono', fontweight='bold')
            y -= 0.06

            # CAM 강도 바
            if mx is not None and my is not None:
                cv = self.cams_full[sel][int(my), int(mx)]
            else:
                cv = 0.0
            bar_w = 0.80
            ax.add_patch(plt.Rectangle(
                (0.05, y - 0.022), bar_w, 0.022,
                transform=ax.transAxes, facecolor='#1c1f23'
            ))
            ax.add_patch(plt.Rectangle(
                (0.05, y - 0.022), bar_w * float(prob), 0.022,
                transform=ax.transAxes, facecolor=color, alpha=0.85
            ))
            txt(0.05, y + 0.008, "PROB",
                fontsize=6.5, color='#555c63', fontfamily='DejaVu Sans')
            y -= 0.055

            # 설명
            for line in CLASS_DESC[sel].split('\n'):
                txt(0.05, y, line,
                    fontsize=8, color='#8b9096', fontfamily='DejaVu Sans',
                    wrap=True)
                y -= 0.045

            y -= 0.01
            ax.axhline(y, color='#2a2d31', lw=0.8, xmin=0.02, xmax=0.98)
            y -= 0.04

        # ── 줌 정보 ──
        zoom = self.ZOOM_LEVELS[self.zoom_idx]
        txt(0.05, y, f"ZOOM  ×{zoom}   (+/- to adjust)",
            fontsize=7.5, color='#555c63', fontfamily='DejaVu Sans')
        y -= 0.06

        # ── 단축키 안내 ──
        shortcuts = [
            ("+/-", "배율 조절"),
            ("S",   "화면 저장"),
            ("ESC", "선택 해제"),
        ]
        for key, desc in shortcuts:
            ax.text(0.05, y, key, fontsize=7.5, color='#4af0c4',
                    fontfamily='DejaVu Mono', transform=ax.transAxes, va='top')
            ax.text(0.22, y, desc, fontsize=7.5, color='#555c63',
                    fontfamily='DejaVu Sans', transform=ax.transAxes, va='top')
            y -= 0.04

        self.fig.canvas.draw_idle()

    # ── 이벤트 연결 ───────────────────────────────────

    def _connect_events(self):
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_hover)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event',    self._on_key)
        self.fig.canvas.mpl_connect('axes_leave_event',   self._on_leave)

    # ── 좌표 변환 헬퍼 ────────────────────────────────

    def _event_to_img_coords(self, event):
        """matplotlib 이벤트 좌표 → 이미지 픽셀 좌표"""
        if event.inaxes != self.ax_main:
            return None, None
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return None, None
        ix = int(np.clip(x, 0, self.W - 1))
        iy = int(np.clip(y, 0, self.H - 1))
        return ix, iy

    def _get_class_at(self, ix, iy):
        """픽셀 위치에서 어떤 클래스 마스크에 속하는지 반환"""
        results = []
        for name, mask in self.masks_dict.items():
            if mask[iy, ix] > 0:
                results.append(name)
        if not results:
            return None, None
        # CAM 강도가 가장 높은 클래스 반환
        best = max(results, key=lambda n: self.cams_full[n][iy, ix])
        region_type = 'overlap' if len(results) > 1 else 'solo'
        return best, region_type

    def _update_lens(self, ix, iy):
        """돋보기 인셋 업데이트"""
        zoom   = self.ZOOM_LEVELS[self.zoom_idx]
        r      = self.LENS_RADIUS
        half_s = r / zoom

        # 데이터 좌표로 렌즈 위치 계산
        x0 = max(0, ix - half_s)
        x1 = min(self.W, ix + half_s)
        y0 = max(0, iy - half_s)
        y1 = min(self.H, iy + half_s)

        # 렌즈가 이미지 밖으로 나가지 않도록 위치 조정
        display_x = ix + r * 1.2
        display_y = iy - r * 1.2
        if display_x + r > self.W:
            display_x = ix - r * 2.2
        if display_y - r < 0:
            display_y = iy + r * 1.2

        # figure 좌표로 변환
        trans = self.ax_main.transData
        fig_t = self.fig.transFigure.inverted()
        p0 = fig_t.transform(trans.transform([[display_x - r, display_y - r]])[0])
        p1 = fig_t.transform(trans.transform([[display_x + r, display_y + r]])[0])

        lens_x = p0[0]; lens_y = p0[1]
        lens_w = p1[0] - p0[0]; lens_h = p1[1] - p0[1]
        if lens_w <= 0 or lens_h <= 0:
            return

        self.ax_lens.set_position([lens_x, lens_y, lens_w, lens_h])
        self.ax_lens.set_visible(True)

        # 크롭 이미지 (윤곽선 포함 이미지에서)
        x0i, x1i = max(0, int(ix - half_s)), min(self.W, int(ix + half_s))
        y0i, y1i = max(0, int(iy - half_s)), min(self.H, int(iy + half_s))
        crop = self.contour_rgb[y0i:y1i, x0i:x1i]

        if crop.size == 0:
            return

        # 십자선 오버레이
        crop_draw = crop.copy()
        ch, cw = crop_draw.shape[:2]
        cy_pos  = int((iy - y0i) / (y1i - y0i + 1e-6) * ch) if (y1i > y0i) else ch // 2
        cx_pos  = int((ix - x0i) / (x1i - x0i + 1e-6) * cw) if (x1i > x0i) else cw // 2
        cv2.line(crop_draw, (0, cy_pos), (cw, cy_pos), (255, 255, 255), 1)
        cv2.line(crop_draw, (cx_pos, 0), (cx_pos, ch), (255, 255, 255), 1)

        self.lens_img_obj.set_data(crop_draw)
        self.ax_lens.set_xlim(0, crop_draw.shape[1])
        self.ax_lens.set_ylim(crop_draw.shape[0], 0)

        # 원형 clip patch — edgecolor를 튜플로 수정
        from matplotlib.patches import Circle
        from matplotlib.transforms import Bbox, TransformedBbox

        for patch in self.ax_lens.patches:
            patch.remove()
        # 렌즈 테두리 원
        circle = Circle((0.5, 0.5), 0.5, transform=self.ax_lens.transAxes,
                         fill=False, edgecolor=(1.0, 1.0, 1.0, 0.6),
                         linewidth=2, zorder=10)
        self.ax_lens.add_patch(circle)
        self.lens_img_obj.set_clip_path(circle)

    # ── 이벤트 핸들러 ─────────────────────────────────

    def _on_hover(self, event):
        ix, iy = self._event_to_img_coords(event)
        if ix is None:
            self._hide_lens()
            return

        # 십자선
        self.ch_h.set_ydata([iy, iy]); self.ch_h.set_visible(True)
        self.ch_v.set_xdata([ix, ix]); self.ch_v.set_visible(True)

        # 어떤 클래스 영역인지 파악
        cls_name, region_type = self._get_class_at(ix, iy)
        cam_val = self.cams_full[cls_name][iy, ix] if cls_name else None

        # 하이라이트 오버레이
        if cls_name and cls_name in self.highlight_overlays:
            self.highlight_obj.set_data(self.highlight_overlays[cls_name])
            self.highlight_obj.set_visible(True)
        else:
            self.highlight_obj.set_visible(False)

        # 툴팁
        if cls_name:
            prob = self.probs[CLASS_NAMES.index(cls_name)]
            type_str = "단독 영역" if region_type == 'solo' else "겹침 영역"
            tip = (f"{cls_name}\n"
                   f"확률: {prob:.1%}\n"
                   f"CAM: {cam_val:.3f}\n"
                   f"{type_str}")
            color = CLASS_COLORS[cls_name]
            self.tooltip_box.set_text(tip)
            self.tooltip_box.set_position((ix + 12, iy - 12))
            self.tooltip_box.get_bbox_patch().set_edgecolor(color)
            self.tooltip_box.set_color(color)
            self.tooltip_box.set_visible(True)
        else:
            self.tooltip_box.set_visible(False)

        # 돋보기
        self._update_lens(ix, iy)

        # 정보 패널
        self._draw_info_panel(
            hover_name=cls_name,
            hover_prob=self.probs[CLASS_NAMES.index(cls_name)] if cls_name else None,
            hover_type=region_type,
            click_name=self.selected_cls,
            cam_val=cam_val,
            mx=ix, my=iy
        )

    def _on_click(self, event):
        ix, iy = self._event_to_img_coords(event)
        if ix is None:
            return
        cls_name, region_type = self._get_class_at(ix, iy)
        if cls_name:
            self.selected_cls = cls_name
            print(f"\n[클릭] {cls_name}  |  확률: {self.probs[CLASS_NAMES.index(cls_name)]:.3f}"
                  f"  |  CAM: {self.cams_full[cls_name][iy, ix]:.3f}"
                  f"  |  {region_type}  |  픽셀 ({ix}, {iy})")
        else:
            self.selected_cls = None
        self._draw_info_panel(
            click_name=self.selected_cls,
            mx=ix, my=iy
        )

    def _on_key(self, event):
        if event.key in ('+', '='):
            self.zoom_idx = min(self.zoom_idx + 1, len(self.ZOOM_LEVELS) - 1)
            print(f"[줌] ×{self.ZOOM_LEVELS[self.zoom_idx]}")
            self._draw_info_panel(click_name=self.selected_cls)
        elif event.key == '-':
            self.zoom_idx = max(self.zoom_idx - 1, 0)
            print(f"[줌] ×{self.ZOOM_LEVELS[self.zoom_idx]}")
            self._draw_info_panel(click_name=self.selected_cls)
        elif event.key in ('s', 'S'):
            base, _ = os.path.splitext(self.image_path)
            save_path = f"{base}_interactive_capture.png"
            self.fig.savefig(save_path, dpi=150, bbox_inches='tight',
                             facecolor=self.fig.get_facecolor())
            print(f"[저장] {save_path}")
        elif event.key == 'escape':
            self.selected_cls = None
            self._draw_info_panel()

    def _on_leave(self, event):
        if event.inaxes == self.ax_main:
            self._hide_lens()
            self.ch_h.set_visible(False)
            self.ch_v.set_visible(False)
            self.tooltip_box.set_visible(False)
            self.highlight_obj.set_visible(False)
            self.fig.canvas.draw_idle()

    def _hide_lens(self):
        self.ax_lens.set_visible(False)
        self.fig.canvas.draw_idle()

    def show(self):
        plt.tight_layout(pad=0)
        plt.show()


# ═══════════════════════════════════════════════════════
# 6. 메인
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",          type=str,   default="test.jpg")
    parser.add_argument("--conf_thresh",    type=float, default=0.05)
    parser.add_argument("--cam_percentile", type=float, default=80)
    parser.add_argument("--min_area_ratio", type=float, default=0.005)
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"이미지를 찾을 수 없습니다: {args.image}"); return
    if not os.path.isfile(MODEL_PATH):
        print(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}"); return

    model = FractographyNet(num_classes=4).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"모델 로드: {MODEL_PATH}  /  디바이스: {DEVICE}")

    gradcam = GradCAMPlusPlus(model)

    pil_img  = Image.open(args.image).convert("RGB")
    img_rgb  = np.array(pil_img)
    H, W     = img_rgb.shape[:2]

    x = transform(pil_img).unsqueeze(0).to(DEVICE)
    cams_dict, probs = gradcam.generate_all_classes(x, num_classes=4)

    pred_idx  = int(probs.argmax())
    print(f"\n주 예측: {CLASS_NAMES[pred_idx]}  ({probs[pred_idx]:.1%})")
    for i, n in enumerate(CLASS_NAMES):
        print(f"  {n:15s}: {probs[i]:.4f}")

    # 마스크 생성
    masks_dict = {}
    for i, name in enumerate(CLASS_NAMES):
        if probs[i] < args.conf_thresh:
            continue
        mask = cam_to_mask(cams_dict[i], (W, H),
                           cam_percentile=args.cam_percentile,
                           min_area_ratio=args.min_area_ratio)
        if mask.sum() > 0:
            masks_dict[name] = mask

    if not masks_dict:
        print("경고: 표시할 영역이 없습니다.")
        solo_masks, overlap_masks = {}, {}
    else:
        solo_masks, overlap_masks = split_solo_overlap(masks_dict)

    contour_rgb = build_contour_image(img_rgb, solo_masks, overlap_masks)

    viewer = InteractiveFractureViewer(
        img_rgb, contour_rgb, masks_dict,
        solo_masks, overlap_masks,
        cams_dict, probs, args.image,
        cam_percentile=args.cam_percentile
    )
    viewer.show()


if __name__ == "__main__":
    main()
