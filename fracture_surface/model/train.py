"""
train.py — 파손단면 CNN 분류 (ConvNeXt-Small + ASPP + GradCAM++)
==================================================================
백본: ConvNeXt-Small
  - 7×7 Depthwise Conv → 국소 텍스처(파손 패턴) 포착에 유리
  - Gradient flow가 깨끗 → GradCAM++ 히트맵 선명
  - SE/CBAM 중복 없음 → ASPP만 붙여 다중 스케일 커버
  - ImageNet 83.1% (EfficientNet-B2: 82.1%)

어텐션: ASPP (1024 → 256, BN+Dropout)
  - CBAM 제거: ConvNeXt LayerScale과 중복
  - ASPP만으로 dilation 1/6/12/18 다중 스케일 커버

시각화: GradCAM++ (visualize.py 분리)
  - 일반 GradCAM보다 작은 물체/국소 영역에 정확
  - 파손단면처럼 텍스처가 분산된 이미지에 적합

파일 구조:
  DATA_DIR/
    Cleavage/
    Ductile/
    Fatigue/
    Intergranular/

사용법:
  python train.py                  # 단일 학습
  python train.py --cv             # 5-Fold CV
  python train.py --epochs 80      # epoch 조정
  python train.py --no-mixup       # Mixup 비활성화
  python train.py --folds 3        # 3-Fold CV
  python train.py --resume         # 마지막 체크포인트에서 이어서 학습
  python train.py --cv --resume    # CV 모드에서 이어서 학습 (완료된 fold 건너뜀)

체크포인트 파일:
  checkpoints/
    checkpoint_latest.pth          # 단일 학습용 (매 epoch 덮어씀)
    checkpoint_fold{N}_latest.pth  # CV fold별 (매 epoch 덮어씀)
    fractography_best.pth          # 최고 성능 모델 (별도 보관)
"""

import os
import copy
import time
import json
import random
import shutil
import argparse
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import convnext_small, ConvNeXt_Small_Weights

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, recall_score, precision_recall_fscore_support,
)

# ================================================================
# 0. 시드 고정
# ================================================================
SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================================================
# 1. 하이퍼파라미터
# ================================================================
IMG_SIZE            = 224
NUM_CLASSES         = 4
BATCH_SIZE          = 16
EPOCHS              = 60
LEARNING_RATE       = 1e-4
WEIGHT_DECAY        = 1e-4
FREEZE_EPOCHS       = 7       # backbone freeze 유지 epoch 수
EARLY_STOP_PATIENCE = 15
VAL_RATIO           = 0.15
N_FOLDS             = 5

# Cleavage Recall 보호 — 데이터 균형 개선으로 패널티 완화
# (Cleavage 188장으로 여전히 가장 적으므로 완전 제거하지 않고 낮게 유지)
CLEAVAGE_RECALL_FLOOR  = 0.50  # 0.70 → 0.50 완화
CLEAVAGE_PENALTY_SCALE = 0.3   # 0.50 → 0.30 완화

DATA_DIR       = r"C:\Project\Fractography"
MODEL_SAVE_DIR = "checkpoints"

CLASS_NAMES = ["Cleavage", "Ductile", "Fatigue", "Intergranular"]

# ================================================================
# 2. Augmentation
# ================================================================

# 전 클래스 공통 augmentation (데이터 균형 개선으로 차등 적용 폐지)
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 64, IMG_SIZE + 64)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(
        degrees=45,
        interpolation=transforms.InterpolationMode.BILINEAR,
        fill=[int(0.485*255), int(0.456*255), int(0.406*255)],
    ),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.15),
    transforms.RandomGrayscale(p=0.15),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0)),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
    transforms.RandomAffine(degrees=0, shear=15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
])

# 검증 / GradCAM용
val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ================================================================
# 3. 커스텀 Dataset
# ================================================================

class UniformTransformDataset(torch.utils.data.Dataset):
    """전 클래스 동일 augmentation (데이터 균형 개선 후 차등 적용 불필요)"""

    def __init__(self, image_folder, indices, transform):
        self.dataset   = image_folder
        self.indices   = indices
        self.transform = transform

    def __getitem__(self, idx):
        path, label = self.dataset.samples[self.indices[idx]]
        return self.transform(self.dataset.loader(path)), label

    def __len__(self):
        return len(self.indices)

    @property
    def targets(self):
        return [self.dataset.targets[i] for i in self.indices]


class SimpleSubset(torch.utils.data.Dataset):
    """검증 / 단일 transform 서브셋"""

    def __init__(self, image_folder, indices, transform):
        self.dataset   = image_folder
        self.indices   = indices
        self.transform = transform

    def __getitem__(self, idx):
        path, label = self.dataset.samples[self.indices[idx]]
        return self.transform(self.dataset.loader(path)), label

    def __len__(self):
        return len(self.indices)

    @property
    def targets(self):
        return [self.dataset.targets[i] for i in self.indices]


# ================================================================
# 4. 모델: ConvNeXt-Small + ASPP
# ================================================================

class ASPP(nn.Module):
    """
    BN + Dropout 포함 ASPP.
    ConvNeXt-Small backbone 출력 채널: 768
    dilation: 1 / 6 / 12 / 18 + GAP branch → 프로젝션 256ch

    ConvNeXt는 LayerScale이 내장되어 있어 CBAM 불필요.
    ASPP만으로 다중 스케일 수용야를 커버한다.
    """

    def __init__(self, in_channels: int = 768, out_channels: int = 256):
        super().__init__()

        def _branch(dilation: int) -> nn.Sequential:
            if dilation == 1:
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),           # ConvNeXt 기본 활성화와 통일
                )
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3,
                          padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )

        self.b1  = _branch(1)
        self.b6  = _branch(6)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        gap  = F.interpolate(self.gap_branch(x), size=(h, w),
                             mode="bilinear", align_corners=False)
        return self.project(
            torch.cat([self.b1(x), self.b6(x),
                       self.b12(x), self.b18(x), gap], dim=1)
        )


class FractographyNet(nn.Module):
    """
    ConvNeXt-Small + ASPP + GAP + Classifier

    forward 흐름:
      backbone  → (B, 768, 7, 7)    ※ 224px 입력 기준
      aspp      → (B, 256, 7, 7)    ← GradCAM++ target: self.aspp
      gap       → (B, 256)
      classifier → (B, num_classes)

    GradCAM++ target layer: model.aspp  (ASPP 출력 직후)
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()
        base           = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
        # ConvNeXt features[0..7] → 마지막 LayerNorm 전까지
        self.backbone  = base.features
        self.aspp      = ASPP(in_channels=768, out_channels=256)
        self.gap       = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.LayerNorm(128),   # ConvNeXt 스타일에 맞춰 LayerNorm 사용
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)   # (B, 768, 7, 7)
        x = self.aspp(x)       # (B, 256, 7, 7)  ← GradCAM++ target
        x = self.gap(x)        # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ================================================================
# 5. FocalLoss
# ================================================================

class FocalLoss(nn.Module):
    """
    FocalLoss + 클래스 가중치 + Label Smoothing.
    gamma=2: 쉬운 샘플(pt 높음) 억제, 어려운 샘플 집중.
    """

    def __init__(self, weight=None, gamma: float = 2.0,
                 label_smoothing: float = 0.05):
        super().__init__()
        self.weight          = weight
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            inputs, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ================================================================
# 6. 유틸리티
# ================================================================

def get_source_id(filename: str) -> str:
    """
    파일명에서 원본 이미지 ID를 추출.
    슬라이딩 윈도우 패치: F_1_x0_y0.jpg  → "F_1"
    원본 이미지:          F_1.JPG         → "F_1"
    언더스코어 없는 경우: img001.jpg      → "img001"  (확장자 제거)
    """
    import re
    stem = os.path.splitext(filename)[0]          # 확장자 제거
    # _x숫자_y숫자 패턴 제거 → 원본 ID만 남김
    source = re.sub(r'_x\d+_y\d+$', '', stem)
    return source


def group_aware_split(full_ds, val_ratio: float, seed: int):
    """
    원본 이미지(source_id) 단위로 train/val 분리.
    같은 원본에서 나온 패치들이 train/val에 동시에 들어가지 않도록 방지.

    Returns:
        tr_idx, val_idx: 파일 인덱스 리스트
    """
    from collections import defaultdict

    # source_id별 (index, label) 그룹핑
    groups = defaultdict(list)
    for i, (path, label) in enumerate(full_ds.samples):
        filename  = os.path.basename(path)
        source_id = get_source_id(filename)
        # 클래스도 key에 포함 → Stratified 분리 가능
        groups[(label, source_id)].append(i)

    # 클래스별로 source 그룹 목록 수집
    from collections import defaultdict as dd
    class_groups = dd(list)
    for (label, sid), idxs in groups.items():
        class_groups[label].append((sid, idxs))

    rng = np.random.RandomState(seed)
    tr_idx, val_idx = [], []

    for label, sid_list in class_groups.items():
        rng.shuffle(sid_list)
        n_val = max(1, int(len(sid_list) * val_ratio))
        val_sids  = sid_list[:n_val]
        train_sids = sid_list[n_val:]
        for _, idxs in val_sids:
            val_idx.extend(idxs)
        for _, idxs in train_sids:
            tr_idx.extend(idxs)

    return tr_idx, val_idx

def make_loss_fn(device):
    """
    데이터 균형 개선 후 → 균등 가중치 FocalLoss.
    Ductile(1119장)이 여전히 많으므로 FocalLoss gamma=2로 다수 클래스 억제 유지.
    클래스 가중치는 제거 (Cleavage/Fatigue/Intergranular 모두 200~400장대로 균형).
    """
    return FocalLoss(weight=None, gamma=2.0, label_smoothing=0.05)


def mixup_data(x, y, alpha: float = 0.2):
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def custom_score(preds, labels, cleavage_idx: int,
                 floor: float = CLEAVAGE_RECALL_FLOOR,
                 scale: float = CLEAVAGE_PENALTY_SCALE):
    macro_f1        = f1_score(labels, preds, average="macro", zero_division=0)
    per_recall      = recall_score(labels, preds, average=None, zero_division=0)
    cleavage_recall = per_recall[cleavage_idx]
    penalty         = max(0.0, floor - cleavage_recall) * scale
    return macro_f1 - penalty, cleavage_recall


# ================================================================
# 7. 체크포인트 저장 / 불러오기
# ================================================================

def save_checkpoint(ckpt_path: str, epoch: int, model, optimizer, scheduler,
                    best_score: float, best_wts, history: dict,
                    backbone_unfrozen: bool, fold_id=""):
    """
    매 epoch 끝에 호출. 중단 후 정확히 다음 epoch부터 재개 가능.
    저장 내용: epoch / model / best_wts / optimizer / scheduler /
               best_score / history / backbone_unfrozen / fold_id
    """
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    torch.save({
        "epoch":             epoch,
        "model_state":       model.state_dict(),
        "best_wts":          best_wts,
        "optimizer_state":   optimizer.state_dict(),
        "scheduler_state":   scheduler.state_dict(),
        "best_score":        best_score,
        "history":           history,
        "backbone_unfrozen": backbone_unfrozen,
        "fold_id":           fold_id,
    }, ckpt_path)


def load_checkpoint(ckpt_path: str, model, optimizer, scheduler, device):
    """
    체크포인트 불러오기.
    Returns: start_epoch, best_score, best_wts, history, backbone_unfrozen
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    print(f"  체크포인트 로드: epoch {ckpt['epoch']+1}까지 완료")
    print(f"  → epoch {ckpt['epoch']+2}부터 재개")
    return (
        ckpt["epoch"] + 1,
        ckpt["best_score"],
        ckpt["best_wts"],
        ckpt["history"],
        ckpt["backbone_unfrozen"],
    )


# ================================================================
# 8. 학습 / 검증
# ================================================================

def train_one_epoch(model, loader, criterion, optimizer, device,
                    use_mixup: bool = True):
    model.train()
    running_loss = correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if use_mixup:
            images, la, lb, lam = mixup_data(images, labels)
            out  = model(images)
            loss = mixup_criterion(criterion, out, la, lb, lam)
            _, preds = torch.max(out, 1)
            correct += (lam * (preds == la).float().sum().item()
                        + (1 - lam) * (preds == lb).float().sum().item())
        else:
            out  = model(images)
            loss = criterion(out, labels)
            _, preds = torch.max(out, 1)
            correct += (preds == labels).sum().item()

        total        += labels.size(0)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = correct = total = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            out   = model(images)
            loss  = criterion(out, labels)
            probs = torch.softmax(out, dim=1)
            _, preds = torch.max(out, 1)

            running_loss += loss.item() * images.size(0)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (running_loss / total, correct / total,
            np.array(all_preds), np.array(all_labels), np.array(all_probs))


# ================================================================
# 8. 학습 루프 (단일 fold)
# ================================================================

def train_fold(model, train_loader, val_loader, criterion, device,
               epochs, freeze_epochs, lr, patience,
               cleavage_idx, use_mixup=True, fold_id="", resume=False):
    """
    resume=True 이면 아래 경로에서 이어서 학습.
      단일 학습: checkpoints/checkpoint_latest.pth
      CV Fold N: checkpoints/checkpoint_foldN_latest.pth
    학습 완료(early stop 또는 전체 epoch) 시 latest 파일은 자동 삭제.
    최고 모델은 best_foldN.pth / fractography_best.pth 로 별도 보존.
    """
    # 체크포인트 경로
    ckpt_name = (f"checkpoint_fold{fold_id}_latest.pth" if fold_id
                 else "checkpoint_latest.pth")
    ckpt_path = os.path.join(MODEL_SAVE_DIR, ckpt_name)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=1, eta_min=1e-6
    )

    prefix            = f"[Fold {fold_id}] " if fold_id else ""
    fatigue_idx       = CLASS_NAMES.index("Fatigue")
    start_epoch       = 0
    best_score        = -999.0
    patience_cnt      = 0
    backbone_unfrozen = False

    history = {k: [] for k in [
        "train_loss", "val_loss", "val_acc",
        "val_f1", "custom_score", "cleavage_recall",
    ]}

    # ── 체크포인트 재개 ──────────────────────────────────────
    if resume and os.path.exists(ckpt_path):
        print(f"\n{prefix}[Resume] 체크포인트 발견: {ckpt_path}")
        start_epoch, best_score, best_wts, history, backbone_unfrozen = (
            load_checkpoint(ckpt_path, model, optimizer, scheduler, device)
        )
        patience_cnt = 0
    elif resume:
        print(f"{prefix}[Resume] 체크포인트 없음 → 처음부터 시작")
        best_wts = copy.deepcopy(model.state_dict())
    else:
        best_wts = copy.deepcopy(model.state_dict())

    # ── backbone 고정/해제 상태 복원 ─────────────────────────
    if backbone_unfrozen:
        for p in model.backbone.parameters():
            p.requires_grad = True
        print(f"{prefix}backbone: 해제 상태로 복원")
    else:
        for p in model.backbone.parameters():
            p.requires_grad = False
        if start_epoch == 0:
            print(f"\n{prefix}처음 {freeze_epochs}ep: ASPP + Classifier만 학습")

    # ── 학습 루프 ─────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        # backbone 해제 시점
        if not backbone_unfrozen and epoch >= freeze_epochs:
            for p in model.backbone.parameters():
                p.requires_grad = True
            for pg in optimizer.param_groups:
                pg["lr"] = lr * 0.1
            backbone_unfrozen = True
            print(f"{prefix}[{epoch+1}ep] 전체 파인튜닝 시작 (LR={lr*0.1:.2e})")

        apply_mixup = use_mixup and backbone_unfrozen
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, apply_mixup
        )
        val_loss, val_acc, preds, labels, _ = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        val_f1           = f1_score(labels, preds, average="macro", zero_division=0)
        score, cl_recall = custom_score(preds, labels, cleavage_idx)

        for k, v in zip(
            ["train_loss", "val_loss", "val_acc", "val_f1", "custom_score", "cleavage_recall"],
            [tr_loss, val_loss, val_acc, val_f1, score, cl_recall]
        ):
            history[k].append(v)

        print(
            f"{prefix}[{epoch+1:02d}/{epochs}] "
            f"TrLoss:{tr_loss:.4f} TrAcc:{tr_acc:.4f} | "
            f"ValLoss:{val_loss:.4f} ValAcc:{val_acc:.4f} "
            f"F1:{val_f1:.4f} CLRec:{cl_recall:.4f} Score:{score:.4f} "
            f"LR:{optimizer.param_groups[0]['lr']:.2e}"
        )

        # 클래스별 정확도
        for i, cls in enumerate(CLASS_NAMES):
            mask = labels == i
            n = mask.sum()
            if n > 0:
                acc = (preds[mask] == i).mean()
                print(f"  {cls:15s}: {int(acc*n):2d}/{n} ({acc:.3f})")

        # Cleavage ↔ Fatigue 혼동 모니터
        c2f = ((labels == cleavage_idx) & (preds == fatigue_idx)).sum()
        f2c = ((labels == fatigue_idx)  & (preds == cleavage_idx)).sum()
        print(f"  [혼동] Cleavage→Fatigue:{c2f}건 | Fatigue→Cleavage:{f2c}건")

        # 최고 모델 갱신
        if score > best_score:
            best_score   = score
            best_wts     = copy.deepcopy(model.state_dict())
            patience_cnt = 0
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            best_name = (f"best_fold{fold_id}.pth" if fold_id else "fractography_best.pth")
            torch.save(best_wts, os.path.join(MODEL_SAVE_DIR, best_name))
            print(f"  ✓ 최고 모델 갱신 (score={best_score:.4f}) → {best_name}")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"\n{prefix}[Early Stop] {patience}ep 개선 없음 → 종료")
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
                    print(f"  latest 체크포인트 삭제 (학습 완료): {ckpt_path}")
                break

        # ── 매 epoch: latest 체크포인트 저장 ─────────────────
        save_checkpoint(
            ckpt_path, epoch, model, optimizer, scheduler,
            best_score, best_wts, history, backbone_unfrozen, fold_id
        )

    else:
        # 모든 epoch 정상 완료 시
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
            print(f"{prefix}latest 체크포인트 삭제 (전체 epoch 완료): {ckpt_path}")

    return best_wts, history, best_score


# ================================================================
# 9. 최종 평가 + 시각화 저장
# ================================================================

def final_evaluation(model, val_loader, criterion, device, class_names,
                     history, save_prefix="result"):

    print("\n" + "=" * 60)
    print("  최종 평가")
    print("=" * 60)

    val_loss, val_acc, preds, labels, probs = validate(
        model, val_loader, criterion, device
    )

    print(f"\n정확도: {val_acc:.4f}  |  손실: {val_loss:.4f}")
    print(f"\n{classification_report(labels, preds, target_names=class_names, digits=4, zero_division=0)}")

    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    header = "예측→".ljust(16) + "".join(f"{n:>14s}" for n in class_names)
    print(header); print("-" * len(header))
    for i, cls in enumerate(class_names):
        row = f"실제 {cls}".ljust(16) + "".join(f"{cm[i,j]:14d}" for j in range(len(class_names)))
        print(row)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(class_names))), zero_division=0
    )

    print(f"\n{'─'*60}")
    weakest_cls, weakest_f1 = None, 1.0
    for i, cls in enumerate(class_names):
        mask = labels == i; n = mask.sum()
        if n == 0: continue
        cm_mask  = (labels == i) & (preds == i)
        avg_conf = probs[cm_mask, i].mean() if cm_mask.sum() > 0 else 0.0
        wm       = (labels == i) & (preds != i)
        confused = (class_names[Counter(preds[wm]).most_common(1)[0][0]],
                    Counter(preds[wm]).most_common(1)[0][1]) if wm.sum() > 0 else ("없음", 0)
        print(f"\n  [{cls}]  n={n}  P:{precision[i]:.4f}  R:{recall[i]:.4f}  F1:{f1[i]:.4f}")
        print(f"    확신도(정답): {avg_conf:.4f}  |  주요혼동: {confused[0]}({confused[1]}건)")
        if f1[i] < weakest_f1:
            weakest_f1 = f1[i]; weakest_cls = cls

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    w_f1     = f1_score(labels, preds, average="weighted", zero_division=0)
    print(f"\n  Accuracy:{val_acc:.4f}  Macro F1:{macro_f1:.4f}  Weighted F1:{w_f1:.4f}")
    if weakest_cls:
        print(f"  가장 약한 클래스: {weakest_cls} (F1={weakest_f1:.4f})")
    print(f"{'─'*60}")

    # ── 학습 곡선 + Confusion Matrix 저장 ──
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    axes[0][0].plot(history["train_loss"], label="Train")
    axes[0][0].plot(history["val_loss"],   label="Val")
    axes[0][0].set_title("Loss"); axes[0][0].legend()

    axes[0][1].plot(history["val_f1"],       label="Macro F1",    color="green")
    axes[0][1].plot(history["custom_score"], label="Custom Score", color="blue", ls="--")
    axes[0][1].set_title("F1 & Custom Score"); axes[0][1].legend()

    axes[0][2].plot(history["val_acc"], label="Val Acc", color="purple")
    axes[0][2].set_title("Val Accuracy"); axes[0][2].legend()

    axes[1][0].plot(history["cleavage_recall"], label="Cleavage Recall", color="orange")
    axes[1][0].axhline(CLEAVAGE_RECALL_FLOOR, color="red", ls="--",
                       label=f"Floor={CLEAVAGE_RECALL_FLOOR}")
    axes[1][0].set_title("Cleavage Recall"); axes[1][0].legend()

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1][1])
    axes[1][1].set_xlabel("Predicted"); axes[1][1].set_ylabel("True")
    axes[1][1].set_title("Confusion Matrix")

    # 클래스별 F1 bar
    axes[1][2].bar(class_names, f1, color=["#4c72b0","#55a868","#c44e52","#8172b2"])
    axes[1][2].axhline(macro_f1, color="red", ls="--", label=f"Macro={macro_f1:.3f}")
    axes[1][2].set_title("Per-Class F1"); axes[1][2].set_ylim(0, 1)
    axes[1][2].legend()

    plt.tight_layout()
    curve_path = f"{save_prefix}_training_curves.png"
    plt.savefig(curve_path, dpi=150); plt.close()
    print(f"\n학습 곡선 저장 → {curve_path}")

    # JSON 저장
    results = {
        "accuracy": float(val_acc), "macro_f1": float(macro_f1),
        "weighted_f1": float(w_f1), "val_loss": float(val_loss),
        "per_class": {
            cls: {"precision": float(precision[i]), "recall": float(recall[i]),
                  "f1": float(f1[i]), "support": int(support[i])}
            for i, cls in enumerate(class_names)
        },
        "confusion_matrix": cm.tolist(),
    }
    json_path = f"{save_prefix}_results.json"
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"결과 JSON 저장 → {json_path}")

    return results


# ================================================================
# 10. 단일 학습 모드
# ================================================================

def run_single_train(args):
    print("\n" + "=" * 60)
    print("  단일 학습 모드 | 백본: ConvNeXt-Small")
    print("=" * 60)

    full_ds = datasets.ImageFolder(args.data_dir, transform=None)
    class_names = full_ds.classes
    print(f"클래스: {class_names}")

    if sorted(class_names) != sorted(CLASS_NAMES):
        raise ValueError(f"클래스 불일치\n현재: {class_names}\n예상: {CLASS_NAMES}")

    cleavage_idx = class_names.index("Cleavage")
    all_labels   = [s[1] for s in full_ds.samples]
    for cls in class_names:
        print(f"  {cls}: {Counter(all_labels)[full_ds.class_to_idx[cls]]}장")

    # train / val 분리 — 원본 이미지 단위로 나눠 슬라이딩 윈도우 패치 누수 방지
    tr_idx, val_idx = group_aware_split(full_ds, VAL_RATIO, SEED)
    print(f"  → train {len(tr_idx)}장 / val {len(val_idx)}장 (원본 단위 분리)")

    # 균형 잡힌 데이터 → 전 클래스 동일 augmentation + 일반 shuffle
    train_data = UniformTransformDataset(full_ds, tr_idx, train_transform)
    val_data   = SimpleSubset(full_ds, val_idx, val_transform)

    criterion  = make_loss_fn(DEVICE)

    train_loader = DataLoader(train_data, BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_data,   BATCH_SIZE, shuffle=False, num_workers=0)

    set_seed(SEED)
    model = FractographyNet(NUM_CLASSES).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nConvNeXt-Small + ASPP  |  파라미터: {n_params:,}")
    print("GradCAM++ target layer: model.aspp  (visualize.py 참조)")

    best_wts, history, _ = train_fold(
        model, train_loader, val_loader, criterion, DEVICE,
        args.epochs, FREEZE_EPOCHS, LEARNING_RATE, EARLY_STOP_PATIENCE,
        cleavage_idx, use_mixup=not args.no_mixup, resume=args.resume,
    )

    model.load_state_dict(best_wts)
    final_evaluation(model, val_loader, criterion, DEVICE,
                     class_names, history, save_prefix="single")


# ================================================================
# 11. K-Fold CV 모드
# ================================================================

def run_kfold_cv(args):
    print("\n" + "=" * 60)
    print(f"  {args.folds}-Fold Stratified CV | 백본: ConvNeXt-Small")
    print("=" * 60)

    full_ds      = datasets.ImageFolder(args.data_dir, transform=None)
    class_names  = full_ds.classes
    cleavage_idx = class_names.index("Cleavage")
    all_targets  = np.array(full_ds.targets)
    all_indices  = np.arange(len(full_ds))

    print(f"전체: {len(full_ds)}장")
    for cls in class_names:
        idx = full_ds.class_to_idx[cls]
        print(f"  {cls}: {(all_targets == idx).sum()}장")

    # 원본 이미지 단위로 fold 분리 (슬라이딩 윈도우 패치 누수 방지)
    from collections import defaultdict
    import re

    # source_id별 그룹 구성
    groups = defaultdict(list)
    for i, (path, label) in enumerate(full_ds.samples):
        stem      = os.path.splitext(os.path.basename(path))[0]
        source_id = re.sub(r'_x\d+_y\d+$', '', stem)
        groups[(label, source_id)].append(i)

    # 클래스별 source 목록
    class_source_list = defaultdict(list)
    for (label, sid), idxs in groups.items():
        class_source_list[label].append((sid, idxs))

    # 각 클래스 내에서 source를 folds개로 나눔 → 패치 인덱스로 변환
    rng = np.random.RandomState(SEED)
    for label in class_source_list:
        rng.shuffle(class_source_list[label])

    def build_cv_folds(class_source_list, n_folds):
        """클래스별로 source를 n_folds로 나눠 fold별 (tr_idx, val_idx) 반환"""
        fold_val_indices = [[] for _ in range(n_folds)]
        for label, sid_list in class_source_list.items():
            for fi, (sid, idxs) in enumerate(sid_list):
                fold_val_indices[fi % n_folds].extend(idxs)
        result = []
        all_idx = [i for idxs in fold_val_indices for i in idxs]
        for fi in range(n_folds):
            val   = fold_val_indices[fi]
            train = [i for i in all_idx if i not in set(val)]
            result.append((np.array(train), np.array(val)))
        return result

    cv_folds     = build_cv_folds(class_source_list, args.folds)
    fold_results = []
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    for fold_idx, (tr_idx, val_idx) in enumerate(cv_folds, start=1):
        print(f"\n{'━'*60}  Fold {fold_idx}/{args.folds}  {'━'*60}")

        # 이미 완료된 fold 건너뜀 (best 파일은 있는데 latest 체크포인트는 없는 경우)
        best_path = os.path.join(MODEL_SAVE_DIR, f"best_fold{fold_idx}.pth")
        ckpt_path = os.path.join(MODEL_SAVE_DIR, f"checkpoint_fold{fold_idx}_latest.pth")
        if args.resume and os.path.exists(best_path) and not os.path.exists(ckpt_path):
            print(f"  Fold {fold_idx}: 이미 완료된 fold → 건너뜀")
            # 결과 복원을 위해 best 모델로 간단히 평가
            model_tmp = FractographyNet(NUM_CLASSES).to(DEVICE)
            model_tmp.load_state_dict(torch.load(best_path, map_location=DEVICE))
            crit_tmp = make_loss_fn(DEVICE)
            vd_tmp = SimpleSubset(full_ds, val_idx.tolist(), val_transform)
            vl_tmp = DataLoader(vd_tmp, BATCH_SIZE, shuffle=False, num_workers=0)
            _, vacc, vp, vl, _ = validate(model_tmp, vl_tmp, crit_tmp, DEVICE)
            vf1 = f1_score(vl, vp, average="macro", zero_division=0)
            sc, _ = custom_score(vp, vl, cleavage_idx)
            pcf1 = f1_score(vl, vp, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
            fold_results.append({
                "fold": fold_idx, "accuracy": float(vacc),
                "macro_f1": float(vf1), "custom_score": float(sc),
                "per_class_f1": {class_names[i]: float(pcf1[i]) for i in range(NUM_CLASSES)},
            })
            continue

        # 균형 잡힌 데이터 → 전 클래스 동일 augmentation + 일반 shuffle
        train_data = UniformTransformDataset(full_ds, tr_idx.tolist(), train_transform)
        val_data   = SimpleSubset(full_ds, val_idx.tolist(), val_transform)

        criterion  = make_loss_fn(DEVICE)

        train_loader = DataLoader(train_data, BATCH_SIZE, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_data,   BATCH_SIZE, shuffle=False, num_workers=0)

        set_seed(SEED + fold_idx)
        model = FractographyNet(NUM_CLASSES).to(DEVICE)

        best_wts, history, best_score = train_fold(
            model, train_loader, val_loader, criterion, DEVICE,
            args.epochs, FREEZE_EPOCHS, LEARNING_RATE, EARLY_STOP_PATIENCE,
            cleavage_idx, use_mixup=not args.no_mixup, fold_id=fold_idx,
            resume=args.resume,
        )

        model.load_state_dict(best_wts)
        _, val_acc, preds, labels, _ = validate(model, val_loader, criterion, DEVICE)
        fold_f1      = f1_score(labels, preds, average="macro", zero_division=0)
        per_cls_f1   = f1_score(labels, preds, average=None,
                                labels=list(range(NUM_CLASSES)), zero_division=0)

        fold_results.append({
            "fold": fold_idx, "accuracy": float(val_acc),
            "macro_f1": float(fold_f1), "custom_score": float(best_score),
            "per_class_f1": {class_names[i]: float(per_cls_f1[i])
                             for i in range(NUM_CLASSES)},
        })

        torch.save(best_wts, os.path.join(MODEL_SAVE_DIR, f"fold{fold_idx}.pth"))
        print(f"\n  Fold {fold_idx}: Acc={val_acc:.4f}  F1={fold_f1:.4f}  Score={best_score:.4f}")

    # ── 요약 ──
    print("\n" + "=" * 60)
    accs    = [r["accuracy"]     for r in fold_results]
    f1s     = [r["macro_f1"]     for r in fold_results]
    scores  = [r["custom_score"] for r in fold_results]

    print(f"  Accuracy:     {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Macro F1:     {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  Custom Score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print(f"\n  클래스별 F1:")
    for cls in class_names:
        vals = [r["per_class_f1"][cls] for r in fold_results]
        print(f"    {cls:15s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    best_fi  = int(np.argmax(scores))
    best_fld = fold_results[best_fi]
    src = os.path.join(MODEL_SAVE_DIR, f"fold{best_fld['fold']}.pth")
    dst = os.path.join(MODEL_SAVE_DIR, "fractography_best.pth")
    shutil.copy2(src, dst)
    print(f"\n  최고 Fold {best_fld['fold']} (Score={best_fld['custom_score']:.4f}) → {dst}")

    cv_results = {
        "n_folds": args.folds, "backbone": "ConvNeXt-Small",
        "accuracy_mean": float(np.mean(accs)),   "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),    "macro_f1_std": float(np.std(f1s)),
        "custom_score_mean": float(np.mean(scores)), "custom_score_std": float(np.std(scores)),
        "per_class_f1_mean": {
            cls: float(np.mean([r["per_class_f1"][cls] for r in fold_results]))
            for cls in class_names
        },
        "folds": fold_results,
    }
    with open("cv_results.json", "w", encoding="utf-8") as fp:
        json.dump(cv_results, fp, ensure_ascii=False, indent=2)
    print("  CV 결과 → cv_results.json")


# ================================================================
# 12. 메인
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(description="파손단면 CNN 분류 (ConvNeXt-Small)")
    p.add_argument("--data-dir",  default=DATA_DIR)
    p.add_argument("--epochs",    type=int, default=EPOCHS)
    p.add_argument("--folds",     type=int, default=N_FOLDS)
    p.add_argument("--cv",        action="store_true", help="K-Fold CV")
    p.add_argument("--no-mixup",  action="store_true", help="Mixup 비활성화")
    p.add_argument("--resume",    action="store_true",
                   help="체크포인트에서 이어서 학습 (checkpoints/checkpoint_*.pth)")
    return p.parse_args()


if __name__ == "__main__":
    args  = parse_args()
    start = time.time()

    print(f"디바이스: {DEVICE}")
    print(f"모드: {'K-Fold CV' if args.cv else '단일 학습'}")

    if args.cv:
        run_kfold_cv(args)
    else:
        run_single_train(args)

    print(f"\n총 소요: {(time.time()-start)/60:.1f}분  |  완료!")
