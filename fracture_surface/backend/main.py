import io, os
from typing import Optional

import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from model import FractographyNet, CLASS_NAMES, KO_NAMES, CLASS_FEATURES, CLASS_CAUSES
from gradcam import (
    GradCAMPlusPlus,
    cam_to_mask, split_solo_overlap,
    build_contour_image, draw_dashed_contour,
    extract_contours_json, to_b64,
)

# [수정 - LLM 연동]: missing된 llm_service 연동 함수 정상 임포트
from llm_service import generate_llm_analysis, generate_compare_analysis

# ═══════════════════════════════════════════════════════
# 1. 설정
# ═══════════════════════════════════════════════════════

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "fractography_best.pth"

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# [수정 - 점선 버그 해결]: 실선과 점선이 겹치는 픽셀 구분을 위해 상단에 색상 스펙 보존
CLASS_COLORS_BGR = {
    "Cleavage":      (245, 130, 59),
    "Ductile":       (94,  197, 34),
    "Fatigue":       (21,  204, 250),
    "Intergranular": (68,  68,  239),
}

# ═══════════════════════════════════════════════════════
# 추가/수정 유틸리티 함수
# ═══════════════════════════════════════════════════════

# [수정 - 점선 버그 해결]: 모든 레이어가 점선으로 뭉개지던 원본 연산 구조를 격리 버퍼 방식으로 전면 전치 수정
def build_layer_rgba(img_rgb, name, solo_mask, overlap_mask):
    H, W = img_rgb.shape[:2]
    bgr_c = CLASS_COLORS_BGR[name]
    canvas = np.zeros((H, W, 4), dtype=np.uint8)

    # 단독 영역 (실선) 처리 전용 독립 버퍼 가동
    if solo_mask is not None and solo_mask.sum() > 0:
        tmp_solo = np.zeros((H, W, 3), dtype=np.uint8)
        cnts, _ = cv2.findContours(solo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            cv2.drawContours(tmp_solo, [cnt], -1, bgr_c, thickness=3, lineType=cv2.LINE_AA)
        
        # 실선이 채워진 좌표 픽셀만 정확히 마스킹
        solo_pixel = np.any(tmp_solo > 0, axis=2)
        canvas[solo_pixel, 0] = bgr_c[2]
        canvas[solo_pixel, 1] = bgr_c[1]
        canvas[solo_pixel, 2] = bgr_c[0]
        canvas[solo_pixel, 3] = 255

    # 겹침 영역 (점선) 처리 전용 독립 버퍼 가동
    if overlap_mask is not None and overlap_mask.sum() > 0:
        tmp_overlap = np.zeros((H, W, 3), dtype=np.uint8)
        cnts, _ = cv2.findContours(overlap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            draw_dashed_contour(tmp_overlap, cnt, bgr_c, thickness=3, dash_length=12)
        
        # 점선이 채워진 좌표 픽셀만 타겟팅하여 알파 투명도 부여 (실선과 간섭 원천 차단)
        overlap_pixel = np.any(tmp_overlap > 0, axis=2)
        canvas[overlap_pixel, 0] = bgr_c[2]
        canvas[overlap_pixel, 1] = bgr_c[1]
        canvas[overlap_pixel, 2] = bgr_c[0]
        canvas[overlap_pixel, 3] = 255

    return canvas


# ═══════════════════════════════════════════════════════
# 2. FastAPI 앱
# ═══════════════════════════════════════════════════════

app = FastAPI(title="Fractography Analysis API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model:   Optional[FractographyNet] = None
_gradcam: Optional[GradCAMPlusPlus] = None


@app.on_event("startup")
def startup():
    global _model, _gradcam
    if not os.path.isfile(MODEL_PATH):
        print(f"[경고] 모델 파일 없음: {MODEL_PATH}")
        return
    _model = FractographyNet(num_classes=4).to(DEVICE)
    _model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    _model.eval()
    _gradcam = GradCAMPlusPlus(_model)
    print(f"[startup] 모델 로드 완료 / {DEVICE}")


@app.post("/analyze")
async def analyze(
    file:           UploadFile = File(...),
    material:       str   = Form("unknown"),
    conf_thresh:    float = Form(0.05),
    cam_percentile: float = Form(80.0),
    min_area_ratio: float = Form(0.005),
):
    if _model is None:
        raise HTTPException(503, detail=f"모델 파일 없음: {MODEL_PATH}")

    raw     = await file.read()
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    img_rgb = np.array(pil_img)
    H, W    = img_rgb.shape[:2]

    x = transform(pil_img).unsqueeze(0).to(DEVICE)
    cams_dict, probs = _gradcam.generate_all_classes(x, num_classes=4)

    pred_idx  = int(probs.argmax())
    pred_name = CLASS_NAMES[pred_idx]
    pred_prob = float(probs[pred_idx])

    # 마스크 생성 구간 (원본 로직 완벽 유지)
    masks_dict = {}
    for i, name in enumerate(CLASS_NAMES):
        if probs[i] < conf_thresh:
            continue
        mask = cam_to_mask(cams_dict[i], (W, H),
                           cam_percentile=cam_percentile,
                           min_area_ratio=min_area_ratio)
        if mask.sum() > 0:
            masks_dict[name] = mask

    if masks_dict:
        solo_masks, overlap_masks = split_solo_overlap(masks_dict)
    else:
        solo_masks, overlap_masks = {}, {}

    gradcam_b64 = to_b64(build_contour_image(img_rgb, solo_masks, overlap_masks))

    # 클래스별 RGBA 레이어 생성 루프
    gradcam_layers = {}
    for name in CLASS_NAMES:
        s = solo_masks.get(name,    np.zeros((H, W), dtype=np.uint8))
        o = overlap_masks.get(name, np.zeros((H, W), dtype=np.uint8))
        # [수정 - 점선 버그 해결]: 위에서 상향 패치된 격리형 build_layer_rgba 함수를 적용하여 레이어 적재
        gradcam_layers[name] = to_b64(build_layer_rgba(img_rgb, name, s, o))

    base_image_b64   = to_b64(img_rgb)
    gradcam_contours = extract_contours_json(masks_dict, (H, W))

    # 혼합 / 신뢰도 판정 구간 (원본 로직 완벽 유지)
    si        = probs.argsort()[::-1]
    top1_name = CLASS_NAMES[si[0]]
    top1_prob = float(probs[si[0]])
    top2_name = CLASS_NAMES[si[1]]
    gap       = top1_prob - float(probs[si[1]])
    is_mixed  = gap < 0.15

    highlighted = [KO_NAMES[top1_name]] + ([KO_NAMES[top2_name]] if is_mixed else [])

    if pred_prob >= 0.75:
        conf_status = "high"
    elif pred_prob >= 0.50:
        conf_status = "medium"
    else:
        conf_status = "low"

    similarities = {KO_NAMES[n]: f"{float(probs[i]):.1%}" for i, n in enumerate(CLASS_NAMES)}

    # [수정 - LLM 연동]: 임포트한 대형언어모델 분석 모듈 정상 동기화 가동 및 호출문 안착
    llm_result = generate_llm_analysis(
        prediction=KO_NAMES[pred_name],
        confidence_percent=pred_prob * 100,
        material=material,
    )

    return {
        "prediction":         pred_name,
        "display_prediction": f"{KO_NAMES[pred_name]} ({pred_name})",
        "confidence":         f"{pred_prob:.1%}",
        "confidence_status":  conf_status,
        "confidence_message": llm_result["explanation"],  # [수정 - LLM 연동]: 동적 LLM 분석 설명문 매핑
        "is_mixed":           is_mixed,
        "top1_type":          KO_NAMES[top1_name],
        "top2_type":          KO_NAMES[top2_name],
        "mixed_gap":          f"{gap:.1%}p",
        "highlighted_types":  highlighted,
        "similarities":       similarities,
        "feature":            llm_result["feature"],          # [수정 - LLM 연동]: 하드코딩 맵 대신 LLM 추출 정보 매핑
        "expected_cause":     llm_result["expected_cause"],   # [수정 - LLM 연동]: 하드코딩 맵 대신 LLM 추출 정보 매핑
        "explanation":        llm_result["explanation"],      # [수정 - LLM 연동]: 하드코딩 맵 대신 LLM 추출 정보 매핑
        "material":           material,
        "gradcam_image":      gradcam_b64,
        "gradcam_layers":     gradcam_layers,
        "base_image":         base_image_b64,
        "gradcam_contours":   gradcam_contours,
        "llm_analysis":       llm_result,
    }


# [수정 - LLM 연동]: 누락되었던 /compare 다중 리포트 연동 엔드포인트 누락 없이 보존 통합
@app.post("/compare")
async def compare_analysis(payload: dict):
    items = payload.get("items", [])

    if len(items) < 2:
        return {
            "compare_summary": "비교하려면 최소 2개의 분석 결과가 필요합니다."
        }

    try:
        compare_summary = generate_compare_analysis(items)
    except Exception as e:
        print(f"비교 설명 생성 오류: {e}")
        compare_summary = (
            "선택한 분석 결과들은 파손 유형, 신뢰도, 재질, 주요 특징에서 차이가 있습니다. "
            "신뢰도가 낮은 결과는 추가 이미지나 전문가 검토가 필요할 수 있으며, "
            "각 결과는 단일 판단보다 비교 관점에서 함께 해석하는 것이 좋습니다."
        )

    return {
        "compare_summary": compare_summary
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
