# ═══════════════════════════════════════════════════════
# 기본 라이브러리 import
# ═══════════════════════════════════════════════════════

import io
import os

import cv2
import numpy as np
import torch

from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from torchvision import transforms


# ═══════════════════════════════════════════════════════
# 프로젝트 내부 모듈 import
# ═══════════════════════════════════════════════════════
# 유사 이미지
from similar_search import SimilarImageSearcher

# CNN 모델 로드 함수
from model import load_model

# Grad-CAM++ 관련 함수
from gradcam import (
    GradCAMPlusPlus,
    cam_to_mask,
    split_solo_overlap,
    build_contour_image,
    draw_dashed_contour,
    extract_contours_json,
    to_b64,
)

# LLM 분석 함수
from llm_service import (
    generate_llm_analysis,
    generate_compare_analysis,
)


# ═══════════════════════════════════════════════════════
# FastAPI 앱 생성
# ═══════════════════════════════════════════════════════

app = FastAPI(title="Fractography Analysis API")

# ═══════════════════════════════════════════════════════
# 유사 이미지 폴더 설정
# 프론트에서 이미지 접근 가능하도록 static 연결
# ex) /similar_db/Fatigue/xxx.jpg
# ═══════════════════════════════════════════════════════

SIMILAR_DIR = os.path.join(
    os.path.dirname(__file__),
    "similar_db",
)

# [수정] 폴더가 없으면 자동 생성 — StaticFiles 마운트 실패 방지
os.makedirs(SIMILAR_DIR, exist_ok=True)
for class_name in ["Cleavage", "Ductile", "Fatigue", "Intergranular"]:
    os.makedirs(os.path.join(SIMILAR_DIR, class_name), exist_ok=True)

app.mount(
    "/similar_db",
    StaticFiles(directory=SIMILAR_DIR),
    name="similar_db",
)


# ═══════════════════════════════════════════════════════
# CORS 설정
# ═══════════════════════════════════════════════════════

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════
# 디바이스 / 모델 경로 / 클래스 설정
# ═══════════════════════════════════════════════════════

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "fractography_best5.pth",
)

IMG_SIZE = 224
NUM_CLASSES = 4

CNN_CLASSES = [
    "Cleavage",
    "Ductile",
    "Fatigue",
    "Intergranular",
]

KO_LABELS = {
    "Cleavage": "취성 파괴",
    "Ductile": "연성 파괴",
    "Fatigue": "피로 파괴",
    "Intergranular": "입계 파괴",
}


# ═══════════════════════════════════════════════════════
# [Ductile 우선 + 박빙 시 우선순위 룰]
# ═══════════════════════════════════════════════════════

DUCTILE_DOMINANT = 0.60
GAP_THRESHOLD = 0.10

PRIORITY_GROUPS = [
    ("Fatigue",),
    ("Cleavage", "Intergranular"),
    ("Ductile",),
]


# ═══════════════════════════════════════════════════════
# Grad-CAM 색상 (BGR — gradcam_layers 컬러 PNG 생성에 사용)
# 동적 시각화에서는 프론트의 CLASS_COLORS(RGB)를 사용함
# ═══════════════════════════════════════════════════════

CLASS_COLORS_BGR = {
    "Cleavage": (245, 130, 59),
    "Ductile": (94, 197, 34),
    "Fatigue": (21, 204, 250),
    "Intergranular": (68, 68, 239),
}


# ═══════════════════════════════════════════════════════
# CNN 모델 로드
# ═══════════════════════════════════════════════════════

model = load_model(
    model_path=MODEL_PATH,
    device=DEVICE,
    num_classes=NUM_CLASSES,
)

similar_searcher = SimilarImageSearcher(
    model=model,
    device=DEVICE,
    class_names=CNN_CLASSES,
    similar_dir=SIMILAR_DIR,
    cache_path=os.path.join(
        os.path.dirname(__file__),
        "similar_cache.pt",
    ),
    image_size=IMG_SIZE,
)

gradcam_all = GradCAMPlusPlus(model)


# ═══════════════════════════════════════════════════════
# 이미지 전처리
# ═══════════════════════════════════════════════════════

preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


# ═══════════════════════════════════════════════════════
# [기존] 컬러 RGBA 레이어 생성 — gradcam_layers (옛 호환·폴백용)
# ═══════════════════════════════════════════════════════

def build_layer_rgba(img_rgb, name, solo_mask, overlap_mask):
    H, W = img_rgb.shape[:2]
    bgr_c = CLASS_COLORS_BGR[name]
    canvas = np.zeros((H, W, 4), dtype=np.uint8)

    if solo_mask is not None and solo_mask.sum() > 0:
        tmp_solo = np.zeros((H, W, 3), dtype=np.uint8)
        cnts, _ = cv2.findContours(
            solo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        for cnt in cnts:
            cv2.drawContours(
                tmp_solo, [cnt], -1, bgr_c,
                thickness=3, lineType=cv2.LINE_AA,
            )
        solo_pixel = np.any(tmp_solo > 0, axis=2)
        canvas[solo_pixel, 0] = bgr_c[2]
        canvas[solo_pixel, 1] = bgr_c[1]
        canvas[solo_pixel, 2] = bgr_c[0]
        canvas[solo_pixel, 3] = 255

    if overlap_mask is not None and overlap_mask.sum() > 0:
        tmp_overlap = np.zeros((H, W, 3), dtype=np.uint8)
        cnts, _ = cv2.findContours(
            overlap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        for cnt in cnts:
            draw_dashed_contour(
                tmp_overlap, cnt, bgr_c,
                thickness=3, dash_length=12,
            )
        overlap_pixel = np.any(tmp_overlap > 0, axis=2)
        canvas[overlap_pixel, 0] = bgr_c[2]
        canvas[overlap_pixel, 1] = bgr_c[1]
        canvas[overlap_pixel, 2] = bgr_c[0]
        canvas[overlap_pixel, 3] = 255

    return canvas


# ═══════════════════════════════════════════════════════
# [신규 추가 — 동적 시각화 활성화]
# 이진 마스크 → 흑백 PNG base64 변환
#
# 프론트엔드 `GradcamView` 컴포넌트는 응답에 `gradcam_masks` 키가 있을 때
# 마스크를 픽셀 데이터로 디코딩해 윤곽선 점 단위 알고리즘으로 실시간 재계산함.
# 이 함수가 없으면 프론트는 모드 B로 폴백되어 백엔드 컬러 PNG를 그대로 그리고,
# 결과적으로 점선/실선 동적 분리가 안 됨.
#
# PIL "L" 모드(8-bit grayscale) + PNG optimize로
# 컬러 PNG 대비 1/3~1/5 용량으로 압축됨.
# ═══════════════════════════════════════════════════════

import base64

def mask_to_b64_png(mask: np.ndarray) -> str:
    """이진 마스크(0/255 uint8 2D 배열) → 흑백 PNG base64 data URL."""
    pil = Image.fromarray(mask.astype(np.uint8), mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


# ═══════════════════════════════════════════════════════
# 서버 동작 확인용 API
# ═══════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "message": "Fractography API is running",
        "model": "ConvNeXt-Small + ASPP",
        "pth": MODEL_PATH,
        "device": str(DEVICE),
    }


# ═══════════════════════════════════════════════════════
# 메인 분석 API
# ═══════════════════════════════════════════════════════

@app.post("/analyze")
async def analyze_fracture(
    file: UploadFile = File(...),
    material: str = Form(""),
    conf_thresh: float = Form(0.05),
    cam_percentile: float = Form(80.0),
    min_area_ratio: float = Form(0.005),
):
    print(f"요청 수신 — 재질: {material}")
    model.eval()

    # 이미지 읽기
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_rgb = np.array(image)
    H, W = img_rgb.shape[:2]

    # 전처리
    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    # CNN 추론
    with torch.no_grad():
        output = model(input_tensor)
        probs_tensor = torch.softmax(output, dim=1)[0]

    sorted_probs, sorted_indices = torch.sort(
        probs_tensor, descending=True,
    )

    top1_idx = sorted_indices[0].item()
    top2_idx = sorted_indices[1].item()
    top1_percent = sorted_probs[0].item() * 100
    top2_percent = sorted_probs[1].item() * 100
    gap = top1_percent - top2_percent

    top1_en = CNN_CLASSES[top1_idx]
    top2_en = CNN_CLASSES[top2_idx]
    top1_label = KO_LABELS[top1_en]
    top2_label = KO_LABELS[top2_en]

    # ═══════════════════════════════════════════════════════
    # Grad-CAM 생성 (prediction 결정보다 앞)
    # ═══════════════════════════════════════════════════════

    gradcam_image = None
    gradcam_layers = {}
    gradcam_masks = {}   # [신규] 동적 시각화용 흑백 마스크
    base_image = None
    gradcam_contours = {}
    masks_dict = {}

    try:
        cams_dict, probs_np = gradcam_all.generate_all_classes(
            input_tensor, num_classes=NUM_CLASSES,
        )

        for i, name in enumerate(CNN_CLASSES):
            if probs_np[i] < conf_thresh:
                continue
            mask = cam_to_mask(
                cams_dict[i], (W, H),
                cam_percentile=cam_percentile,
                min_area_ratio=min_area_ratio,
            )
            if mask.sum() > 0:
                masks_dict[name] = mask

        if masks_dict:
            solo_masks, overlap_masks = split_solo_overlap(masks_dict)
        else:
            solo_masks, overlap_masks = {}, {}

        gradcam_image = to_b64(
            build_contour_image(img_rgb, solo_masks, overlap_masks)
        )

        # 클래스별 컬러 레이어 (옛 호환·폴백)
        for name in CNN_CLASSES:
            s = solo_masks.get(name, np.zeros((H, W), dtype=np.uint8))
            o = overlap_masks.get(name, np.zeros((H, W), dtype=np.uint8))
            gradcam_layers[name] = to_b64(
                build_layer_rgba(img_rgb, name, s, o)
            )

        # [신규] 클래스별 흑백 마스크 PNG — 동적 시각화의 핵심 입력
        # 프론트의 GradcamView가 이 마스크를 디코딩해
        # 활성 클래스 조합으로 실시간 solo/overlap 재계산
        for name in CNN_CLASSES:
            mask = masks_dict.get(name, np.zeros((H, W), dtype=np.uint8))
            gradcam_masks[name] = mask_to_b64_png(mask)

        base_image = to_b64(img_rgb)
        gradcam_contours = extract_contours_json(masks_dict, (H, W))

    except Exception as e:
        print(f"Grad-CAM++ 레이어 생성 오류: {e}")

    # ═══════════════════════════════════════════════════════
    # Ductile 우선 + 박빙 시 우선순위 룰
    # ═══════════════════════════════════════════════════════

    ductile_prob = probs_tensor[CNN_CLASSES.index("Ductile")].item()

    if ductile_prob >= DUCTILE_DOMINANT:
        final_en = "Ductile"
        decision_path = f"STEP 1 (Ductile {ductile_prob*100:.1f}% ≥ {DUCTILE_DOMINANT*100:.0f}%)"
    else:
        candidates = [
            (name, probs_tensor[CNN_CLASSES.index(name)].item())
            for name in CNN_CLASSES
            if name in masks_dict
        ]

        if not candidates:
            final_en = CNN_CLASSES[int(probs_tensor.argmax().item())]
            decision_path = "STEP 2 폴백 (GradCAM 통과 클래스 없음 → argmax)"
            print("[경고] GradCAM 통과 클래스 없음 → softmax argmax 폴백")
        else:
            candidates.sort(key=lambda x: x[1], reverse=True)
            top1_name, top1_prob = candidates[0]

            if len(candidates) == 1:
                final_en = top1_name
                decision_path = f"STEP 3 단일 후보 ({top1_name})"
            else:
                top2_prob = candidates[1][1]
                gap_top12 = top1_prob - top2_prob

                if gap_top12 >= GAP_THRESHOLD:
                    final_en = top1_name
                    decision_path = (
                        f"STEP 3 압도 (차이 {gap_top12*100:.1f}%p ≥ "
                        f"{GAP_THRESHOLD*100:.0f}%p)"
                    )
                else:
                    close_group = [
                        (name, prob) for name, prob in candidates
                        if (top1_prob - prob) < GAP_THRESHOLD
                    ]
                    final_en = None
                    for group in PRIORITY_GROUPS:
                        in_group = [
                            (n, p) for n, p in close_group if n in group
                        ]
                        if in_group:
                            in_group.sort(key=lambda x: x[1], reverse=True)
                            final_en = in_group[0][0]
                            break
                    if final_en is None:
                        final_en = top1_name
                    decision_path = (
                        f"STEP 3 박빙 우선순위 (박빙 {len(close_group)}개, "
                        f"선택: {final_en})"
                    )

    final_label = KO_LABELS[final_en]
    final_idx = CNN_CLASSES.index(final_en)
    final_percent = probs_tensor[final_idx].item() * 100

    print(f"[결정 경로] {decision_path}")

    is_mixed = False
    highlighted_types = [final_label]
    display_prediction = final_label
    pred_en = final_en
    prediction = final_label
    confidence = f"{final_percent:.1f}%"

    # 유사 이미지 검색
    try:
        similar_images = similar_searcher.find_similar_images(
            image=image,
            predicted_class=pred_en,
            top_k=3,
        )
    except Exception as e:
        print(f"유사 이미지 검색 오류: {e}")
        similar_images = []

    # 예측확률 상태 분류
    if final_percent >= 80:
        confidence_status = "high"
        confidence_message = "현재 분석은 신뢰할 수 있는 결과입니다."
    elif final_percent >= 60:
        confidence_status = "medium"
        confidence_message = "결과 해석에 주의가 필요합니다."
    else:
        confidence_status = "low"
        confidence_message = (
            "예측확률이 낮아 오분류 가능성이 있습니다. "
            "추가 이미지나 전문가 검토가 필요할 수 있습니다."
        )

    similarities = {
        KO_LABELS[CNN_CLASSES[i]]:
        f"{probs_tensor[i].item() * 100:.1f}%"
        for i in range(len(CNN_CLASSES))
    }

    # LLM 설명 생성
    llm_result = generate_llm_analysis(
        prediction=prediction,
        confidence_percent=final_percent,
        material=material,
    )

    # 응답
    return {
        "prediction": prediction,
        "prediction_en": pred_en,
        "display_prediction": display_prediction,
        "confidence": confidence,
        "similarities": similarities,
        "is_mixed": is_mixed,
        "mixed_gap": f"{gap:.1f}%",
        "top1_type": top1_label,
        "top2_type": top2_label,
        "highlighted_types": highlighted_types,
        "feature": llm_result["feature"],
        "cause": llm_result.get(
            "cause",
            llm_result.get("expected_cause", ""),
        ),
        "expected_cause": llm_result["expected_cause"],
        "explanation": llm_result["explanation"],
        "llm_analysis": llm_result,
        "material": material,
        "confidence_status": confidence_status,
        "confidence_message": confidence_message,
        "gradcam_image": gradcam_image,
        "gradcam_layers": gradcam_layers,
        "gradcam_masks": gradcam_masks,    # [신규] 동적 시각화 활성화
        "base_image": base_image,
        "gradcam_contours": gradcam_contours,
        "similar_images": similar_images,
    }


# ═══════════════════════════════════════════════════════
# 결과 비교 API (변경 없음)
# ═══════════════════════════════════════════════════════

@app.post("/compare")
async def compare_analysis(payload: dict):
    items = payload.get("items", [])

    if len(items) < 2:
        return {
            "type_difference": "비교하려면 최소 2개의 분석 결과가 필요합니다.",
            "confidence_difference": "",
            "cause_difference": "",
            "final_opinion": "",
            "compare_summary": "비교하려면 최소 2개의 분석 결과가 필요합니다.",
        }

    try:
        compare_result = generate_compare_analysis(items)
    except Exception as e:
        print(f"비교 설명 생성 오류: {e}")
        compare_result = {
            "type_difference": "선택한 결과들은 예측된 파손 유형에서 차이가 있을 수 있습니다.",
            "confidence_difference": "예측확률 차이에 따라 해석 우선순위를 다르게 볼 필요가 있습니다.",
            "cause_difference": "예상 원인은 각 파손 유형의 특징에 따라 다르게 해석될 수 있습니다.",
            "final_opinion": (
                "두 결과는 함께 비교해서 보되, "
                "실제 판단에는 추가 이미지나 전문가 검토가 필요할 수 있습니다."
            ),
        }

    if isinstance(compare_result, str):
        return {
            "type_difference": "",
            "confidence_difference": "",
            "cause_difference": "",
            "final_opinion": compare_result,
            "compare_summary": compare_result,
        }

    compare_result["compare_summary"] = compare_result.get(
        "final_opinion",
        "분석 결과 비교가 완료되었습니다.",
    )
    return compare_result


# ═══════════════════════════════════════════════════════
# 직접 실행
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )