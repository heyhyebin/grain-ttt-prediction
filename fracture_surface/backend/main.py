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

app.mount(
    "/similar_db",
    StaticFiles(directory=SIMILAR_DIR),
    name="similar_db",
)


# ═══════════════════════════════════════════════════════
# CORS 설정
# React(localhost:3000)에서 API 접근 허용
# ═══════════════════════════════════════════════════════

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════
# 디바이스 설정 (GPU 사용 가능하면 CUDA 사용)
# ═══════════════════════════════════════════════════════

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ═══════════════════════════════════════════════════════
# 모델 경로
# ═══════════════════════════════════════════════════════

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "fractography_best5.pth",
)


# ═══════════════════════════════════════════════════════
# 모델 설정값
# ═══════════════════════════════════════════════════════

IMG_SIZE = 224
NUM_CLASSES = 4


# CNN 클래스 이름
CNN_CLASSES = [
    "Cleavage",
    "Ductile",
    "Fatigue",
    "Intergranular",
]


# 영어 → 한국어 라벨 변환
KO_LABELS = {
    "Cleavage": "취성 파괴",
    "Ductile": "연성 파괴",
    "Fatigue": "피로 파괴",
    "Intergranular": "입계 파괴",
}


# ═══════════════════════════════════════════════════════
# [신규 - 도메인 우선순위 룰]
# GradCAM에 잡힌 클래스 후보들 중 이 순서대로 우선순위를 매겨
# 가장 높은 클래스를 최종 단일 유형으로 결정한다.
# 도메인 전문가(교수님) 피드백 반영:
#   - Fatigue: 반복 하중의 근본 원인 신호 → 최우선
#   - Cleavage: 위험한 취성 거동 신호 → 차순위
#   - Intergranular: 재료 열화 신호
#   - Ductile: 정상 과하중 모드 → 최후순위
# ═══════════════════════════════════════════════════════
PRIORITY = ["Fatigue", "Cleavage", "Intergranular", "Ductile"]


# ═══════════════════════════════════════════════════════
# [주석처리 - 구 혼합 판정용 임계값]
# 우선순위 룰 도입으로 더 이상 사용하지 않음.
# 향후 혼합 판정 방식으로 돌아갈 경우 참조용으로 보존.
# ═══════════════════════════════════════════════════════
# MIXED_GAP_THRESHOLD = 15.0       # top1-top2 차이 임계값 (혼합 판정용)
# DUCTILE_SINGLE_THRESHOLD = 60.0  # Ductile 단독 판정 임계값


# ═══════════════════════════════════════════════════════
# Grad-CAM 색상 설정
# 클래스별 contour 색상
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

# ═══════════════════════════════════════════════════════
# 유사 이미지 검색 객체 생성
# CNN feature vector 기반 유사 사례 검색
# ═══════════════════════════════════════════════════════

similar_searcher = SimilarImageSearcher(
    model=model,
    device=DEVICE,

    # 클래스 이름
    class_names=CNN_CLASSES,

    # 유사 이미지 폴더
    similar_dir=SIMILAR_DIR,

    # feature cache 저장 파일
    cache_path=os.path.join(
        os.path.dirname(__file__),
        "similar_cache.pt",
    ),

    image_size=IMG_SIZE,
)

# Grad-CAM++ 객체 생성
gradcam_all = GradCAMPlusPlus(model)


# ═══════════════════════════════════════════════════════
# 이미지 전처리
# 학습 시 사용한 방식과 동일해야 함
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
# Grad-CAM 레이어 생성 함수
# 클래스별 contour만 따로 RGBA 레이어로 생성
# ═══════════════════════════════════════════════════════

def build_layer_rgba(
    img_rgb,
    name,
    solo_mask,
    overlap_mask,
):
    H, W = img_rgb.shape[:2]

    # 현재 클래스 색상
    bgr_c = CLASS_COLORS_BGR[name]

    # RGBA 캔버스 생성
    canvas = np.zeros((H, W, 4), dtype=np.uint8)

    # ═══════════════════════════════════
    # 단독 영역 (실선 contour)
    # ═══════════════════════════════════

    if solo_mask is not None and solo_mask.sum() > 0:

        tmp_solo = np.zeros((H, W, 3), dtype=np.uint8)

        cnts, _ = cv2.findContours(
            solo_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for cnt in cnts:
            cv2.drawContours(
                tmp_solo,
                [cnt],
                -1,
                bgr_c,
                thickness=3,
                lineType=cv2.LINE_AA,
            )

        solo_pixel = np.any(tmp_solo > 0, axis=2)

        canvas[solo_pixel, 0] = bgr_c[2]
        canvas[solo_pixel, 1] = bgr_c[1]
        canvas[solo_pixel, 2] = bgr_c[0]
        canvas[solo_pixel, 3] = 255

    # ═══════════════════════════════════
    # 겹침 영역 (점선 contour)
    # ═══════════════════════════════════

    if overlap_mask is not None and overlap_mask.sum() > 0:

        tmp_overlap = np.zeros((H, W, 3), dtype=np.uint8)

        cnts, _ = cv2.findContours(
            overlap_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for cnt in cnts:
            draw_dashed_contour(
                tmp_overlap,
                cnt,
                bgr_c,
                thickness=3,
                dash_length=12,
            )

        overlap_pixel = np.any(tmp_overlap > 0, axis=2)

        canvas[overlap_pixel, 0] = bgr_c[2]
        canvas[overlap_pixel, 1] = bgr_c[1]
        canvas[overlap_pixel, 2] = bgr_c[0]
        canvas[overlap_pixel, 3] = 255

    return canvas


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
# 이미지 업로드 → CNN 분석 → Grad-CAM → 우선순위 룰 → LLM 설명 생성
# [수정] GradCAM 블록이 prediction 결정보다 앞으로 이동.
#        우선순위 룰이 masks_dict에 의존하기 때문.
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

    # ═══════════════════════════════════
    # 이미지 읽기
    # ═══════════════════════════════════

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    img_rgb = np.array(image)

    H, W = img_rgb.shape[:2]

    # ═══════════════════════════════════
    # 전처리 후 Tensor 변환
    # ═══════════════════════════════════

    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    # ═══════════════════════════════════
    # CNN 추론
    # ═══════════════════════════════════

    with torch.no_grad():

        output = model(input_tensor)

        probs_tensor = torch.softmax(output, dim=1)[0]

    # 확률 정렬 (참고용 — top1/top2 정보는 응답에 함께 보냄)
    sorted_probs, sorted_indices = torch.sort(
        probs_tensor,
        descending=True,
    )

    top1_idx = sorted_indices[0].item()
    top2_idx = sorted_indices[1].item()

    top1_percent = sorted_probs[0].item() * 100
    top2_percent = sorted_probs[1].item() * 100

    # softmax 기준 top1 vs top2 차이 (참고 정보용으로만 응답에 포함)
    gap = top1_percent - top2_percent

    top1_en = CNN_CLASSES[top1_idx]
    top2_en = CNN_CLASSES[top2_idx]

    top1_label = KO_LABELS[top1_en]
    top2_label = KO_LABELS[top2_en]

    # ═══════════════════════════════════════════════════════
    # [수정 - 순서 재배치] Grad-CAM 먼저 생성
    # 우선순위 룰이 masks_dict 정보에 의존하므로
    # prediction 결정 전에 GradCAM을 만들어야 한다.
    # ═══════════════════════════════════════════════════════

    gradcam_image = None
    gradcam_layers = {}
    base_image = None
    gradcam_contours = {}
    masks_dict = {}

    try:

        cams_dict, probs_np = (
            gradcam_all.generate_all_classes(
                input_tensor,
                num_classes=NUM_CLASSES,
            )
        )

        for i, name in enumerate(CNN_CLASSES):

            if probs_np[i] < conf_thresh:
                continue

            mask = cam_to_mask(
                cams_dict[i],
                (W, H),
                cam_percentile=cam_percentile,
                min_area_ratio=min_area_ratio,
            )

            if mask.sum() > 0:
                masks_dict[name] = mask

        # 단독 영역 / 겹침 영역 분리
        if masks_dict:

            solo_masks, overlap_masks = (
                split_solo_overlap(masks_dict)
            )

        else:

            solo_masks, overlap_masks = {}, {}

        # 전체 contour 이미지 생성
        gradcam_image = to_b64(
            build_contour_image(
                img_rgb,
                solo_masks,
                overlap_masks,
            )
        )

        # 클래스별 레이어 생성
        for name in CNN_CLASSES:

            solo_mask = solo_masks.get(
                name,
                np.zeros((H, W), dtype=np.uint8),
            )

            overlap_mask = overlap_masks.get(
                name,
                np.zeros((H, W), dtype=np.uint8),
            )

            gradcam_layers[name] = to_b64(
                build_layer_rgba(
                    img_rgb,
                    name,
                    solo_mask,
                    overlap_mask,
                )
            )

        base_image = to_b64(img_rgb)

        gradcam_contours = extract_contours_json(
            masks_dict,
            (H, W),
        )

    except Exception as e:

        print(f"Grad-CAM++ 레이어 생성 오류: {e}")
        # masks_dict는 빈 채로 폴백 로직으로 진행

    # ═══════════════════════════════════════════════════════
    # [신규] 우선순위 룰 기반 단일 분류 결정
    #
    # 1. GradCAM에 잡힌 클래스(masks_dict의 키)만 후보로 본다.
    # 2. PRIORITY = [Fatigue, Cleavage, Intergranular, Ductile]
    #    순서로 우선순위가 높은 클래스를 최종 분류로 선택한다.
    # 3. 만약 GradCAM이 어떤 클래스도 잡지 못한 드문 경우에는
    #    softmax argmax로 폴백한다.
    # ═══════════════════════════════════════════════════════

    if masks_dict:
        detected = [name for name in PRIORITY if name in masks_dict]
        if detected:
            final_en = detected[0]
        else:
            # masks_dict에 있는 클래스가 모두 PRIORITY에 없는 비정상 케이스
            # (현재 클래스 구성에서는 발생 안 함, 방어적 처리)
            final_en = CNN_CLASSES[int(probs_tensor.argmax().item())]
    else:
        # GradCAM이 어느 클래스도 임계값 통과 못한 경우 → argmax 폴백
        print("[경고] GradCAM에 잡힌 클래스 없음 → softmax argmax 폴백")
        final_en = CNN_CLASSES[int(probs_tensor.argmax().item())]

    final_label = KO_LABELS[final_en]
    final_idx = CNN_CLASSES.index(final_en)
    final_percent = probs_tensor[final_idx].item() * 100

    # 혼합 개념 제거 — 항상 단일
    is_mixed = False
    highlighted_types = [final_label]
    display_prediction = final_label

    pred_en = final_en
    prediction = final_label

    confidence = f"{final_percent:.1f}%"

    # ═══════════════════════════════════
    # 유사 이미지 검색
    # 최종 판정된 유형 내부에서만 Top3 검색
    # ═══════════════════════════════════

    try:

        similar_images = (
            similar_searcher.find_similar_images(
                image=image,

                # 최종 영어 클래스명 사용
                predicted_class=pred_en,

                top_k=3,
            )
        )

    except Exception as e:

        print(f"유사 이미지 검색 오류: {e}")

        similar_images = []

    # ═══════════════════════════════════
    # 신뢰도 상태 분류
    # [수정] is_mixed가 항상 False가 되어
    # 기존의 "혼합 가능성" 메시지 분기는 자동으로 죽음.
    # 단일 유형 기준의 메시지로 정리.
    # ═══════════════════════════════════

    if final_percent >= 80:

        confidence_status = "high"

        confidence_message = (
            "현재 분석은 신뢰할 수 있는 결과입니다."
        )

    elif final_percent >= 60:

        confidence_status = "medium"

        confidence_message = (
            "결과 해석에 주의가 필요합니다."
        )

    else:

        confidence_status = "low"

        confidence_message = (
            "신뢰도가 낮아 오분류 가능성이 있습니다. "
            "추가 이미지나 전문가 검토가 필요할 수 있습니다."
        )

    # ═══════════════════════════════════
    # 클래스별 확률 정리 (카드의 유사도 표시용 — softmax 그대로)
    # ═══════════════════════════════════

    similarities = {
        KO_LABELS[CNN_CLASSES[i]]:
        f"{probs_tensor[i].item() * 100:.1f}%"

        for i in range(len(CNN_CLASSES))
    }

    # ═══════════════════════════════════
    # LLM 설명 생성 (최종 판정된 유형 기준)
    # ═══════════════════════════════════

    llm_result = generate_llm_analysis(
        prediction=prediction,
        confidence_percent=final_percent,
        material=material,
    )

    # ═══════════════════════════════════
    # 최종 응답 반환
    # [참고] is_mixed/top1_type/top2_type/mixed_gap 키는
    # 프론트엔드 호환성을 위해 유지하되, is_mixed는 항상 False.
    # top1/top2는 softmax 기준 정보로 그대로 응답.
    # ═══════════════════════════════════

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

        "base_image": base_image,

        "gradcam_contours": gradcam_contours,

        "similar_images": similar_images,
    }


# ═══════════════════════════════════════════════════════
# 결과 비교 API
# 여러 분석 결과를 LLM으로 비교
# ═══════════════════════════════════════════════════════

@app.post("/compare")
async def compare_analysis(payload: dict):

    items = payload.get("items", [])

    # 최소 2개 필요
    if len(items) < 2:

        return {
            "type_difference":
            "비교하려면 최소 2개의 분석 결과가 필요합니다.",

            "confidence_difference": "",
            "cause_difference": "",
            "final_opinion": "",

            "compare_summary":
            "비교하려면 최소 2개의 분석 결과가 필요합니다.",
        }

    # LLM 비교 생성
    try:

        compare_result = generate_compare_analysis(items)

    except Exception as e:

        print(f"비교 설명 생성 오류: {e}")

        compare_result = {
            "type_difference":
            "선택한 결과들은 예측된 파손 유형에서 차이가 있을 수 있습니다.",

            "confidence_difference":
            "신뢰도 차이에 따라 해석 우선순위를 다르게 볼 필요가 있습니다.",

            "cause_difference":
            "예상 원인은 각 파손 유형의 특징에 따라 다르게 해석될 수 있습니다.",

            "final_opinion":
            "두 결과는 함께 비교해서 보되, "
            "실제 판단에는 추가 이미지나 전문가 검토가 필요할 수 있습니다.",
        }

    # 문자열만 반환된 경우 처리
    if isinstance(compare_result, str):

        return {
            "type_difference": "",
            "confidence_difference": "",
            "cause_difference": "",

            "final_opinion": compare_result,

            "compare_summary": compare_result,
        }

    # 최종 요약
    compare_result["compare_summary"] = (
        compare_result.get(
            "final_opinion",
            "분석 결과 비교가 완료되었습니다.",
        )
    )

    return compare_result


# ═══════════════════════════════════════════════════════
# 직접 실행 시 FastAPI 서버 실행
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )