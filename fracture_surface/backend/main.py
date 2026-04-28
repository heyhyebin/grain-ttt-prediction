"""
실행:
    uvicorn main:app --reload --port 8000

폴더 구조:
    backend/
        main.py              ← 이 파일
        model.py          ← 모델 정의 (단일 소스)
        llm_service.py       ← LLM 분석 생성
        model/
            fractography_best.pth  ← 학습 완료 가중치
"""

import io
import os

import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms

from llm_service import generate_llm_analysis


# ── model_v5에서 모델 클래스 import ──
# train_v5.py와 동일한 모델 정의를 사용하므로 가중치 호환이 보장됩니다.
from model import FractographyNet


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# .pth 파일 경로 (학습 후 checkpoints/fractography_best.pth를 여기로 복사)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "fractography_best.pth")

CNN_CLASSES = ["Cleavage", "Ductile", "Fatigue", "Intergranular"]

KO_LABELS = {
    "Cleavage": "취성 파괴",
    "Ductile": "연성 파괴",
    "Fatigue": "피로 파괴",
    "Intergranular": "입계 파괴",
}

# 학습 시 사용한 백본 버전과 동일하게 설정하세요.
# train_v5.py에서 --backbone v2로 학습했으면 "v2",
# --backbone v1으로 학습했으면 "v1"
BACKBONE_VERSION = "auto"


# --------------------------------------------------
# 모델 로드
# --------------------------------------------------
# pretrained=False: 추론 서버에서는 ImageNet 가중치를 다운로드할 필요 없음
# .pth 파일에 모든 가중치가 포함되어 있음
model = FractographyNet(
    num_classes=len(CNN_CLASSES),
    pretrained=False,               # ← 핵심: 사전학습 가중치 다운로드 안 함
    backbone_version=BACKBONE_VERSION,
).to(DEVICE)

try:
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✅ 모델 로드 완료: {model.backbone_name} ({DEVICE})")
    print(f"   가중치: {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ 가중치 파일을 찾을 수 없습니다: {MODEL_PATH}")
    print(f"   학습 후 checkpoints/fractography_best.pth를 model/ 폴더로 복사하세요.")
except RuntimeError as e:
    print(f"❌ 가중치 로드 실패 (모델 구조 불일치 가능): {e}")
    print(f"   학습 시 사용한 백본 버전과 BACKBONE_VERSION이 일치하는지 확인하세요.")


# --------------------------------------------------
# 이미지 전처리 (train_v5.py val_transform과 동일)
# --------------------------------------------------
preprocess = transforms.Compose([
    transforms.Resize(256),           # IMG_SIZE + 32
    transforms.CenterCrop(224),       # IMG_SIZE
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# --------------------------------------------------
# API
# --------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "Fractography API is running",
        "model": model.backbone_name,
        "device": str(DEVICE),
    }


@app.post("/analyze")
async def analyze_fracture(
    file: UploadFile = File(...),
    material: str = Form(""),
    environment: str = Form(""),
):
    print(f"요청 수신 — 재질: {material}, 환경: {environment}")

    # 이미지 읽기 및 전처리
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    # CNN 예측
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence_val = probs[0, pred_idx].item()

    pred_en = CNN_CLASSES[pred_idx]
    prediction = KO_LABELS[pred_en]
    confidence_percent = confidence_val * 100
    confidence = f"{confidence_percent:.1f}%"

    # 신뢰도 상태 메시지
    if confidence_percent >= 80:
        confidence_status = "high"
        confidence_message = "현재 분석은 신뢰할 수 있는 결과입니다."
    elif confidence_percent >= 60:
        confidence_status = "medium"
        confidence_message = "결과 해석에 주의가 필요합니다."
    else:
        confidence_status = "low"
        confidence_message = (
            "신뢰도가 낮아 오분류 가능성이 있습니다. "
            "추가 이미지나 전문가 검토가 필요할 수 있습니다."
        )

    # 클래스별 확률
    similarities = {
        KO_LABELS[CNN_CLASSES[i]]: f"{probs[0, i].item() * 100:.1f}%"
        for i in range(len(CNN_CLASSES))
    }

    # LLM 설명 생성
    llm_result = generate_llm_analysis(
        prediction=prediction,
        confidence_percent=confidence_percent,
        material=material,
        environment=environment,
    )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "similarities": similarities,
        "feature": llm_result["feature"],
        "cause": llm_result["cause"],
        "expected_cause": llm_result["expected_cause"],
        "explanation": llm_result["explanation"],
        "material": material,
        "environment": environment,
        "confidence_status": confidence_status,
        "confidence_message": confidence_message,
    }
