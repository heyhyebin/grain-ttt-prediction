import io

import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms, models

from llm_service import generate_llm_analysis


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

MODEL_PATH = r"C:\Users\LHB\Desktop\Project\backend\fractography_cnn_best_test.pth"

CNN_CLASSES = ["Cleavage", "Ductile", "Fatigue", "Intergranular"]

KO_LABELS = {
    "Cleavage": "취성 파괴",
    "Ductile": "연성 파괴",
    "Fatigue": "피로 파괴",
    "Intergranular": "입계 파괴",
}


# --------------------------------------------------
# 모델 정의
# --------------------------------------------------
def build_model(num_classes=4):
    model = models.efficientnet_b2(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes),
    )

    return model


# --------------------------------------------------
# 모델 로드
# --------------------------------------------------
model = build_model(num_classes=4).to(DEVICE)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

model.load_state_dict(state_dict)
model.eval()

print(f"✅ 모델 로드 완료 ({DEVICE})")


# --------------------------------------------------
# 이미지 전처리
# --------------------------------------------------
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# --------------------------------------------------
# API
# --------------------------------------------------
@app.get("/")
async def root():
    return {"message": "Fractography API is running"}


@app.post("/analyze")
async def analyze_fracture(
    file: UploadFile = File(...),
    material: str = Form(""),
    environment: str = Form(""),
):
    print("재질:", material)
    print("환경:", environment)

    # 이미지 읽기
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # CNN 입력 형태로 변환
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
