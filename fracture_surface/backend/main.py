# main.py

import io
import re
import ollama
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms

# ==========================================
# [수정된 부분] model.py에서 신경망 구조 불러오기
# ==========================================
from model import FractographyNet

app = FastAPI()

# React(3000) 통신 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# 텍스트 후처리 로직
# --------------------------------------------------
def clean_text(text: str) -> str:
    if not text:
        return text

    text = text.replace("**", "")
    text = text.replace("*", "")
    text = text.replace("#", "")

    text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-•]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 학습 완료된 실제 pth 파일 경로 (경로 확인 필수)
MODEL_PATH = r"C:\Users\LHB\Desktop\Project\backend\fractography_cnn_best_test.pth"

CNN_CLASSES = ["Cleavage", "Ductile", "Fatigue", "Intergranular"]

KO_LABELS = {
    "Cleavage": "취성 파괴",
    "Ductile": "연성 파괴",
    "Fatigue": "피로 파괴",
    "Intergranular": "입계 파괴",
}

FEATURES = {
    "취성 파괴": "균열이 빠르게 진행되고 소성 변형이 거의 없음",
    "연성 파괴": "큰 소성 변형과 함께 파단 발생",
    "피로 파괴": "반복 하중으로 균열이 점진적으로 성장",
    "입계 파괴": "결정립 경계를 따라 균열 진행",
}

CAUSES = {
    "취성 파괴": "충격 하중, 응력 집중",
    "연성 파괴": "과도한 하중, 재료 연성",
    "피로 파괴": "반복 응력",
    "입계 파괴": "고온, 재료 열화, 불순물",
}

EXPECTED_CAUSE = {
    "취성 파괴": "충격 하중 또는 급격한 응력 집중 가능성",
    "연성 파괴": "과도한 하중 또는 구조적 결함 가능성",
    "피로 파괴": "장기간 반복 하중에 의한 손상 가능성",
    "입계 파괴": "고온 환경 또는 재료 열화 가능성",
}

RULE_BASED_EXPLANATIONS = {
    "취성 파괴": "취성 파괴는 소성 변형 없이 균열이 빠르게 진행되는 특징이 있으며, 주로 충격 하중이나 응력 집중에 의해 발생한 것으로 보입니다.",
    "연성 파괴": "연성 파괴는 재료가 늘어나며 큰 소성 변형 후 파단되는 특징이 있으며, 과도한 하중이 원인일 가능성이 있습니다.",
    "피로 파괴": "피로 파괴는 반복 하중으로 인해 균열이 점진적으로 성장한 뒤 파손되는 특징이 있으며, 장기간 반복 응력이 원인일 가능성이 있습니다.",
    "입계 파괴": "입계 파괴는 결정립 경계를 따라 균열이 진행되는 특징이 있으며, 재료 열화나 고온 환경의 영향일 가능성이 있습니다.",
}

# --------------------------------------------------
# LLM 프롬프트 생성 (생략 처리됨 - 기존 코드와 동일)
# --------------------------------------------------
def build_prompt(prediction: str, confidence_percent: float) -> str:
    # (기존 코드와 동일하게 유지 - 분량상 내용은 생략하지 않고 원본대로 넣으시면 됩니다)
    # ... 기존 프롬프트 코드 그대로 사용 ...
    pass 

def validate_explanation(prediction: str, text: str) -> str:
    # (기존 코드와 동일)
    # ... 기존 검증 코드 그대로 사용 ...
    pass

# ==========================================
# [수정된 부분] 외부 model.py 기반의 모델 로드
# ==========================================
# 기존에 있던 build_model() 함수는 삭제하고, model.py에서 가져온 구조를 사용합니다.
# 추론 시에는 pretrained 가중치 다운로드가 불필요하므로 pretrained=False로 설정합니다.

print("모델 초기화 중...")
model = FractographyNet(num_classes=len(CNN_CLASSES), pretrained=False).to(DEVICE)

# pth 파일(학습된 가중치)을 불러와 FractographyNet에 덮어씌웁니다.
state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)

model.eval() # 추론 모드로 전환
print(f"✅ 모델 로드 완료 ({DEVICE})")


# --------------------------------------------------
# 전처리
# --------------------------------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# API 라우터 (결과를 JSON으로 반환)
# --------------------------------------------------
@app.get("/")
async def root():
    return {"message": "Fractography API is running"}

@app.post("/analyze")
async def analyze_fracture(file: UploadFile = File(...)):
    # 1. 이미지 읽기
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    # 2. CNN 추론 (model.py의 코드가 실행됨)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence_val = probs[0, pred_idx].item()

    pred_en = CNN_CLASSES[pred_idx]
    prediction = KO_LABELS[pred_en]

    confidence_percent = confidence_val * 100
    confidence = f"{confidence_percent:.1f}%"

    explanation = RULE_BASED_EXPLANATIONS[prediction]

    # 3. 카드용 유사도 퍼센트
    similarities = {
        KO_LABELS[CNN_CLASSES[i]]: f"{probs[0, i].item() * 100:.0f}%"
        for i in range(len(CNN_CLASSES))
    }

    # 4. LLM 설명 생성 (Ollama)
    # (여기서 Ollama API를 호출하는 로직은 작성자님의 원본 유지)
    prompt = build_prompt(prediction, confidence_percent)
    korean_explanation = explanation

    try:
        response = ollama.chat(
            model="exaone3.5:2.4b",
            messages=[{"role": "user", "content": prompt}]
        )
        raw_explanation = response["message"]["content"].strip()
        korean_explanation = clean_text(raw_explanation)
        korean_explanation = validate_explanation(prediction, korean_explanation)
    except Exception as e:
        print(f"Ollama 오류: {e}")
        korean_explanation = (
            f"{prediction}으로 분류되었습니다. "
            f"신뢰도는 {confidence}입니다. "
            "설명 생성 중 오류가 발생하여 기본 설명을 제공합니다."
        )

    # 5. 최종 결과 반환 (이 Dictionary 형태가 자동으로 JSON으로 변환되어 프론트엔드로 갑니다)
    return {
        "prediction": prediction,
        "confidence": confidence,
        "similarities": similarities,
        "feature": FEATURES[prediction],
        "cause": CAUSES[prediction],
        "expected_cause": EXPECTED_CAUSE[prediction],
        "explanation": korean_explanation,
    }