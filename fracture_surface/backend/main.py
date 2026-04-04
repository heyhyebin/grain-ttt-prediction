import torch
import torch.nn as nn
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import ollama
import io

app = FastAPI()

# React(3000) 통신 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 1. 설정 ───────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"C:\Users\LHB\Desktop\Project\backend\fractography_cnn_best1.pth"

# train.py의 ImageFolder가 자동 인식한 클래스 순서와 반드시 동일해야 함
CNN_CLASSES  = ["Cleavage", "Ductile", "Fatigue", "Intergranular"]

# 프론트엔드 카드 표시용 한글 매핑
KO_LABELS = {
    "Cleavage":      "취성 파괴",
    "Ductile":       "연성 파괴",
    "Fatigue":       "피로 파괴",
    "Intergranular": "입계 파괴",
}

FEATURES = {
    "취성 파괴": "균열이 빠르게 진행되고 소성 변형이 거의 없음",
    "연성 파괴": "큰 소성 변형과 함께 파단 발생",
    "피로 파괴": "반복 하중으로 균열이 점진적으로 성장",
    "입계 파괴": "결정립 경계를 따라 균열 진행",
}

CAUSES = {
    "취성 파괴": "저온 환경, 충격 하중",
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
    "취성 파괴": "취성 파괴는 소성 변형 없이 균열이 빠르게 진행되는 특징이 있으며, 주로 저온 환경이나 충격 하중에 의해 발생합니다.",
    "연성 파괴": "연성 파괴는 재료가 늘어나며 큰 소성 변형 후 파단되는 특징이 있으며, 과도한 하중이 주요 원인입니다.",
    "피로 파괴": "피로 파괴는 반복 하중으로 인해 균열이 점진적으로 성장한 뒤 파손되는 특징이 있으며, 반복 응력이 주요 원인입니다.",
    "입계 파괴": "입계 파괴는 결정립 경계를 따라 균열이 진행되는 특징이 있으며, 고온 환경이나 재료 열화가 원인이 됩니다.",
}

# ── 2. CNN 모델 정의 (train.py / predict.py와 동일한 구조) ────────
class FractographyCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ── 3. 모델 로드 (서버 시작 시 1회만 실행) ───────────────────────
model = FractographyCNN(num_classes=4).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"✅ 모델 로드 완료 ({DEVICE})")

# ── 4. 이미지 전처리 (train.py val_transform과 동일) ─────────────
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── 5. API 엔드포인트 ─────────────────────────────────────────────
@app.post("/analyze")
async def analyze_fracture(file: UploadFile = File(...)):

    # 5-1. 이미지 읽기 및 전처리
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    # 5-2. CNN 추론 (predict.py와 동일한 방식)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence_val = probs[0, pred_idx].item()

    pred_en = CNN_CLASSES[pred_idx]           # ex) "Cleavage"
    prediction = KO_LABELS[pred_en]           # ex) "취성 파괴"

    explanation = RULE_BASED_EXPLANATIONS[prediction]

    confidence = f"{confidence_val * 100:.1f}%"

    # 5-3. 프론트 카드 4개용 유사도 딕셔너리 (한글 키)
    similarities = {
        KO_LABELS[CNN_CLASSES[i]]: f"{probs[0, i].item() * 100:.0f}%"
        for i in range(len(CNN_CLASSES))
    }

    # 5-4. Ollama LLM 해석 생성
    prompt = f"""
    당신은 재료공학 전문가입니다.

    금속 파손 분석 결과, {prediction}일 확률이 {confidence}입니다.
    이 파손의 미세구조적 특징과 발생 원인을 사용자에게 쉽게 설명하세요.

    조건:
    - 반드시 한국어로만 3줄 이내
    - 한국어 외 다른 언어 절대 금지
    - 원인은 하나만 선택
    - 특징은 구체적으로 표현
    - 매번 다른 표현 사용
    """

    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )
        explanation = response["message"]["content"]
    except Exception as e:
        explanation = (
            f"{prediction}으로 분류되었습니다 (신뢰도 {confidence}). "
            "LLM 리포트 생성 중 오류가 발생했습니다."
        )
        print(f"Ollama 오류: {e}")

    return {
        "prediction": prediction,
        "confidence": confidence,
        "similarities": similarities,
        "feature": FEATURES[prediction],
        "cause": CAUSES[prediction],
        "expected_cause": EXPECTED_CAUSE[prediction],
        "explanation": explanation,
    }
