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
# [중요] model.py에서 신경망 구조 불러오기
# ==========================================
from model import FractographyNet

# from deep_translator import GoogleTranslator
# 나중에 영어 생성 → 번역기 방식으로 다시 사용할 때 주석 해제

app = FastAPI()

# React(3000) 통신 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# 번역 함수 (현재 미사용 - 필요시 주석 해제)
# --------------------------------------------------
"""
def translate_to_korean(text: str) -> str:
    try:
        translated = GoogleTranslator(source='en', target='ko').translate(text)
        return translated
    except Exception as e:
        print("번역 오류:", e)
        return text
"""

# --------------------------------------------------
# 텍스트 후처리
# --------------------------------------------------
def clean_text(text: str) -> str:
    if not text:
        return text

    text = text.replace("**", "")
    text = text.replace("*", "")
    text = text.replace("#", "")

    # 줄 앞 번호 제거
    text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)
    # 줄 앞 bullet 제거
    text = re.sub(r'^\s*[-•]\s*', '', text, flags=re.MULTILINE)
    # 줄바꿈 / 공백 정리
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 실제 pth 파일명 확인해서 필요하면 경로 수정
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
# LLM 프롬프트 생성 (Ollama용)
# --------------------------------------------------
def build_prompt(prediction: str, confidence_percent: float) -> str:
    common_prefix = f"""
너는 파손 단면 분석 결과를 설명하는 도우미다.
설명 대상은 전문가가 아니라 일반 사용자다.

예측된 파손 유형은 {prediction}이고, 신뢰도는 {confidence_percent:.1f}%이다.

다음 규칙을 반드시 지켜라:
- 반드시 한국어로만 작성
- 2문장 또는 3문장으로 작성
- 어려운 전문용어는 가급적 사용하지 말 것
- 꼭 필요한 경우에도 쉬운 말로 풀어서 설명할 것
- 실제 표면에서 보일 수 있는 특징을 먼저 설명하고, 그 다음 원인을 설명할 것
- 문장은 짧고 자연스럽게 작성할 것
- 같은 뜻의 반복 표현은 금지
- 확정하지 말고 "~로 보입니다", "~가능성이 있습니다"처럼 추정형으로 작성할 것
- "파손 원인:", "설명:" 같은 형식적인 말은 쓰지 말 것
- 설명문만 출력할 것
- 불필요한 특정 조건을 단정하지 말 것
"""

    prompt_map = {
        "연성 파괴": common_prefix + """
이 유형은 연성 파괴이다.

반드시 포함할 내용:
- 재료가 늘어나면서 파손된 것으로 보인다는 점
- 표면에 늘어난 흔적, 찢어진 듯한 모습, 움푹한 부분 등이 보일 수 있다는 점
- 강한 하중이나 과도한 힘이 원인일 가능성
- 충격에 의해 발생했다고 설명하지 말 것

좋은 예시:
표면에 재료가 늘어난 듯한 흔적과 일부 찢어진 모양이 보입니다.
이는 재료가 힘을 받으며 늘어나다가 파손된 경우로 보입니다.
강한 하중이 가해졌을 가능성이 있습니다.

절대 쓰지 말 것:
- 충격
- 반복 하중
- 급격히 깨짐
- 쉽게 깨진다
- 응력 변화
- 저온
""",
        "취성 파괴": common_prefix + """
이 유형은 취성 파괴이다.

반드시 포함할 내용:
- 재료가 거의 늘어나지 않은 상태에서 파손된 것으로 보인다는 점
- 표면이 비교적 평평하거나 날카롭게 갈라진 모습이 보일 수 있다는 점
- 충격 하중이나 응력 집중이 원인일 가능성
- 특정 환경을 단정하지 말 것

좋은 예시:
표면이 비교적 평평하고 날카롭게 갈라진 모습이 보입니다.
이는 재료가 거의 늘어나지 않은 상태에서 갑자기 파손된 경우로 보입니다.
충격 하중이나 응력 집중의 영향이 있었을 가능성이 있습니다.

절대 쓰지 말 것:
- 늘어남
- 찢어진 흔적
- 큰 소성 변형
- 저온
- 낮은 온도
- 매우 낮은 온도
""",
        "피로 파괴": common_prefix + """
이 유형은 피로 파괴이다.

반드시 포함할 내용:
- 한 번에 깨진 것이 아니라 반복된 힘으로 서서히 손상된 것으로 보인다는 점
- 표면에 반복된 흔적이나 일정한 줄무늬 같은 모습이 보일 수 있다는 점
- 장기간 반복 하중이 원인일 가능성
- 반복 하중이 핵심 원인임을 반드시 반영할 것

좋은 예시:
표면에 반복된 흔적이나 줄무늬처럼 보이는 부분이 관찰됩니다.
이는 재료가 한 번에 깨진 것이 아니라, 반복된 힘으로 서서히 손상된 경우로 보입니다.
오랜 시간 반복해서 힘을 받은 영향일 가능성이 있습니다.

절대 쓰지 말 것:
- 한 번의 큰 하중
- 충격으로 바로 파손
- 크게 늘어난 뒤 파손
""",
        "입계 파괴": common_prefix + """
이 유형은 입계 파괴이다.

반드시 포함할 내용:
- 균열이 재료 내부의 경계를 따라 진행된 것으로 보인다는 점
- 표면이 알갱 경계를 따라 나뉜 듯한 모습으로 보일 수 있다는 점
- 재료 열화, 불순물, 또는 고온 환경의 영향 가능성
- 원인을 하나만 단정하지 말 것

좋은 예시:
표면이 작은 경계를 따라 나뉜 듯한 모습이 보입니다.
이는 균열이 재료 내부의 경계를 따라 진행된 경우로 보입니다.
재료 열화나 불순물, 또는 고온 환경의 영향이 있었을 가능성이 있습니다.

절대 쓰지 말 것:
- 재료가 늘어나며 파손
- 찢어진 흔적
- 반복 하중이 주원인
""",
    }

    return prompt_map[prediction]

# --------------------------------------------------
# 설명 검증
# --------------------------------------------------
def validate_explanation(prediction: str, text: str) -> str:
    invalid_keywords = {
        "연성 파괴": ["반복 하중"],
        "취성 파괴": ["늘어난 흔적"],
        "피로 파괴": ["한 번의 큰 하중"],
        "입계 파괴": ["반복 하중"],
    }

    banned = invalid_keywords.get(prediction, [])
    for word in banned:
        if word in text:
            print(f"[validate] 금지어 감지: {word}")
            return RULE_BASED_EXPLANATIONS[prediction]

    return text

# ==========================================
# 모델 로드 (model.py 연동)
# ==========================================
print("모델 초기화 중...")
model = FractographyNet(num_classes=len(CNN_CLASSES), pretrained=False).to(DEVICE)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)

model.eval()
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
# API 라우터
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

    # 2. CNN 추론
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

    # 3. 카드용 유사도
    similarities = {
        KO_LABELS[CNN_CLASSES[i]]: f"{probs[0, i].item() * 100:.0f}%"
        for i in range(len(CNN_CLASSES))
    }

    # 4. LLM 설명 생성 (Ollama 연동 부분)
    prompt = build_prompt(prediction, confidence_percent)
    korean_explanation = explanation

    try:
        response = ollama.chat(
            model="exaone3.5:2.4b",
            messages=[{"role": "user", "content": prompt}]
        )

        raw_explanation = response["message"]["content"].strip()
        print("LLM 원본 응답:", raw_explanation)

        korean_explanation = clean_text(raw_explanation)
        print("후처리 후:", korean_explanation)

        korean_explanation = validate_explanation(prediction, korean_explanation)
        print("최종 설명:", korean_explanation)

        # 만약 영어로 뱉는 문제가 생겨서 번역기를 켜야 한다면 아래 주석을 해제하세요.
        # korean_explanation = translate_to_korean(korean_explanation)

    except Exception as e:
        korean_explanation = (
            f"{prediction}으로 분류되었습니다. "
            f"신뢰도는 {confidence}입니다. "
            "설명 생성 중 오류가 발생했습니다."
        )
        print(f"Ollama 오류: {e}")

    # 5. JSON 반환
    return {
        "prediction": prediction,
        "confidence": confidence,
        "similarities": similarities,
        "feature": FEATURES[prediction],
        "cause": CAUSES[prediction],
        "expected_cause": EXPECTED_CAUSE[prediction],
        "explanation": korean_explanation,
    }