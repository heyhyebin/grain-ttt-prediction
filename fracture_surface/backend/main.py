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

# ── 1. CNN 모델 정의 ──────────────────────────────────────────────
class FractographyCNN(nn.Module):
    def __init__(self):
        super(FractographyCNN, self).__init__()
        # 학교에서 쓴 아키텍처 코드를 여기에 복사해서 넣으세요.
        pass

    def forward(self, x):
        return self.model(x)

# 모델 로드
MODEL_PATH = r"backend\fractography_cnn_best1.pth"
device = torch.device("cpu")
model = FractographyCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ✅ 수정 1: transforms.Compose 올바른 문법으로 수정
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

FRACTURE_TYPES = ["취성 파괴", "연성 파괴", "피로 파괴", "크리프 파괴"]

@app.post("/analyze")
async def analyze_fracture(file: UploadFile = File(...)):
    # ── 2. 이미지 읽기 및 CNN 추론 ──
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        # ✅ 수정 2: softmax 뒤 [9, 12] 인덱싱 제거
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    prediction = FRACTURE_TYPES[pred.item()]
    confidence = f"{conf.item()*100:.1f}%"

    # 유사도 딕셔너리 생성
    similarities = {
        FRACTURE_TYPES[i]: f"{probs[0][i].item()*100:.0f}%"
        for i in range(len(FRACTURE_TYPES))
    }

    # ── 3. Ollama LLM 해석 생성 ──
    # ✅ 수정 3: prompt 닫는 괄호 뒤 [14, 15, 16] 제거
    prompt = (
        f"금속 파손 분석 결과, {prediction}일 확률이 {confidence}입니다. "
        f"파면학 기술자의 관점에서 이 파괴의 미세구조적 특징과 발생 원인을 "
        f"3줄 내외로 아주 쉽게 설명해줘."
    )

    try:
        response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
        explanation = response['message']['content']
    except:
        explanation = "LLM 분석 리포트를 생성하는 중 오류가 발생했습니다."

    return {
        "prediction": prediction,
        "confidence": confidence,
        "similarities": similarities,
        "explanation": explanation
    }