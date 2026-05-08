import io
import os

import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms

from model import load_model
from gradcam import make_gradcampp_base64
from llm_service import generate_llm_analysis


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "fractography_best5.pth",
)

IMG_SIZE = 224
NUM_CLASSES = 4

CNN_CLASSES = ["Cleavage", "Ductile", "Fatigue", "Intergranular"]

KO_LABELS = {
    "Cleavage": "취성 파괴",
    "Ductile": "연성 파괴",
    "Fatigue": "피로 파괴",
    "Intergranular": "입계 파괴",
}

MIXED_GAP_THRESHOLD = 15.0


model = load_model(
    model_path=MODEL_PATH,
    device=DEVICE,
    num_classes=NUM_CLASSES,
)


preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


@app.get("/")
async def root():
    return {
        "message": "Fractography API is running",
        "model": "ConvNeXt-Small + ASPP",
        "pth": MODEL_PATH,
        "device": str(DEVICE),
    }


@app.post("/analyze")
async def analyze_fracture(
    file: UploadFile = File(...),
    material: str = Form(""),
):
    print(f"요청 수신 — 재질: {material}")

    model.eval()

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    top1_idx = sorted_indices[0].item()
    top2_idx = sorted_indices[1].item()

    top1_percent = sorted_probs[0].item() * 100
    top2_percent = sorted_probs[1].item() * 100
    gap = top1_percent - top2_percent

    pred_en = CNN_CLASSES[top1_idx]
    prediction = KO_LABELS[pred_en]

    top1_label = KO_LABELS[CNN_CLASSES[top1_idx]]
    top2_label = KO_LABELS[CNN_CLASSES[top2_idx]]

    is_mixed = gap <= MIXED_GAP_THRESHOLD

    if is_mixed:
        highlighted_types = [top1_label, top2_label]
        display_prediction = f"혼합 가능성: {top1_label} + {top2_label}"
    else:
        highlighted_types = [top1_label]
        display_prediction = top1_label

    confidence = f"{top1_percent:.1f}%"

    if top1_percent >= 80 and not is_mixed:
        confidence_status = "high"
        confidence_message = "현재 분석은 신뢰할 수 있는 결과입니다."
    elif top1_percent >= 60:
        confidence_status = "medium"
        confidence_message = (
            "상위 두 유형의 확률 차이가 크지 않아 혼합 파손 가능성을 함께 확인해야 합니다."
            if is_mixed
            else "결과 해석에 주의가 필요합니다."
        )
    else:
        confidence_status = "low"
        confidence_message = (
            "신뢰도가 낮아 오분류 가능성이 있습니다. "
            "추가 이미지나 전문가 검토가 필요할 수 있습니다."
        )

    similarities = {
        KO_LABELS[CNN_CLASSES[i]]: f"{probs[i].item() * 100:.1f}%"
        for i in range(len(CNN_CLASSES))
    }

    llm_result = generate_llm_analysis(
        prediction=prediction,
        confidence_percent=top1_percent,
        material=material,
    )

    gradcam_image = make_gradcampp_base64(
        model=model,
        input_tensor=input_tensor,
        original_image=image,
        target_class_idx=top1_idx,
        img_size=IMG_SIZE,
    )

    return {
        "prediction": prediction,
        "display_prediction": display_prediction,
        "confidence": confidence,
        "similarities": similarities,

        "is_mixed": is_mixed,
        "mixed_gap": f"{gap:.1f}%",
        "top1_type": top1_label,
        "top2_type": top2_label,
        "highlighted_types": highlighted_types,

        "feature": llm_result["feature"],
        "cause": llm_result["cause"],
        "expected_cause": llm_result["expected_cause"],
        "explanation": llm_result["explanation"],
        "material": material,

        "confidence_status": confidence_status,
        "confidence_message": confidence_message,

        "gradcam_image": gradcam_image,
    }
