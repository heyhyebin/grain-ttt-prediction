import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import FractographyNet


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURE_PATH = os.path.join(BASE_DIR, "image_index", "image_features.npy")
PATH_JSON = os.path.join(BASE_DIR, "image_index", "image_paths.json")

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


database_features = None
image_paths = None


def load_index():
    global database_features, image_paths

    database_features = np.load(FEATURE_PATH)

    with open(PATH_JSON, "r", encoding="utf-8") as f:
        image_paths = json.load(f)

    print(f"[retrival] index loaded: {database_features.shape}")


def extract_feature(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    model.eval()

    with torch.no_grad():
        x = model.backbone(image_tensor)
        x = model.aspp(x)

        if hasattr(model, "attention"):
            x = model.attention(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = F.normalize(x, dim=1)

    return x.cpu().numpy()[0]


def search_similar(model, query_image_path, top_k=5, threshold=0.90):
    global database_features, image_paths

    if database_features is None or image_paths is None:
        load_index()

    query_feature = extract_feature(model, query_image_path)

    similarities = database_features @ query_feature

    valid_indices = np.where(similarities >= threshold)[0]

    if len(valid_indices) == 0:
        return []

    valid_sims = similarities[valid_indices]
    sorted_indices = valid_indices[np.argsort(valid_sims)[::-1]]
    top_indices = sorted_indices[:top_k]

    results = []

    for idx in top_indices:
        path = image_paths[idx].replace("\\", "/")
        class_name = os.path.basename(os.path.dirname(path))

        results.append({
            "class_name": class_name,
            "similarity": float(similarities[idx]),
            "image_path": path,
            "image_url": f"http://127.0.0.1:8000/{path}",
        })

    return results