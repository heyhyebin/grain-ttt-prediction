import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models


class SimilarImageSearcher:
    def __init__(
        self,
        model,
        device,
        class_names,
        similar_dir="similar_db",
        cache_path="similar_cache.pt",
        image_size=224,
    ):
        self.device = device
        self.class_names = class_names
        self.similar_dir = similar_dir
        self.cache_path = cache_path
        self.image_size = image_size
        self.cache = None

        # 유사 이미지 검색용 feature extractor
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        resnet = models.resnet18(weights=weights)

        # 마지막 분류층 제거 → feature vector만 사용
        self.feature_extractor = torch.nn.Sequential(
            *list(resnet.children())[:-1]
        ).to(self.device)

        self.feature_extractor.eval()

        self.transform = weights.transforms()

    def extract_feature(self, image: Image.Image):
        """
        이미지를 ResNet18 feature vector로 변환
        """
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.feature_extractor(x)
            feat = torch.flatten(feat, 1)
            feat = F.normalize(feat, p=2, dim=1)

        return feat.cpu()

    def build_cache(self):
        """
        similar_db 내부 이미지들의 feature를 미리 계산하여 저장
        """
        cache = {}

        for class_name in self.class_names:
            class_dir = os.path.join(self.similar_dir, class_name)
            cache[class_name] = []

            if not os.path.exists(class_dir):
                print(f"[유사 이미지] 폴더 없음: {class_dir}")
                continue

            for filename in os.listdir(class_dir):
                if not filename.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp", ".webp")
                ):
                    continue

                img_path = os.path.join(class_dir, filename)

                try:
                    image = Image.open(img_path).convert("RGB")
                    feature = self.extract_feature(image)

                    relative_path = os.path.relpath(
                        img_path,
                        self.similar_dir
                    ).replace("\\", "/")

                    cache[class_name].append({
                        "path": f"similar_db/{relative_path}",
                        "feature": feature
                    })

                except Exception as e:
                    print(f"[유사 이미지] 처리 실패: {img_path}")
                    print(e)

        torch.save(cache, self.cache_path)
        self.cache = cache
        print("[유사 이미지] 캐시 생성 완료")

    def load_cache(self):
        """
        저장된 캐시 불러오기
        없으면 새로 생성
        """
        if os.path.exists(self.cache_path):
            self.cache = torch.load(
                self.cache_path,
                map_location="cpu",
                weights_only=False,
            )
            print("[유사 이미지] 캐시 불러오기 완료")
        else:
            self.build_cache()

    def find_similar_images(
        self,
        image: Image.Image,
        predicted_class: str,
        top_k=3,
        threshold=0.75,
    ):
        """
        예측된 유형 내부에서만 유사 이미지 top_k 검색
        ResNet18 feature vector 기준 cosine similarity 사용
        """
        if self.cache is None:
            self.load_cache()

        if predicted_class not in self.cache:
            return []

        class_items = self.cache[predicted_class]

        if len(class_items) == 0:
            return []

        input_feature = self.extract_feature(image)

        results = []

        for item in class_items:
            db_feature = item["feature"]

            similarity = torch.sum(
                input_feature * db_feature
            ).item()

            if similarity < threshold:
                continue

            results.append({
                "image_url": "/" + item["path"],
                "similarity": round(similarity, 4)
            })

        results.sort(
            key=lambda x: x["similarity"],
            reverse=True
        )

        top_results = results[:top_k]

        print(f"[유사 이미지 Top{top_k}] threshold={threshold}")
        for r in top_results:
            print(r)

        return top_results