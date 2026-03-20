# 🔧 파손 단면 이미지 분류 CNN 프로젝트

## 📌 프로젝트 개요

본 프로젝트는 금속/재료의 파손 단면 이미지(Fractography)를 입력으로 받아
다음 4가지 유형 중 하나로 자동 분류하는 **CNN 기반 딥러닝 모델**입니다.

### 🎯 분류 대상 (4 클래스)

* Cleavage (취성 파괴)
* Ductile (연성 파괴)
* Fatigue (피로 파괴)
* Intergranular (입계 파괴)

---

## 📁 폴더 구조

```
파손단면/
 ├─ train.py
 ├─ predict.py
 ├─ Fractography/
 │   ├─ Cleavage/
 │   ├─ Ductile/
 │   ├─ Fatigue/
 │   └─ Intergranular/
```

* 각 폴더 안에는 해당 클래스 이미지들이 들어있어야 합니다.
* `ImageFolder` 구조를 따릅니다.

---

## ⚙️ 필수 라이브러리 설치

```bash
pip install torch torchvision numpy pillow matplotlib
```

---

## 🚀 학습 실행 방법

```bash
python train.py
```

### 📊 학습 결과 출력 예시

```
[1/30] Train Loss: 1.23, Train Acc: 0.45 | Val Loss: 1.10, Val Acc: 0.52
...
최고 검증 정확도: 0.87
```

### 📦 결과

* 최고 성능 모델이 자동 저장됨:

```
fractography_cnn_best.pth
```

---

## 🔍 예측 실행 방법

```bash
python predict.py
```

### ⚠️ 반드시 수정해야 할 부분

```python
image_path = r"C:\Users\COM\Desktop\파손단면\test.jpg"
```

👉 본인이 테스트할 이미지 경로로 변경

---

## 📈 예측 결과 출력 예시

```
예측 클래스: Fatigue
신뢰도: 0.8732

클래스별 확률:
Cleavage: 0.0123
Ductile: 0.0451
Fatigue: 0.8732
Intergranular: 0.0694
```

---

## ⚠️ 주의사항

### 1. Windows 사용자 필수 설정

* `num_workers=0` 사용 (멀티프로세싱 에러 방지)

```python
DataLoader(..., num_workers=0)
```

---

### 2. 경로 설정 주의

* 반드시 **절대경로 사용 권장**

```python
DATA_DIR = r"C:\Users\COM\Desktop\파손단면\Fractography"
```

---

### 3. 데이터 품질 중요

* 이미지 수가 적으면 정확도 낮아짐
* 클래스 불균형 시 성능 저하

---

## 🔥 성능 향상 팁

* ResNet (전이학습) 사용 시 정확도 크게 향상
* 데이터 증강 (augmentation) 다양화
* 클래스별 데이터 균형 맞추기



---

## 👨‍💻 실행 흐름 요약

1. 데이터 준비 (Fractography 폴더)
2. `train.py` 실행 → 모델 학습
3. `predict.py` 실행 → 이미지 분류

---

## ✅ 요구 환경

* Python 3.8 ~ 3.11 권장
* PyTorch
* GPU 있으면 학습 속도 향상

---


🚀 **이 프로젝트는 재료 파손 분석 자동화를 위한 기초 모델입니다.**
