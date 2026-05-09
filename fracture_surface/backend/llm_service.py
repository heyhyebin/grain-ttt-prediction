import re
import json
import ollama


FEATURES = {
    "취성 파괴": "균열이 빠르게 진행되고 소성 변형이 거의 없음",
    "연성 파괴": "큰 소성 변형과 함께 파단 발생",
    "피로 파괴": "반복 하중으로 균열이 점진적으로 성장",
    "입계 파괴": "결정립 경계를 따라 균열 진행",
}

EXPECTED_CAUSE = {
    "취성 파괴": "충격 하중 또는 급격한 응력 집중",
    "연성 파괴": "과도한 하중 또는 구조적 결함",
    "피로 파괴": "장기간 반복 하중에 의한 손상",
    "입계 파괴": "고온 영향 또는 재료 열화",
}

RULE_BASED_EXPLANATIONS = {
    "취성 파괴": "취성 파괴는 소성 변형 없이 균열이 빠르게 진행되는 특징이 있으며, 충격 하중이나 응력 집중에 의해 발생했을 가능성이 있습니다.",
    "연성 파괴": "연성 파괴는 재료가 늘어나며 큰 소성 변형 후 파단되는 특징이 있으며, 과도한 하중이 원인일 가능성이 있습니다.",
    "피로 파괴": "피로 파괴는 반복 하중으로 인해 균열이 점진적으로 성장한 뒤 파손되는 특징이 있으며, 장기간 반복 응력이 원인일 가능성이 있습니다.",
    "입계 파괴": "입계 파괴는 결정립 경계를 따라 균열이 진행되는 특징이 있으며, 재료 열화나 결정립 경계 약화의 영향일 가능성이 있습니다.",
}

MATERIAL_LABELS = {
    "steel": "강",
    "stainless_steel": "스테인리스강",
    "aluminum": "알루미늄",
    "titanium": "티타늄",
    "cast_iron": "주철",
    "copper": "구리",
    "magnesium": "마그네슘 합금",
    "nickel_alloy": "니켈 합금",
    "tool_steel": "공구강",
    "unknown": "정보 없음",
    "": "미선택",
}


def clean_text(text: str) -> str:
    if not text:
        return text

    text = text.replace("**", "")
    text = text.replace("*", "")
    text = text.replace("#", "")

    text = re.sub(r"^\s*\d+\.\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def parse_llm_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass

    return None


def get_confidence_level(confidence_percent: float) -> str:
    if confidence_percent < 40:
        return "low"
    elif confidence_percent < 70:
        return "medium"
    else:
        return "high"


def get_confidence_instruction(confidence_percent: float) -> str:
    level = get_confidence_level(confidence_percent)

    if level == "low":
        return (
            "신뢰도가 낮으므로 반드시 '가능성 중 하나로 보입니다' 또는 "
            "'추가 검토가 필요합니다'라는 뉘앙스로 작성한다."
        )

    if level == "medium":
        return (
            "신뢰도가 중간 수준이므로 확정 표현은 피하고 "
            "'가능성이 있습니다', '추정됩니다' 형태로 작성한다."
        )

    return (
        "신뢰도가 비교적 높더라도 단정하지 말고 "
        "'가능성이 높습니다', '추정됩니다' 형태로 작성한다."
    )


def get_rule_based_analysis(prediction: str, material: str):
    rules = []

    if prediction == "피로 파괴":
        rules.append("반복 하중에 의해 균열이 점진적으로 성장했을 가능성이 있습니다.")
    elif prediction == "취성 파괴":
        rules.append("재료가 거의 변형 없이 급격히 파손되었을 가능성이 있습니다.")
    elif prediction == "연성 파괴":
        rules.append("재료가 큰 변형을 겪은 후 파손되었을 가능성이 있습니다.")
    elif prediction == "입계 파괴":
        rules.append("균열이 결정립 경계를 따라 진행되었을 가능성이 있습니다.")

    if material == "cast_iron":
        rules.append("주철은 상대적으로 취성적인 거동을 보일 수 있어 급격한 균열 진행과 관련될 수 있습니다.")

    if material == "stainless_steel":
        rules.append("스테인리스강은 조건에 따라 결정립계 손상이나 국부적인 재료 열화와 관련될 수 있습니다.")

    if material == "aluminum":
        rules.append("알루미늄은 가벼운 금속 재료로, 반복 하중이나 과하중 조건에서 손상 양상이 나타날 수 있습니다.")

    if material == "titanium":
        rules.append("티타늄은 강도 대비 가벼운 특성이 있어 반복 하중이 작용하는 부품에서 피로 손상을 고려할 수 있습니다.")

    if material == "copper":
        rules.append("구리는 비교적 연성이 큰 재료이므로 큰 변형을 동반한 파손 가능성을 함께 고려할 수 있습니다.")

    if material == "magnesium":
        rules.append("마그네슘 합금은 가볍지만 조건에 따라 취성적인 파손 양상이 나타날 수 있습니다.")

    if material == "nickel_alloy":
        rules.append("니켈 합금은 고온 부품에 자주 사용되므로 재료 열화나 반복 하중에 의한 손상을 고려할 수 있습니다.")

    if material == "tool_steel":
        rules.append("공구강은 높은 경도 때문에 응력 집중 조건에서 취성적 균열 진행 가능성을 고려할 수 있습니다.")

    return rules


def build_condition_text(material: str) -> str:
    material_ko = MATERIAL_LABELS.get(material, material)

    if material not in ["", "unknown"]:
        return f"- 참고 재질: {material_ko}"

    return "없음"


def build_prompt(prediction: str, confidence_percent: float, material: str) -> str:
    rules = get_rule_based_analysis(prediction, material)
    rules_text = "\n".join([f"- {r}" for r in rules]) if rules else "없음"
    condition_text = build_condition_text(material)
    confidence_instruction = get_confidence_instruction(confidence_percent)

    return f"""
너는 파손 단면 분석 결과를 설명하는 도우미다.
설명 대상은 전문가가 아니라 일반 사용자다.

CNN 분석 결과:
- 예측 파손 유형: {prediction}
- 신뢰도: {confidence_percent:.1f}%

참고 입력 정보:
{condition_text}

반드시 반영해야 하는 분석 근거:
{rules_text}

신뢰도 표현 규칙:
{confidence_instruction}

작성해야 할 출력:
1. expected_cause
   - 15~25자 정도의 짧은 원인 요약
   - 명사형 또는 짧은 문장형
   - 예: "장기간 반복 하중", "급격한 응력 집중"

2. explanation
   - 반드시 3문장으로 작성
   - 1문장: 관찰된 표면 특징
   - 2문장: 해당 특징이 예측 파손 유형과 연결되는 이유
   - 3문장: 가능한 원인 또는 재질과의 관련성
   - 신뢰도가 낮으면 불확실성을 반드시 표현

중요 규칙:
- 반드시 한국어로만 작성한다.
- JSON 이외의 문장은 출력하지 않는다.
- 확정 표현을 사용하지 않는다.
- "발생했습니다", "원인입니다", "확실합니다" 같은 단정 표현은 금지한다.
- "가능성이 있습니다", "추정됩니다", "보입니다" 같은 표현을 사용한다.
- 재질을 단순히 나열하지 않는다.
- 재질이 정보 없음 또는 미선택이면 재질을 언급하지 않는다.
- 같은 의미를 반복하지 않는다.
- 설명은 너무 길지 않게 작성한다.

반드시 아래 JSON 형식으로만 출력하라.

{{
  "expected_cause": "짧은 원인 요약",
  "explanation": "관찰된 특징 문장. 파손 유형과 연결되는 해석 문장. 가능한 원인 추정 문장."
}}
"""


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


def validate_confidence_tone(confidence_percent: float, text: str) -> str:
    if confidence_percent >= 40:
        return text

    uncertainty_words = ["가능성", "추정", "보입니다", "검토"]

    if not any(word in text for word in uncertainty_words):
        text += " 다만 신뢰도가 낮아 추가 이미지 또는 전문가 검토가 필요합니다."

    return text


def limit_to_three_sentences(text: str) -> str:
    sentences = re.split(r"(?<=[.!?。])\s+", text)

    if len(sentences) <= 3:
        return text

    return " ".join(sentences[:3]).strip()


def fallback_analysis(prediction: str, confidence_percent: float):
    if confidence_percent < 40:
        return (
            f"{prediction} 가능성 중 하나로 분류되었습니다. "
            f"다만 신뢰도는 {confidence_percent:.1f}%로 낮아 추가 이미지나 전문가 검토가 필요합니다. "
            f"{RULE_BASED_EXPLANATIONS[prediction]}"
        )

    return RULE_BASED_EXPLANATIONS[prediction]


def generate_llm_analysis(
    prediction: str,
    confidence_percent: float,
    material: str = "",
):
    llm_expected_cause = EXPECTED_CAUSE[prediction]
    korean_explanation = RULE_BASED_EXPLANATIONS[prediction]

    prompt = build_prompt(prediction, confidence_percent, material)

    try:
        response = ollama.chat(
            model="exaone3.5:2.4b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "너는 파손 단면 분석 설명 생성기다. "
                        "반드시 한국어 JSON만 출력해야 한다. "
                        "확정 표현은 금지하고 추정형 표현만 사용한다."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
                "temperature": 0.2,
                "top_p": 0.8,
            },
        )

        raw_response = response["message"]["content"].strip()
        print("LLM 원본 응답:", raw_response)

        parsed = parse_llm_json(raw_response)

        if parsed:
            llm_expected_cause = clean_text(
                parsed.get("expected_cause", llm_expected_cause)
            )
            korean_explanation = clean_text(
                parsed.get("explanation", korean_explanation)
            )
        else:
            print("JSON 파싱 실패 → LLM 원문 정리 후 사용")
            korean_explanation = clean_text(raw_response)

        korean_explanation = validate_explanation(prediction, korean_explanation)
        korean_explanation = limit_to_three_sentences(korean_explanation)
        korean_explanation = validate_confidence_tone(
            confidence_percent,
            korean_explanation,
        )

    except Exception as e:
        print(f"Ollama 오류: {e}")

        llm_expected_cause = EXPECTED_CAUSE[prediction]
        korean_explanation = fallback_analysis(prediction, confidence_percent)

    return {
        "feature": FEATURES[prediction],
        "cause": llm_expected_cause,
        "expected_cause": llm_expected_cause,
        "explanation": korean_explanation,
    }
