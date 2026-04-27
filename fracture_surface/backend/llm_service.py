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
    "취성 파괴": "충격 하중 또는 급격한 응력 집중 가능성",
    "연성 파괴": "과도한 하중 또는 구조적 결함 가능성",
    "피로 파괴": "장기간 반복 하중에 의한 손상 가능성",
    "입계 파괴": "고온 환경 또는 재료 열화 가능성",
}

RULE_BASED_EXPLANATIONS = {
    "취성 파괴": "취성 파괴는 소성 변형 없이 균열이 빠르게 진행되는 특징이 있으며, 충격 하중이나 응력 집중에 의해 발생했을 가능성이 있습니다.",
    "연성 파괴": "연성 파괴는 재료가 늘어나며 큰 소성 변형 후 파단되는 특징이 있으며, 과도한 하중이 원인일 가능성이 있습니다.",
    "피로 파괴": "피로 파괴는 반복 하중으로 인해 균열이 점진적으로 성장한 뒤 파손되는 특징이 있으며, 장기간 반복 응력이 원인일 가능성이 있습니다.",
    "입계 파괴": "입계 파괴는 결정립 경계를 따라 균열이 진행되는 특징이 있으며, 재료 열화나 고온 환경의 영향일 가능성이 있습니다.",
}

MATERIAL_LABELS = {
    "steel": "강",
    "aluminum": "알루미늄",
    "titanium": "티타늄",
    "unknown": "정보 없음",
    "": "미선택",
}

ENVIRONMENT_LABELS = {
    "high_temp": "고온 환경",
    "corrosion": "부식 환경",
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


def get_rule_based_analysis(prediction: str, material: str, environment: str):
    rules = []

    if prediction == "피로 파괴":
        rules.append("반복 하중에 의해 균열이 점진적으로 성장했을 가능성이 있습니다.")
    elif prediction == "취성 파괴":
        rules.append("재료가 거의 변형 없이 급격히 파손되었을 가능성이 있습니다.")
    elif prediction == "연성 파괴":
        rules.append("재료가 큰 변형을 겪은 후 파손되었을 가능성이 있습니다.")
    elif prediction == "입계 파괴":
        rules.append("균열이 결정립 경계를 따라 진행되었을 가능성이 있습니다.")

    if environment == "corrosion":
        rules.append("부식 환경이 표면 손상이나 균열 시작에 영향을 주었을 가능성이 있습니다.")

    if environment == "high_temp":
        rules.append("고온 환경으로 인해 재료 특성이 변화했을 가능성이 있습니다.")

    if material == "steel" and environment == "corrosion":
        rules.append("강 재질은 부식 환경에서 표면 손상이 발생하기 쉬울 수 있습니다.")

    if prediction == "피로 파괴" and environment == "corrosion":
        rules.append("부식과 반복 하중이 함께 작용했을 가능성도 고려할 수 있습니다.")

    if prediction == "입계 파괴" and environment == "high_temp":
        rules.append("고온 환경은 결정립 경계 손상과 관련될 가능성이 있습니다.")

    return rules


def build_condition_text(material: str, environment: str) -> str:
    condition_lines = []

    material_ko = MATERIAL_LABELS.get(material, material)
    environment_ko = ENVIRONMENT_LABELS.get(environment, environment)

    if material not in ["", "unknown"]:
        condition_lines.append(f"- 참고 재질: {material_ko}")

    if environment not in ["", "unknown"]:
        condition_lines.append(f"- 참고 환경: {environment_ko}")

    if not condition_lines:
        return "없음"

    return "\n".join(condition_lines)


def build_prompt(prediction: str, confidence_percent: float, material: str, environment: str) -> str:
    rules = get_rule_based_analysis(prediction, material, environment)
    rules_text = "\n".join([f"- {r}" for r in rules]) if rules else "없음"
    condition_text = build_condition_text(material, environment)

    return f"""
너는 파손 단면 분석 결과를 설명하는 도우미다.
설명 대상은 전문가가 아니라 일반 사용자다.

CNN이 예측한 파손 유형은 {prediction}이고, 신뢰도는 {confidence_percent:.1f}%이다.

참고 입력 정보
{condition_text}

추가 분석 정보
{rules_text}

해야 할 일
1. 예상 원인은 15~25자 정도의 짧은 키워드형 문장으로 작성한다.
2. 판단 근거 설명은 3문장으로 작성한다.
   - 1문장: 표면에서 보이는 특징
   - 2문장: 그 특징이 해당 파손 유형과 연결되는 이유
   - 3문장: 가능한 원인 또는 재질/환경과의 관련성

중요 규칙
- 반드시 한국어로만 작성한다.
- 확정하지 말고 추정형으로 작성한다.
- 재질과 환경을 단순히 나열하지 않는다.
- 환경이 정보 없음 또는 미선택이면 환경 조건을 언급하지 않는다.
- 재질이 정보 없음 또는 미선택이면 재질을 언급하지 않는다.
- 같은 의미를 반복하지 않는다.

반드시 아래 JSON 형식으로만 출력하라.

{{
  "expected_cause": "짧은 원인 요약",
  "explanation": "표면 특징 설명. 파손 유형과 연결되는 이유. 가능한 원인 추정."
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


def generate_llm_analysis(
    prediction: str,
    confidence_percent: float,
    material: str,
    environment: str,
):
    llm_expected_cause = EXPECTED_CAUSE[prediction]
    korean_explanation = RULE_BASED_EXPLANATIONS[prediction]

    prompt = build_prompt(prediction, confidence_percent, material, environment)

    try:
        response = ollama.chat(
            model="exaone3.5:2.4b",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
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
            print("JSON 파싱 실패 → 기존 규칙 기반 설명 사용")
            korean_explanation = clean_text(raw_response)

        korean_explanation = validate_explanation(prediction, korean_explanation)

    except Exception as e:
        print(f"Ollama 오류: {e}")
        korean_explanation = (
            f"{prediction}으로 분류되었습니다. "
            f"신뢰도는 {confidence_percent:.1f}%입니다. "
            f"설명 생성 중 오류가 발생하여 기본 원인 설명을 표시합니다."
        )

    return {
        "feature": FEATURES[prediction],
        "cause": llm_expected_cause,
        "expected_cause": llm_expected_cause,
        "explanation": korean_explanation,
    }