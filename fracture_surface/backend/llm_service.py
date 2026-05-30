import re
import json
import ollama

## 파손 유형 설명

FEATURES = {
    "취성 파괴": "균열이 빠르게 진행되고 소성 변형이 거의 없음",
    "연성 파괴": "큰 소성 변형과 함께 파단 발생",
    "피로 파괴": "반복 하중으로 균열이 점진적으로 성장",
    "입계 파괴": "결정립계 중심의 분리 형태",
}

EXPECTED_CAUSE = {
    "취성 파괴": "급격한 응력 집중",
    "연성 파괴": "과도한 하중",
    "피로 파괴": "장기간 반복 하중",
    "입계 파괴": "결정립 경계 약화",
}

RULE_BASED_EXPLANATIONS = {
    "취성 파괴": "취성 파괴는 소성 변형 없이 균열이 빠르게 진행되는 특징이 있으며, 충격 하중이나 응력 집중에 의해 발생했을 가능성이 있습니다.",
    "연성 파괴": "연성 파괴는 재료가 큰 소성 변형을 겪은 뒤 파단되는 특징이 있으며, 과도한 하중이나 국부적인 변형 집중과 관련될 가능성이 있습니다.",
    "피로 파괴": "피로 파괴는 반복 하중으로 인해 균열이 점진적으로 성장한 뒤 파손되는 특징이 있으며, 장기간 반복 응력이 원인일 가능성이 있습니다.",
    "입계 파괴": "균열이 결정립 경계를 따라 분리된 형태로 관찰될 가능성이 있으며, 재료 내부 경계면 약화와 관련될 수 있습니다.",
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

MATERIAL_CONTEXT = {
    "steel": "강은 구조용 재료로 널리 사용되며, 반복 하중 조건에서는 피로 균열 성장 가능성을 고려할 수 있습니다.",
    "stainless_steel": "스테인리스강은 조건에 따라 결정립계 부식이나 국부적인 열화가 문제가 될 수 있습니다.",
    "aluminum": "알루미늄은 낮은 밀도와 비교적 높은 연성을 가지는 재료입니다.",
    "titanium": "티타늄은 가볍고 강도 대비 성능이 좋은 재료로, 항공·고성능 부품에서 반복 하중에 의한 피로 손상을 고려할 수 있습니다.",
    "cast_iron": "주철은 흑연 조직이 응력 집중원처럼 작용할 수 있어 취성적 균열 진행 가능성을 고려할 수 있습니다.",
    "copper": "구리는 연성과 가공성이 좋은 금속이므로, 큰 소성 변형이나 반복 하중 조건에서의 손상 가능성을 함께 검토할 수 있습니다.",
    "magnesium": "마그네슘 합금은 매우 가벼운 재료이며, 조건에 따라 취성적 파손이나 피로 손상 가능성을 검토할 수 있습니다.",
    "nickel_alloy": "니켈 합금은 고온 강도와 내식성이 요구되는 부품에 사용되므로, 고온 환경에서의 열화나 반복 하중 손상을 고려할 수 있습니다.",
    "tool_steel": "공구강은 높은 경도와 내마모성이 요구되는 재료로, 응력 집중 조건에서는 취성적 균열 가능성을 고려할 수 있습니다.",
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
            "신뢰도가 낮으므로 반드시 '가능성 중 하나로 보입니다', "
            "'추가 검토가 필요합니다'와 같은 불확실한 표현을 사용한다."
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
        rules.append("소성 변형이 거의 없이 균열이 빠르게 진행되었을 가능성이 있습니다.")
    elif prediction == "연성 파괴":
        rules.append("재료가 큰 소성 변형을 겪은 후 파손되었을 가능성이 있습니다.")
    elif prediction == "입계 파괴":
        rules.append("균열이 결정립 경계를 따라 진행되었을 가능성이 있습니다.")

    material_context = MATERIAL_CONTEXT.get(material)
    if material_context:
        rules.append(
            "재질 정보는 보조 참고 정보로만 사용해야 하며, 이미지에서 직접 확인되지 않는 재질 특성은 단정하지 않습니다."
        )
        rules.append(material_context)

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
- 재질의 일반적인 특성을 과도하게 끌어오지 않는다.
- 재질 정보는 보조 설명으로만 사용하고, 파손 유형과 직접 연결되지 않으면 언급하지 않는다.
- 이미지에서 직접 확인되지 않는 재질 특성, 사용 환경, 열처리 상태, 부식 상태는 단정하지 않는다.
- 재질이 정보 없음 또는 미선택이면 재질을 언급하지 않는다.
- 같은 의미를 반복하지 않는다.
- 설명은 너무 길지 않게 작성한다.
- "결정립 경계", "균열 진행" 같은 표현을 여러 항목에서 반복 사용하지 않는다.

반드시 아래 JSON 형식으로만 출력하라.

{{
  "expected_cause": "짧은 원인 요약",
  "explanation": "관찰된 특징 문장. 파손 유형과 연결되는 해석 문장. 가능한 원인 추정 문장."
}}
"""


def validate_explanation(prediction: str, text: str) -> str:
    invalid_keywords = {
    "연성 파괴": ["결정립 경계", "급격히 진행", "반복 하중", "반복 응력", "피로", "반복적인 변형"],
    "취성 파괴": ["큰 소성 변형", "늘어난"],
    "피로 파괴": ["한 번의 큰 하중", "급격한 파손"],
    "입계 파괴": ["큰 소성 변형"],
    }
    banned = invalid_keywords.get(prediction, [])

    for word in banned:
        if word in text:
            print(f"[validate] 부적절 표현 감지: {word}")
            return RULE_BASED_EXPLANATIONS[prediction]

    return text


def validate_confidence_tone(confidence_percent: float, text: str) -> str:
    uncertainty_words = ["가능성", "추정", "보입니다", "검토"]

    if not any(word in text for word in uncertainty_words):
        text += " 다만 실제 판정에는 추가 검토가 필요할 수 있습니다."

    if confidence_percent < 40 and "추가" not in text:
        text += " 신뢰도가 낮아 추가 이미지 또는 전문가 검토가 필요합니다."

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
                        "확정 표현은 금지하고 추정형 표현만 사용한다. "
                        "재질 특성은 참고 정보로만 사용하고 과도하게 단정하지 않는다."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
                "temperature": 0.3,
                "top_p": 0.9,
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


## 분석 결과 비교 설명
def generate_compare_analysis(compare_items: list):
    summary_lines = []

    # 신뢰도 차이 계산
    conf1 = float(compare_items[0].get("confidence", 0))
    conf2 = float(compare_items[1].get("confidence", 0))

    conf_gap = abs(conf1 - conf2)

    if conf_gap < 5:
        confidence_gap_text = (
            "두 결과의 신뢰도는 유사한 수준으로 해석할 수 있다."
        )
    elif conf_gap < 15:
        confidence_gap_text = (
            "두 결과의 신뢰도에는 다소 차이가 존재한다."
        )
    else:
        confidence_gap_text = (
            "두 결과의 신뢰도 차이가 비교적 큰 편이다."
        )

    for idx, item in enumerate(compare_items, start=1):
        summary_lines.append(
            f"""
비교 {idx}
- 파손 유형: {item.get("prediction", "-")}
- 표시 유형: {item.get("display_prediction", item.get("prediction", "-"))}
- 신뢰도: {item.get("confidence", "-")}
- 재질: {MATERIAL_LABELS.get(item.get("material", ""), item.get("material", "-"))}
- 혼합 여부: {"혼합 가능성 있음" if item.get("is_mixed") else "단일 유형 가능성 높음"}
- 주요 특징: {item.get("feature", "-")}
- 예상 원인: {item.get("expected_cause", "-")}
"""
        )

    prompt = f"""
너는 파손단면 분석 결과를 비교 설명하는 도우미다.
아래 여러 분석 결과를 단순 나열하지 말고, 사용자가 한눈에 이해할 수 있도록 비교하라.

분석 결과 목록:
{chr(10).join(summary_lines)}

신뢰도 참고:
{confidence_gap_text}

작성해야 할 내용:
1. summary
   - 두 결과를 각각 나열하지 말고, 비교했을 때 가장 큰 차이 1가지만 결론형으로 작성한다.
   - "비교 1은", "비교 2는" 표현을 사용하지 않는다.
   - 신뢰도 수치나 우선 검토 여부는 포함하지 않는다.
   - 1문장으로 작성한다.

2. mechanism_compare_1
   - 비교 1의 파손 메커니즘을 설명하되, 단순 설명이 아니라 비교 2와 다른 점을 중심으로 작성한다.
   - 주요 차이가 없으면 억지로 만들지 말고 공통점 안에서 신뢰도나 재질 차이만 언급한다.

3. mechanism_compare_2
   - 비교 2의 파손 메커니즘을 설명하되, 단순 설명이 아니라 비교 1과 다른 점을 중심으로 작성한다.
   - 주요 차이가 없으면 억지로 만들지 말고 공통점 안에서 신뢰도나 재질 차이만 언급한다.

4. confidence_opinion
   - 두 결과의 신뢰도 차이를 비교한다.
   - 어떤 결과를 더 우선적으로 참고할 수 있는지 설명한다.
   - 신뢰도가 낮은 결과는 추가 검토가 필요하다고 설명한다.
   - 신뢰도 차이가 매우 작으면 우열을 과도하게 강조하지 않는다.

5. final_opinion
   - 사용자가 최종적으로 어떻게 해석하면 좋을지 1~2문장으로 작성한다.
   - summary와 같은 말을 반복하지 않는다.
   - 단정하지 말고 추가 검토 필요성을 포함한다.

중요 규칙:
- 반드시 한국어로 작성한다.
- JSON 이외의 문장은 출력하지 않는다.
- 단순히 "비교 1은 ~, 비교 2는 ~" 형식으로 나열하지 않는다.
- summary에는 신뢰도 관련 내용을 넣지 않는다.
- confidence_opinion에서만 신뢰도 차이를 설명한다.
- final_opinion은 짧게 작성하고, 전체 해석 시 주의할 점만 말한다.
- "반면", "상대적으로", "더 강하게", "차이를 보입니다" 같은 비교 표현을 사용한다.
- 확정 표현은 피하고 "가능성이 있습니다", "해석할 수 있습니다", "검토가 필요합니다" 형태로 작성한다.
- 너무 길게 쓰지 않는다.
- 같은 단어나 문장 구조를 반복하지 않는다.
- 동일한 의미를 여러 문장에서 반복 설명하지 않는다.
- "결정립 경계", "균열 진행", "반복 하중" 같은 표현을 여러 항목에서 반복 사용하지 않는다.
- 재질의 일반 특성을 과도하게 일반화하지 않는다.
- 이미지에서 직접 확인되지 않은 기계적 특성 차이를 단정하지 않는다.
- "더 취약하다", "더 약하다" 같은 표현은 명확한 근거가 있을 때만 사용한다.
- 각 항목은 1~3문장 이내로 작성한다.

반드시 아래 JSON 형식으로만 출력하라.

{{
  "summary": "핵심 요약",
  "mechanism_compare_1": "비교 1의 파손 메커니즘 설명",
  "mechanism_compare_2": "비교 2의 파손 메커니즘 설명",
  "confidence_opinion": "신뢰도 차이와 해석 주의점",
  "final_opinion": "최종 해석"
}}
"""

    fallback_result = {
        "summary": "두 결과는 파손이 진행된 방식에서 차이를 보이며, 하나는 순간적인 응력 집중, 다른 하나는 반복 하중에 의한 점진적 균열 성장 가능성을 더 강하게 보여줍니다.",
        "mechanism_compare_1": "비교 1은 순간적인 응력 집중이나 국부적인 균열 진행과 관련된 특징으로 해석될 수 있습니다.",
        "mechanism_compare_2": "비교 2는 반복 하중이나 점진적인 균열 성장과 관련된 특징으로 해석될 수 있습니다.",
        "confidence_opinion": "신뢰도가 더 높은 결과는 상대적으로 우선 참고할 수 있지만, 신뢰도가 낮은 결과는 추가 검토가 필요할 수 있습니다.",
        "final_opinion": "두 결과는 단일 판단보다 비교 관점에서 함께 확인하는 것이 좋습니다. 실제 판정에는 추가 이미지나 전문가 검토가 필요할 수 있습니다.",
    }

    try:
        response = ollama.chat(
            model="exaone3.5:2.4b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "너는 파손단면 분석 결과 비교 설명 생성기다. "
                        "반드시 한국어 JSON만 출력한다. "
                        "결과를 나열하지 말고 차이점 중심으로 비교한다. "
                        "신뢰도 내용은 confidence_opinion에서만 설명한다. "
                        "확정 표현은 피하고 추정형 표현을 사용한다."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
                "temperature": 0.35,
                "top_p": 0.9,
            },
        )

        raw_response = response["message"]["content"].strip()
        print("LLM 비교 원본 응답:", raw_response)

        parsed = parse_llm_json(raw_response)

        if parsed:
            return {
                "summary": clean_text(
                    parsed.get("summary", fallback_result["summary"])
                ),
                "mechanism_compare_1": clean_text(
                    parsed.get(
                        "mechanism_compare_1",
                        fallback_result["mechanism_compare_1"],
                    )
                ),
                "mechanism_compare_2": clean_text(
                    parsed.get(
                        "mechanism_compare_2",
                        fallback_result["mechanism_compare_2"],
                    )
                ),
                "confidence_opinion": clean_text(
                    parsed.get(
                        "confidence_opinion",
                        fallback_result["confidence_opinion"],
                    )
                ),
                "final_opinion": clean_text(
                    parsed.get("final_opinion", fallback_result["final_opinion"])
                ),
            }

        return fallback_result

    except Exception as e:
        print(f"LLM 비교 분석 오류: {e}")
        return fallback_result
