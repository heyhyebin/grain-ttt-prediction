import { useState } from "react";

// ── 파괴 유형별 상세 설명 데이터 ──────────────────────────────────
const FRACTURE_DETAILS = {
  "취성 파괴": {
    en: "Brittle Fracture",
    cause: "충격 하중, 저온 환경, 재료 내부 결함",
    features: [
      "소성 변형 거의 없음",
      "균열이 직선적·빠르게 전파",
      "파면이 거칠고 결정립 경계를 따라 분리",
    ],
    prevention: "인성(Toughness) 높은 소재 선택, 노치(notch) 제거 설계",
    color: "blue",
  },
  "연성 파괴": {
    en: "Ductile Fracture",
    cause: "과도한 인장 하중, 재료 항복 초과",
    features: [
      "상당한 소성 변형 동반",
      "컵-콘(cup-and-cone) 파면 형성",
      "파면이 섬유질 형태로 나타남",
    ],
    prevention: "적절한 단면적 설계, 과부하 방지 안전장치 적용",
    color: "green",
  },
  "피로 파괴": {
    en: "Fatigue Fracture",
    cause: "반복 하중(진동·굴곡), 응력 집중부 균열 시작",
    features: [
      "낮은 응력에서도 장기 반복 시 발생",
      "파면에 줄무늬(Striations)·조개껍질 무늬",
      "최종 파단부만 거칠고 나머지는 매끄러움",
    ],
    prevention: "표면 처리(shot peening), 응력 집중 최소화 설계",
    color: "amber",
  },
  "입계 파괴": {
    en: "Creep Fracture",
    cause: "고온 환경에서 장기간 하중 작용",
    features: [
      "고온(재결정 온도 이상)에서 발생",
      "시간 경과에 따른 변형 누적",
      "결정립계를 따라 기공(void) 생성·합체",
    ],
    prevention: "내열 합금 사용, 냉각 시스템 설계, 운용 온도 관리",
    color: "red",
  },
};

const COLOR_MAP = {
  blue: {
    badge: "bg-blue-100 text-blue-700",
    title: "text-blue-700",
    border: "border-blue-400",
    dot: "bg-blue-400",
  },
  green: {
    badge: "bg-green-100 text-green-700",
    title: "text-green-700",
    border: "border-green-400",
    dot: "bg-green-400",
  },
  amber: {
    badge: "bg-amber-100 text-amber-700",
    title: "text-amber-700",
    border: "border-amber-400",
    dot: "bg-amber-400",
  },
  red: {
    badge: "bg-red-100 text-red-700",
    title: "text-red-700",
    border: "border-red-400",
    dot: "bg-red-400",
  },
};

// ── FlipCard 컴포넌트 ────────────────────────────────────────────
export default function FlipCard({ type, similarity, isBest, imageSlot }) {
  const [flipped, setFlipped] = useState(false);
  const detail = FRACTURE_DETAILS[type];
  const c = COLOR_MAP[detail.color];

  return (
    <div
      className="cursor-pointer"
      style={{ perspective: "900px" }}
      onClick={() => setFlipped((f) => !f)}
    >
      <div
        style={{
          position: "relative",
          transformStyle: "preserve-3d",
          transition: "transform 0.55s cubic-bezier(0.4,0.2,0.2,1)",
          transform: flipped ? "rotateY(180deg)" : "rotateY(0deg)",
          height: "360px",
        }}
      >
        {/* ── FRONT ─────────────────────────────── */}
        <div
          className={`absolute inset-0 rounded-2xl border shadow-sm p-4 flex flex-col ${
            isBest ? "border-blue-500 bg-blue-50" : "bg-white border-slate-200"
          }`}
          style={{ backfaceVisibility: "hidden" }}
        >
          <p className="text-sm text-slate-500 mb-2">{type}</p>

          <div className="h-44 bg-slate-200 rounded-xl flex items-center justify-center mb-3 overflow-hidden flex-shrink-0">
            {imageSlot ?? (
              <span className="text-slate-400 text-sm">대표 이미지</span>
            )}
          </div>

          <div className="text-sm text-slate-600">유사도</div>
          <div className="text-xl font-bold">{similarity}</div>

          {isBest && (
            <span className="mt-2 text-xs font-semibold text-blue-600 bg-blue-100 rounded-full px-2 py-0.5 self-start">
              최고 유사도
            </span>
          )}

          <p className="mt-2 text-xs text-slate-400 text-right select-none">
            클릭하여 상세 정보 보기 ▶
          </p>
        </div>

        {/* ── BACK ──────────────────────────────── */}
        <div
          className={`absolute inset-0 rounded-2xl border shadow-sm p-4 flex flex-col bg-white ${c.border}`}
          style={{
            backfaceVisibility: "hidden",
            transform: "rotateY(180deg)",
          }}
        >
          {/* 스크롤 영역 */}
          <div className="flex-1 overflow-y-auto pr-1">
            {/* 헤더 */}
            <div className="flex items-center justify-between mb-3">
              <div>
                <p className={`text-xs font-semibold uppercase tracking-widest ${c.title}`}>
                  {detail.en}
                </p>
                <p className="text-base font-bold">{type}</p>
              </div>
              <span className={`text-xs font-bold rounded-full px-2 py-0.5 ${c.badge}`}>
                유사도 {similarity}
              </span>
            </div>

            {/* 주요 원인 */}
            <div className="mb-2">
              <p className="text-xs font-semibold text-slate-500 uppercase mb-1">
                주요 원인
              </p>
              <p className="text-sm text-slate-700">{detail.cause}</p>
            </div>

            {/* 파면 특징 */}
            <div className="mb-2">
              <p className="text-xs font-semibold text-slate-500 uppercase mb-1">
                파면 특징
              </p>
              <ul className="space-y-1">
                {detail.features.map((f) => (
                  <li key={f} className="flex gap-1.5 items-start text-sm text-slate-700">
                    <span
                      className={`mt-1.5 w-1.5 h-1.5 rounded-full flex-shrink-0 ${c.dot}`}
                    />
                    {f}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* 하단 고정 영역 */}
          <div className="pt-2 border-t border-slate-100">
            <p className="text-xs font-semibold text-slate-500 uppercase mb-1">
              예방 대책
            </p>
            <p className="text-sm text-slate-700">{detail.prevention}</p>
          </div>

          <p className="mt-2 text-xs text-slate-400 text-right select-none">
            ◀ 클릭하여 되돌아가기
          </p>
        </div>
      </div>
    </div>
  );
}
