import { useState } from "react";

// ── 백엔드 CNN 모델 분류 기준 및 상세 데이터 (입계 파괴 포함) ──
const FRACTURE_DETAILS = {
  "취성 파괴": {
    en: "Cleavage Fracture",
    cause: "저온 환경, 충격 하중, 재료의 인성 부족",
    features: [
      "소성 변형이 거의 없이 갑작스럽게 발생",
      "결정면을 따라 쪼개지는 벽개(Cleavage) 파면",
      "강에서 강 가에 얼음이 깨진 듯한 반짝이는 외관",
    ],
    prevention: "연성-취성 천이 온도(DBTT) 고려, 노치 설계 지양",
    color: "blue",
  },
  "연성 파괴": {
    en: "Ductile Fracture",
    cause: "과도한 인장력, 연성 재료의 한계 응력 초과",
    features: [
      "파단 전 현저한 목수축(Necking) 현상 발생",
      "미세공공(Void)의 합체로 인한 딤플(Dimple) 구조",
      "거시적으로 컵-콘(Cup-and-Cone) 형태 형성",
    ],
    prevention: "재료의 항복 강도 준수, 단면적 보강 설계",
    color: "green",
  },
  "피로 파괴": {
    en: "Fatigue Fracture",
    cause: "반복적인 변동 응력, 진동, 장기 가동 부품",
    features: [
      "파면에서 조개껍데기 모양의 비치 마크(Beach marks)",
      "현미경 상의 미세한 줄무늬(Striations) 존재",
      "정적 하중보다 훨씬 낮은 응력에서 급작스럽게 발생",
    ],
    prevention: "표면 거칠기 개선, 잔류 압축 응력(Shot Peening) 부여",
    color: "amber",
  },
  "입계 파괴": { 
    en: "Intergranular Fracture",
    cause: "결정립계의 취화, 부식 환경(응력부식균열), 입계 석출물",
    features: [
      "균열이 결정립 경계(Grain Boundary)를 따라 전파",
      "설탕 덩어리가 부서진 듯한 입체적인 암석 모양 파면",
      "입계 부식이나 수소 취성에 의해 주로 발생",
    ],
    prevention: "결정립 미세화, 열처리 제어, 부식 환경 차단",
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

export default function FlipCard({ type, similarity, isBest, imageSlot }) {
  const [flipped, setFlipped] = useState(false);
  
  // 데이터가 없을 경우를 대비한 방어 코드
  const detail = FRACTURE_DETAILS[type] || FRACTURE_DETAILS["취성 파괴"];
  const c = COLOR_MAP[detail.color];

  return (
    <div
      className="cursor-pointer group"
      style={{ perspective: "1000px" }}
      onClick={() => setFlipped((f) => !f)}
    >
      <div
        style={{
          position: "relative",
          transformStyle: "preserve-3d",
          transition: "transform 0.6s cubic-bezier(0.4, 0, 0.2, 1)",
          transform: flipped ? "rotateY(180deg)" : "rotateY(0deg)",
          height: "380px",
        }}
      >
        {/* ── FRONT ─────────────────────────────── */}
        <div
          className={`absolute inset-0 rounded-2xl border shadow-sm p-5 flex flex-col ${
            isBest ? "border-blue-500 bg-blue-50/50 ring-2 ring-blue-200" : "bg-white border-slate-200"
          }`}
          style={{ backfaceVisibility: "hidden" }}
        >
          <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">{detail.en}</p>
          <h3 className="text-lg font-bold text-slate-800 mb-3">{type}</h3>

          <div className="h-40 bg-slate-100 rounded-xl flex items-center justify-center mb-4 overflow-hidden flex-shrink-0 border border-slate-50">
            {imageSlot ?? (
              <span className="text-slate-400 text-xs">Reference Image</span>
            )}
          </div>

          <div className="mt-auto">
            <div className="flex justify-between items-end">
              <div>
                <p className="text-xs text-slate-500 mb-1">유사도 점수</p>
                <p className={`text-2xl font-black ${isBest ? "text-blue-600" : "text-slate-700"}`}>
                  {similarity}
                </p>
              </div>
              {isBest && (
                <span className="mb-1 text-[10px] font-bold text-white bg-blue-600 rounded-lg px-2 py-1 uppercase">
                  Best Match
                </span>
              )}
            </div>
          </div>

          <p className="mt-4 text-[10px] text-slate-400 text-center font-medium group-hover:text-blue-500 transition-colors">
            카드를 클릭하여 엔지니어 가이드 보기
          </p>
        </div>

        {/* ── BACK ──────────────────────────────── */}
        <div
          className={`absolute inset-0 rounded-2xl border-2 shadow-xl p-5 flex flex-col bg-white ${c.border}`}
          style={{
            backfaceVisibility: "hidden",
            transform: "rotateY(180deg)",
          }}
        >
          <div className="flex-1 overflow-y-auto custom-scrollbar pr-1">
            <div className="flex items-center justify-between mb-4">
              <div>
                <p className={`text-[10px] font-black uppercase tracking-tighter ${c.title}`}>
                  {detail.en}
                </p>
                <p className="text-lg font-bold text-slate-900">{type}</p>
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <p className="text-[11px] font-bold text-slate-400 uppercase mb-1.5 flex items-center gap-1">
                   발생 원인
                </p>
                <p className="text-sm text-slate-700 leading-snug">{detail.cause}</p>
              </div>

              <div>
                <p className="text-[11px] font-bold text-slate-400 uppercase mb-1.5 flex items-center gap-1">
                   미세구조적 특징
                </p>
                <ul className="space-y-1.5">
                  {detail.features.map((f) => (
                    <li key={f} className="flex gap-2 items-start text-sm text-slate-600 leading-snug">
                      <span className={`mt-1.5 w-1.5 h-1.5 rounded-full flex-shrink-0 ${c.dot}`} />
                      {f}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-100">
            <p className="text-[11px] font-bold text-slate-400 uppercase mb-1.5">권장 예방 대책</p>
            <p className="text-sm font-medium text-slate-800 bg-slate-50 p-2 rounded-lg border border-slate-100">
              {detail.prevention}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}