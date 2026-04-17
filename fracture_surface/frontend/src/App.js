import { useState, useRef } from "react";
import FlipCard from "./components/FlipCard";

// ── 1. 내부 시스템용 영어 키 통일 ──
const FRACTURE_TYPES = ["Cleavage", "Ductile", "Fatigue", "Intergranular"];

// ── 2. 프론트엔드 카드 표시용 한글 매핑 ──
const KO_LABELS = {
  Cleavage: "취성 파괴",
  Ductile: "연성 파괴",
  Fatigue: "피로 파괴",
  Intergranular: "입계 파괴", // 백엔드 구조에 맞게 '입계 파괴' 사용
};

// ── 3. 이미지 주소 매핑 (영어 키 기준) ──
const FRACTURE_IMAGES = {
  Cleavage: "/images/cleavage.jpg",
  Ductile: "/images/ductile.jpg",
  Fatigue: "/images/fatigue.jpg",
  Intergranular: "/images/Intergranular.jpg",
};

const DEFAULT_SIMILARITIES = Object.fromEntries(
  FRACTURE_TYPES.map((t) => [t, { sim: "—", best: false }])
);

export default function App() {
  const fileRef = useRef(null);

  const [previewUrl, setPreviewUrl] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);
  const [similarities, setSimilarities] = useState(DEFAULT_SIMILARITIES);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setSimilarities(DEFAULT_SIMILARITIES);
  };

  const handleUpload = async () => {
    const file = fileRef.current?.files[0];
    if (!file) return alert("이미지를 먼저 선택해주세요.");

    const formData = new FormData();
    formData.append("file", file);

    try {
      setUploading(true);

      const res = await fetch("http://localhost:8000/analyze", { method: "POST", body: formData });
      const data = await res.json();

      setResult(data);

      const bestKo = data.prediction; 
      const mapped = {};

      if (data.similarities) {
        // 백엔드의 한국어 데이터를 영어 키에 맞춰서 매핑
        FRACTURE_TYPES.forEach((t) => {
          const koName = KO_LABELS[t]; 
          mapped[t] = {
            sim: data.similarities[koName] ?? "—",
            best: koName === bestKo, 
          };
        });
      } else {
        Object.assign(mapped, DEFAULT_SIMILARITIES);
        const bestEnKey = Object.keys(KO_LABELS).find(key => KO_LABELS[key] === bestKo);
        if (bestEnKey) {
          mapped[bestEnKey] = { sim: data.confidence, best: true };
        }
      }
      setSimilarities(mapped);

    } catch (err) {
      alert("분석 중 오류가 발생했습니다.\n백엔드 서버가 켜져 있는지 확인해주세요.");
      console.error(err);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 pb-20">
      <header className="border-b border-slate-200 bg-white/90 backdrop-blur sticky top-0 z-20">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <p className="text-sm font-semibold tracking-[0.2em] text-blue-600 uppercase">
              Failure Analysis System
            </p>
            <h1 className="text-2xl font-bold">파손단면 이미지 분석 웹 시스템</h1>
          </div>
          <button
            onClick={handleUpload}
            disabled={uploading}
            className="rounded-xl bg-slate-900 text-white px-6 py-2.5 text-sm font-bold
                       disabled:opacity-50 disabled:cursor-not-allowed hover:bg-slate-800 transition"
          >
            {uploading ? "분석 중…" : "분석 시작"}
          </button>
        </div>
      </header>

      <section className="max-w-7xl mx-auto px-6 py-10">
        <div className="bg-white rounded-3xl shadow-sm border p-8 text-center">
          <h2 className="text-2xl font-bold mb-4">이미지 업로드</h2>
          <label
            htmlFor="file-input"
            className="block border-2 border-dashed border-slate-300 rounded-2xl p-10
                       cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition"
          >
            {previewUrl ? (
              <img
                src={previewUrl}
                alt="미리보기"
                className="mx-auto max-h-64 rounded-xl object-contain shadow-sm"
              />
            ) : (
              <div className="py-8">
                <p className="text-slate-500 mb-4 text-lg">파손단면 이미지를 이곳에 업로드하세요</p>
                <span className="px-6 py-3 bg-blue-600 text-white rounded-xl text-sm font-bold shadow-md">
                  컴퓨터에서 파일 선택
                </span>
              </div>
            )}
          </label>
          <input
            id="file-input"
            type="file"
            accept="image/*"
            ref={fileRef}
            onChange={handleFileChange}
            className="hidden"
          />
        </div>
      </section>

      <section className="max-w-7xl mx-auto px-6 py-4">
        <div className="mb-6">
          <h2 className="text-2xl font-bold">유사도 기반 파손 유형 비교</h2>
          <p className="text-slate-500 mt-2">
            입력된 이미지의 질감, 균열 패턴 등을 기준 데이터와 비교합니다.
            <span className="ml-2 text-blue-500 font-medium">카드를 클릭하면 상세 설명을 볼 수 있습니다.</span>
          </p>
        </div>

        <div className="grid lg:grid-cols-5 gap-4 items-start">
          <div className="bg-slate-800 p-4 rounded-2xl border border-slate-700 shadow-sm flex flex-col">
            <p className="text-sm font-bold text-slate-300 mb-2">업로드한 이미지</p>
            <div className="flex-1 bg-slate-900 rounded-xl flex items-center justify-center overflow-hidden min-h-[140px]">
              {previewUrl
                ? <img src={previewUrl} alt="입력" className="w-full h-full object-cover rounded-xl opacity-90" />
                : <span className="text-slate-600 text-sm">대기 중...</span>
              }
            </div>
          </div>

          {FRACTURE_TYPES.map((type) => (
            <FlipCard
              key={type}
              type={KO_LABELS[type]} // FlipCard가 인식할 수 있는 한국어로 변환하여 전달
              similarity={similarities[type].sim}
              isBest={similarities[type].best}
              imageSlot={
                <img
                  src={FRACTURE_IMAGES[type]} 
                  alt={KO_LABELS[type]}
                  className="w-full h-full object-cover rounded-xl"
                  onError={(e) => {
                    e.target.src = "https://via.placeholder.com/150?text=No+Image"; 
                  }}
                />
              }
            />
          ))}
        </div>
      </section>

      {result && (
        <section className="max-w-7xl mx-auto px-6 py-10 animate-fade-in">
          <div className="bg-white rounded-3xl border p-8 shadow-sm border-blue-100">
            <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
              <span className="w-2 h-8 bg-blue-600 rounded-full"></span>
              최종 분석 결과 보고서
            </h3>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="p-5 bg-slate-50 rounded-2xl border border-slate-100">
                <p className="text-sm font-bold text-slate-400 mb-1">파손 유형 (분류)</p>
                <p className="text-2xl font-bold text-blue-600">{result.prediction}</p>
              </div>
              <div className="p-5 bg-slate-50 rounded-2xl border border-slate-100">
                <p className="text-sm font-bold text-slate-400 mb-1">신뢰도 점수</p>
                <p className="text-2xl font-bold text-slate-800">{result.confidence}</p>
              </div>
            </div>

            <div className="mt-8 p-6 bg-slate-50 rounded-2xl border border-slate-200">
              <h4 className="text-lg font-bold mb-3 text-slate-700">전문가 판단 근거 (설명형 AI)</h4>
              <p className="text-slate-600 leading-relaxed text-lg">{result.explanation}</p>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}