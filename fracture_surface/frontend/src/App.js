import { useState, useRef } from "react";
import FlipCard from "./components/FlipCard";

// 비교 대상 카드 데이터 (유사도는 분석 후 백엔드로부터 받아옴)
const FRACTURE_TYPES = ["취성 파괴", "연성 파괴", "피로 파괴", "크리프 파괴"];

const DEFAULT_SIMILARITIES = {
  "취성 파괴":  { sim: "—", best: false },
  "연성 파괴":  { sim: "—", best: false },
  "피로 파괴":  { sim: "—", best: false },
  "크리프 파괴":{ sim: "—", best: false },
};

export default function App() {
  const fileRef = useRef(null);

  const [previewUrl, setPreviewUrl]     = useState(null);   // 업로드 미리보기
  const [uploading, setUploading]       = useState(false);  // 로딩 상태
  const [result, setResult]             = useState(null);   // 백엔드 응답
  const [similarities, setSimilarities] = useState(DEFAULT_SIMILARITIES);

  // ── 파일 선택 핸들러 ──────────────────────────────────────────
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setSimilarities(DEFAULT_SIMILARITIES);
  };

  // ── 분석 요청 핸들러 ──────────────────────────────────────────
  const handleUpload = async () => {
    const file = fileRef.current?.files[0];
    if (!file) return alert("이미지를 먼저 선택해주세요.");

    const formData = new FormData();
    formData.append("file", file);

    try {
      setUploading(true);

      const res  = await fetch("http://localhost:8000/analyze", { method: "POST", body: formData });
      const data = await res.json();

      // 백엔드 응답 예시:
      // { prediction: "취성 파괴", confidence: "89%", explanation: "...",
      //   similarities: { "취성 파괴": "89%", "연성 파괴": "52%", ... } }

      setResult(data);

      // 유사도 매핑 (백엔드가 similarities 필드를 내려주는 경우)
      if (data.similarities) {
        const best = data.prediction;
        const mapped = {};
        FRACTURE_TYPES.forEach((t) => {
          mapped[t] = {
            sim:  data.similarities[t] ?? "—",
            best: t === best,
          };
        });
        setSimilarities(mapped);
      } else {
        // 단순 prediction만 내려오는 현재 테스트 백엔드 대응
        const mapped = { ...DEFAULT_SIMILARITIES };
        mapped[data.prediction] = { sim: data.confidence, best: true };
        setSimilarities(mapped);
      }
    } catch (err) {
      alert("분석 중 오류가 발생했습니다.\n백엔드 서버가 실행 중인지 확인해주세요.");
      console.error(err);
    } finally {
      setUploading(false);
    }
  };

  const bestType = result?.prediction ?? null;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">

      {/* ── Header ───────────────────────────────────────── */}
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
            className="rounded-2xl bg-slate-900 text-white px-5 py-2.5 text-sm font-medium
                       disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {uploading ? "분석 중…" : "분석 시작"}
          </button>
        </div>
      </header>

      {/* ── Upload ───────────────────────────────────────── */}
      <section className="max-w-7xl mx-auto px-6 py-10">
        <div className="bg-white rounded-[28px] shadow border p-8 text-center">
          <h2 className="text-3xl font-bold mb-4">이미지 업로드</h2>

          <label
            htmlFor="file-input"
            className="block border-2 border-dashed border-slate-300 rounded-2xl p-10
                       cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition"
          >
            {previewUrl ? (
              <img
                src={previewUrl}
                alt="미리보기"
                className="mx-auto max-h-52 rounded-xl object-contain"
              />
            ) : (
              <>
                <p className="text-slate-500 mb-4">파손단면 이미지를 업로드하세요</p>
                <span className="px-5 py-3 bg-slate-900 text-white rounded-xl text-sm">
                  파일 선택
                </span>
              </>
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

          {previewUrl && (
            <p className="mt-3 text-sm text-slate-400">
              다른 이미지를 선택하려면 위 영역을 다시 클릭하세요.
            </p>
          )}
        </div>
      </section>

      {/* ── Comparison (FlipCard 영역) ────────────────────── */}
      <section className="max-w-7xl mx-auto px-6 py-10">
        <div className="mb-6">
          <h2 className="text-3xl font-bold">유사도 기반 파손 유형 비교</h2>
          <p className="text-slate-500 mt-2">
            이미지의 질감, 균열 패턴, 표면 거칠기 등을 기반으로 유사도를 계산합니다.
            <span className="ml-2 text-blue-500 font-medium">카드를 클릭하면 상세 설명을 볼 수 있습니다.</span>
          </p>
        </div>

        <div className="grid lg:grid-cols-5 gap-6">

          {/* 입력 이미지 카드 */}
          <div className="bg-white p-4 rounded-2xl border border-slate-200 shadow-sm flex flex-col">
            <p className="text-sm text-slate-500 mb-2">입력 이미지</p>
            <div className="flex-1 bg-slate-200 rounded-xl flex items-center justify-center overflow-hidden min-h-[140px]">
              {previewUrl
                ? <img src={previewUrl} alt="입력" className="w-full h-full object-cover rounded-xl" />
                : <span className="text-slate-400 text-sm">사용자 이미지</span>
              }
            </div>
          </div>

          {/* FlipCard × 4 */}
          {FRACTURE_TYPES.map((type) => (
            <FlipCard
              key={type}
              type={type}
              similarity={similarities[type].sim}
              isBest={similarities[type].best}
            />
          ))}
        </div>

        {/* 최고 유사도 결과 배너 */}
        {bestType && (
          <div className="mt-6 p-5 bg-white rounded-2xl border">
            <p className="text-lg font-semibold text-blue-600">
              → 입력 이미지가 &ldquo;{bestType}&rdquo;와 가장 유사합니다.
            </p>
            <p className="text-slate-600 mt-2">{result?.explanation}</p>
          </div>
        )}
      </section>

      {/* ── Final Result ─────────────────────────────────── */}
      {result && (
        <section className="max-w-7xl mx-auto px-6 py-10">
          <div className="bg-white rounded-2xl border p-6 shadow-sm">
            <h3 className="text-2xl font-bold mb-4">최종 분석 결과</h3>

            <div className="grid md:grid-cols-4 gap-4">
              {[
                ["파손 유형", result.prediction],
                ["신뢰도",   result.confidence],
                ["특징",     "균열 전파"],
                ["원인",     "충격/과하중"],
              ].map(([t, v]) => (
                <div key={t} className="p-4 bg-slate-50 rounded-xl">
                  <p className="text-sm text-slate-500">{t}</p>
                  <p className="text-lg font-bold">{v}</p>
                </div>
              ))}
            </div>

            <div className="mt-6 rounded-2xl border border-amber-200 bg-amber-50 p-4">
              <p className="text-sm font-semibold text-amber-700">신뢰도 상태 안내</p>
              <p className="text-sm text-amber-700 mt-1">
                현재 분석은 충분한 유사도를 보여 비교적 신뢰할 수 있는 결과입니다.
                유사도가 낮은 경우에는 추가 이미지나 데이터가 필요할 수 있습니다.
              </p>
            </div>

            <div className="grid lg:grid-cols-2 gap-6 mt-8">
              <div className="p-5 bg-slate-50 rounded-2xl border">
                <h4 className="text-xl font-semibold mb-3">판단 근거 설명</h4>
                <p className="text-slate-700 leading-7">{result.explanation}</p>
              </div>
              <div className="p-5 bg-blue-50 rounded-2xl border border-blue-100">
                <h4 className="text-xl font-semibold mb-3 text-blue-700">예상 사고 원인</h4>
                <p className="text-slate-700 leading-7">
                  분석 결과로 보아 본 파손은{" "}
                  <span className="font-semibold">충격 하중</span>,{" "}
                  <span className="font-semibold">과도한 응력 집중</span>, 또는{" "}
                  <span className="font-semibold">재료 내부 결함</span> 등에 의해 발생했을 가능성이 있습니다.
                  실제 원인은 재료 정보, 사용 환경, 반복 하중 여부와 함께 종합적으로 판단해야 합니다.
                </p>
              </div>
            </div>
          </div>
        </section>
      )}

    </div>
  );
}