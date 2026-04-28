import { useState, useRef, useEffect } from "react";
import FlipCard from "./components/FlipCard";

const FRACTURE_TYPES = ["취성 파괴", "연성 파괴", "피로 파괴", "입계 파괴"];

const DEFAULT_SIMILARITIES = {
  "취성 파괴": { sim: "—", best: false },
  "연성 파괴": { sim: "—", best: false },
  "피로 파괴": { sim: "—", best: false },
  "입계 파괴": { sim: "—", best: false },
};

const FRACTURE_IMAGES = {
  "취성 파괴": "/images/cleavage.jpg",
  "연성 파괴": "/images/ductile.jpg",
  "피로 파괴": "/images/fatigue.jpg",
  "입계 파괴": "/images/Intergranular.jpg",
};

const MATERIAL_LABELS = {
  steel: "강 (Steel)",
  aluminum: "알루미늄",
  titanium: "티타늄",
  unknown: "모름",
};

const ENVIRONMENT_LABELS = {
  high_temp: "고온 환경",
  corrosion: "부식 환경",
  unknown: "모름",
};

const CONFIDENCE_BOX_STYLES = {
  high: "border-green-200 bg-green-50 text-green-700",
  medium: "border-amber-200 bg-amber-50 text-amber-700",
  low: "border-red-200 bg-red-50 text-red-700",
};

const layout = {
  page: "min-h-screen bg-slate-50 text-slate-900",
  container: "max-w-7xl mx-auto px-6",
  section: "max-w-7xl mx-auto px-6 py-10",
  card: "bg-white rounded-2xl border p-6 shadow-sm",
  resultBox: "p-5 bg-slate-50 rounded-xl",
  select: "p-3 border rounded-xl bg-white",
  button:
    "rounded-2xl bg-slate-900 text-white px-5 py-2.5 text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed",
};

export default function App() {
  const fileRef = useRef(null);

  const [previewUrl, setPreviewUrl] = useState(null);
  const [imageBase64, setImageBase64] = useState(null);
  const [uploading, setUploading] = useState(false);

  const [material, setMaterial] = useState("");
  const [environment, setEnvironment] = useState("");

  const [result, setResult] = useState(null);
  const [similarities, setSimilarities] = useState(DEFAULT_SIMILARITIES);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const saved = localStorage.getItem("analysisHistory");
    if (saved) {
      setHistory(JSON.parse(saved));
    }
  }, []);

  const materialText = MATERIAL_LABELS[result?.material] || result?.material || "-";
  const environmentText =
    ENVIRONMENT_LABELS[result?.environment] || result?.environment || "-";

  const confidenceStyle =
    CONFIDENCE_BOX_STYLES[result?.confidence_status] ||
    CONFIDENCE_BOX_STYLES.medium;

  const fileToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();

      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;

      reader.readAsDataURL(file);
    });
  };

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const base64 = await fileToBase64(file);

    setPreviewUrl(base64);
    setImageBase64(base64);
    setResult(null);
    setSimilarities(DEFAULT_SIMILARITIES);
  };

  const updateSimilarities = (data) => {
    if (!data.similarities) {
      const mapped = { ...DEFAULT_SIMILARITIES };
      mapped[data.prediction] = { sim: data.confidence, best: true };
      setSimilarities(mapped);
      return;
    }

    const mapped = {};

    FRACTURE_TYPES.forEach((type) => {
      mapped[type] = {
        sim: data.similarities[type] ?? "—",
        best: type === data.prediction,
      };
    });

    setSimilarities(mapped);
  };

  const saveHistory = (data, image) => {
    const newItem = {
      id: Date.now(),
      time: new Date().toLocaleString(),
      image,
      result: data,
    };

    const updatedHistory = [newItem, ...history].slice(0, 10);

    setHistory(updatedHistory);
    localStorage.setItem("analysisHistory", JSON.stringify(updatedHistory));
  };

  const handleHistoryClick = (item) => {
    setResult(item.result);
    setPreviewUrl(item.image);
    setImageBase64(item.image);
    setMaterial(item.result.material || "");
    setEnvironment(item.result.environment || "");
    updateSimilarities(item.result);
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem("analysisHistory");
  };

  const handleUpload = async () => {
    const file = fileRef.current?.files[0];

    if (!file) return alert("이미지를 먼저 선택해주세요.");
    if (!material) return alert("재질을 선택해주세요.");
    if (!environment) return alert("환경을 선택해주세요.");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("material", material);
    formData.append("environment", environment);

    try {
      setUploading(true);

      const res = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`서버 오류: ${res.status}`);
      }

      const data = await res.json();

      setResult(data);
      updateSimilarities(data);
      saveHistory(data, imageBase64);
    } catch (err) {
      alert("분석 중 오류가 발생했습니다.\n백엔드 서버가 실행 중인지 확인해주세요.");
      console.error(err);
    } finally {
      setUploading(false);
    }
  };

  const reportText = result
    ? `
[분석 리포트]

1. 입력 조건
- 재질: ${materialText}
- 사용 환경: ${environmentText}

2. 분석 결과
- 예측된 파손 유형: ${result.prediction}
- 신뢰도: ${result.confidence}
- 신뢰도 상태: ${result.confidence_message}

3. 주요 특징
- ${result.feature}

4. 예상 원인
- ${result.expected_cause}

5. 종합 설명
- ${result.explanation}

6. 해석 의견
- 본 결과는 업로드된 파손단면 이미지와 사용자가 선택한 참고 정보를 바탕으로 생성된 분석 결과입니다.
- 실제 판정 시에는 추가 실험 및 전문가 검토가 필요할 수 있습니다.
`.trim()
    : "";

  const handleCopyReport = async () => {
    if (!reportText) return;

    try {
      await navigator.clipboard.writeText(reportText);
      alert("분석 리포트가 복사되었습니다.");
    } catch (err) {
      console.error(err);
      alert("복사에 실패했습니다.");
    }
  };

  return (
    <div className={layout.page}>
      <div className="flex min-h-screen">
        {/* 사이드바 */}
        <aside className="w-80 bg-white border-r border-slate-200 p-5 hidden lg:block">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-bold">분석 기록</h2>

            {history.length > 0 && (
              <button
                onClick={clearHistory}
                className="text-xs text-slate-400 hover:text-red-500"
              >
                전체 삭제
              </button>
            )}
          </div>

          <div className="space-y-3">
            {history.length === 0 && (
              <p className="text-sm text-slate-400">아직 분석 기록이 없습니다.</p>
            )}

            {history.map((item) => {
              const itemResult = item.result;
              const itemMaterial =
                MATERIAL_LABELS[itemResult.material] || itemResult.material || "-";
              const itemEnvironment =
                ENVIRONMENT_LABELS[itemResult.environment] ||
                itemResult.environment ||
                "-";

              return (
                <button
                  key={item.id}
                  onClick={() => handleHistoryClick(item)}
                  className="w-full text-left p-3 bg-slate-50 rounded-xl border hover:border-blue-300 hover:bg-blue-50 transition"
                >
                  <div className="flex gap-3">
                    <div className="w-14 h-14 rounded-lg bg-slate-200 overflow-hidden shrink-0">
                      {item.image && (
                        <img
                          src={item.image}
                          alt="기록 이미지"
                          className="w-full h-full object-cover"
                        />
                      )}
                    </div>

                    <div className="min-w-0">
                      <p className="font-bold text-sm">{itemResult.prediction}</p>
                      <p className="text-sm text-blue-600 font-semibold">
                        {itemResult.confidence}
                      </p>
                      <p className="text-xs text-slate-500 truncate">
                        {itemMaterial} / {itemEnvironment}
                      </p>
                      <p className="text-xs text-slate-400 mt-1">{item.time}</p>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </aside>

        {/* 메인 화면 */}
        <main className="flex-1">
          <header className="border-b border-slate-200 bg-white/90 backdrop-blur sticky top-0 z-20">
            <div
              className={`${layout.container} py-4 flex items-center justify-between`}
            >
              <div>
                <p className="text-sm font-semibold tracking-[0.2em] text-blue-600 uppercase">
                  Failure Analysis System
                </p>
                <h1 className="text-2xl font-bold">
                  파손단면 이미지 분석 웹 시스템
                </h1>
              </div>

              <button
                onClick={handleUpload}
                disabled={uploading}
                className={layout.button}
              >
                {uploading ? "분석 중…" : "분석 시작"}
              </button>
            </div>
          </header>

          <section className={layout.section}>
            <div className="bg-white rounded-[28px] shadow border p-8 text-center">
              <h2 className="text-3xl font-bold mb-4">이미지 업로드</h2>

              <label
                htmlFor="file-input"
                className="block border-2 border-dashed border-slate-300 rounded-2xl p-10 cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition"
              >
                {previewUrl ? (
                  <img
                    src={previewUrl}
                    alt="미리보기"
                    className="mx-auto max-h-64 md:max-h-72 object-contain rounded-xl"
                  />
                ) : (
                  <>
                    <p className="text-slate-500 mb-4">
                      파손단면 이미지를 업로드하세요
                    </p>
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

              <div className="mt-6 grid md:grid-cols-2 gap-4">
                <select
                  value={material}
                  onChange={(e) => setMaterial(e.target.value)}
                  className={layout.select}
                >
                  <option value="">재질 선택</option>
                  <option value="steel">강 (Steel)</option>
                  <option value="aluminum">알루미늄</option>
                  <option value="titanium">티타늄</option>
                  <option value="unknown">모름</option>
                </select>

                <select
                  value={environment}
                  onChange={(e) => setEnvironment(e.target.value)}
                  className={layout.select}
                >
                  <option value="">환경 선택</option>
                  <option value="high_temp">고온 환경</option>
                  <option value="corrosion">부식 환경</option>
                  <option value="unknown">모름</option>
                </select>
              </div>
            </div>
          </section>

          <section className={layout.section}>
            <div className="mb-6">
              <h2 className="text-3xl font-bold">유사도 기반 파손 유형 비교</h2>
              <p className="text-slate-500 mt-2">
                이미지의 질감, 균열 패턴, 표면 거칠기 등을 기반으로 유사도를
                계산합니다.
                <span className="ml-2 text-blue-500 font-medium">
                  카드를 클릭하면 상세 설명을 볼 수 있습니다.
                </span>
              </p>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 items-start">
              {FRACTURE_TYPES.map((type) => (
                <FlipCard
                  key={type}
                  type={type}
                  similarity={similarities[type].sim}
                  isBest={similarities[type].best}
                  imageSlot={
                    <img
                      src={FRACTURE_IMAGES[type]}
                      alt={type}
                      className="w-full h-full object-cover rounded-xl"
                    />
                  }
                />
              ))}
            </div>
          </section>

          {result && (
            <section className={layout.section}>
              <div className={layout.card}>
                <h3 className="text-2xl font-bold mb-6">최종 분석 결과</h3>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  <div className={layout.resultBox}>
                    <p className="text-sm text-slate-500">파손 유형</p>
                    <p className="text-lg font-bold">{result.prediction}</p>
                  </div>

                  <div className={layout.resultBox}>
                    <p className="text-sm text-slate-500">신뢰도</p>
                    <p className="text-lg font-bold">{result.confidence}</p>
                  </div>

                  <div className={layout.resultBox}>
                    <p className="text-sm text-slate-500">재질</p>
                    <p className="text-lg font-bold">{materialText}</p>
                  </div>

                  <div className={layout.resultBox}>
                    <p className="text-sm text-slate-500">환경</p>
                    <p className="text-lg font-bold">{environmentText}</p>
                  </div>
                </div>

                <div className="grid md:grid-cols-2 gap-4 mb-6">
                  <div className={layout.resultBox}>
                    <p className="text-sm text-slate-500">주요 특징</p>
                    <p className="text-lg font-semibold leading-7">
                      {result.feature}
                    </p>
                  </div>

                  <div className={layout.resultBox}>
                    <p className="text-sm text-slate-500">유형 기반 예상 사고 원인</p>
                    <p className="text-lg font-semibold leading-7">
                      {result.expected_cause}
                    </p>
                  </div>
                </div>

                <div className={`rounded-2xl border p-4 mb-6 ${confidenceStyle}`}>
                  <p className="text-sm font-semibold">신뢰도 상태 안내</p>
                  <p className="text-sm mt-1">{result.confidence_message}</p>
                </div>

                <div className="grid lg:grid-cols-2 gap-6">
                  <div className="p-5 bg-slate-50 rounded-2xl border">
                    <h4 className="text-xl font-semibold mb-3">판단 근거 설명</h4>
                    <p className="text-slate-700 leading-7">
                      {result.explanation}
                    </p>
                  </div>

                  <div className="p-5 bg-slate-50 rounded-2xl border">
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="text-xl font-semibold">Grad-CAM 시각화</h4>
                      <span className="text-xs px-3 py-1 rounded-full bg-slate-200 text-slate-500">
                        추후 추가 예정
                      </span>
                    </div>

                    <div className="h-[260px] rounded-xl border-2 border-dashed border-slate-300 bg-white flex items-center justify-center">
                      <div className="text-center px-6">
                        <p className="text-slate-500 font-medium">
                          모델이 주목한 영역을 시각화할 공간입니다.
                        </p>
                        <p className="text-sm text-slate-400 mt-2">
                          추후 Grad-CAM 결과 이미지를 이 영역에 표시할 수 있습니다.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </section>
          )}

          {result && (
            <section className="max-w-7xl mx-auto px-6 pb-12">
              <div className={layout.card}>
                <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 mb-4">
                  <h3 className="text-2xl font-bold">분석 리포트</h3>

                  <button
                    onClick={handleCopyReport}
                    className="rounded-xl bg-blue-600 text-white px-4 py-2 text-sm font-medium hover:bg-blue-700 transition"
                  >
                    리포트 복사
                  </button>
                </div>

                <div className="rounded-2xl border bg-slate-50 p-5">
                  <pre className="whitespace-pre-wrap break-keep text-slate-700 leading-7 font-sans">
                    {reportText}
                  </pre>
                </div>
              </div>
            </section>
          )}
        </main>
      </div>
    </div>
  );
}
