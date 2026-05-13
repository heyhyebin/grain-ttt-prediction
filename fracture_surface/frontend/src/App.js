import { useState, useRef, useEffect } from "react";
import FlipCard from "./components/FlipCard";

const FRACTURE_TYPES = ["취성 파괴", "연성 파괴", "피로 파괴", "입계 파괴"];

const DEFAULT_SIMILARITIES = {
  "취성 파괴": { sim: "—", best: false, mixed: false },
  "연성 파괴": { sim: "—", best: false, mixed: false },
  "피로 파괴": { sim: "—", best: false, mixed: false },
  "입계 파괴": { sim: "—", best: false, mixed: false },
};

const FRACTURE_IMAGES = {
  "취성 파괴": "/images/cleavage.jpg",
  "연성 파괴": "/images/ductile.jpg",
  "피로 파괴": "/images/fatigue.jpg",
  "입계 파괴": "/images/Intergranular.jpg",
};

const MATERIAL_LABELS = {
  steel: "강 (Steel)",
  stainless_steel: "스테인리스강",
  aluminum: "알루미늄",
  titanium: "티타늄",
  cast_iron: "주철",
  copper: "구리",
  magnesium: "마그네슘 합금",
  nickel_alloy: "니켈 합금",
  tool_steel: "공구강",
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
};

export default function App() {
  const fileRef = useRef(null);

  const [previewUrl, setPreviewUrl] = useState(null);
  const [imageBase64, setImageBase64] = useState(null);
  const [thumbnailBase64, setThumbnailBase64] = useState(null);
  const [uploading, setUploading] = useState(false);

  const [material, setMaterial] = useState("");
  const [result, setResult] = useState(null);
  const [similarities, setSimilarities] = useState(DEFAULT_SIMILARITIES);
  const [history, setHistory] = useState([]);

  const [showGradcamModal, setShowGradcamModal] = useState(false);

  useEffect(() => {
    try {
      const saved = localStorage.getItem("analysisHistory");
      if (saved) {
        setHistory(JSON.parse(saved));
      }
    } catch (err) {
      console.error("기록 불러오기 실패:", err);
      localStorage.removeItem("analysisHistory");
    }
  }, []);

  const materialText =
    MATERIAL_LABELS[result?.material] || result?.material || "-";

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

  const makeThumbnail = (file, maxSize = 180, quality = 0.65) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      const img = new Image();

      reader.onload = () => {
        img.onload = () => {
          const canvas = document.createElement("canvas");

          const scale = Math.min(maxSize / img.width, maxSize / img.height);
          canvas.width = Math.round(img.width * scale);
          canvas.height = Math.round(img.height * scale);

          const ctx = canvas.getContext("2d");
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

          resolve(canvas.toDataURL("image/jpeg", quality));
        };

        img.onerror = reject;
        img.src = reader.result;
      };

      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    try {
      const base64 = await fileToBase64(file);
      const thumbnail = await makeThumbnail(file);

      setPreviewUrl(base64);
      setImageBase64(base64);
      setThumbnailBase64(thumbnail);
      setResult(null);
      setSimilarities(DEFAULT_SIMILARITIES);
      setShowGradcamModal(false);
    } catch (err) {
      console.error("이미지 처리 실패:", err);
      alert("이미지를 불러오는 중 오류가 발생했습니다.");
    }
  };

  const updateSimilarities = (data) => {
    const mapped = {};
    const highlightedTypes = data.highlighted_types || [data.prediction];

    FRACTURE_TYPES.forEach((type) => {
      mapped[type] = {
        sim: data.similarities?.[type] ?? "—",
        best: type === data.prediction,
        mixed: highlightedTypes.includes(type),
      };
    });

    setSimilarities(mapped);
  };

  const saveHistory = (data, thumbnail) => {
    const historyResult = {
      ...data,
      gradcam_image: null,
    };

    const newItem = {
      id: Date.now(),
      time: new Date().toLocaleString(),
      image: thumbnail,
      result: historyResult,
    };

    const updatedHistory = [newItem, ...history].slice(0, 10);

    setHistory(updatedHistory);

    try {
      localStorage.setItem("analysisHistory", JSON.stringify(updatedHistory));
    } catch (err) {
      console.error("기록 저장 실패:", err);

      const lighterHistory = [newItem, ...history]
        .slice(0, 5)
        .map((item) => ({
          ...item,
          result: {
            ...item.result,
            gradcam_image: null,
          },
        }));

      setHistory(lighterHistory);
      localStorage.setItem("analysisHistory", JSON.stringify(lighterHistory));

      alert("이미지 용량이 커서 최근 5개 기록만 저장했습니다.");
    }
  };

  const handleHistoryClick = (item) => {
    setResult(item.result);
    setPreviewUrl(item.image);
    setImageBase64(item.image);
    setThumbnailBase64(item.image);
    setMaterial(item.result.material || "");
    updateSimilarities(item.result);
    setShowGradcamModal(false);
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem("analysisHistory");
  };

  const handleUpload = async () => {
    const file = fileRef.current?.files[0];

    if (!file) return alert("이미지를 먼저 선택해주세요.");
    if (!material) return alert("재질을 선택해주세요.");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("material", material);

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

      console.log("백엔드 응답:", data);

      setResult(data);
      updateSimilarities(data);
      saveHistory(data, thumbnailBase64);
    } catch (err) {
      console.error("분석 결과 처리 오류:", err);
      alert("분석 결과 처리 중 오류가 발생했습니다. 콘솔을 확인해주세요.");
    } finally {
      setUploading(false);
    }
  };

  const reportText = result
    ? `
[분석 리포트]

1. 입력 조건
- 재질: ${materialText}

2. 분석 결과
- 예측된 파손 유형: ${result.display_prediction || result.prediction}
- 신뢰도: ${result.confidence}
- 혼합 여부: ${result.is_mixed ? "혼합 가능성 있음" : "단일 유형 가능성 높음"}
- 신뢰도 상태: ${result.confidence_message}

3. 주요 특징
- ${result.feature}

4. 예상 원인
- ${result.expected_cause}

5. 종합 설명
- ${result.explanation}

6. 해석 의견
- 본 결과는 업로드된 파손단면 이미지와 사용자가 선택한 재질 정보를 바탕으로 생성된 분석 결과입니다.
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
              <p className="text-sm text-slate-400">
                아직 분석 기록이 없습니다.
              </p>
            )}

            {history.map((item) => {
              const itemResult = item.result;
              const itemMaterial =
                MATERIAL_LABELS[itemResult.material] ||
                itemResult.material ||
                "-";

              return (
                <button
                  key={item.id}
                  onClick={() => handleHistoryClick(item)}
                  className="w-full text-left p-3 bg-slate-50 rounded-xl border hover:border-blue-300 hover:bg-blue-50 transition"
                >
                  <div className="flex gap-3">
                    <div className="w-14 h-14 rounded-lg bg-slate-200 overflow-hidden shrink-0">
                      {item.image ? (
                        <img
                          src={item.image}
                          alt="기록 이미지"
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center text-[10px] text-slate-400">
                          No Image
                        </div>
                      )}
                    </div>

                    <div className="min-w-0">
                      <p className="font-bold text-sm">
                        {itemResult.display_prediction || itemResult.prediction}
                      </p>
                      <p className="text-sm text-blue-600 font-semibold">
                        {itemResult.confidence}
                      </p>
                      <p className="text-xs text-slate-500 truncate">
                        {itemMaterial}
                      </p>
                      <p className="text-xs text-slate-400 mt-1">
                        {item.time}
                      </p>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </aside>

        <main className="flex-1">
          <header className="border-b border-slate-200 bg-white/90 backdrop-blur sticky top-0 z-20">
            <div className={`${layout.container} py-4`}>
              <p className="text-sm font-semibold tracking-[0.2em] text-blue-600 uppercase">
                Failure Analysis System
              </p>
              <h1 className="text-2xl font-bold">
                파손단면 이미지 분석 웹 시스템
              </h1>
            </div>
          </header>

          <section className={layout.section}>
            <div className="bg-white rounded-[28px] shadow border p-8">
              <h2 className="text-3xl font-bold mb-6 text-center">
                이미지 업로드
              </h2>

              <div className="grid lg:grid-cols-[1fr_280px] gap-6 items-stretch">
                <label
                  htmlFor="file-input"
                  className="min-h-[220px] flex items-center justify-center border-2 border-dashed border-slate-300 rounded-2xl p-10 cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition"
                >
                  {previewUrl ? (
                    <img
                      src={previewUrl}
                      alt="미리보기"
                      className="mx-auto max-h-72 object-contain rounded-xl"
                    />
                  ) : (
                    <div className="text-center">
                      <p className="text-slate-500 mb-4">
                        파손단면 이미지를 업로드하세요
                      </p>
                      <span className="inline-block px-5 py-3 bg-slate-900 text-white rounded-xl text-sm">
                        파일 선택
                      </span>
                    </div>
                  )}
                </label>

                <div className="bg-slate-50 border rounded-2xl p-5 flex flex-col justify-center gap-4">
                  <div className="text-left">
                    <p className="text-sm font-semibold text-slate-500 mb-2">
                      분석 조건
                    </p>

                    <select
                      value={material}
                      onChange={(e) => setMaterial(e.target.value)}
                      className={`${layout.select} w-full`}
                    >
                      <option value="">재질 선택</option>
                      <option value="steel">강 (Steel)</option>
                      <option value="stainless_steel">스테인리스강</option>
                      <option value="aluminum">알루미늄</option>
                      <option value="titanium">티타늄</option>
                      <option value="cast_iron">주철</option>
                      <option value="copper">구리</option>
                      <option value="magnesium">마그네슘 합금</option>
                      <option value="nickel_alloy">니켈 합금</option>
                      <option value="tool_steel">공구강</option>
                      <option value="unknown">모름</option>
                    </select>
                  </div>

                  <button
                    onClick={handleUpload}
                    disabled={uploading}
                    className="w-full rounded-xl bg-slate-900 text-white py-3 text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:bg-slate-800 transition"
                  >
                    {uploading ? "분석 중…" : "분석 시작"}
                  </button>

                  <p className="text-xs text-slate-400 leading-5">
                    재질 정보는 파손 유형 설명을 보조하는 참고 정보로 사용됩니다.
                  </p>
                </div>
              </div>

              <input
                id="file-input"
                type="file"
                accept="image/*"
                ref={fileRef}
                onChange={handleFileChange}
                className="hidden"
              />

              {previewUrl && (
                <p className="mt-3 text-sm text-slate-400 text-center">
                  다른 이미지를 선택하려면 이미지 영역을 다시 클릭하세요.
                </p>
              )}
            </div>
          </section>

          <section className={layout.section}>
            <div className="mb-6">
              <h2 className="text-3xl font-bold">유사도 기반 파손 유형 비교</h2>
              <p className="text-slate-500 mt-2">
                혼합 가능성이 있으면 새 카드를 추가하지 않고 상위 두 유형 카드가
                함께 강조됩니다.
              </p>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 items-start">
              {FRACTURE_TYPES.map((type) => (
                <FlipCard
                  key={type}
                  type={type}
                  similarity={similarities[type].sim}
                  isBest={similarities[type].best}
                  isMixed={similarities[type].mixed}
                  mixedMode={result?.is_mixed}
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

                <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6">
                  <div className={layout.resultBox}>
                    <p className="text-sm text-slate-500">파손 유형</p>
                    <p className="text-lg font-bold">
                      {result.display_prediction || result.prediction}
                    </p>
                  </div>

                  <div className={layout.resultBox}>
                    <p className="text-sm text-slate-500">신뢰도</p>
                    <p className="text-lg font-bold">{result.confidence}</p>
                  </div>

                  <div className={layout.resultBox}>
                    <p className="text-sm text-slate-500">재질</p>
                    <p className="text-lg font-bold">{materialText}</p>
                  </div>
                </div>

                {result.is_mixed && (
                  <div className="rounded-2xl border border-blue-200 bg-blue-50 p-4 mb-6 text-blue-700">
                    <p className="text-sm font-semibold">혼합 파손 가능성</p>
                    <p className="text-sm mt-1">
                      {result.top1_type}와 {result.top2_type}의 확률 차이가{" "}
                      {result.mixed_gap}로 작아 두 유형을 함께 확인하는 것이
                      좋습니다.
                    </p>
                  </div>
                )}

                <div className="grid md:grid-cols-2 gap-4 mb-6">
                  <div className={layout.resultBox}>
                    <p className="text-sm text-slate-500">주요 특징</p>
                    <p className="text-lg font-semibold leading-7">
                      {result.feature}
                    </p>
                  </div>

                  <div className={layout.resultBox}>
                    <p className="text-sm text-slate-500">
                      유형 기반 예상 사고 원인
                    </p>
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
                    <h4 className="text-xl font-semibold mb-3">
                      판단 근거 설명
                    </h4>
                    <p className="text-slate-700 leading-7">
                      {result.explanation}
                    </p>
                  </div>

                  <div className="p-5 bg-slate-50 rounded-2xl border">
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="text-xl font-semibold">
                        Grad-CAM++ 영역 시각화
                      </h4>

                      {result.gradcam_image && (
                        <button
                          onClick={() => setShowGradcamModal(true)}
                          className="text-xs px-3 py-1 rounded-full bg-slate-900 text-white hover:bg-slate-700 transition"
                        >
                          크게 보기
                        </button>
                      )}
                    </div>

                    <button
                      type="button"
                      onClick={() =>
                        result.gradcam_image && setShowGradcamModal(true)
                      }
                      className="w-full h-[260px] rounded-xl border bg-white flex items-center justify-center overflow-hidden hover:border-blue-300 transition"
                    >
                      {result.gradcam_image ? (
                        <img
                          src={result.gradcam_image}
                          alt="Grad-CAM++"
                          className="w-full h-full object-contain"
                        />
                      ) : (
                        <div className="text-center px-6">
                          <p className="text-slate-500 font-medium">
                            Grad-CAM++ 결과가 없습니다.
                          </p>
                        </div>
                      )}
                    </button>
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

      {showGradcamModal && result?.gradcam_image && (
        <div className="fixed inset-0 z-50 bg-black/70 flex items-center justify-center p-6">
          <div className="bg-white rounded-3xl max-w-5xl w-full p-5 shadow-2xl">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold">Grad-CAM++ 확대 보기</h3>

              <button
                onClick={() => setShowGradcamModal(false)}
                className="px-4 py-2 rounded-xl bg-slate-900 text-white text-sm hover:bg-slate-700 transition"
              >
                닫기
              </button>
            </div>

            <div className="bg-slate-100 rounded-2xl p-4">
              <img
                src={result.gradcam_image}
                alt="Grad-CAM++ 확대"
                className="w-full max-h-[75vh] object-contain rounded-xl"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
