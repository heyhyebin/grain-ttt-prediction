import { useState, useRef, useEffect } from "react";
import FlipCard from "./components/FlipCard";

const FRACTURE_TYPES = ["취성 파괴", "연성 파괴", "피로 파괴", "입계 파괴"];

const EN_NAMES = {
  "취성 파괴": "Cleavage",
  "연성 파괴": "Ductile",
  "피로 파괴": "Fatigue",
  "입계 파괴": "Intergranular",
};

const CLASS_COLORS = {
  Cleavage: "#f59b3b",
  Ductile: "#22c55e",
  Fatigue: "#15ccfa",
  Intergranular: "#4444ef",
};

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

function GradcamView({ result, chipSize = "text-xs", canvasClass = "h-[260px]" }) {
  const canvasRef = useRef(null);
  const baseImgRef = useRef(null);
  const layerImgsRef = useRef({});

  const allClasses = Object.keys(result.gradcam_layers || {});
  const activeClasses = allClasses.filter((name) => {
    const contours = result.gradcam_contours?.[name];
    return contours && contours.length > 0;
  });

  const [checked, setChecked] = useState(() =>
    Object.fromEntries(allClasses.map((name) => [name, true]))
  );

  useEffect(() => {
    const loadImg = (src) =>
      new Promise((resolve) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.src = src;
      });

    const draw = async () => {
      if (!result.base_image) return;

      baseImgRef.current = await loadImg(result.base_image);
      layerImgsRef.current = {};

      for (const name of allClasses) {
        const src = result.gradcam_layers?.[name];
        if (src) {
          layerImgsRef.current[name] = await loadImg(src);
        }
      }

      const canvas = canvasRef.current;
      const base = baseImgRef.current;

      if (!canvas || !base) return;

      canvas.width = base.naturalWidth;
      canvas.height = base.naturalHeight;

      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(base, 0, 0);

      for (const name of allClasses) {
        const layerImg = layerImgsRef.current[name];
        if (layerImg) {
          ctx.drawImage(layerImg, 0, 0);
        }
      }
    };

    draw();
  }, [result]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const base = baseImgRef.current;

    if (!canvas || !base || base.naturalWidth === 0) return;

    canvas.width = base.naturalWidth;
    canvas.height = base.naturalHeight;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(base, 0, 0);

    for (const name of allClasses) {
      if (!checked[name]) continue;

      const layerImg = layerImgsRef.current[name];
      if (layerImg) {
        ctx.drawImage(layerImg, 0, 0);
      }
    }
  }, [checked, allClasses]);

  const toggle = (name) => {
    setChecked((prev) => ({
      ...prev,
      [name]: !prev[name],
    }));
  };

  return (
    <div>
      {activeClasses.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-3">
          {activeClasses.map((name) => {
            const koName = Object.keys(EN_NAMES).find(
              (key) => EN_NAMES[key] === name
            );

            const color = CLASS_COLORS[name];
            const on = checked[name];

            return (
              <button
                key={name}
                onClick={() => toggle(name)}
                style={{
                  borderColor: color,
                  backgroundColor: on ? color : "transparent",
                  color: on ? "#fff" : color,
                }}
                className={`px-3 py-1 rounded-full font-semibold border-2 transition ${chipSize}`}
              >
                {koName || name}
              </button>
            );
          })}
        </div>
      )}

      <div
        className={`w-full ${canvasClass} rounded-xl border bg-white overflow-hidden flex items-center justify-center`}
      >
        <canvas
          ref={canvasRef}
          className="w-full h-full object-contain"
          style={{ display: "block" }}
        />
      </div>
    </div>
  );
}

function GradcamModal({ result, onClose }) {
  return (
    <div className="fixed inset-0 z-50 bg-black/70 flex items-center justify-center p-6">
      <div className="bg-white rounded-3xl max-w-5xl w-full p-5 shadow-2xl">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-bold">Grad-CAM++ 확대 보기</h3>

          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl bg-slate-900 text-white text-sm hover:bg-slate-700 transition"
          >
            닫기
          </button>
        </div>

        <GradcamView
          result={result}
          chipSize="text-sm"
          canvasClass="max-h-[70vh]"
        />
      </div>
    </div>
  );
}

export default function App() {
  const fileRef = useRef(null);

  const [previewUrl, setPreviewUrl] = useState(null);
  const [thumbnailBase64, setThumbnailBase64] = useState(null);
  const [uploading, setUploading] = useState(false);

  const [material, setMaterial] = useState("");
  const [result, setResult] = useState(null);
  const [similarities, setSimilarities] = useState(DEFAULT_SIMILARITIES);
  const [history, setHistory] = useState([]);

  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [selectedCompareIds, setSelectedCompareIds] = useState([]);
  const [showCompareModal, setShowCompareModal] = useState(false);

  const [compareSummary, setCompareSummary] = useState(null);
  const [compareLoading, setCompareLoading] = useState(false);

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

  const compareItems = history.filter((item) =>
    selectedCompareIds.includes(item.id)
  );

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
      gradcam_layers: null,
      base_image: null,
      gradcam_contours: null,
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

      const lighterHistory = [newItem, ...history].slice(0, 5);
      setHistory(lighterHistory);
      localStorage.setItem("analysisHistory", JSON.stringify(lighterHistory));

      alert("이미지 용량이 커서 최근 5개 기록만 저장했습니다.");
    }
  };

  const handleHistoryClick = (item) => {
    setResult(item.result);
    setPreviewUrl(item.image);
    setThumbnailBase64(item.image);
    setMaterial(item.result.material || "");
    updateSimilarities(item.result);
    setShowGradcamModal(false);
  };

  const toggleCompareSelect = (id) => {
    setSelectedCompareIds((prev) => {
      if (prev.includes(id)) {
        return prev.filter((itemId) => itemId !== id);
      }

      if (prev.length >= 3) {
        alert("비교는 최대 3개까지 선택할 수 있습니다.");
        return prev;
      }

      return [...prev, id];
    });
  };

  const clearHistory = () => {
    setHistory([]);
    setSelectedCompareIds([]);
    setShowCompareModal(false);
    setCompareSummary(null);
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

  const handleCompareWithLLM = async () => {
    if (compareItems.length < 2) {
      alert("비교할 기록을 2개 이상 선택해주세요.");
      return;
    }

    try {
      setCompareLoading(true);
      setCompareSummary(null);

      const payload = {
        items: compareItems.map((item) => item.result),
      };

      const res = await fetch("http://localhost:8000/compare", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error(`비교 분석 오류: ${res.status}`);
      }

      const data = await res.json();
      setCompareSummary(data);
    } catch (err) {
      console.error("LLM 비교 설명 오류:", err);
      alert("LLM 비교 설명 생성 중 오류가 발생했습니다.");
    } finally {
      setCompareLoading(false);
    }
  };

  return (
    <div className={layout.page}>
      <div className="flex min-h-screen">
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className={`fixed top-5 z-50 rounded-r-xl bg-slate-900 text-white px-3 py-3 text-sm shadow-lg transition-all duration-300 ${
            sidebarOpen ? "left-80" : "left-0"
          }`}
        >
          {sidebarOpen ? "‹" : "›"}
        </button>

        <aside
          className={`fixed lg:sticky top-0 left-0 z-40 h-screen bg-white border-r border-slate-200 transition-all duration-300 overflow-y-auto overflow-x-hidden ${
            sidebarOpen
              ? "w-80 translate-x-0 p-5"
              : "w-0 -translate-x-full p-0 border-none"
          }`}
        >
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-bold whitespace-nowrap">분석 기록</h2>

            {history.length > 0 && (
              <button
                onClick={clearHistory}
                className="text-xs text-slate-400 hover:text-red-500 whitespace-nowrap"
              >
                전체 삭제
              </button>
            )}
          </div>

          {history.length > 0 && (
            <div className="mb-4 p-3 rounded-xl bg-blue-50 border border-blue-100">
              <p className="text-xs text-blue-700 mb-2">
                비교할 기록을 2개 선택하세요.
              </p>

              <button
                onClick={() => {
                  setCompareSummary(null);
                  setShowCompareModal(true);
                }}
                disabled={selectedCompareIds.length < 2}
                className="w-full rounded-lg bg-blue-600 text-white py-2 text-xs font-semibold disabled:opacity-40 disabled:cursor-not-allowed hover:bg-blue-700 transition"
              >
                비교하기 ({selectedCompareIds.length})
              </button>
            </div>
          )}

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
              const checked = selectedCompareIds.includes(item.id);

              return (
                <div
                  key={item.id}
                  onClick={() => handleHistoryClick(item)}
                  className={`relative w-full text-left p-3 rounded-xl border transition cursor-pointer ${
                    checked
                      ? "bg-blue-50 border-blue-400"
                      : "bg-slate-50 hover:border-blue-300 hover:bg-blue-50"
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onClick={(e) => e.stopPropagation()}
                    onChange={() => toggleCompareSelect(item.id)}
                    className="absolute top-3 right-3 w-4 h-4 accent-blue-600 cursor-pointer"
                  />

                  <div className="flex gap-3 pr-6">
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
                </div>
              );
            })}
          </div>
        </aside>

        <main className="flex-1 min-w-0">
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
              <h2 className="text-3xl font-bold">
                유사도 기반 파손 유형 비교
              </h2>

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

                      {(result.gradcam_layers || result.gradcam_image) && (
                        <button
                          onClick={() => setShowGradcamModal(true)}
                          className="text-xs px-3 py-1 rounded-full bg-slate-900 text-white hover:bg-slate-700 transition"
                        >
                          크게 보기
                        </button>
                      )}
                    </div>

                    {result.gradcam_layers && result.base_image ? (
                      <GradcamView
                        result={result}
                        chipSize="text-xs"
                        canvasClass="h-[260px]"
                      />
                    ) : result.gradcam_image ? (
                      <button
                        type="button"
                        onClick={() => setShowGradcamModal(true)}
                        className="w-full h-[260px] rounded-xl border bg-white flex items-center justify-center overflow-hidden hover:border-blue-300 transition"
                      >
                        <img
                          src={result.gradcam_image}
                          alt="Grad-CAM++"
                          className="w-full h-full object-contain"
                        />
                      </button>
                    ) : (
                      <div className="w-full h-[260px] rounded-xl border bg-white flex items-center justify-center">
                        <p className="text-slate-500 font-medium">
                          Grad-CAM++ 결과가 없습니다.
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </section>
          )}
        </main>
      </div>

      {showCompareModal && (
        <div className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center p-6">
          <div className="bg-white rounded-3xl max-w-[70vw] w-full max-h-[90vh] overflow-y-auto p-8 shadow-2xl">
            <div className="flex items-start justify-between gap-4 mb-5">
              <div>
                <h3 className="text-2xl font-bold">분석 결과 비교</h3>
                <p className="text-sm text-slate-500 mt-1">
                  선택한 분석 기록의 이미지, 예측 유형, 신뢰도, 재질, 설명을
                  비교합니다.
                </p>

                <button
                  onClick={handleCompareWithLLM}
                  disabled={compareItems.length < 2 || compareLoading}
                  className="mt-4 rounded-xl bg-blue-600 text-white px-4 py-2 text-sm font-semibold disabled:opacity-40 disabled:cursor-not-allowed hover:bg-blue-700 transition"
                >
                  {compareLoading
                    ? "비교 설명 생성 중..."
                    : "LLM으로 비교 설명 생성"}
                </button>
              </div>

              <button
                onClick={() => setShowCompareModal(false)}
                className="px-4 py-2 rounded-xl bg-slate-900 text-white text-sm hover:bg-slate-700 transition"
              >
                닫기
              </button>
            </div>

            {compareSummary && (
              <div className="mb-6 space-y-5">
                <div className="rounded-3xl border border-blue-200 bg-blue-50 p-6 text-blue-900">
                  <p className="text-sm font-bold mb-2">핵심 요약</p>

                  <p className="text-2xl font-bold leading-9">
                    {compareSummary.summary}
                  </p>

                  <p className="text-sm leading-6 mt-4 text-blue-700">
                    {compareSummary.final_opinion}
                  </p>
                </div>

                <div className="grid lg:grid-cols-2 gap-5">
                  <div className="rounded-3xl border border-purple-200 bg-purple-50 p-5 text-purple-900">
                    <p className="text-lg font-bold mb-4">파손 메커니즘 차이</p>

                    <div className="grid grid-cols-[1fr_auto_1fr] gap-4 items-stretch">
                      <div className="bg-white/80 rounded-2xl p-4 border border-purple-100">
                        <p className="text-sm font-bold text-purple-700 mb-2">
                          비교 1
                        </p>

                        <p className="text-sm leading-7">
                          {compareSummary.mechanism_compare_1}
                        </p>
                      </div>

                      <div className="flex items-center justify-center">
                        <span className="rounded-full bg-purple-500 text-white text-xs font-bold px-4 py-3">
                          VS
                        </span>
                      </div>

                      <div className="bg-white/80 rounded-2xl p-4 border border-purple-100">
                        <p className="text-sm font-bold text-blue-700 mb-2">
                          비교 2
                        </p>

                        <p className="text-sm leading-7">
                          {compareSummary.mechanism_compare_2}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="rounded-3xl border border-amber-200 bg-amber-50 p-5 text-amber-900">
                    <p className="text-lg font-bold mb-4">
                      신뢰도 및 해석 주의점
                    </p>

                    <div className="grid grid-cols-[1fr_auto_1fr] gap-4 items-center mb-4">
                      <div>
                        <p className="text-sm font-bold mb-2">비교 1</p>

                        <div className="h-4 rounded-full bg-white overflow-hidden">
                          <div
                            className="h-full rounded-full bg-purple-500"
                            style={{
                              width:
                                compareItems[0]?.result?.confidence || "0%",
                            }}
                          />
                        </div>

                        <p className="text-lg font-bold mt-3">
                          {compareItems[0]?.result?.confidence}
                        </p>
                      </div>

                      <span className="rounded-full bg-white border border-amber-200 px-4 py-3 text-xs font-bold">
                        VS
                      </span>

                      <div>
                        <p className="text-sm font-bold mb-2">비교 2</p>

                        <div className="h-4 rounded-full bg-white overflow-hidden">
                          <div
                            className="h-full rounded-full bg-blue-500"
                            style={{
                              width:
                                compareItems[1]?.result?.confidence || "0%",
                            }}
                          />
                        </div>

                        <p className="text-lg font-bold mt-3">
                          {compareItems[1]?.result?.confidence}
                        </p>
                      </div>
                    </div>

                    <div className="rounded-2xl bg-white/80 border border-amber-100 p-5">
                      <p className="text-sm leading-7">
                        {compareSummary.confidence_opinion}
                      </p>
                    </div>
                  </div>
                </div>

                <p className="text-xs text-slate-400">
                  본 비교 분석은 이미지와 입력 정보를 바탕으로 한 AI 추정
                  결과입니다. 실제 판정에는 추가 실험 및 전문가 검토가 필요할 수
                  있습니다.
                </p>
              </div>
            )}

            {compareItems.length < 2 ? (
              <div className="rounded-2xl bg-slate-50 border p-8 text-center text-slate-500">
                비교할 기록을 2개 이상 선택해주세요.
              </div>
            ) : (
              <div
                className={`grid gap-4 ${
                  compareItems.length === 2 ? "md:grid-cols-2" : "md:grid-cols-3"
                }`}
              >
                {compareItems.map((item, index) => {
                  const itemResult = item.result;
                  const itemMaterial =
                    MATERIAL_LABELS[itemResult.material] ||
                    itemResult.material ||
                    "-";

                  return (
                    <div
                      key={item.id}
                      className="rounded-2xl border bg-slate-50 p-4"
                    >
                      <div className="flex items-center justify-between mb-3">
                        <p className="text-sm font-bold text-blue-600">
                          비교 {index + 1}
                        </p>
                        <p className="text-xs text-slate-400">{item.time}</p>
                      </div>

                      <div className="h-40 bg-white rounded-xl border overflow-hidden mb-4 flex items-center justify-center">
                        {item.image ? (
                          <img
                            src={item.image}
                            alt="비교 이미지"
                            className="w-full h-full object-contain"
                          />
                        ) : (
                          <p className="text-sm text-slate-400">No Image</p>
                        )}
                      </div>

                      <div className="space-y-3">
                        <div className="bg-white rounded-xl border p-3">
                          <p className="text-xs text-slate-500">파손 유형</p>
                          <p className="font-bold">
                            {itemResult.display_prediction ||
                              itemResult.prediction}
                          </p>
                        </div>

                        <div className="bg-white rounded-xl border p-3">
                          <p className="text-xs text-slate-500">신뢰도</p>
                          <p className="font-bold text-blue-600">
                            {itemResult.confidence}
                          </p>
                        </div>

                        <div className="bg-white rounded-xl border p-3">
                          <p className="text-xs text-slate-500">재질</p>
                          <p className="font-bold">{itemMaterial}</p>
                        </div>

                        <div className="bg-white rounded-xl border p-3">
                          <p className="text-xs text-slate-500">혼합 여부</p>
                          <p className="font-bold">
                            {itemResult.is_mixed
                              ? "혼합 가능성 있음"
                              : "단일 유형 가능성 높음"}
                          </p>
                        </div>

                        <div className="bg-white rounded-xl border p-3">
                          <p className="text-xs text-slate-500">주요 특징</p>
                          <p className="text-sm leading-6">
                            {itemResult.feature || "-"}
                          </p>
                        </div>

                        <div className="bg-white rounded-xl border p-3">
                          <p className="text-xs text-slate-500">예상 원인</p>
                          <p className="text-sm leading-6">
                            {itemResult.expected_cause || "-"}
                          </p>
                        </div>

                        <div className="bg-white rounded-xl border p-3">
                          <p className="text-xs text-slate-500">설명</p>
                          <p className="text-sm leading-6">
                            {itemResult.explanation || "-"}
                          </p>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}

      {showGradcamModal &&
        result &&
        (result.gradcam_layers && result.base_image ? (
          <GradcamModal
            result={result}
            onClose={() => setShowGradcamModal(false)}
          />
        ) : (
          result.gradcam_image && (
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
          )
        ))}
    </div>
  );
}
