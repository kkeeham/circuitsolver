"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import "katex/dist/katex.min.css";
import { InlineMath, BlockMath } from "react-katex";

// ─── 백엔드 API 응답 타입 (해석 JSON만 — SVG/회로 이미지 필드 없음) ─────────
interface StepData {
  title: string;
  description: string;
  applied_theory_ids?: string[];
}

interface TheoryData {
  theory_id?: string;
  chapter_name?: string;
  description?: string;
  formula_latex?: string;
}

interface SubSolution {
  menu_id: number;
  menu_title: string;
  solution: Record<string, number>;
  steps: StepData[];
  applied_theories: TheoryData[];
}

interface CircuitApiResponse {
  problem_type: "CONVERSION" | "CALCULATION";
  parsed_question_text: string;
  sub_solutions: SubSolution[];
}

interface AccordionStep {
  tag: string;
  title: string;
  formula: string;
  formulaHtml: string;
  detail: string;
}

const API_ENDPOINT = "http://localhost:8000/api/upload-image";

async function uploadCircuitImage(file: File): Promise<CircuitApiResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(API_ENDPOINT, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`서버 에러 (${response.status}): ${errorText}`);
  }

  return response.json();
}

function renderNaturalText(text: string): string {
  // 마크다운 **bold** 제거, 자연어로 변환
  return text
    .replace(/\*\*(.*?)\*\*/g, "$1")   // **bold** → 일반 텍스트
    .replace(/\*(.*?)\*/g, "$1")        // *italic* → 일반 텍스트
    .replace(/`(.*?)`/g, "$1")          // `code` → 일반 텍스트
    .trim();
}

// ─── 혼합 텍스트(한글 + LaTeX) 렌더링용 헬퍼 컴포넌트 ─────────
const LatexFormatter = ({ text }: { text: string }) => {
  if (!text) return null;

  // $$...$$, \[...\], $...$, \(...\) 패턴을 찾아 배열로 쪼갭니다.
  const regex = /(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\$[\s\S]*?\$|\\\([\s\S]*?\\\))/g;
  const parts = text.split(regex);

  return (
    <>
      {parts.map((part, index) => {
        if (part.startsWith("$$") && part.endsWith("$$")) {
          return <BlockMath key={index} math={part.slice(2, -2)} />;
        } else if (part.startsWith("\\[") && part.endsWith("\\]")) {
          return <BlockMath key={index} math={part.slice(2, -2)} />;
        } else if (part.startsWith("$") && part.endsWith("$")) {
          return <InlineMath key={index} math={part.slice(1, -1)} />;
        } else if (part.startsWith("\\(") && part.endsWith("\\)")) {
          return <InlineMath key={index} math={part.slice(2, -2)} />;
        }
        // 일반 텍스트
        return <span key={index}>{part}</span>;
      })}
    </>
  );
};

function normalizeStep(s: StepData, i: number): AccordionStep {
  const cleanDescription = renderNaturalText(s.description ?? "");
  return {
    tag: `Step ${i + 1}`,
    title: renderNaturalText(s.title ?? "제목 없음"),
    formula: "",                  // 수식 칸은 비움 (description을 detail로 이동)
    formulaHtml: "",
    detail: cleanDescription,     // description을 자연어 텍스트로 표시
  };
}

function formatSolutionValue(key: string, value: number): string {
  // 💡 맨 앞의 별(★) 기호나 공백이 있다면 판별할 때만 임시로 제거하고 검사합니다.
  const cleanKey = key.replace(/^★\s*/, "");

  const unit = 
    /^[vV]/.test(cleanKey) || /volt|voltage|_v/i.test(cleanKey) ? "V" :
    /^[iI]/.test(cleanKey) || /current|_i|amp/i.test(cleanKey) ? "A" :
    /power|_p$/i.test(cleanKey) ? "W" :
    /res|ohm|_r$/i.test(cleanKey) ? "Ω" : "V"; // 기본 단위 폴백

  const formatted =
    Math.abs(value) >= 1000 || (Math.abs(value) < 0.01 && value !== 0)
      ? value.toExponential(2)
      : Number(value.toPrecision(4)).toString();
      
  return unit ? `${formatted} ${unit}` : formatted;
}

const FontLink = () => (
  <style>{`
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=JetBrains+Mono:wght@400;600&family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Noto Sans KR', sans-serif; }

    :root {
      --bg: #f5f4f0;
      --surface: #ffffff;
      --border: #e0ddd6;
      --text: #1a1814;
      --muted: #7a7670;
      --accent: #e63946;
      --accent2: #2d6a4f;
      --accent3: #457b9d;
      --mono: 'JetBrains Mono', monospace;
    }

    @keyframes fadeSlideUp {
      from { opacity: 0; transform: translateY(18px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
      from { opacity: 0; } to { opacity: 1; }
    }
    @keyframes pulse-ring {
      0%   { transform: scale(1);   opacity: .5; }
      70%  { transform: scale(1.18); opacity: 0; }
      100% { transform: scale(1.18); opacity: 0; }
    }
    @keyframes pageIn {
      from { opacity: 0; transform: translateX(20px); }
      to   { opacity: 1; transform: translateX(0); }
    }
  `}</style>
);

const ResistorAnimated = () => {
  const [cycleKey, setCycleKey] = useState(0);
  const [phase, setPhase] = useState<"idle" | "scan" | "hold">("idle");
  const [ripples, setRipples] = useState<number[]>([]);
  
  const animRef = useRef<number | null>(null);
  
  // 💡 [수정] 상태(State) 대신 직접 DOM을 조작하기 위한 Ref 연결
  const eraserRef = useRef<SVGRectElement>(null);
  const tracerRef = useRef<SVGRectElement>(null);
  const dotRef = useRef<SVGCircleElement>(null);

  const SCAN_MS = 2000;
  const RIPPLE_MS = 700;
  const IDLE_MS = 1000;
  const GAP_PERCENT = 18;
  const WIDTH = 220;
  const HEIGHT = 60;
  const strokeW = 4;

  const pathPoints: [number, number][] = [
    [0, 30], [34, 30],
    [47, 8], [60, 52], [73, 8], [86, 52], [99, 8], [112, 52], [125, 8], [138, 52], [151, 8], [164, 52], [186, 30],
    [WIDTH, 30],
  ];

  const getYAtX = useCallback((x: number): number => {
    if (x <= 0) return 30;
    if (x >= WIDTH) return 30;
    for (let i = 0; i < pathPoints.length - 1; i++) {
      const [x1, y1] = pathPoints[i];
      const [x2, y2] = pathPoints[i + 1];
      if (x >= x1 && x <= x2) {
        const t = (x - x1) / (x2 - x1);
        return y1 + t * (y2 - y1);
      }
    }
    return 30;
  }, []);

  const gaussianEase = (t: number): number => {
    if (t <= 0) return 0;
    if (t >= 1) return 1;
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  };

  useEffect(() => {
    let startTime: number | null = null;
    let t1: ReturnType<typeof setTimeout>;
    let t2: ReturnType<typeof setTimeout>;
    let t3: ReturnType<typeof setTimeout>;

    const animate = (timestamp: number) => {
      if (!startTime) startTime = timestamp;
      const elapsed = timestamp - startTime;
      const rawProgress = Math.min(elapsed / SCAN_MS, 1);
      const currentProgress = gaussianEase(rawProgress) * 100;

      // 💡 [수정] setProgress(상태 업데이트) 대신 Ref를 통해 브라우저 DOM 속성 직접 변경
      const eraserX = (currentProgress / 100) * WIDTH;
      const tracerX = Math.max(0, ((currentProgress - GAP_PERCENT) / 100) * WIDTH);
      const dotY = getYAtX(eraserX);

      if (eraserRef.current) eraserRef.current.setAttribute("width", String(eraserX));
      if (tracerRef.current) tracerRef.current.setAttribute("width", String(tracerX));
      if (dotRef.current) {
        dotRef.current.setAttribute("cx", String(eraserX));
        dotRef.current.setAttribute("cy", String(dotY));
      }

      if (rawProgress < 1) {
        animRef.current = requestAnimationFrame(animate);
      } else {
        setPhase("hold");
        const rippleId = Date.now();
        setRipples((r) => [...r, rippleId]);
        t1 = setTimeout(() => {
          setRipples((r) => r.filter((id) => id !== rippleId));
        }, RIPPLE_MS);
        t2 = setTimeout(() => {
          setPhase("idle");
          // DOM 속성 수동 초기화
          if (eraserRef.current) eraserRef.current.setAttribute("width", "0");
          if (tracerRef.current) tracerRef.current.setAttribute("width", "0");
          t3 = setTimeout(runCycle, 50);
        }, IDLE_MS);
      }
    };

    const runCycle = () => {
      setCycleKey((k) => k + 1);
      setPhase("scan");
      startTime = null;
      animRef.current = requestAnimationFrame(animate);
    };

    const initTimer = setTimeout(runCycle, 100);
    return () => {
      clearTimeout(initTimer);
      [t1, t2, t3].forEach((t) => t && clearTimeout(t));
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [getYAtX]); // 의존성 배열에 getYAtX 추가

  const pts = "34,30 47,8 60,52 73,8 86,52 99,8 112,52 125,8 138,52 151,8 164,52 186,30";
  const maskId = `eraser-mask-${cycleKey}`;
  const tracerId = `tracer-mask-${cycleKey}`;

  return (
    <div style={{ position: "relative", display: "inline-block" }}>
      <style>{`
        @keyframes rippleExponential-${cycleKey} {
          0% { transform: scale(1); opacity: 0.55; }
          100% { transform: scale(1.35); opacity: 0; }
        }
      `}</style>
      {ripples.map((id) => (
        <svg
          key={id}
          width={WIDTH}
          height={HEIGHT}
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          fill="none"
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            pointerEvents: "none",
            animation: `rippleExponential-${cycleKey} ${RIPPLE_MS}ms cubic-bezier(0.16, 1, 0.3, 1) forwards`,
            transformOrigin: `${(34 + 186) / 2}px 30px`,
          }}
        >
          <polyline points={pts} stroke="#1a1814" strokeWidth={strokeW} strokeLinecap="round" strokeLinejoin="round" fill="none" />
        </svg>
      ))}
      <svg width={WIDTH} height={HEIGHT} viewBox={`0 0 ${WIDTH} ${HEIGHT}`} fill="none" style={{ display: "block" }}>
        <defs>
          <mask id={maskId}>
            <rect x="0" y="0" width={WIDTH} height={HEIGHT} fill="white" />
            {/* 💡 [수정] 너비를 직접 제어하기 위해 ref 연결 및 초기 width="0" 설정 */}
            {phase === "scan" && <rect ref={eraserRef} x="0" y="0" width="0" height={HEIGHT} fill="black" />}
          </mask>
          <mask id={tracerId}>
            <rect x="0" y="0" width={WIDTH} height={HEIGHT} fill="black" />
            {/* 💡 [수정] ref 연결 및 초기 width="0" 설정 */}
            {phase === "scan" && <rect ref={tracerRef} x="0" y="0" width="0" height={HEIGHT} fill="white" />}
            {phase === "hold" && <rect x="0" y="0" width={WIDTH} height={HEIGHT} fill="white" />}
          </mask>
        </defs>
        <g mask={`url(#${maskId})`}>
          <line x1="0" y1="30" x2="34" y2="30" stroke="#1a1814" strokeWidth={strokeW} strokeLinecap="round" />
          <polyline points={pts} stroke="#1a1814" strokeWidth={strokeW} strokeLinecap="round" strokeLinejoin="round" fill="none" />
          <line x1="186" y1="30" x2={WIDTH} y2="30" stroke="#1a1814" strokeWidth={strokeW} strokeLinecap="round" />
        </g>
        <g mask={`url(#${tracerId})`}>
          <line x1="0" y1="30" x2="34" y2="30" stroke="#1a1814" strokeWidth={strokeW} strokeLinecap="round" />
          <polyline points={pts} stroke="#1a1814" strokeWidth={strokeW} strokeLinecap="round" strokeLinejoin="round" fill="none" />
          <line x1="186" y1="30" x2={WIDTH} y2="30" stroke="#1a1814" strokeWidth={strokeW} strokeLinecap="round" />
        </g>
        {phase === "scan" && (
          <circle
            ref={dotRef}
            cx="0" // 💡 [수정] 초기값 0
            cy="30"
            r="6"
            fill="var(--accent, #e63946)"
            style={{ filter: "drop-shadow(0 0 10px var(--accent))" }}
          />
        )}
      </svg>
    </div>
  );
};

const CircuitLogo = () => (
  <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect x="4" y="4" width="40" height="40" rx="8" fill="#1a1814" />
    <line x1="10" y1="24" x2="16" y2="24" stroke="#f5f4f0" strokeWidth="2" strokeLinecap="round" />
    <polyline points="16,24 18,18 20,30 22,18 24,30 26,18 28,30 30,24" stroke="#e63946" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    <line x1="30" y1="24" x2="38" y2="24" stroke="#f5f4f0" strokeWidth="2" strokeLinecap="round" />
    <circle cx="10" cy="24" r="2" fill="#2d6a4f" />
    <circle cx="38" cy="24" r="2" fill="#2d6a4f" />
  </svg>
);

const Navbar = () => (
  <nav
    style={{
      position: "fixed",
      top: 0,
      left: 0,
      right: 0,
      zIndex: 100,
      background: "rgba(245,244,240,0.96)",
      backdropFilter: "blur(12px)",
      borderBottom: "1px solid var(--border)",
      display: "flex",
      alignItems: "center",
      padding: "0 32px",
      height: "56px",
      gap: "8px",
    }}
  >
    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginRight: "auto" }}>
      <CircuitLogo />
    </div>
    {["회로 풀이", "단원별 핵심 요약", "계산기"].map((label) => (
      <div key={label} style={{ position: "relative", padding: "6px 16px", cursor: "pointer" }}>
        <span
          style={{
            fontFamily: "'Noto Sans KR', sans-serif",
            fontSize: "14px",
            fontWeight: label === "회로 풀이" ? 700 : 400,
            color: label === "회로 풀이" ? "var(--text)" : "var(--muted)",
          }}
        >
          {label}
        </span>
        {label === "회로 풀이" && (
          <div
            style={{
              position: "absolute",
              bottom: "0px",
              left: "16px",
              right: "16px",
              height: "2.5px",
              background: "var(--accent)",
              borderRadius: "2px",
            }}
          />
        )}
      </div>
    ))}
  </nav>
);

const Page1 = ({ onFileSelect }: { onFileSelect: (file: File) => void }) => {
  const inputRef = useRef<HTMLInputElement>(null);
  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        background: "var(--bg)",
        paddingTop: "56px",
        animation: "pageIn 0.5s ease",
      }}
    >
      <div style={{ textAlign: "center", marginBottom: "48px", animation: "fadeSlideUp 0.6s ease" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "14px", marginBottom: "16px" }}>
          <CircuitLogo />
          <h1
            style={{
              fontFamily: "'DM Serif Display', serif",
              fontSize: "clamp(28px, 5vw, 44px)",
              color: "var(--text)",
              letterSpacing: "-0.02em",
              fontWeight: 900,
            }}
          >
            Circuit Solver
          </h1>
        </div>
        <p style={{ fontFamily: "'Noto Sans KR', sans-serif", fontSize: "18px", color: "var(--muted)", fontWeight: 700 }}>
          &quot;회로 공부를 더욱 쾌적하게&quot;
        </p>
      </div>

      <div style={{ position: "relative", marginBottom: "32px", animation: "fadeSlideUp 0.7s ease" }}>
        <div
          style={{
            position: "absolute",
            inset: "-16px",
            borderRadius: "50%",
            border: "2px solid var(--accent)",
            animation: "pulse-ring 2s cubic-bezier(0.215,0.61,0.355,1) infinite",
            pointerEvents: "none",
          }}
        />
        <button
          onClick={() => inputRef.current?.click()}
          style={{
            width: "120px",
            height: "120px",
            borderRadius: "50%",
            background: "var(--text)",
            border: "none",
            cursor: "pointer",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: "8px",
            boxShadow: "0 12px 40px rgba(26,24,20,0.18)",
          }}
        >
          <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
            <path d="M6 14h4l3-4h14l3 4h4a2 2 0 0 1 2 2v16a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V16a2 2 0 0 1 2-2z" stroke="#f5f4f0" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            <circle cx="20" cy="22" r="6" stroke="#f5f4f0" strokeWidth="2" />
            <circle cx="20" cy="22" r="2.5" fill="#f5f4f0" />
          </svg>
          <span style={{ fontFamily: "var(--mono)", fontSize: "9px", color: "#f5f4f0", opacity: 0.7, letterSpacing: "0.1em" }}>
            UPLOAD
          </span>
        </button>
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          style={{ display: "none" }}
          onChange={(e) => {
            if (e.target.files?.[0]) onFileSelect(e.target.files[0]);
          }}
        />
      </div>

      <p style={{ fontFamily: "var(--mono)", fontSize: "12px", color: "var(--muted)", animation: "fadeSlideUp 0.8s ease" }}>
        회로 문제 이미지를 업로드하세요
      </p>
    </div>
  );
};

const Page2 = ({ isSuccess, onDone }: { isSuccess: boolean; onDone: () => void }) => {
  const loadingTexts = [
    "회로 문제를 분석하는 중...",
    "수학 연산 엔진에 데이터를 매핑하는 중...",
    "마디 해석법(Nodal Analysis) 행렬식을 빌드하는 중...",
    "수식 연산을 거쳐 풀이를 도출하는 중...",
  ];

  const [loadingText] = useState(() => loadingTexts[Math.floor(Math.random() * loadingTexts.length)]);

  useEffect(() => {
    if (isSuccess) onDone();
  }, [isSuccess, onDone]);

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        background: "var(--bg)",
        paddingTop: "56px",
        animation: "pageIn 0.5s ease",
      }}
    >
      <div style={{ animation: "fadeSlideUp 0.6s ease", textAlign: "center" }}>
        <ResistorAnimated />
        <p style={{ marginTop: "24px", fontFamily: "'Noto Sans KR', sans-serif", fontSize: "14px", color: "var(--muted)", fontWeight: 500 }}>
          {loadingText}
        </p>
      </div>
    </div>
  );
};

const AccordionItem = ({ step, idx }: { step: AccordionStep; idx: number }) => {
  const [open, setOpen] = useState(idx === 0);
  return (
    <div
      style={{
        border: "1px solid var(--border)",
        borderRadius: "10px",
        overflow: "hidden",
        background: "var(--surface)",
        marginBottom: "10px",
        boxShadow: open ? "0 4px 20px rgba(26,24,20,0.07)" : "none",
      }}
    >
      <button
        onClick={() => setOpen((o) => !o)}
        style={{
          width: "100%",
          background: "none",
          border: "none",
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          padding: "16px 20px",
          gap: "12px",
          textAlign: "left",
        }}
      >
        <span
          style={{
            fontFamily: "var(--mono)",
            fontSize: "10px",
            fontWeight: 600,
            background: open ? "var(--text)" : "var(--border)",
            color: open ? "var(--bg)" : "var(--muted)",
            borderRadius: "4px",
            padding: "2px 7px",
          }}
        >
          {step.tag}
        </span>
        <span style={{ fontFamily: "'Noto Sans KR'", fontSize: "13px", fontWeight: 500, color: "var(--text)", flex: 1 }}>
          {step.title}
        </span>
        <span style={{ fontFamily: "var(--mono)", fontSize: "16px", color: "var(--muted)", transform: open ? "rotate(90deg)" : "rotate(0)", display: "inline-block" }}>
          ›
        </span>
      </button>
      <div style={{ overflow: "hidden", maxHeight: open ? "400px" : "0", opacity: open ? 1 : 0, transition: "max-height 0.4s ease, opacity 0.3s ease" }}>
        <div style={{ padding: "0 20px 20px", borderTop: "1px solid var(--border)" }}>
          {step.formula && (
            <div
              style={{
                fontFamily: "'DM Serif Display', Georgia, serif",
                fontSize: "22px",
                textAlign: "center",
                padding: "18px 24px",
                background: "linear-gradient(135deg, #fafaf8 0%, #f0efeb 100%)",
                borderRadius: "10px",
                margin: "14px 0",
                border: "1px solid var(--border)",
              }}
              dangerouslySetInnerHTML={{ __html: step.formulaHtml || step.formula }}
            />
          )}
          {step.detail && (
            <div style={{ fontFamily: "'Noto Sans KR'", fontSize: "13px", color: "var(--muted)", lineHeight: 1.75, whiteSpace: "pre-wrap", overflowX: "auto" }}>
              <LatexFormatter text={step.detail} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const SolutionSummary = ({ solution }: { solution: Record<string, number> }) => {
  const entries = Object.entries(solution);
  if (entries.length === 0) return null;

  return (
    <div
      style={{
        background: "var(--surface)",
        borderRadius: "12px",
        border: "1.5px solid var(--accent2)",
        padding: "20px 24px",
        marginBottom: "32px",
        boxShadow: "0 2px 12px rgba(45,106,79,0.08)",
        animation: "fadeSlideUp 0.55s ease",
      }}
    >
      <p style={{ fontFamily: "var(--mono)", fontSize: "10px", color: "var(--accent2)", letterSpacing: "0.1em", marginBottom: "14px" }}>
        ✦ 해석 결과
      </p>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "12px" }}>
        {entries.map(([key, value]) => (
          <div
            key={key}
            style={{
              flex: "1 1 140px",
              minWidth: "120px",
              padding: "12px 16px",
              background: "linear-gradient(135deg, #fafaf8 0%, #f0efeb 100%)",
              borderRadius: "8px",
              border: "1px solid var(--border)",
            }}
          >
            <p style={{ fontFamily: "var(--mono)", fontSize: "11px", color: "var(--muted)", marginBottom: "4px" }}>{key}</p>
            <p style={{ fontFamily: "'DM Serif Display', serif", fontSize: "20px", color: "var(--text)" }}>
              {formatSolutionValue(key, value)}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
};

const Page3 = ({
  onReset,
  apiData,
  apiError,
  isLoading,
  uploadedImageUrl,
}: {
  onReset: () => void;
  apiData: CircuitApiResponse | null;
  apiError: string | null;
  isLoading: boolean;
  uploadedImageUrl: string | null;
}) => {
  const [showModal, setShowModal] = useState(false);

  if (isLoading) {
    return (
      <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", background: "var(--bg)", paddingTop: "72px" }}>
        <ResistorAnimated />
        <p style={{ marginTop: "24px", fontFamily: "'Noto Sans KR'", color: "var(--muted)" }}>백엔드 서버로부터 데이터를 수신하는 중...</p>
      </div>
    );
  }

  if (apiError) {
    return (
      <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", background: "var(--bg)", paddingTop: "72px", padding: "20px" }}>
        <div style={{ background: "#fee2e2", border: "1px solid #fecaca", borderRadius: "12px", padding: "24px", maxWidth: "500px", textAlign: "center" }}>
          <p style={{ fontFamily: "'Noto Sans KR'", fontSize: "16px", fontWeight: 600, color: "#dc2626", marginBottom: "12px" }}>오류가 발생했습니다</p>
          <p style={{ fontFamily: "var(--mono)", fontSize: "13px", color: "#7f1d1d" }}>{apiError}</p>
          <button
            onClick={onReset}
            style={{ marginTop: "20px", padding: "10px 24px", background: "#dc2626", color: "#fff", border: "none", borderRadius: "8px", fontFamily: "'Noto Sans KR'", fontWeight: 600, cursor: "pointer" }}
          >
            다시 시도
          </button>
        </div>
      </div>
    );
  }

  const questionText = apiData?.parsed_question_text?.replace(/\*\*/g, "").trim() ?? "회로 문제를 분석하고 있습니다...";
  const activeSubSolution = apiData?.sub_solutions?.[0];
  const accordionSteps = (activeSubSolution?.steps ?? []).map(normalizeStep);
  const solution = activeSubSolution?.solution ?? {};
  const appliedTheories = activeSubSolution?.applied_theories ?? [];
  const problemTypeLabel = apiData?.problem_type === "CONVERSION" ? "변환 문제" : apiData?.problem_type === "CALCULATION" ? "계산 문제" : null;

  return (
    <div style={{ minHeight: "100vh", background: "var(--bg)", paddingTop: "72px", paddingBottom: "60px", animation: "pageIn 0.5s ease" }}>
      <div style={{ maxWidth: "720px", margin: "0 auto", padding: "0 20px" }}>

        {/* ── 업로드 이미지 표시 영역 ── */}
        {uploadedImageUrl && (
          <div style={{ marginTop: "24px", marginBottom: "28px", animation: "fadeSlideUp 0.4s ease" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "12px" }}>
              <span style={{ fontFamily: "var(--mono)", fontSize: "10px", color: "var(--muted)", letterSpacing: "0.15em" }}>INPUT</span>
              <div style={{ flex: 1, height: "1px", background: "var(--border)" }} />
            </div>
            <div style={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: "12px",
              overflow: "hidden",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              padding: "16px",
            }}>
              <img
                src={uploadedImageUrl}
                alt="업로드한 회로 이미지"
                style={{
                  maxWidth: "100%",
                  maxHeight: "320px",
                  objectFit: "contain",
                  borderRadius: "6px",
                }}
              />
            </div>
          </div>
        )}
        <div style={{ marginBottom: "28px", marginTop: "24px", animation: "fadeSlideUp 0.5s ease" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "14px" }}>
            <span style={{ fontFamily: "var(--mono)", fontSize: "10px", color: "var(--accent)", letterSpacing: "0.15em" }}>RESULT</span>
            <div style={{ flex: 1, height: "1px", background: "var(--border)" }} />
            {problemTypeLabel && (
              <span style={{ fontFamily: "var(--mono)", fontSize: "10px", color: "var(--muted)", padding: "2px 8px", border: "1px solid var(--border)", borderRadius: "4px" }}>
                {problemTypeLabel}
              </span>
            )}
          </div>
          {activeSubSolution?.menu_title && (
            <p style={{ fontFamily: "var(--mono)", fontSize: "11px", color: "var(--accent3)", marginBottom: "8px", textAlign: "center" }}>
              {activeSubSolution.menu_title}
            </p>
          )}
          <h2 style={{ fontFamily: "'Noto Sans KR', sans-serif", fontSize: "clamp(16px, 3vw, 22px)", color: "var(--text)", lineHeight: 1.4, fontWeight: 800, textAlign: "center" }}>
            {`"${questionText}"`}
          </h2>
        </div>

        <SolutionSummary solution={solution} />

      
        {appliedTheories.length > 0 && (
          <div style={{ marginBottom: "28px", animation: "fadeSlideUp 0.6s ease" }}>
            <p style={{ fontFamily: "var(--mono)", fontSize: "10px", color: "var(--accent3)", letterSpacing: "0.1em", marginBottom: "12px" }}>APPLIED THEORIES</p>
            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
              {appliedTheories.map((t, i) => (
                <div key={t.theory_id ?? i} style={{ padding: "12px 16px", background: "var(--surface)", borderRadius: "8px", border: "1px solid var(--border)" }}>
                  {/* 💡 여기에 (i + 1) 을 추가하여 1. 2. 3. 순서를 명시하고, chapter_name을 출력합니다 */}
                  <p style={{ fontFamily: "'Noto Sans KR'", fontSize: "13px", fontWeight: 600, color: "var(--text)" }}>
                    {i + 1}. {t.chapter_name ?? "이론"}
                  </p>
                  
                  {/* 💡 수식 데이터의 올바른 키(formula_latex)를 매핑합니다 */}
                  {t.formula_latex && (
                    <div style={{ color: "var(--text)", marginTop: "12px", marginBottom: "8px", textAlign: "center", overflowX: "auto" }}>
                      <BlockMath math={t.formula_latex} />
                    </div>
                  )}
                  
                  {t.description && (
                    <div style={{ fontFamily: "'Noto Sans KR'", fontSize: "12px", color: "var(--muted)", marginTop: "4px", lineHeight: 1.6, overflowX: "auto" }}>
                      <LatexFormatter text={t.description} />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        <div style={{ animation: "fadeSlideUp 0.7s ease" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "16px" }}>
            <span style={{ fontFamily: "var(--mono)", fontSize: "10px", color: "var(--accent3)", letterSpacing: "0.15em" }}>SOLUTION</span>
            <div style={{ flex: 1, height: "1px", background: "var(--border)" }} />
            {accordionSteps.length > 0 && (
              <button
                onClick={() => setShowModal(true)}
                style={{ fontFamily: "var(--mono)", fontSize: "10px", color: "var(--muted)", background: "none", border: "1px solid var(--border)", borderRadius: "4px", padding: "4px 10px", cursor: "pointer" }}
              >
                전체 보기
              </button>
            )}
          </div>
          <h3 style={{ fontFamily: "'Noto Sans KR'", fontSize: "15px", fontWeight: 700, color: "var(--text)", marginBottom: "16px" }}>
            단계별 풀이
          </h3>

          {accordionSteps.length > 0 ? (
            accordionSteps.map((s, i) => <AccordionItem key={i} idx={i} step={s} />)
          ) : (
            <div style={{ padding: "32px", textAlign: "center", background: "var(--surface)", borderRadius: "10px", border: "1px solid var(--border)", fontFamily: "'Noto Sans KR'", fontSize: "14px", color: "var(--muted)" }}>
              {apiData ? "풀이 단계 데이터가 없습니다." : "데이터를 불러오는 중..."}
            </div>
          )}
        </div>

        <div style={{ textAlign: "center", marginTop: "40px" }}>
          <button
            onClick={onReset}
            style={{ background: "none", border: "1.5px solid var(--border)", borderRadius: "8px", padding: "10px 24px", cursor: "pointer", fontFamily: "var(--mono)", fontSize: "12px", color: "var(--muted)" }}
          >
            ← 새 문제 업로드
          </button>
        </div>
      </div>

      {showModal && (
        <div
          style={{ position: "fixed", inset: 0, background: "rgba(26,24,20,0.7)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 200 }}
          onClick={() => setShowModal(false)}
        >
          <div
            style={{ background: "var(--surface)", borderRadius: "16px", padding: "32px", maxWidth: "520px", width: "90%", maxHeight: "80vh", overflowY: "auto" }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "20px" }}>
              <h3 style={{ fontFamily: "'DM Serif Display', serif", fontSize: "20px", fontStyle: "italic" }}>풀이 과정</h3>
              <button onClick={() => setShowModal(false)} style={{ background: "none", border: "none", fontSize: "20px", cursor: "pointer", color: "var(--muted)" }}>
                ✕
              </button>
            </div>
            {accordionSteps.map((s, i) => (
              <div key={i} style={{ display: "flex", gap: "12px", marginBottom: "14px", alignItems: "flex-start" }}>
                <div style={{ width: "24px", height: "24px", borderRadius: "50%", background: "var(--text)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                  <span style={{ fontFamily: "var(--mono)", fontSize: "10px", color: "var(--bg)" }}>{i + 1}</span>
                </div>
                <div>
                  <p style={{ fontFamily: "'Noto Sans KR'", fontSize: "13px", fontWeight: 500 }}>{s.title}</p>
                  {s.detail && <p style={{ fontFamily: "'Noto Sans KR'", fontSize: "12px", color: "var(--muted)", marginTop: "4px", lineHeight: 1.7, whiteSpace: "pre-wrap" }}>{s.detail}</p>}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default function App() {
  const [page, setPage] = useState(1);
  const [apiData, setApiData] = useState<CircuitApiResponse | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const [uploadedImageUrl, setUploadedImageUrl] = useState<string | null>(null);

  const handleFileSelect = useCallback(async (file: File) => {
    // 업로드한 이미지 미리보기 URL 생성
    const objectUrl = URL.createObjectURL(file);
    setUploadedImageUrl(objectUrl);

    setPage(2);
    setIsLoading(true);
    setApiError(null);
    setIsSuccess(false);
    setApiData(null);

    try {
      const response = await uploadCircuitImage(file);
      setApiData(response);
      setIsSuccess(true);
    } catch (error) {
      setApiError(error instanceof Error ? error.message : "알 수 없는 오류가 발생했습니다.");
      setIsSuccess(true);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleLoadingDone = useCallback(() => setPage(3), []);
  const handleReset = useCallback(() => {
    setPage(1);
    setApiData(null);
    setApiError(null);
    setIsSuccess(false);
    if (uploadedImageUrl) URL.revokeObjectURL(uploadedImageUrl);
    setUploadedImageUrl(null);
  }, [uploadedImageUrl]);

  return (
    <>
      <FontLink />
      <Navbar />
      {page === 1 && <Page1 onFileSelect={handleFileSelect} />}
      {page === 2 && <Page2 isSuccess={isSuccess} onDone={handleLoadingDone} />}
      {page === 3 && <Page3 onReset={handleReset} apiData={apiData} apiError={apiError} isLoading={isLoading} uploadedImageUrl={uploadedImageUrl} />}
    </>
  );
}
