/**
 * PlantDiseaseDetector.jsx  — PlantMD v2 (Redesigned)
 * Dark AI SaaS aesthetic: glassmorphism · bioluminescent greens · animated data
 * All API/logic from the original is fully preserved.
 */
import { useState, useRef, useEffect } from "react";
import { analyseFile, analyseBase64 } from "../utils/api.js";
import { downloadHtmlReport } from "../utils/report.js";

// ─── Global Styles (injected once) ───────────────────────────────────────────
const GlobalStyle = () => (
  <style>{`
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Space+Mono:wght@400;700&family=Newsreader:ital,wght@0,400;0,500;1,400&display=swap');

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html { scroll-behavior: smooth; }

    :root {
      --bg-void:      #070c0a;
      --bg-deep:      #0b1410;
      --bg-surface:   #0f1e17;
      --bg-raised:    #152b1e;
      --bg-hover:     #1b3526;

      --green-dim:    #1a5c38;
      --green-mid:    #22a05a;
      --green-bright: #3ddc84;
      --green-glow:   #4dffa0;
      --green-muted:  #2a7a48;

      --amber:        #f59e0b;
      --red:          #ef4444;
      --blue:         #3b82f6;
      --purple:       #a78bfa;

      --text-primary:   #e8f5ee;
      --text-secondary: #7ab893;
      --text-dim:       #3d6b50;
      --text-mono:      #5ddc96;

      --border-subtle: rgba(61, 220, 132, 0.08);
      --border-mid:    rgba(61, 220, 132, 0.15);
      --border-bright: rgba(61, 220, 132, 0.35);

      --glow-sm:  0 0 12px rgba(61, 220, 132, 0.15);
      --glow-md:  0 0 28px rgba(61, 220, 132, 0.2);
      --glow-lg:  0 0 60px rgba(61, 220, 132, 0.25);

      --glass-bg: rgba(15, 30, 23, 0.65);
      --glass-border: rgba(61, 220, 132, 0.12);

      --radius-sm: 8px;
      --radius-md: 14px;
      --radius-lg: 20px;
      --radius-xl: 28px;

      --font-display: 'Syne', sans-serif;
      --font-mono:    'Space Mono', monospace;
      --font-serif:   'Newsreader', Georgia, serif;
    }

    body {
      font-family: var(--font-display);
      background: var(--bg-void);
      color: var(--text-primary);
      min-height: 100vh;
      -webkit-font-smoothing: antialiased;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--green-dim); border-radius: 2px; }

    /* ── Keyframes ── */
    @keyframes spin       { to { transform: rotate(360deg); } }
    @keyframes fadeUp     { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: none; } }
    @keyframes fadeIn     { from { opacity: 0; } to { opacity: 1; } }
    @keyframes slideRight { from { width: 0; } to { width: var(--bar-w); } }
    @keyframes pulse-glow { 0%,100% { box-shadow: var(--glow-sm); } 50% { box-shadow: var(--glow-lg); } }
    @keyframes scan-line  { from { transform: translateY(-100%); } to { transform: translateY(400%); } }
    @keyframes flicker    { 0%,100%{opacity:1} 92%{opacity:1} 93%{opacity:.85} 94%{opacity:1} 97%{opacity:.9} 98%{opacity:1} }
    @keyframes blink-cursor { 0%,100%{opacity:1} 50%{opacity:0} }
    @keyframes drift      { 0%,100%{transform:translateY(0) rotate(0deg)} 33%{transform:translateY(-6px) rotate(1deg)} 66%{transform:translateY(3px) rotate(-1deg)} }
    @keyframes shimmer    { 0%{background-position:-200% 0} 100%{background-position:200% 0} }
    @keyframes ring-spin  { from{stroke-dashoffset:280} to{stroke-dashoffset:0} }

    .fade-up   { animation: fadeUp .5s cubic-bezier(.22,1,.36,1) forwards; }
    .fade-in   { animation: fadeIn .4s ease forwards; }

    button { cursor: pointer; transition: all 0.2s ease; }
    button:active { transform: scale(0.97); }

    /* Noise overlay */
    .noise-bg::after {
      content: '';
      position: absolute;
      inset: 0;
      background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
      pointer-events: none;
      border-radius: inherit;
      z-index: 0;
    }

    /* Glass card */
    .glass {
      background: var(--glass-bg);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--glass-border);
    }

    /* Glow border animation */
    .glow-border {
      position: relative;
    }
    .glow-border::before {
      content: '';
      position: absolute;
      inset: -1px;
      border-radius: inherit;
      padding: 1px;
      background: linear-gradient(135deg, var(--green-bright), transparent 40%, transparent 60%, var(--green-dim));
      -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
      -webkit-mask-composite: xor;
      mask-composite: exclude;
      pointer-events: none;
    }

    @media (max-width: 640px) {
      html { font-size: 15px; }
    }
  `}</style>
);

// ─── Background Grid ──────────────────────────────────────────────────────────
const GridBackground = () => (
  <div style={{
    position: "fixed", inset: 0, zIndex: 0, overflow: "hidden", pointerEvents: "none",
  }}>
    {/* Grid lines */}
    <svg width="100%" height="100%" style={{ position: "absolute", inset: 0, opacity: 0.035 }}>
      <defs>
        <pattern id="grid" width="48" height="48" patternUnits="userSpaceOnUse">
          <path d="M 48 0 L 0 0 0 48" fill="none" stroke="#3ddc84" strokeWidth="0.5"/>
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill="url(#grid)"/>
    </svg>
    {/* Radial glow blobs */}
    <div style={{ position:"absolute", top:"10%", left:"15%", width:600, height:600, borderRadius:"50%", background:"radial-gradient(circle, rgba(34,160,90,0.06) 0%, transparent 70%)" }} />
    <div style={{ position:"absolute", bottom:"20%", right:"10%", width:400, height:400, borderRadius:"50%", background:"radial-gradient(circle, rgba(61,220,132,0.04) 0%, transparent 70%)" }} />
  </div>
);

// ─── Loading Messages ─────────────────────────────────────────────────────────
const LOADING_MESSAGES = [
  "Scanning leaf morphology…",
  "Identifying disease patterns…",
  "Checking against plant database…",
  "Generating treatment plan…",
];

// ─── Icon Components ──────────────────────────────────────────────────────────
const LeafIcon = ({ size = 20 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M2 22c1.25-1.25 2.5-2.5 3.5-3.5C7 17 8.5 16 11 15c2.5-1 5.5-1 8-3.5 2.5-2.5 3-6 3-9.5 0 0-4 .5-7 2.5S10 10 8 13c-2 3-4 6-6 9z"/>
    <line x1="2" y1="22" x2="10" y2="14"/>
  </svg>
);
const UploadIcon = () => (
  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/>
    <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>
  </svg>
);
const CameraIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
    <circle cx="12" cy="13" r="4"/>
  </svg>
);
const ScanIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <path d="M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2"/>
    <rect x="7" y="7" width="10" height="10" rx="1"/>
  </svg>
);
const DownloadIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
  </svg>
);
const TrashIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4h6v2"/>
  </svg>
);

// ─── Spinner ──────────────────────────────────────────────────────────────────
const Spinner = ({ size = 20 }) => (
  <svg width={size} height={size} viewBox="0 0 50 50" style={{ animation: "spin 0.8s linear infinite", flexShrink: 0 }}>
    <circle cx="25" cy="25" r="20" fill="none" stroke="rgba(61,220,132,0.2)" strokeWidth="4"/>
    <circle cx="25" cy="25" r="20" fill="none" stroke="#3ddc84" strokeWidth="4"
      strokeDasharray="80 45" strokeLinecap="round"/>
  </svg>
);

// ─── Confidence Bar ───────────────────────────────────────────────────────────
const ConfidenceBar = ({ value }) => {
  const color = value >= 80 ? "#3ddc84" : value >= 60 ? "#f59e0b" : "#ef4444";
  return (
    <div style={{ marginTop: 10 }}>
      <div style={{ display:"flex", justifyContent:"space-between", marginBottom:5 }}>
        <span style={{ fontFamily:"var(--font-mono)", fontSize:11, color:"var(--text-dim)", letterSpacing:"0.08em" }}>CONFIDENCE</span>
        <span style={{ fontFamily:"var(--font-mono)", fontSize:11, color, fontWeight:700 }}>{value}%</span>
      </div>
      <div style={{ height:4, borderRadius:99, background:"rgba(61,220,132,0.1)", overflow:"hidden", position:"relative" }}>
        <div style={{
          height:"100%", width:`${value}%`, background:`linear-gradient(90deg, ${color}88, ${color})`,
          borderRadius:99, transition:"width 1.2s cubic-bezier(.22,1,.36,1)",
          boxShadow:`0 0 8px ${color}66`,
        }}/>
      </div>
    </div>
  );
};

// ─── Result Card ──────────────────────────────────────────────────────────────
const ResultCard = ({ icon, title, items, accent }) => (
  <div className="glass fade-up" style={{
    borderRadius:"var(--radius-lg)", padding:"20px 22px",
    border:`1px solid rgba(${accent}, 0.18)`,
    position:"relative", overflow:"hidden",
  }}>
    <div style={{ position:"absolute", top:0, right:0, width:80, height:80, borderRadius:"0 var(--radius-lg) 0 80px", background:`rgba(${accent}, 0.05)` }}/>
    <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:14 }}>
      <div style={{ width:34, height:34, borderRadius:"var(--radius-sm)", background:`rgba(${accent}, 0.12)`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:16 }}>
        {icon}
      </div>
      <h3 style={{ fontFamily:"var(--font-display)", fontSize:13, fontWeight:700, letterSpacing:"0.06em", textTransform:"uppercase", color:"var(--text-secondary)" }}>{title}</h3>
    </div>
    <ul style={{ listStyle:"none", display:"flex", flexDirection:"column", gap:8 }}>
      {items.map((item, i) => (
        <li key={i} style={{ display:"flex", gap:10, fontSize:13, color:"var(--text-primary)", lineHeight:1.6 }}>
          <span style={{ color:`rgb(${accent})`, marginTop:3, flexShrink:0, fontSize:8 }}>◆</span>
          <span style={{ fontFamily:"var(--font-serif)", fontStyle:"italic" }}>{item}</span>
        </li>
      ))}
    </ul>
  </div>
);

// ─── Status Badge ─────────────────────────────────────────────────────────────
const StatusBadge = ({ isHealthy }) => (
  <span style={{
    fontFamily:"var(--font-mono)", fontSize:10, fontWeight:700, letterSpacing:"0.12em",
    padding:"3px 10px", borderRadius:99, textTransform:"uppercase",
    background: isHealthy ? "rgba(61,220,132,0.12)" : "rgba(239,68,68,0.12)",
    color: isHealthy ? "var(--green-bright)" : "#ef4444",
    border: `1px solid ${isHealthy ? "rgba(61,220,132,0.25)" : "rgba(239,68,68,0.25)"}`,
  }}>
    {isHealthy ? "● HEALTHY" : "● DISEASED"}
  </span>
);

// ─── Urgency Colors ───────────────────────────────────────────────────────────
const urgencyColor = (u) => ({
  Low: "#3ddc84", Medium: "#f59e0b", High: "#f97316", Critical: "#ef4444"
})[u] || "#3ddc84";

// ─── Main Component ───────────────────────────────────────────────────────────
export default function PlantDiseaseDetector() {
  const [image, setImage]             = useState(null);
  const [imageFile, setImageFile]     = useState(null);
  const [dragging, setDragging]       = useState(false);
  const [loading, setLoading]         = useState(false);
  const [result, setResult]           = useState(null);
  const [error, setError]             = useState(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [loadingMsg, setLoadingMsg]   = useState(LOADING_MESSAGES[0]);
  const [msgIndex, setMsgIndex]       = useState(0);

  const fileRef   = useRef(null);
  const videoRef  = useRef(null);
  const streamRef = useRef(null);
  const resultRef = useRef(null);

  // Rotate loading messages
  useEffect(() => {
    if (!loading) return;
    setMsgIndex(0);
    setLoadingMsg(LOADING_MESSAGES[0]);
    let i = 0;
    const iv = setInterval(() => {
      i = (i + 1) % LOADING_MESSAGES.length;
      setMsgIndex(i);
      setLoadingMsg(LOADING_MESSAGES[i]);
    }, 1800);
    return () => clearInterval(iv);
  }, [loading]);

  // Scroll to results
  useEffect(() => {
    if (result && resultRef.current) {
      setTimeout(() => resultRef.current.scrollIntoView({ behavior:"smooth", block:"start" }), 150);
    }
  }, [result]);

  // ── File handling ────────────────────────────────────────────────────────
  const processFile = (file) => {
    if (!file || !file.type.startsWith("image/")) {
      setError("Please upload a valid image file."); return;
    }
    setError(null); setResult(null);
    setImageFile(file);
    const reader = new FileReader();
    reader.onload = (e) => setImage(e.target.result);
    reader.readAsDataURL(file);
  };

  const onDrop = (e) => {
    e.preventDefault(); setDragging(false);
    processFile(e.dataTransfer.files[0]);
  };

  // ── Camera ───────────────────────────────────────────────────────────────
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode:"environment" } });
      streamRef.current = stream;
      setCameraActive(true);
      setTimeout(() => { if (videoRef.current) videoRef.current.srcObject = stream; }, 100);
    } catch {
      setError("Camera access denied. Please use file upload instead.");
    }
  };

  const stopCamera = () => {
    if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    setCameraActive(false);
    streamRef.current = null;
  };

  const captureFrame = () => {
    const canvas = document.createElement("canvas");
    canvas.width  = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    canvas.getContext("2d").drawImage(videoRef.current, 0, 0);
    const dataUrl = canvas.toDataURL("image/jpeg");
    setImage(dataUrl);
    setImageFile(null);
    stopCamera();
    setResult(null); setError(null);
  };

  // ── Analysis ─────────────────────────────────────────────────────────────
  const analyse = async () => {
    if (!image) return;
    setLoading(true); setResult(null); setError(null);
    try {
      const prediction = imageFile
        ? await analyseFile(imageFile)
        : await analyseBase64(image);
      setResult(prediction);
    } catch (e) {
      console.error(e);
      setError(e.message || "Analysis failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const downloadReport = () => result && downloadHtmlReport(result, image);

  const clearAll = () => {
    setImage(null); setImageFile(null); setResult(null); setError(null);
  };

  return (
    <div style={{ minHeight:"100vh", background:"var(--bg-void)", color:"var(--text-primary)", position:"relative" }}>
      <GlobalStyle />
      <GridBackground />

      {/* ── NAVBAR ──────────────────────────────────────────────────────── */}
      <nav style={{
        position:"sticky", top:0, zIndex:200,
        background:"rgba(7,12,10,0.85)", backdropFilter:"blur(20px)",
        borderBottom:"1px solid var(--border-subtle)",
        display:"flex", alignItems:"center", justifyContent:"space-between",
        padding:"0 28px", height:58,
      }}>
        {/* Logo */}
        <div style={{ display:"flex", alignItems:"center", gap:10 }}>
          <div style={{ color:"var(--green-bright)", animation:"drift 4s ease-in-out infinite" }}>
            <LeafIcon size={22} />
          </div>
          <span style={{ fontFamily:"var(--font-display)", fontWeight:800, fontSize:17, letterSpacing:"-0.02em", color:"var(--text-primary)" }}>
            Plant<span style={{ color:"var(--green-bright)" }}>MD</span>
          </span>
          <span style={{
            fontFamily:"var(--font-mono)", fontSize:9, letterSpacing:"0.15em",
            background:"rgba(61,220,132,0.1)", color:"var(--green-bright)",
            border:"1px solid rgba(61,220,132,0.2)", padding:"2px 8px", borderRadius:99,
          }}>v2.0</span>
        </div>
        {/* Nav right */}
        <div style={{ display:"flex", alignItems:"center", gap:16 }}>
          <span style={{ fontFamily:"var(--font-mono)", fontSize:11, color:"var(--text-dim)", letterSpacing:"0.06em", display:"flex", alignItems:"center", gap:6 }}>
            <span style={{ width:6, height:6, borderRadius:"50%", background:"var(--green-bright)", display:"inline-block", boxShadow:"0 0 6px var(--green-bright)", animation:"pulse-glow 2s ease-in-out infinite" }}/>
            SYSTEM ONLINE
          </span>
        </div>
      </nav>

      <div style={{ position:"relative", zIndex:1 }}>

        {/* ── HERO ──────────────────────────────────────────────────────── */}
        <header style={{ textAlign:"center", padding:"72px 24px 56px", position:"relative" }}>
          {/* Badge */}
          <div style={{ display:"inline-flex", alignItems:"center", gap:8, marginBottom:24 }}>
            <span style={{
              fontFamily:"var(--font-mono)", fontSize:11, letterSpacing:"0.1em", textTransform:"uppercase",
              background:"rgba(61,220,132,0.08)", border:"1px solid rgba(61,220,132,0.2)",
              color:"var(--green-bright)", padding:"5px 14px", borderRadius:99,
              display:"flex", alignItems:"center", gap:8,
            }}>
              <ScanIcon />
              TensorFlow · MobileNetV2 · 38 Plant Classes
            </span>
          </div>

          <h1 style={{
            fontFamily:"var(--font-display)", fontSize:"clamp(38px, 6vw, 68px)",
            fontWeight:800, lineHeight:1.05, letterSpacing:"-0.035em",
            color:"var(--text-primary)", marginBottom:20,
          }}>
            Plant Disease<br/>
            <span style={{
              background:"linear-gradient(135deg, var(--green-bright), var(--green-mid))",
              WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent",
              backgroundClip:"text",
            }}>Detection System</span>
          </h1>

          <p style={{
            fontFamily:"var(--font-serif)", fontStyle:"italic",
            fontSize:17, color:"var(--text-secondary)", maxWidth:520,
            margin:"0 auto", lineHeight:1.75,
          }}>
            Upload a leaf photograph. Our AI instantly identifies diseases,
            assesses severity, and prescribes personalised treatment.
          </p>
        </header>

        {/* ── MAIN ────────────────────────────────────────────────────────── */}
        <main style={{ maxWidth:880, margin:"0 auto", padding:"0 20px 100px" }}>

          {/* Upload Panel */}
          <div style={{ display:"grid", gridTemplateColumns:"1fr auto", gap:14, marginBottom:20, alignItems:"start" }}>

            {/* Drop Zone */}
            <div
              className={`glass glow-border ${dragging ? "fade-in" : ""}`}
              onDragOver={e => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => !cameraActive && fileRef.current.click()}
              style={{
                borderRadius:"var(--radius-xl)", overflow:"hidden", cursor:"pointer",
                minHeight:240, display:"flex", alignItems:"center", justifyContent:"center",
                transition:"all 0.25s ease", position:"relative",
                border: dragging ? "1px solid var(--border-bright)" : "1px solid var(--glass-border)",
                boxShadow: dragging ? "var(--glow-md)" : "none",
              }}
            >
              {cameraActive ? (
                <div style={{ width:"100%", position:"relative" }}>
                  <video ref={videoRef} autoPlay playsInline
                    style={{ width:"100%", display:"block", borderRadius:"var(--radius-xl)" }}/>
                  {/* Scan line overlay */}
                  <div style={{ position:"absolute", inset:0, overflow:"hidden", borderRadius:"var(--radius-xl)", pointerEvents:"none" }}>
                    <div style={{
                      position:"absolute", left:0, right:0, height:2,
                      background:"linear-gradient(90deg, transparent, var(--green-bright), transparent)",
                      animation:"scan-line 2s linear infinite", opacity:0.6,
                    }}/>
                  </div>
                  <div style={{ position:"absolute", bottom:16, left:"50%", transform:"translateX(-50%)", display:"flex", gap:10 }}>
                    <button
                      onClick={e => { e.stopPropagation(); captureFrame(); }}
                      style={{
                        background:"var(--green-bright)", color:"var(--bg-void)",
                        border:"none", borderRadius:99, padding:"10px 24px",
                        fontFamily:"var(--font-display)", fontSize:13, fontWeight:700, letterSpacing:"0.06em",
                        boxShadow:"var(--glow-md)",
                      }}
                    >📸 CAPTURE</button>
                    <button
                      onClick={e => { e.stopPropagation(); stopCamera(); }}
                      style={{
                        background:"rgba(0,0,0,0.7)", color:"var(--text-primary)",
                        border:"1px solid var(--border-mid)", borderRadius:99, padding:"10px 18px",
                        fontFamily:"var(--font-display)", fontSize:13,
                      }}
                    >✕</button>
                  </div>
                </div>

              ) : image ? (
                <div style={{ width:"100%", padding:16, position:"relative" }}>
                  <img src={image} alt="Plant preview" style={{
                    width:"100%", maxHeight:320, objectFit:"contain",
                    borderRadius:"var(--radius-md)", display:"block",
                  }}/>
                  <div style={{
                    position:"absolute", top:24, right:24,
                    background:"rgba(7,12,10,0.8)", border:"1px solid var(--border-mid)",
                    borderRadius:99, padding:"4px 12px",
                    fontFamily:"var(--font-mono)", fontSize:10, letterSpacing:"0.08em", color:"var(--text-secondary)",
                  }}>CLICK TO REPLACE</div>
                </div>

              ) : (
                <div style={{ textAlign:"center", padding:"48px 24px", color:"var(--text-dim)" }}>
                  <div style={{ marginBottom:16, color:"var(--green-dim)" }}>
                    <UploadIcon />
                  </div>
                  <p style={{ fontFamily:"var(--font-display)", fontSize:15, fontWeight:600, color:"var(--text-secondary)", marginBottom:6, letterSpacing:"-0.01em" }}>
                    Drop leaf image here
                  </p>
                  <p style={{ fontFamily:"var(--font-mono)", fontSize:11, letterSpacing:"0.06em", color:"var(--text-dim)" }}>
                    OR CLICK TO BROWSE · JPG · PNG · WEBP
                  </p>
                </div>
              )}
              <input ref={fileRef} type="file" accept="image/*" style={{ display:"none" }}
                onChange={e => processFile(e.target.files[0])}/>
            </div>

            {/* Side Controls */}
            <div style={{ display:"flex", flexDirection:"column", gap:10 }}>
              <button
                onClick={cameraActive ? stopCamera : startCamera}
                style={{
                  background: cameraActive ? "rgba(239,68,68,0.1)" : "var(--bg-surface)",
                  border: `1px solid ${cameraActive ? "rgba(239,68,68,0.3)" : "var(--border-mid)"}`,
                  color: cameraActive ? "#ef4444" : "var(--text-secondary)",
                  borderRadius:"var(--radius-md)", padding:"12px 16px",
                  fontFamily:"var(--font-display)", fontSize:12, fontWeight:600,
                  letterSpacing:"0.04em", display:"flex", flexDirection:"column",
                  alignItems:"center", gap:7, minWidth:72,
                }}
              >
                <CameraIcon />
                {cameraActive ? "STOP" : "CAMERA"}
              </button>
              {image && !cameraActive && (
                <button
                  onClick={clearAll}
                  style={{
                    background:"transparent", border:"1px solid var(--border-subtle)",
                    color:"var(--text-dim)", borderRadius:"var(--radius-md)", padding:"12px 16px",
                    fontFamily:"var(--font-display)", fontSize:12, fontWeight:600,
                    letterSpacing:"0.04em", display:"flex", flexDirection:"column",
                    alignItems:"center", gap:7, minWidth:72,
                  }}
                >
                  <TrashIcon />
                  CLEAR
                </button>
              )}
            </div>
          </div>

          {/* Error Banner */}
          {error && (
            <div className="fade-up" style={{
              background:"rgba(239,68,68,0.08)", border:"1px solid rgba(239,68,68,0.25)",
              borderRadius:"var(--radius-md)", padding:"12px 16px", marginBottom:16,
              display:"flex", alignItems:"center", gap:10,
            }}>
              <span style={{ fontSize:16 }}>⚠</span>
              <span style={{ fontFamily:"var(--font-mono)", fontSize:12, color:"#ef4444", letterSpacing:"0.04em" }}>{error}</span>
            </div>
          )}

          {/* Analyse Button */}
          {image && !cameraActive && (
            <div style={{ marginBottom:36 }}>
              <button
                onClick={analyse}
                disabled={loading}
                style={{
                  width:"100%", border:"none", borderRadius:"var(--radius-lg)",
                  padding:"17px 28px", fontSize:15, fontWeight:700,
                  letterSpacing:"0.05em", fontFamily:"var(--font-display)",
                  display:"flex", alignItems:"center", justifyContent:"center", gap:12,
                  background: loading
                    ? "rgba(61,220,132,0.08)"
                    : "linear-gradient(135deg, rgba(34,160,90,0.25), rgba(61,220,132,0.15))",
                  color: loading ? "var(--text-dim)" : "var(--green-bright)",
                  border: `1px solid ${loading ? "var(--border-subtle)" : "rgba(61,220,132,0.35)"}`,
                  boxShadow: loading ? "none" : "var(--glow-sm), inset 0 1px 0 rgba(61,220,132,0.1)",
                  cursor: loading ? "not-allowed" : "pointer",
                  transition:"all 0.2s ease",
                }}
              >
                {loading ? (
                  <>
                    <Spinner size={18} />
                    <span style={{ fontFamily:"var(--font-mono)", fontSize:12, letterSpacing:"0.08em" }}>
                      {loadingMsg.toUpperCase()}
                    </span>
                    <span style={{ fontFamily:"var(--font-mono)", fontSize:12, animation:"blink-cursor 1s step-end infinite" }}>_</span>
                  </>
                ) : (
                  <>
                    <LeafIcon size={18} />
                    ANALYSE PLANT DISEASE
                  </>
                )}
              </button>
            </div>
          )}

          {/* ── RESULTS ──────────────────────────────────────────────────── */}
          {result && (
            <div ref={resultRef} className="fade-up">

              {/* Hero Result Card */}
              <div className="glass glow-border" style={{
                borderRadius:"var(--radius-xl)", padding:"28px 28px 24px",
                marginBottom:16, position:"relative", overflow:"hidden",
              }}>
                {/* Decorative corner glow */}
                <div style={{
                  position:"absolute", top:-40, right:-40, width:180, height:180, borderRadius:"50%",
                  background: result.isHealthy
                    ? "radial-gradient(circle, rgba(61,220,132,0.08) 0%, transparent 70%)"
                    : "radial-gradient(circle, rgba(239,68,68,0.08) 0%, transparent 70%)",
                  pointerEvents:"none",
                }}/>

                <div style={{ display:"flex", alignItems:"flex-start", gap:18, flexWrap:"wrap", position:"relative" }}>
                  {/* Icon */}
                  <div style={{
                    width:60, height:60, borderRadius:"var(--radius-md)", flexShrink:0,
                    background: result.isHealthy ? "rgba(61,220,132,0.1)" : "rgba(239,68,68,0.1)",
                    border: `1px solid ${result.isHealthy ? "rgba(61,220,132,0.2)" : "rgba(239,68,68,0.2)"}`,
                    display:"flex", alignItems:"center", justifyContent:"center", fontSize:28,
                  }}>
                    {result.isPlant === false ? "🚫" : result.isHealthy ? "🌿" : "🍂"}
                  </div>

                  <div style={{ flex:1 }}>
                    <div style={{ display:"flex", alignItems:"center", gap:10, flexWrap:"wrap", marginBottom:8 }}>
                      <h2 style={{
                        fontFamily:"var(--font-display)", fontSize:22, fontWeight:800,
                        letterSpacing:"-0.03em",
                        color: result.isHealthy ? "var(--green-bright)" : "#ef4444",
                      }}>
                        {result.diseaseName}
                      </h2>
                      <StatusBadge isHealthy={result.isHealthy} />
                    </div>

                    {/* Metadata row */}
                    <div style={{ display:"flex", flexWrap:"wrap", gap:16, fontSize:12, color:"var(--text-secondary)", fontFamily:"var(--font-mono)", letterSpacing:"0.04em" }}>
                      {result.plantType && (
                        <span>🌱 {result.plantType.toUpperCase()}</span>
                      )}
                      {result.severity && result.severity !== "None" && (
                        <span>SEVERITY: <strong style={{ color:"var(--text-primary)" }}>{result.severity.toUpperCase()}</strong></span>
                      )}
                      <span style={{ color: urgencyColor(result.urgency) }}>
                        URGENCY: <strong>{result.urgency?.toUpperCase()}</strong>
                      </span>
                      {result.inferenceMs > 0 && (
                        <span style={{ color:"var(--text-dim)" }}>{result.inferenceMs}ms</span>
                      )}
                    </div>

                    <ConfidenceBar value={result.confidence} />
                  </div>
                </div>

                {result.additionalNotes && (
                  <p style={{
                    marginTop:20, fontFamily:"var(--font-serif)", fontStyle:"italic",
                    fontSize:14, color:"var(--text-secondary)", lineHeight:1.7,
                    borderTop:"1px solid var(--border-subtle)", paddingTop:16,
                  }}>
                    {result.additionalNotes}
                  </p>
                )}
              </div>

              {/* Detail Cards Grid */}
              <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fit, minmax(255px, 1fr))", gap:12, marginBottom:16 }}>
                {result.symptoms?.length  > 0 && <ResultCard icon="🔬" title="Observed Symptoms"     items={result.symptoms}   accent="245, 158, 11" />}
                {result.causes?.length    > 0 && <ResultCard icon="🧬" title="Likely Causes"         items={result.causes}     accent="167, 139, 250" />}
                {result.treatment?.length > 0 && <ResultCard icon="💊" title="Treatment Protocol"    items={result.treatment}  accent="61, 220, 132" />}
                {result.prevention?.length> 0 && <ResultCard icon="🛡" title="Prevention Strategy"   items={result.prevention} accent="59, 130, 246" />}
              </div>

              {/* Top Predictions */}
              {result.topPredictions?.length > 0 && (
                <div className="glass" style={{ borderRadius:"var(--radius-lg)", padding:"22px 24px", marginBottom:16 }}>
                  <h3 style={{
                    fontFamily:"var(--font-display)", fontSize:11, fontWeight:700, letterSpacing:"0.12em",
                    textTransform:"uppercase", color:"var(--text-dim)", marginBottom:18,
                  }}>Top Predictions</h3>
                  <div style={{ display:"flex", flexDirection:"column", gap:10 }}>
                    {result.topPredictions.slice(0, 5).map((p, i) => (
                      <div key={i} style={{ display:"flex", alignItems:"center", gap:12 }}>
                        <span style={{ fontFamily:"var(--font-mono)", fontSize:10, color:"var(--text-dim)", width:14, textAlign:"right" }}>{i+1}</span>
                        <span style={{ fontFamily:"var(--font-serif)", fontSize:13, color: i === 0 ? "var(--text-primary)" : "var(--text-secondary)", flex:1 }}>
                          {p.label.replace(/_/g, " ").replace(/___/g, " — ")}
                        </span>
                        <div style={{ width:90, height:4, borderRadius:99, background:"rgba(61,220,132,0.08)", overflow:"hidden" }}>
                          <div style={{
                            height:"100%", width:`${p.confidence}%`,
                            background: i === 0 ? "var(--green-bright)" : "var(--green-dim)",
                            borderRadius:99, transition:"width 1s ease",
                            boxShadow: i === 0 ? "0 0 6px rgba(61,220,132,0.5)" : "none",
                          }}/>
                        </div>
                        <span style={{ fontFamily:"var(--font-mono)", fontSize:10, color:"var(--text-dim)", width:34, textAlign:"right" }}>{p.confidence}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Download */}
              <button
                onClick={downloadReport}
                style={{
                  width:"100%", background:"transparent",
                  border:"1px solid var(--border-mid)",
                  color:"var(--text-secondary)", borderRadius:"var(--radius-md)",
                  padding:"13px 20px", fontSize:12, fontWeight:600,
                  fontFamily:"var(--font-display)", letterSpacing:"0.06em",
                  display:"flex", alignItems:"center", justifyContent:"center", gap:8,
                }}
              >
                <DownloadIcon />
                DOWNLOAD DIAGNOSIS REPORT (HTML)
              </button>
            </div>
          )}

          {/* ── Empty State ───────────────────────────────────────────── */}
          {!image && !cameraActive && (
            <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fit, minmax(195px, 1fr))", gap:12, marginTop:8 }}>
              {[
                { icon:"🌾", label:"Wheat & Corn",   desc:"Rust · Blight · Smut" },
                { icon:"🍅", label:"Tomatoes",        desc:"Early Blight · Mosaic" },
                { icon:"🍇", label:"Grapes & Fruit",  desc:"Powdery Mildew · Rot" },
                { icon:"🌿", label:"Leafy Crops",     desc:"Downy Mildew · Spots" },
              ].map(({ icon, label, desc }) => (
                <div
                  key={label}
                  className="glass"
                  style={{ borderRadius:"var(--radius-md)", padding:"16px 18px", display:"flex", alignItems:"center", gap:12, transition:"border-color 0.2s" }}
                >
                  <span style={{ fontSize:22, flexShrink:0 }}>{icon}</span>
                  <div>
                    <p style={{ fontFamily:"var(--font-display)", fontSize:13, fontWeight:600, letterSpacing:"-0.01em", color:"var(--text-primary)", marginBottom:2 }}>{label}</p>
                    <p style={{ fontFamily:"var(--font-mono)", fontSize:10, letterSpacing:"0.06em", color:"var(--text-dim)" }}>{desc}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </main>
      </div>

      {/* ── FOOTER ────────────────────────────────────────────────────────── */}
      <footer style={{
        textAlign:"center", padding:"24px",
        borderTop:"1px solid var(--border-subtle)",
        position:"relative", zIndex:1,
      }}>
        <div style={{ display:"flex", alignItems:"center", justifyContent:"center", gap:8, marginBottom:6 }}>
          <span style={{ color:"var(--green-bright)" }}><LeafIcon size={13} /></span>
          <span style={{ fontFamily:"var(--font-display)", fontWeight:800, fontSize:13, letterSpacing:"-0.02em" }}>
            Plant<span style={{ color:"var(--green-bright)" }}>MD</span>
          </span>
        </div>
        <p style={{ fontFamily:"var(--font-mono)", fontSize:10, letterSpacing:"0.06em", color:"var(--text-dim)", lineHeight:1.8 }}>
          AI-POWERED PLANT DISEASE DETECTION<br/>
          FOR CRITICAL AGRICULTURAL DECISIONS, CONSULT A CERTIFIED AGRONOMIST.
        </p>
      </footer>
    </div>
  );
}
