/**
 * PlantMD – Node.js / Express Backend
 * Proxies prediction requests to the Python AI server,
 * handles auth, rate limiting, and prediction history.
 */

const express    = require("express");
const cors       = require("cors");
const multer     = require("multer");
const rateLimit  = require("express-rate-limit");
const helmet     = require("helmet");
const morgan     = require("morgan");
const axios      = require("axios");
const sharp      = require("sharp");
const path       = require("path");
const { v4: uuidv4 } = require("uuid");

const app = express();
const PORT      = process.env.PORT      || 5000;
const AI_SERVER = process.env.AI_SERVER || "http://localhost:8000";

// ── Middleware ─────────────────────────────────────────────────────────────────
app.use(helmet({ crossOriginEmbedderPolicy: false }));
app.use(cors({
  origin: process.env.CORS_ORIGIN || "*",
  methods: ["GET", "POST", "OPTIONS"],
}));
app.use(morgan("dev"));
app.use(express.json({ limit: "20mb" }));

// ── Rate limiting ──────────────────────────────────────────────────────────────
const analysisLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,    // 15 minutes
  max: 30,                       // max 30 analyses per IP
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "Too many requests. Please wait 15 minutes before trying again." },
});

// ── File upload (in-memory, max 10 MB) ────────────────────────────────────────
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    const allowed = ["image/jpeg", "image/png", "image/webp", "image/gif"];
    if (allowed.includes(file.mimetype)) cb(null, true);
    else cb(new Error("Only JPEG, PNG, WebP, and GIF images are allowed."));
  },
});

// ── In-memory prediction history (replace with DB in production) ───────────────
const history = [];

// ─────────────────────────────────────────────────────────────────────────────
// Routes
// ─────────────────────────────────────────────────────────────────────────────

/** Health check – also pings the AI server */
app.get("/api/health", async (req, res) => {
  try {
    const { data } = await axios.get(`${AI_SERVER}/health`, { timeout: 3000 });
    res.json({ backend: "ok", ai_server: data });
  } catch (err) {
    res.status(503).json({ backend: "ok", ai_server: "unreachable", detail: err.message });
  }
});

/**
 * POST /api/analyse/upload
 * Accepts multipart/form-data with a single 'image' field.
 * Preprocesses the image, forwards to Python AI server, returns result.
 */
app.post("/api/analyse/upload", analysisLimiter, upload.single("image"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No image file provided." });

  try {
    // Normalise image with sharp: resize to 512px max, convert to JPEG
    const processed = await sharp(req.file.buffer)
      .resize({ width: 512, height: 512, fit: "inside", withoutEnlargement: true })
      .jpeg({ quality: 90 })
      .toBuffer();

    const b64 = processed.toString("base64");

    // Forward to FastAPI
    const { data: prediction } = await axios.post(
      `${AI_SERVER}/predict`,
      { image: b64, mediaType: "image/jpeg" },
      { timeout: 30_000 },
    );

    const entry = {
      id: uuidv4(),
      timestamp: new Date().toISOString(),
      filename: req.file.originalname,
      result: prediction,
    };
    history.unshift(entry);
    if (history.length > 100) history.pop();   // keep last 100

    res.json({ success: true, id: entry.id, result: prediction });
  } catch (err) {
    console.error("[analyse/upload]", err.message);
    if (err.response) {
      return res.status(err.response.status).json({ error: err.response.data?.detail || "AI server error." });
    }
    if (err.code === "ECONNREFUSED") {
      return res.status(503).json({ error: "AI inference server is not running. Please start the Python server." });
    }
    res.status(500).json({ error: "Analysis failed. Please try again." });
  }
});

/**
 * POST /api/analyse/base64
 * Accepts JSON body { image: "<base64 with data URI prefix>" }
 * Used by the frontend when capturing from camera or pasting base64.
 */
app.post("/api/analyse/base64", analysisLimiter, async (req, res) => {
  const { image } = req.body;
  if (!image) return res.status(400).json({ error: "No image data provided." });

  try {
    // Strip data URI header to get raw base64, then re-encode via sharp
    const rawB64   = image.includes(",") ? image.split(",")[1] : image;
    const imgBuf   = Buffer.from(rawB64, "base64");
    const processed = await sharp(imgBuf)
      .resize({ width: 512, height: 512, fit: "inside", withoutEnlargement: true })
      .jpeg({ quality: 90 })
      .toBuffer();

    const b64 = processed.toString("base64");
    const { data: prediction } = await axios.post(
      `${AI_SERVER}/predict`,
      { image: b64, mediaType: "image/jpeg" },
      { timeout: 30_000 },
    );

    const entry = {
      id: uuidv4(),
      timestamp: new Date().toISOString(),
      filename: "camera-capture.jpg",
      result: prediction,
    };
    history.unshift(entry);
    if (history.length > 100) history.pop();

    res.json({ success: true, id: entry.id, result: prediction });
  } catch (err) {
    console.error("[analyse/base64]", err.message);
    if (err.code === "ECONNREFUSED") {
      return res.status(503).json({ error: "AI inference server is not running." });
    }
    res.status(500).json({ error: "Analysis failed. Please try again." });
  }
});

/** GET /api/history – return last N predictions */
app.get("/api/history", (req, res) => {
  const limit = Math.min(parseInt(req.query.limit) || 20, 100);
  res.json({ predictions: history.slice(0, limit), total: history.length });
});

/** GET /api/history/:id – return a single past prediction */
app.get("/api/history/:id", (req, res) => {
  const entry = history.find(h => h.id === req.params.id);
  if (!entry) return res.status(404).json({ error: "Prediction not found." });
  res.json(entry);
});

/** Serve React build in production */
if (process.env.NODE_ENV === "production") {
  app.use(express.static(path.join(__dirname, "../frontend/build")));
  app.get("*", (_req, res) =>
    res.sendFile(path.join(__dirname, "../frontend/build/index.html"))
  );
}

// ── Error handler ─────────────────────────────────────────────────────────────
app.use((err, _req, res, _next) => {
  if (err instanceof multer.MulterError) {
    return res.status(400).json({ error: `Upload error: ${err.message}` });
  }
  console.error(err);
  res.status(500).json({ error: err.message || "Internal server error" });
});

// ── Start ─────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n🌿 PlantMD Backend running on http://localhost:${PORT}`);
  console.log(`   AI Server: ${AI_SERVER}`);
  console.log(`   Environment: ${process.env.NODE_ENV || "development"}\n`);
});

module.exports = app;
