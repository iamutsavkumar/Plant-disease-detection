/**
 * api.js — Axios instance + PlantMD API helpers
 */
import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

// ── Axios instance ────────────────────────────────────────────────────────────
const api = axios.create({
  baseURL: BASE_URL,
  timeout: 60_000,
  headers: { "Content-Type": "application/json" },
});

// ── Response interceptor: normalise errors ────────────────────────────────────
api.interceptors.response.use(
  (res) => res,
  (err) => {
    const msg =
      err.response?.data?.error ||
      err.response?.data?.detail ||
      err.message ||
      "Unknown error";
    return Promise.reject(new Error(msg));
  }
);

// ── API helpers ───────────────────────────────────────────────────────────────

/**
 * Analyse a plant image via file upload (multipart/form-data).
 * @param {File} file
 * @returns {Promise<object>} prediction result
 */
export async function analyseFile(file) {
  const form = new FormData();
  form.append("image", file);
  const { data } = await api.post("/api/analyse/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data.result;
}

/**
 * Analyse a plant image via base64 string.
 * @param {string} base64 – data URI or raw base64
 * @returns {Promise<object>} prediction result
 */
export async function analyseBase64(base64) {
  const { data } = await api.post("/api/analyse/base64", { image: base64 });
  return data.result;
}

/**
 * Fetch prediction history.
 * @param {number} limit – max entries (default 20)
 */
export async function fetchHistory(limit = 20) {
  const { data } = await api.get("/api/history", { params: { limit } });
  return data;
}

/**
 * Fetch a single past prediction by ID.
 * @param {string} id
 */
export async function fetchPrediction(id) {
  const { data } = await api.get(`/api/history/${id}`);
  return data;
}

/**
 * Health-check both backend and AI server.
 */
export async function healthCheck() {
  const { data } = await api.get("/api/health");
  return data;
}

export default api;
