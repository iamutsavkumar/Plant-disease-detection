/**
 * useAnalysis.js — custom hook for plant disease analysis logic
 */
import { useState, useCallback } from "react";
import { analyseFile, analyseBase64 } from "../utils/api.js";

const LOADING_MESSAGES = [
  "Scanning leaf morphology…",
  "Identifying disease patterns…",
  "Checking against plant database…",
  "Generating treatment plan…",
];

export default function useAnalysis() {
  const [result, setResult]       = useState(null);
  const [error, setError]         = useState(null);
  const [loading, setLoading]     = useState(false);
  const [loadingMsg, setLoadingMsg] = useState(LOADING_MESSAGES[0]);

  /** Rotate loading messages while analysing */
  const startMessageCycle = () => {
    let i = 0;
    setLoadingMsg(LOADING_MESSAGES[0]);
    return setInterval(() => {
      i = (i + 1) % LOADING_MESSAGES.length;
      setLoadingMsg(LOADING_MESSAGES[i]);
    }, 1800);
  };

  /**
   * Analyse via File object (from <input type="file"> or drag-drop)
   * @param {File} file
   */
  const analyseByFile = useCallback(async (file) => {
    setLoading(true);
    setError(null);
    setResult(null);
    const iv = startMessageCycle();
    try {
      const prediction = await analyseFile(file);
      setResult(prediction);
    } catch (err) {
      setError(err.message);
    } finally {
      clearInterval(iv);
      setLoading(false);
    }
  }, []);

  /**
   * Analyse via base64 string (from camera capture)
   * @param {string} base64 – data URI or raw base64
   */
  const analyseByBase64 = useCallback(async (base64) => {
    setLoading(true);
    setError(null);
    setResult(null);
    const iv = startMessageCycle();
    try {
      const prediction = await analyseBase64(base64);
      setResult(prediction);
    } catch (err) {
      setError(err.message);
    } finally {
      clearInterval(iv);
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
    setLoading(false);
  }, []);

  return {
    result,
    error,
    loading,
    loadingMsg,
    analyseByFile,
    analyseByBase64,
    reset,
  };
}
