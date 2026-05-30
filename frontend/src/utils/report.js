/**
 * report.js — Generate and download a plant disease diagnosis report as HTML.
 */

/**
 * Build a styled HTML report string from a prediction result.
 * @param {object} result  – prediction object from the API
 * @param {string} imageDataUrl – optional base64 data URI of the analysed image
 * @returns {string} full HTML document
 */
export function buildHtmlReport(result, imageDataUrl = null) {
  const now = new Date().toLocaleString();
  const urgencyColor = { Critical: "#dc2626", High: "#ea580c", Medium: "#d97706", Low: "#16a34a" };
  const uc = urgencyColor[result.urgency] || "#555";

  const listItems = (arr = []) =>
    arr.map((x) => `<li>${escHtml(x)}</li>`).join("");

  const section = (title, emoji, items, color) =>
    items?.length
      ? `<div class="card">
           <h3>${emoji} ${escHtml(title)}</h3>
           <ul>${listItems(items)}</ul>
         </div>`
      : "";

  const topPreds =
    result.topPredictions?.length
      ? `<div class="card">
           <h3>📊 Top Predictions</h3>
           <table>
             <thead><tr><th>Class</th><th>Confidence</th></tr></thead>
             <tbody>
               ${result.topPredictions
                 .map(
                   (p) => `<tr>
                     <td>${escHtml(p.label.replace(/_/g, " ").replace(/___/g, " — "))}</td>
                     <td>
                       <div class="bar-wrap">
                         <div class="bar" style="width:${p.confidence}%;background:#2d6a4f"></div>
                         <span>${p.confidence}%</span>
                       </div>
                     </td>
                   </tr>`
                 )
                 .join("")}
             </tbody>
           </table>
         </div>`
      : "";

  const imageSection = imageDataUrl
    ? `<div class="card image-card">
         <h3>🖼 Analysed Image</h3>
         <img src="${imageDataUrl}" alt="analysed plant leaf" style="max-width:320px;border-radius:12px;margin-top:8px;" />
       </div>`
    : "";

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>PlantMD Report — ${escHtml(result.diseaseName)}</title>
  <style>
    body{font-family:system-ui,sans-serif;background:#f0f4f1;color:#1a2e1f;margin:0;padding:24px}
    .page{max-width:800px;margin:0 auto;background:#fff;border-radius:16px;padding:40px;box-shadow:0 4px 24px rgba(26,71,49,.12)}
    h1{font-size:26px;color:#1a4731;margin-bottom:4px}
    .meta{font-size:13px;color:#4a6358;margin-bottom:28px}
    .hero{background:${result.isHealthy ? "#edfbf1" : "#fff8f5"};border:1.5px solid ${result.isHealthy ? "#52b788" : "#f4845f"};border-radius:14px;padding:20px 24px;margin-bottom:24px}
    .hero h2{font-size:22px;margin:0 0 8px;color:${result.isHealthy ? "#1a4731" : "#b34020"}}
    .badges{display:flex;gap:10px;flex-wrap:wrap;font-size:13px;margin-bottom:12px}
    .badge{padding:3px 12px;border-radius:99px;font-weight:500}
    .badge.healthy{background:#52b78830;color:#1a4731}
    .badge.disease{background:#f4845f30;color:#b34020}
    .urgency{color:${uc};font-weight:600}
    .bar-wrap{display:flex;align-items:center;gap:8px}
    .bar-wrap .bar{height:8px;border-radius:99px;min-width:4px}
    .bar-wrap span{font-size:12px;font-weight:500;white-space:nowrap}
    .card{border:1px solid #d4e9dc;border-radius:12px;padding:18px 20px;margin-bottom:16px}
    .card h3{font-size:15px;margin-bottom:12px;color:#1a4731}
    .card ul{list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:7px}
    .card li{font-size:13.5px;line-height:1.55;padding-left:16px;position:relative;color:#3a5a46}
    .card li::before{content:"●";position:absolute;left:0;color:#52b788;font-size:8px;top:5px}
    table{width:100%;border-collapse:collapse;font-size:13px}
    th,td{text-align:left;padding:8px 10px;border-bottom:1px solid #d4e9dc}
    th{color:#4a6358;font-weight:500}
    .note{font-size:13px;color:#4a6358;border-top:1px solid #d4e9dc;margin-top:12px;padding-top:12px;line-height:1.6}
    .footer{margin-top:32px;text-align:center;font-size:12px;color:#7aab8e}
    @media print{body{background:#fff;padding:0}.page{box-shadow:none}}
  </style>
</head>
<body>
<div class="page">
  <h1>🌿 PlantMD Diagnosis Report</h1>
  <p class="meta">Generated: ${now} · AI-powered plant disease detection</p>

  <div class="hero">
    <h2>${escHtml(result.diseaseName)}</h2>
    <div class="badges">
      <span class="badge ${result.isHealthy ? "healthy" : "disease"}">${result.isHealthy ? "✓ Healthy" : "⚠ Diseased"}</span>
      ${result.plantType ? `<span class="badge" style="background:#eaf4ef;color:#1a4731">🌱 ${escHtml(result.plantType)}</span>` : ""}
      ${result.severity !== "None" ? `<span class="badge" style="background:#fef3c7;color:#92400e">📊 ${escHtml(result.severity)} severity</span>` : ""}
    </div>
    <p>Confidence: <strong>${result.confidence}%</strong> &nbsp;|&nbsp; Urgency: <span class="urgency">${escHtml(result.urgency)}</span></p>
    ${result.additionalNotes ? `<p class="note">💬 ${escHtml(result.additionalNotes)}</p>` : ""}
  </div>

  ${imageSection}
  ${section("Observed Symptoms", "🔍", result.symptoms, "#f59e0b")}
  ${section("Likely Causes", "🧬", result.causes, "#8b5cf6")}
  ${section("Recommended Treatment", "💊", result.treatment, "#10b981")}
  ${section("Prevention Tips", "🛡", result.prevention, "#3b82f6")}
  ${topPreds}

  <div class="footer">
    PlantMD · AI-powered plant disease detection<br/>
    For critical agricultural decisions, consult a certified agronomist.
  </div>
</div>
</body>
</html>`;
}

/**
 * Trigger a browser download of the HTML report.
 * @param {object} result
 * @param {string|null} imageDataUrl
 */
export function downloadHtmlReport(result, imageDataUrl = null) {
  const html = buildHtmlReport(result, imageDataUrl);
  const blob = new Blob([html], { type: "text/html" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `plantmd-report-${result.diseaseName.replace(/\s+/g, "-").toLowerCase()}.html`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function escHtml(str = "") {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
