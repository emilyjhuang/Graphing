export async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return data;
}

export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

export function fmt(v, digits = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (typeof v !== "number") return String(v);
  if (Number.isInteger(v)) return v.toLocaleString();
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return v.toFixed(digits);
}
export const pct = (v, digits = 1) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(digits)}%`);

export const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/** The 8 categorical slots, in fixed order. Index >= 8 folds into "other". */
export function categorical(i) {
  return i < 8 ? css(`--s${i + 1}`) : css("--other");
}

export function sequential(t) {
  return d3.interpolateLab(css("--seq-lo"), css("--seq-hi"))(Math.max(0, Math.min(1, t)));
}

const tip = () => document.getElementById("tooltip");
export function showTip(html, event) {
  const el = tip();
  el.innerHTML = html;
  el.hidden = false;
  moveTip(event);
}
export function moveTip(event) {
  const el = tip();
  const pad = 14;
  const { innerWidth: w, innerHeight: h } = window;
  const r = el.getBoundingClientRect();
  let x = event.clientX + pad, y = event.clientY + pad;
  if (x + r.width > w - 8) x = event.clientX - r.width - pad;
  if (y + r.height > h - 8) y = event.clientY - r.height - pad;
  el.style.left = `${x}px`;
  el.style.top = `${y}px`;
}
export const hideTip = () => { tip().hidden = true; };

export function tipRows(title, rows, swatch) {
  const sw = swatch ? `<span class="sw" style="background:${swatch}"></span>` : "";
  return `<div class="t">${sw}${esc(title)}</div>` +
    rows.map(([k, v]) => `<div class="r"><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join("");
}

export const METRIC_LABELS = {
  degree: "Degree centrality",
  betweenness: "Betweenness",
  closeness: "Closeness",
  eigenvector: "Eigenvector",
  pagerank: "PageRank",
  clustering: "Local clustering",
  core: "k-core",
};
