import { api, esc, fmt, pct, categorical, sequential, css, showTip, hideTip, tipRows, METRIC_LABELS } from "./util.js";
import { GraphView, edgeKey } from "./graph.js";
import { lineChart, barChart, scatter, legend } from "./charts.js";

const $ = (id) => document.getElementById(id);
const METRICS = ["degree", "betweenness", "closeness", "eigenvector", "pagerank", "clustering", "core"];

const state = {
  catalog: [],
  algorithms: [],
  spec: null,
  specKey: "",
  data: null,
  byId: new Map(),
  ranks: {},
  tab: "overview",
  results: {}, // per-graph results for heavy tabs: robustness, links, gcn, compare
  path: { source: "", target: "", next: "source", result: null, active: -1 },
  selected: null,
};

const view = new GraphView($("graph"), {
  onHover: (event, d) => showTip(nodeTip(d), event),
  onLeave: hideTip,
  onClick: (d) => onNodeClick(d),
});

// ------------------------------------------------------------------ helpers

function busy(on, text = "Computing…") {
  $("loading").hidden = !on;
  $("loading-text").textContent = text;
}

async function withButton(btn, fn) {
  const label = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Running…";
  try { return await fn(); } finally { btn.disabled = false; btn.textContent = label; }
}

function errorBox(el, err) {
  el.innerHTML = `<p class="error">⚠ ${esc(err.message || err)}</p>`;
}

function nodeTip(d) {
  const rows = [
    ["Degree", fmt(d.degree * (state.data.nodes.length - 1), 0)],
    ["Betweenness", fmt(d.betweenness)],
    ["PageRank", fmt(d.pagerank, 4)],
    ["Community", `#${d.community + 1}`],
  ];
  if (d.truth !== null && d.truth !== undefined) rows.push(["Ground truth", d.truth]);
  const gcn = state.results.gcn;
  if (gcn) {
    const i = gcn.index.get(d.id);
    rows.push(["GCN predicts", gcn.classes[gcn.final_pred[i]]]);
  }
  return tipRows(d.id, rows, view.color(d));
}

function datasetSpec() {
  const key = $("dataset").value;
  const ds = state.catalog.find((d) => d.key === key);
  const params = {};
  for (const p of ds.params) params[p.name] = +$(`param-${p.name}`).value;
  return { key, params, edges: key === "custom" ? $("custom-edges").value : null };
}

// ------------------------------------------------------------------ sidebar

function renderParams() {
  const ds = state.catalog.find((d) => d.key === $("dataset").value);
  $("dataset-desc").textContent = ds.description;
  $("custom-field").hidden = ds.key !== "custom";
  $("dataset-params").innerHTML = ds.params.map((p) => `
    <div class="field">
      <label for="param-${p.name}">${esc(p.label)} <output id="out-${p.name}">${p.default}</output></label>
      <input type="range" id="param-${p.name}" min="${p.min}" max="${p.max}" step="${p.step}" value="${p.default}">
    </div>`).join("");
  for (const p of ds.params) {
    $(`param-${p.name}`).addEventListener("input", (e) => { $(`out-${p.name}`).textContent = e.target.value; });
  }
}

async function loadGraph({ keepLayout = false } = {}) {
  const spec = keepLayout ? state.spec : datasetSpec();
  busy(true, "Analysing graph…");
  try {
    const data = await api("/api/graph", { dataset: spec, community: $("community").value });
    const specKey = JSON.stringify(spec);
    const sameGraph = keepLayout && specKey === state.specKey;
    state.spec = spec;
    state.specKey = specKey;
    state.data = data;
    state.byId = new Map(data.nodes.map((d) => [d.id, d]));
    computeRanks();
    if (sameGraph) {
      for (const n of view.nodes) Object.assign(n, state.byId.get(n.id));
    } else {
      state.results = {};
      state.path = { source: "", target: "", next: "source", result: null, active: -1 };
      state.selected = null;
      $("show-labels").checked = data.nodes.length <= 40;
      view.load(data, data.summary.weighted);
      renderInspector();
    }
    if ($("color-by").value === "truth" && !data.summary.has_truth) $("color-by").value = "community";
    applyEncoding();
    $("status").textContent = `${fmt(data.summary.nodes)} nodes · ${fmt(data.summary.edges)} edges · ${data.elapsed_ms} ms`;
    renderTab();
  } catch (err) {
    $("status").textContent = "";
    alertInPanel(err);
  } finally {
    busy(false);
  }
}

function alertInPanel(err) {
  const el = $(`tab-${state.tab}`);
  el.insertAdjacentHTML("afterbegin", `<p class="error">⚠ ${esc(err.message)}</p>`);
}

function computeRanks() {
  state.ranks = {};
  const n = state.data.nodes.length;
  for (const m of METRICS) {
    const sorted = [...state.data.nodes].sort((a, b) => b[m] - a[m]);
    const r = new Map();
    let rank = 0;
    sorted.forEach((d, i) => { if (i === 0 || d[m] !== sorted[i - 1][m]) rank = i; r.set(d.id, rank); });
    state.ranks[m] = { rank: r, n };
  }
}

// ------------------------------------------------------------------ encoding

function classList() {
  return [...new Set(state.data.nodes.map((d) => d.truth))].filter((t) => t !== null).sort((a, b) => String(a).localeCompare(String(b), undefined, { numeric: true }));
}

function applyEncoding() {
  const data = state.data;
  if (!data) return;
  const n = data.nodes.length;
  const sizeBy = $("size-by").value;
  const [rMin, rMax] = n > 1000 ? [2, 9] : n > 400 ? [2.5, 11] : n > 150 ? [3, 13] : n > 50 ? [4, 16] : [5, 22];
  let radius = () => (n > 1000 ? 3 : n > 300 ? 4.5 : 8);
  if (sizeBy !== "none") {
    const s = d3.scaleSqrt().domain(d3.extent(data.nodes, (d) => d[sizeBy])).range([rMin, rMax]).clamp(true);
    if (s.domain()[0] === s.domain()[1]) s.domain([0, s.domain()[1] || 1]);
    radius = (d) => s(d[sizeBy]);
  }

  let colorBy = $("color-by").value;
  const gcn = state.results.gcn;
  if (colorBy === "gcn" && !gcn) colorBy = "community";
  let color;
  let legendHtml;
  if (colorBy === "community") {
    color = (d) => categorical(d.community);
    const c = data.communities;
    const items = c.sizes.slice(0, 8).map((size, i) => ({ color: categorical(i), label: `Community ${i + 1}`, note: `${size} nodes` }));
    legendHtml = legendList(c.label, items, c.count > 8 ? `+ ${c.count - 8} smaller communities in grey` : null);
  } else if (colorBy === "truth") {
    const classes = classList();
    color = (d) => categorical(classes.indexOf(d.truth));
    legendHtml = legendList("Ground-truth label", classes.slice(0, 8).map((c, i) => ({ color: categorical(i), label: c })), classes.length > 8 ? "Extra labels in grey" : null);
  } else if (colorBy === "gcn") {
    color = (d) => categorical(gcn.final_pred[gcn.index.get(d.id)]);
    const acc = gcn.results[0].test_acc;
    legendHtml = legendList("GCN predicted class", gcn.classes.slice(0, 8).map((c, i) => ({ color: categorical(i), label: c })),
      `Test accuracy ${pct(acc)} · ringed = labelled for training`);
  } else {
    const max = d3.max(data.nodes, (d) => d[colorBy]) || 1;
    color = (d) => sequential(Math.sqrt(d[colorBy] / max));
    legendHtml = `<div class="title">${esc(METRIC_LABELS[colorBy])}</div>
      <div class="ramp" style="background:linear-gradient(90deg, ${[0, 0.25, 0.5, 0.75, 1].map((t) => sequential(t)).join(",")})"></div>
      <div class="ramp-labels"><span>0</span><span>${fmt(max)}</span></div><div class="note">√-scaled</div>`;
  }
  $("legend").innerHTML = legendHtml;
  view.setEncoding({ radius, color, labels: $("show-labels").checked });
  if (state.tab === "gcn" && gcn) view.setRings(gcn.labelledIds);
  if (state.selected) view.select(state.selected);
}

function legendList(title, items, note) {
  return `<div class="title">${esc(title)}</div>` +
    items.map((it) => `<div class="item"><span class="swatch" style="background:${it.color}"></span>${esc(it.label)}${it.note ? ` <span class="muted">· ${esc(it.note)}</span>` : ""}</div>`).join("") +
    (note ? `<div class="note">${esc(note)}</div>` : "");
}

// ------------------------------------------------------------------ inspector

function onNodeClick(d) {
  if (!d) {
    state.selected = null;
    view.select(null);
    renderInspector();
    return;
  }
  state.selected = d.id;
  view.select(d.id);
  renderInspector();
  if (state.tab === "paths") {
    state.path[state.path.next] = d.id;
    state.path.next = state.path.next === "source" ? "target" : "source";
    renderPaths();
    if (state.path.source && state.path.target) findPath();
  }
}

function renderInspector() {
  const el = $("inspector");
  const d = state.selected && state.byId.get(state.selected);
  if (!d) { el.innerHTML = `<span class="muted">Click a node to inspect it.</span>`; return; }
  const deg = view.adj.get(d.id).size;
  const rows = METRICS.map((m) => {
    const { rank, n } = state.ranks[m];
    const r = rank.get(d.id);
    return `<span class="k">${esc(METRIC_LABELS[m])}</span><span>${fmt(d[m], m === "pagerank" ? 4 : 3)}</span><span class="pct">#${r + 1}/${n}</span>`;
  }).join("");
  el.innerHTML = `<h4>${esc(d.id)}</h4>
    <div class="row" style="margin-bottom:8px"><span class="chip">${deg} neighbours</span><span class="chip">community ${d.community + 1}</span>${d.truth !== null && d.truth !== undefined ? `<span class="chip">${esc(d.truth)}</span>` : ""}</div>
    <div class="kv">${rows}</div>
    <div class="row" style="margin-top:10px"><button class="btn" id="center-node">Centre</button><button class="btn" id="ego-node">Show ego network</button></div>`;
  $("center-node").onclick = () => view.centerOn(d.id);
  $("ego-node").onclick = () => {
    const nb = view.adj.get(d.id);
    const nodes = new Set([d.id, ...nb]);
    const edges = new Set(view.edges.filter((e) => nodes.has(e.source.id) && nodes.has(e.target.id)).map((e) => e.key));
    view.highlight(view.focus ? null : { nodes, edges });
  };
}

// ------------------------------------------------------------------ tabs

const TABS = {
  overview: renderOverview,
  centrality: renderCentrality,
  communities: renderCommunities,
  paths: renderPaths,
  robustness: renderRobustness,
  links: renderLinks,
  gcn: renderGCN,
};

function clearOverlays() {
  view.highlight(null);
  view.setRemoved(null);
  view.setOverlay([]);
  view.setRings(null);
}

function switchTab(tab) {
  state.tab = tab;
  for (const b of document.querySelectorAll(".tab")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
  for (const key of Object.keys(TABS)) $(`tab-${key}`).hidden = key !== tab;
  renderTab();
}

function renderTab() {
  if (!state.data) return;
  clearOverlays();
  TABS[state.tab]();
  applyEncoding();
}

// ---- Overview

function renderOverview() {
  const s = state.data.summary;
  const el = $("tab-overview");
  const cRand = s.nodes > 1 ? s.avg_degree / (s.nodes - 1) : 0;
  const lRand = s.avg_degree > 1 ? Math.log(s.nodes) / Math.log(s.avg_degree) : null;
  const sigma = cRand > 0 && lRand && s.avg_path_length ? (s.avg_clustering / cRand) / (s.avg_path_length / lRand) : null;
  const tiles = [
    ["Nodes", fmt(s.nodes)], ["Edges", fmt(s.edges)], ["Density", fmt(s.density, 4)],
    ["Mean degree", fmt(s.avg_degree, 2), `max ${s.max_degree}`],
    ["Avg clustering", fmt(s.avg_clustering), `transitivity ${fmt(s.transitivity)}`],
    ["Assortativity", fmt(s.assortativity), s.assortativity === null ? "" : s.assortativity < -0.05 ? "hubs link to leaves" : s.assortativity > 0.05 ? "hubs link to hubs" : "neutral mixing"],
    ["Avg path length", fmt(s.avg_path_length, 2), s.path_stats_approx ? "sampled, largest comp." : "largest component"],
    ["Diameter", fmt(s.diameter), s.path_stats_approx ? "lower bound" : ""],
    ["Components", fmt(s.components), `largest holds ${pct(s.lcc_fraction, 0)}`],
  ];
  el.innerHTML = `
    <p class="lede">Global structure of the network. Hover the histogram for exact counts.</p>
    <section><div class="tiles">${tiles.map(([l, v, sub]) => `<div class="tile"><div class="v">${v}</div><div class="l">${l}</div>${sub ? `<div class="s">${esc(sub)}</div>` : ""}</div>`).join("")}</div></section>
    <section>
      <div class="row" style="justify-content:space-between"><h3 class="section-title">Degree distribution</h3>
      <label class="check"><input type="checkbox" id="deg-log"> log count</label></div>
      <div id="deg-chart"></div>
    </section>
    <section class="callout" id="insight"></section>`;
  const hist = s.degree_histogram.map((y, x) => ({ x, y }));
  const draw = () => barChart($("deg-chart"), {
    data: hist, xLabel: "degree k", yLabel: "nodes", logY: $("deg-log").checked,
    tip: (d) => tipRows(`Degree ${d.x}`, [["Nodes", fmt(d.y)], ["Share", pct(d.y / s.nodes)]]),
  });
  draw();
  $("deg-log").onchange = draw;

  const notes = [];
  if (sigma) {
    notes.push(`Clustering is <strong>${fmt(s.avg_clustering / cRand, 1)}×</strong> what an Erdős–Rényi graph of the same density would give, with path lengths ${fmt(s.avg_path_length / lRand, 2)}× the random expectation. Small-world coefficient σ ≈ <strong>${fmt(sigma, 2)}</strong>${sigma > 1.5 ? ": a small-world network." : "."}`);
  }
  const hub = [...state.data.nodes].sort((a, b) => b.betweenness - a.betweenness)[0];
  if (hub) notes.push(`<strong>${esc(hub.id)}</strong> has the highest betweenness: ${pct(hub.betweenness)} of all shortest paths route through it.`);
  $("insight").innerHTML = notes.join("<br><br>");
}

// ---- Centrality

function spearman(a, b) {
  const rank = (arr) => {
    const idx = arr.map((v, i) => [v, i]).sort((x, y) => x[0] - y[0]);
    const r = new Array(arr.length);
    for (let i = 0; i < idx.length;) {
      let j = i;
      while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
      for (let k = i; k <= j; k++) r[idx[k][1]] = (i + j) / 2;
      i = j + 1;
    }
    return r;
  };
  const ra = rank(a), rb = rank(b);
  const ma = d3.mean(ra), mb = d3.mean(rb);
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < ra.length; i++) { num += (ra[i] - ma) * (rb[i] - mb); da += (ra[i] - ma) ** 2; db += (rb[i] - mb) ** 2; }
  return da && db ? num / Math.sqrt(da * db) : 0;
}

function renderCentrality() {
  const el = $("tab-centrality");
  const current = METRICS.includes($("size-by").value) ? $("size-by").value : "pagerank";
  el.innerHTML = `
    <p class="lede">Who matters, and by which definition? Click a row to locate the node.</p>
    <div class="field"><span class="label">Rank by</span><select id="cent-metric">${METRICS.map((m) => `<option value="${m}" ${m === current ? "selected" : ""}>${METRIC_LABELS[m]}</option>`).join("")}</select></div>
    <section class="table-scroll" id="cent-table"></section>
    <section><h3 class="section-title">Do the measures agree? (Spearman ρ)</h3><div class="table-scroll" id="corr"></div>
    <p class="muted" style="font-size:12px;margin-top:6px">Low agreement between betweenness and degree flags brokers: nodes that bridge groups without being popular.</p></section>`;
  const drawTable = () => {
    const m = $("cent-metric").value;
    $("size-by").value = m;
    applyEncoding();
    const top = [...state.data.nodes].sort((a, b) => b[m] - a[m]).slice(0, 15);
    const max = top[0]?.[m] || 1;
    $("cent-table").innerHTML = `<table class="data"><thead><tr><th>#</th><th>Node</th><th class="num">${esc(METRIC_LABELS[m])}</th><th></th><th class="num">Degree</th></tr></thead><tbody>
      ${top.map((d, i) => `<tr class="clickable" data-id="${esc(d.id)}"><td class="muted">${i + 1}</td><td>${esc(d.id)}</td><td class="num">${fmt(d[m], m === "pagerank" ? 4 : 3)}</td>
        <td class="bar-cell"><div class="bar" style="width:${(100 * d[m]) / max}%"></div></td><td class="num">${view.adj.get(d.id).size}</td></tr>`).join("")}
      </tbody></table>`;
    for (const tr of $("cent-table").querySelectorAll("tr[data-id]")) {
      tr.onclick = () => { onNodeClick(state.byId.get(tr.dataset.id)); view.centerOn(tr.dataset.id); };
    }
  };
  $("cent-metric").onchange = drawTable;
  drawTable();

  const cols = METRICS;
  const vals = Object.fromEntries(cols.map((m) => [m, state.data.nodes.map((d) => d[m])]));
  const short = { degree: "Deg", betweenness: "Btw", closeness: "Clo", eigenvector: "Eig", pagerank: "PR", clustering: "CC", core: "Core" };
  const div = d3.scaleLinear().domain([-1, 0, 1]).range([css("--s8"), css("--surface-2"), css("--s1")]).interpolate(d3.interpolateLab);
  $("corr").innerHTML = `<table class="data"><thead><tr><th></th>${cols.map((c) => `<th class="num" title="${METRIC_LABELS[c]}">${short[c]}</th>`).join("")}</tr></thead><tbody>
    ${cols.map((r) => `<tr><th title="${METRIC_LABELS[r]}">${short[r]}</th>${cols.map((c) => {
      const rho = r === c ? 1 : spearman(vals[r], vals[c]);
      const strong = Math.abs(rho) > 0.6;
      return `<td class="num" style="background:${div(rho)};color:${strong ? "#fff" : "var(--ink)"}">${rho.toFixed(2)}</td>`;
    }).join("")}</tr>`).join("")}</tbody></table>`;
}

// ---- Communities

function renderCommunities() {
  const el = $("tab-communities");
  const c = state.data.communities;
  const hasTruth = c.nmi !== null;
  el.innerHTML = `
    <p class="lede">Partition found by <strong>${esc(c.label)}</strong>. Switch algorithms in the sidebar; node colours follow.</p>
    <section><div class="tiles">
      <div class="tile"><div class="v">${c.count}</div><div class="l">Communities</div></div>
      <div class="tile"><div class="v">${fmt(c.modularity)}</div><div class="l">Modularity Q</div>${hasTruth ? `<div class="s">truth partition: ${fmt(c.truth_modularity)}</div>` : ""}</div>
      <div class="tile"><div class="v">${hasTruth ? fmt(c.nmi) : "—"}</div><div class="l">NMI vs truth</div><div class="s">${hasTruth ? `ARI ${fmt(c.ari)}` : "no labels"}</div></div>
    </div></section>
    <section><h3 class="section-title">Community sizes</h3><div id="comm-sizes"></div></section>
    <section>
      <div class="row" style="justify-content:space-between"><h3 class="section-title">Benchmark every algorithm</h3><button class="btn" id="compare">Run benchmark</button></div>
      <div id="compare-out" class="table-scroll"></div>
    </section>
    <section class="callout">Modularity Q measures how many more edges fall inside communities than a degree-preserving random graph would put there. ${hasTruth ? "NMI and ARI score agreement with the known labels (1 = perfect)." : "Load Karate Club, the SBM, or an edge list with labels to score against ground truth."} Higher Q is not always better: ${hasTruth && c.truth_modularity < c.modularity ? "here the <strong>true</strong> partition has lower Q than the one found, so maximising modularity alone over-splits." : "Q-maximisers can merge small groups (the resolution limit)."}</section>`;
  const max = c.sizes[0] || 1;
  $("comm-sizes").innerHTML = `<table class="data"><tbody>${c.sizes.slice(0, 12).map((s, i) => `<tr><td style="width:110px"><span class="chip" style="border-left:4px solid ${categorical(i)}">Community ${i + 1}</span></td>
    <td class="bar-cell"><div class="bar" style="width:${(100 * s) / max}%;background:${categorical(i)}"></div></td><td class="num" style="width:60px">${s}</td></tr>`).join("")}
    ${c.sizes.length > 12 ? `<tr><td colspan="3" class="muted">+ ${c.sizes.length - 12} more</td></tr>` : ""}</tbody></table>`;
  const drawCompare = (rows) => {
    const best = (k) => d3.max(rows, (r) => r[k] ?? -Infinity);
    const bestQ = best("modularity"), bestN = best("nmi");
    $("compare-out").innerHTML = `<table class="data"><thead><tr><th>Algorithm</th><th class="num">k</th><th class="num">Q</th><th class="num">NMI</th><th class="num">ARI</th><th class="num">ms</th></tr></thead><tbody>
      ${rows.map((r) => `<tr class="clickable" data-key="${r.key}"><td>${esc(r.label)}</td><td class="num">${r.count}</td>
        <td class="num">${r.modularity === bestQ ? "<strong>" + fmt(r.modularity) + "</strong>" : fmt(r.modularity)}</td>
        <td class="num">${r.nmi !== null && r.nmi === bestN ? "<strong>" + fmt(r.nmi) + "</strong>" : fmt(r.nmi)}</td>
        <td class="num">${fmt(r.ari)}</td><td class="num muted">${fmt(r.ms, 0)}</td></tr>`).join("")}</tbody></table>
      <p class="muted" style="font-size:12px">Best value in bold. Click a row to colour the graph with that partition.</p>`;
    for (const tr of $("compare-out").querySelectorAll("tr[data-key]")) {
      tr.onclick = () => { $("community").value = tr.dataset.key; loadGraph({ keepLayout: true }); };
    }
  };
  if (state.results.compare) drawCompare(state.results.compare);
  $("compare").onclick = (e) => withButton(e.target, async () => {
    try {
      const r = await api("/api/communities/compare", { dataset: state.spec });
      state.results.compare = r.rows;
      drawCompare(r.rows);
    } catch (err) { errorBox($("compare-out"), err); }
  });
}

// ---- Paths

function renderPaths() {
  const el = $("tab-paths");
  const p = state.path;
  const weighted = state.data.summary.weighted;
  el.innerHTML = `
    <p class="lede">Click two nodes on the graph (or type them) to find every shortest path between them.</p>
    <datalist id="node-ids">${state.data.nodes.slice(0, 2000).map((d) => `<option value="${esc(d.id)}">`).join("")}</datalist>
    <div class="controls-grid" style="grid-template-columns:1fr 1fr">
      <div class="field"><span class="label">From ${p.next === "source" ? "<span class='chip'>next click</span>" : ""}</span><input type="text" id="path-source" list="node-ids" value="${esc(p.source)}"></div>
      <div class="field"><span class="label">To ${p.next === "target" ? "<span class='chip'>next click</span>" : ""}</span><input type="text" id="path-target" list="node-ids" value="${esc(p.target)}"></div>
    </div>
    <div class="row" style="margin-bottom:14px">
      ${weighted ? `<label class="check"><input type="checkbox" id="path-weighted" ${p.weighted ? "checked" : ""}> Weighted (Dijkstra, cost = 1/weight)</label>` : ""}
      <button class="btn primary" id="path-go">Find paths</button><button class="btn" id="path-clear">Clear</button>
    </div>
    <div id="path-out"></div>`;
  $("path-go").onclick = () => {
    p.source = $("path-source").value.trim();
    p.target = $("path-target").value.trim();
    findPath();
  };
  $("path-clear").onclick = () => { Object.assign(p, { source: "", target: "", next: "source", result: null, active: -1 }); renderPaths(); view.highlight(null); };
  if (weighted) $("path-weighted").onchange = (e) => { p.weighted = e.target.checked; if (p.source && p.target) findPath(); };
  if (p.result) drawPathResult();
}

async function findPath() {
  const p = state.path;
  if (!p.source || !p.target) return;
  try {
    p.result = await api("/api/path", { dataset: state.spec, source: p.source, target: p.target, weighted: !!p.weighted });
    p.active = -1;
    if (state.tab === "paths") renderPaths();
  } catch (err) {
    p.result = null;
    errorBox($("path-out"), err);
  }
}

function drawPathResult() {
  const r = state.path.result;
  const out = $("path-out");
  if (!r.paths.length) {
    out.innerHTML = `<div class="callout">No path: <strong>${esc(state.path.source)}</strong> and <strong>${esc(state.path.target)}</strong> are in different components.</div>`;
    return;
  }
  const show = (i) => {
    state.path.active = i;
    const paths = i < 0 ? r.paths : [r.paths[i]];
    const nodes = new Set(paths.flat());
    const edges = new Set(paths.flatMap((pp) => pp.slice(1).map((v, j) => edgeKey(pp[j], v))));
    view.highlight({ nodes, edges });
    for (const tr of out.querySelectorAll("tr[data-i]")) tr.classList.toggle("best", +tr.dataset.i === i);
  };
  out.innerHTML = `
    <div class="tiles" style="margin-bottom:12px">
      <div class="tile"><div class="v">${r.hops}</div><div class="l">Hops</div></div>
      <div class="tile"><div class="v">${r.paths.length}${r.truncated ? "+" : ""}</div><div class="l">Shortest paths</div></div>
      <div class="tile"><div class="v">${r.cost !== null ? fmt(r.cost) : "—"}</div><div class="l">Weighted cost</div></div>
    </div>
    <table class="data"><tbody>
      <tr class="clickable ${state.path.active < 0 ? "best" : ""}" data-i="-1"><td colspan="2">Show all paths</td></tr>
      ${r.paths.map((pp, i) => `<tr class="clickable" data-i="${i}"><td class="muted">${i + 1}</td><td class="mono" style="font-size:12px">${pp.map(esc).join(" → ")}</td></tr>`).join("")}
    </tbody></table>`;
  for (const tr of out.querySelectorAll("tr[data-i]")) tr.onclick = () => show(+tr.dataset.i);
  show(state.path.active);
}

// ---- Robustness

const STRATEGIES = [
  { key: "random", label: "Random failure", short: "Random" },
  { key: "degree", label: "Degree attack", short: "Degree" },
  { key: "adaptive_degree", label: "Adaptive degree attack", short: "Adaptive" },
  { key: "betweenness", label: "Betweenness attack", short: "Betweenness" },
  { key: "pagerank", label: "PageRank attack", short: "PageRank" },
];

function renderRobustness() {
  const el = $("tab-robustness");
  el.innerHTML = `
    <p class="lede">Remove nodes one at a time and track how much of the network stays connected. Random failures vs targeted attacks.</p>
    <div id="rob-body"><div class="muted">Simulating…</div></div>`;
  if (state.results.robustness) return drawRobustness();
  const key = state.specKey;
  api("/api/robustness", { dataset: state.spec }).then((r) => {
    if (key !== state.specKey) return;
    state.results.robustness = r;
    if (state.tab === "robustness") drawRobustness();
  }).catch((err) => errorBox($("rob-body"), err));
}

function drawRobustness() {
  const r = state.results.robustness;
  const body = $("rob-body");
  const series = STRATEGIES.map((s, i) => ({
    key: s.key, label: s.label, shortLabel: s.short, color: categorical(i),
    points: r.x.map((x, j) => [x, r.curves[s.key][j]]),
  }));
  const bestAttack = STRATEGIES.slice(1).reduce((a, b) => (r.R[a.key] < r.R[b.key] ? a : b));
  const half = (k) => { const i = r.curves[k].findIndex((v) => v < 0.5); return i < 0 ? null : r.x[i]; };
  body.innerHTML = `
    <section><div class="chart-legend" id="rob-legend"></div><div id="rob-chart"></div></section>
    <section>
      <div class="row"><span class="label" style="font-size:12px;color:var(--ink-2)">Replay on graph:</span>
      <select id="rob-strategy" style="width:auto">${STRATEGIES.map((s) => `<option value="${s.key}" ${s.key === bestAttack.key ? "selected" : ""}>${s.label}</option>`).join("")}</select>
      <button class="btn" id="rob-play">▶ Play</button></div>
      <div class="scrubber"><input type="range" id="rob-k" min="0" max="${r.n}" value="0"><output id="rob-k-out"></output></div>
    </section>
    <section class="table-scroll"><table class="data"><thead><tr><th>Strategy</th><th class="num">R index</th><th></th><th class="num">LCC &lt; 50% after</th></tr></thead><tbody>
      ${STRATEGIES.map((s, i) => `<tr><td><span class="chip" style="border-left:4px solid ${categorical(i)}">${s.label}</span></td><td class="num">${fmt(r.R[s.key])}</td>
        <td class="bar-cell"><div class="bar" style="width:${200 * r.R[s.key]}%;background:${categorical(i)}"></div></td><td class="num">${half(s.key) === null ? "never" : pct(half(s.key), 0)}</td></tr>`).join("")}
    </tbody></table></section>
    <section class="callout">R = mean largest-component fraction over the whole removal sequence (Schneider et al., 2011; max 0.5). The most damaging strategy here is <strong>${bestAttack.label.toLowerCase()}</strong>, which needs to remove only ${half(bestAttack.key) === null ? "—" : pct(half(bestAttack.key), 0)} of nodes to break the network below half size, versus ${half("random") === null ? "never" : pct(half("random"), 0)} for random failures. Curves are computed in near-linear time by replaying removals backwards with a union–find.</section>`;
  legend($("rob-legend"), series);
  const chart = lineChart($("rob-chart"), {
    series, height: 230, xDomain: [0, 1], yDomain: [0, 1], directLabels: false,
    xLabel: "fraction removed", yLabel: "largest component", xFormat: d3.format(".0%"), yFormat: d3.format(".0%"),
  });
  const slider = $("rob-k");
  const update = () => {
    const k = +slider.value;
    const order = r.orders[$("rob-strategy").value];
    view.setRemoved(new Set(order.slice(0, k)));
    chart.setMarker(k / r.n);
    $("rob-k-out").textContent = `${k} removed (${pct(k / r.n, 0)})`;
  };
  slider.oninput = update;
  $("rob-strategy").onchange = update;
  let timer = null;
  $("rob-play").onclick = (e) => {
    if (timer) { clearInterval(timer); timer = null; e.target.textContent = "▶ Play"; return; }
    if (+slider.value >= r.n) slider.value = 0;
    e.target.textContent = "❚❚ Pause";
    const step = Math.max(1, Math.round(r.n / 80));
    timer = setInterval(() => {
      if (state.tab !== "robustness" || !document.body.contains(slider)) { clearInterval(timer); timer = null; return; }
      slider.value = Math.min(r.n, +slider.value + step);
      update();
      if (+slider.value >= r.n) { clearInterval(timer); timer = null; e.target.textContent = "▶ Play"; }
    }, 60);
  };
  update();
}

// ---- Link prediction

function renderLinks() {
  const el = $("tab-links");
  const prev = state.results.links;
  el.innerHTML = `
    <p class="lede">Hide a slice of real edges, then ask: can structure alone predict them? Seven classic heuristics vs a model trained on all of them.</p>
    <div class="controls-grid">
      <div class="field"><label>Held-out edges <output id="lp-frac-out">${pct(prev?.frac ?? 0.1, 0)}</output></label><input type="range" id="lp-frac" min="0.05" max="0.3" step="0.05" value="${prev?.frac ?? 0.1}"></div>
      <div class="field"><span class="label">Seed</span><input type="number" id="lp-seed" value="${prev?.seed ?? 0}" min="0"></div>
      <div class="field"><span class="label">&nbsp;</span><button class="btn primary" id="lp-run">Train &amp; evaluate</button></div>
    </div>
    <div id="lp-out"></div>`;
  $("lp-frac").oninput = (e) => { $("lp-frac-out").textContent = pct(+e.target.value, 0); };
  $("lp-run").onclick = (e) => withButton(e.target, runLinks);
  if (prev) drawLinks(); else runLinks();
}

async function runLinks() {
  const frac = +$("lp-frac").value, seed = +$("lp-seed").value;
  const key = state.specKey;
  $("lp-out").innerHTML = `<div class="muted">Training…</div>`;
  try {
    const r = await api("/api/link-prediction", { dataset: state.spec, test_frac: frac, seed });
    if (key !== state.specKey) return;
    state.results.links = { ...r, frac, seed };
    if (state.tab === "links") drawLinks();
  } catch (err) { errorBox($("lp-out"), err); }
}

function drawLinks() {
  const r = state.results.links;
  const model = r.methods[r.methods.length - 1];
  const heuristics = r.methods.slice(0, -1);
  const bestH = heuristics.reduce((a, b) => (a.auc > b.auc ? a : b));
  const maxImp = Math.max(...r.importances.map((d) => d.importance), 1e-9);
  const out = $("lp-out");
  out.innerHTML = `
    <section><div class="chart-legend" id="roc-legend"></div><div id="roc"></div></section>
    <section class="table-scroll"><table class="data"><thead><tr><th>Method</th><th class="num">ROC-AUC</th><th></th><th class="num">Avg precision</th></tr></thead><tbody>
      ${[...r.methods].sort((a, b) => b.auc - a.auc).map((m) => `<tr class="${m.key === "model" ? "best" : ""}"><td>${esc(m.label)}</td><td class="num">${fmt(m.auc)}</td>
        <td class="bar-cell"><div class="bar ${m.key === "model" ? "" : m.key === bestH.key ? "alt" : "muted"}" style="width:${Math.max(0, (m.auc - 0.5) * 200)}%"></div></td><td class="num">${fmt(m.ap)}</td></tr>`).join("")}
    </tbody></table><p class="muted" style="font-size:12px">Bars show lift over a coin flip (AUC 0.5 → 1.0). ${r.test_edges} test edges + ${r.test_edges} sampled non-edges; model fit on ${r.fit_edges} separately held-out edges.</p></section>
    <section><h3 class="section-title">What the model relies on (permutation importance, Δ AUC)</h3>
      <table class="data"><tbody>${r.importances.map((d) => `<tr><td style="width:46%">${esc(d.feature)}</td><td class="bar-cell"><div class="bar" style="width:${Math.max(0, (100 * d.importance) / maxImp)}%"></div></td><td class="num" style="width:60px">${fmt(d.importance)}</td></tr>`).join("")}</tbody></table></section>
    <section>
      <div class="row" style="justify-content:space-between"><h3 class="section-title">Top ${r.top_predictions.length} predicted links · precision ${pct(r.precision_at_k, 0)}</h3>
      <label class="check"><input type="checkbox" id="lp-overlay" checked> draw on graph</label></div>
      <table class="data"><thead><tr><th>Pair</th><th class="num">Score</th><th>Actually hidden?</th></tr></thead><tbody>
      ${r.top_predictions.map((t) => `<tr class="clickable" data-s="${esc(t.source)}" data-t="${esc(t.target)}"><td>${esc(t.source)} — ${esc(t.target)}</td><td class="num">${fmt(t.score)}</td>
        <td class="${t.hidden_edge ? "hit" : "miss"}">${t.hidden_edge ? "✓ real edge" : "✗ non-edge"}</td></tr>`).join("")}
      </tbody></table>
    </section>
    <section class="callout">The split is leakage-safe: features for training pairs come from a graph with those edges removed, and test features from a graph that never contained the test edges. Dashed arcs on the graph are the top predictions (<span class="hit">green = a real hidden edge</span>, <span class="miss">red = false positive</span>).</section>`;

  const series = [
    ...heuristics.filter((m) => m.key !== bestH.key).map((m) => ({ ...m, color: css("--axis"), muted: true, width: 1.5 })),
    { ...bestH, label: `${bestH.label} (best heuristic)`, color: categorical(1) },
    { ...model, color: categorical(0), width: 2.5 },
  ].map((m) => ({ key: m.key, label: m.label, color: m.color, muted: m.muted, width: m.width, points: m.roc }));
  legend($("roc-legend"), [
    { color: categorical(0), label: `${model.label} · AUC ${fmt(model.auc)}` },
    { color: categorical(1), label: `${bestH.label} · AUC ${fmt(bestH.auc)}` },
    { color: css("--axis"), label: "Other heuristics" },
  ]);
  lineChart($("roc"), {
    series, height: 240, xDomain: [0, 1], yDomain: [0, 1], diagonal: true, directLabels: false,
    xLabel: "false positive rate", yLabel: "true positive rate", xFormat: d3.format(".0%"), yFormat: d3.format(".0%"),
  });

  const overlay = () => view.setOverlay($("lp-overlay").checked
    ? r.top_predictions.map((t) => ({ source: t.source, target: t.target, cls: t.hidden_edge ? "hit" : "miss" }))
    : []);
  $("lp-overlay").onchange = overlay;
  overlay();
  for (const tr of out.querySelectorAll("tr[data-s]")) {
    tr.onclick = () => {
      const nodes = new Set([tr.dataset.s, tr.dataset.t]);
      view.highlight({ nodes, edges: new Set() });
      setTimeout(() => view.highlight(null), 1500);
    };
  }
}

// ---- GCN

function renderGCN() {
  const el = $("tab-gcn");
  if (!state.data.summary.has_truth) {
    el.innerHTML = `<p class="lede">Semi-supervised node classification needs ground-truth labels.</p>
      <div class="callout">Load <strong>Zachary's Karate Club</strong> or a <strong>Stochastic Block Model</strong>, or paste an edge list with a <code># labels</code> section.</div>`;
    return;
  }
  const c = state.results.gcn?.config || { per_class: 1, hidden: 16, epochs: 200, lr: 0.01, dropout: 0.5, seed: 0 };
  el.innerHTML = `
    <p class="lede">A 2-layer Graph Convolutional Network written from scratch in NumPy (forward pass, hand-derived backprop, Adam). It sees <strong>no node features</strong>, only the graph and a handful of labels.</p>
    <div class="controls-grid">
      <div class="field"><span class="label">Labels / class</span><input type="number" id="g-per" min="1" max="50" value="${c.per_class}"></div>
      <div class="field"><span class="label">Hidden units</span><input type="number" id="g-hidden" min="2" max="64" value="${c.hidden}"></div>
      <div class="field"><span class="label">Epochs</span><input type="number" id="g-epochs" min="10" max="1000" step="10" value="${c.epochs}"></div>
      <div class="field"><span class="label">Learning rate</span><input type="number" id="g-lr" min="0.001" max="1" step="0.005" value="${c.lr}"></div>
      <div class="field"><span class="label">Dropout</span><input type="number" id="g-drop" min="0" max="0.9" step="0.1" value="${c.dropout}"></div>
      <div class="field"><span class="label">Seed</span><input type="number" id="g-seed" min="0" value="${c.seed}"></div>
    </div>
    <button class="btn primary block" id="g-run">Train GCN</button>
    <div id="g-out" style="margin-top:16px"></div>`;
  $("g-run").onclick = (e) => withButton(e.target, runGCN);
  if (state.results.gcn) drawGCN(); else runGCN();
}

async function runGCN() {
  const config = {
    per_class: +$("g-per").value, hidden: +$("g-hidden").value, epochs: +$("g-epochs").value,
    lr: +$("g-lr").value, dropout: +$("g-drop").value, seed: +$("g-seed").value,
  };
  const key = state.specKey;
  $("g-out").innerHTML = `<div class="muted">Training…</div>`;
  try {
    const r = await api("/api/gcn", { dataset: state.spec, ...config });
    if (key !== state.specKey) return;
    r.config = config;
    r.index = new Map(r.nodes.map((id, i) => [id, i]));
    r.labelledIds = new Set(r.nodes.filter((_, i) => r.train_mask[i]));
    state.results.gcn = r;
    $("color-by").value = "gcn";
    if (state.tab === "gcn") { drawGCN(); applyEncoding(); }
  } catch (err) { errorBox($("g-out"), err); }
}

function drawGCN() {
  const r = state.results.gcn;
  const out = $("g-out");
  const bench = r.benchmark;
  const score = (x) => (bench ? bench.scores[x.key].mean : x.test_acc);
  const maxAcc = Math.max(...r.results.map(score));
  out.innerHTML = `
    <section class="table-scroll"><table class="data"><thead><tr><th>Model (${r.n_labelled} labelled nodes)</th><th class="num">This run</th>${bench ? `<th class="num">Mean ± std, ${bench.repeats} splits</th>` : ""}<th></th></tr></thead><tbody>
      ${r.results.map((x) => `<tr class="${score(x) === maxAcc ? "best" : ""}"><td>${esc(x.label)}</td><td class="num">${pct(x.test_acc)}</td>
        ${bench ? `<td class="num">${pct(bench.scores[x.key].mean)} <span class="muted">± ${pct(bench.scores[x.key].std, 0)}</span></td>` : ""}
        <td class="bar-cell"><div class="bar ${x.key === "gcn" ? "" : "muted"}" style="width:${100 * score(x)}%"></div></td></tr>`).join("")}
    </tbody></table>
    ${bench ? `<p class="muted" style="font-size:12px">With ${r.config.per_class} label(s) per class, which nodes get labelled swings accuracy a lot, so bars show the ${bench.repeats}-split mean.</p>` : ""}</section>
    <section><h3 class="section-title">Hidden-layer embedding during training</h3>
      <div class="chart-legend" id="emb-legend"></div>
      <div id="emb"></div>
      <div class="scrubber"><button class="btn" id="emb-play">▶ Play</button><input type="range" id="emb-frame" min="0" max="${r.frames.length - 1}" value="${r.frames.length - 1}"><output id="emb-out"></output></div>
      <p class="muted" style="font-size:12px;margin:0">PCA of the 16-d hidden layer, fixed basis across frames. Colour = true class · ring = labelled for training · ✕ = currently misclassified.</p>
    </section>
    <section><h3 class="section-title">Accuracy</h3><div class="chart-legend" id="acc-legend"></div><div id="acc"></div></section>
    <section><h3 class="section-title">Training loss</h3><div id="loss"></div></section>`;

  const epochs = r.history.map((h) => h.epoch);
  const accSeries = [
    { key: "train", label: "Train (labelled)", shortLabel: "Train", color: categorical(0), points: r.history.map((h) => [h.epoch, h.train_acc]) },
    { key: "test", label: "Test (unlabelled)", shortLabel: "Test", color: categorical(1), points: r.history.map((h) => [h.epoch, h.test_acc]) },
  ];
  legend($("acc-legend"), accSeries);
  lineChart($("acc"), { series: accSeries, height: 170, xDomain: d3.extent(epochs), yDomain: [0, 1], xLabel: "epoch", yFormat: d3.format(".0%") });
  lineChart($("loss"), {
    series: [{ key: "loss", label: "Cross-entropy + L2", color: categorical(0), points: r.history.map((h) => [h.epoch, h.loss]) }],
    height: 140, xDomain: d3.extent(epochs), xLabel: "epoch", directLabels: false, yFormat: d3.format(".2f"),
  });

  // Frame the converged embedding; early, wilder frames are clamped to the edge.
  const all = r.frames[r.frames.length - 1].xy;
  const pad = (ext) => { const d = (ext[1] - ext[0]) * 0.12 || 1; return [ext[0] - d, ext[1] + d]; };
  const sc = scatter($("emb"), { height: 260, xDomain: pad(d3.extent(all, (p) => p[0])), yDomain: pad(d3.extent(all, (p) => p[1])) });
  legend($("emb-legend"), r.classes.slice(0, 8).map((c, i) => ({ color: categorical(i), label: c })), { dots: true });

  const draw = (fi, duration = 0) => {
    const f = r.frames[fi];
    sc.update(r.nodes.map((id, i) => ({
      id, x: f.xy[i][0], y: f.xy[i][1], color: categorical(r.labels[i]), ring: r.train_mask[i], cross: f.pred[i] !== r.labels[i],
      tip: tipRows(id, [["True class", r.classes[r.labels[i]]], ["Predicted", r.classes[f.pred[i]]], ["Labelled", r.train_mask[i] ? "yes" : "no"]], categorical(r.labels[i])),
    })), duration);
    const h = r.history[f.epoch - 1];
    $("emb-out").textContent = `epoch ${f.epoch} · ${pct(h.test_acc, 0)}`;
  };
  const slider = $("emb-frame");
  slider.oninput = () => draw(+slider.value);
  draw(r.frames.length - 1);
  let timer = null;
  $("emb-play").onclick = (e) => {
    if (timer) { clearInterval(timer); timer = null; e.target.textContent = "▶ Play"; return; }
    slider.value = 0;
    draw(0);
    e.target.textContent = "❚❚ Pause";
    timer = setInterval(() => {
      if (!document.body.contains(slider)) { clearInterval(timer); timer = null; return; }
      const next = +slider.value + 1;
      if (next >= r.frames.length) { clearInterval(timer); timer = null; e.target.textContent = "▶ Play"; return; }
      slider.value = next;
      draw(next, 110);
    }, 120);
  };
  view.setRings(r.labelledIds);
}

// ------------------------------------------------------------------ boot

function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("theme"); } catch { /* storage unavailable */ }
  if (saved) document.documentElement.dataset.theme = saved;
  $("theme-toggle").onclick = () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch { /* ignore */ }
    renderTab();
  };
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => renderTab());
}

async function init() {
  initTheme();
  const { datasets, community_algorithms } = await api("/api/datasets");
  state.catalog = datasets;
  state.algorithms = community_algorithms;
  $("dataset").innerHTML = datasets.map((d) => `<option value="${d.key}">${esc(d.name)}</option>`).join("");
  $("community").innerHTML = community_algorithms.map((a) => `<option value="${a.key}">${esc(a.label)}</option>`).join("");
  $("custom-edges").value = "alice,bob\nalice,carol\nbob,carol\ncarol,dave\ndave,erin\ndave,frank\nerin,frank\n# labels\nalice,red\nbob,red\ncarol,red\ndave,blue\nerin,blue\nfrank,blue";

  $("dataset").onchange = () => { renderParams(); if ($("dataset").value !== "custom") loadGraph(); };
  $("load").onclick = () => loadGraph();
  $("community").onchange = () => loadGraph({ keepLayout: true });
  $("size-by").onchange = () => { applyEncoding(); if (state.tab === "centrality") renderCentrality(); };
  $("color-by").onchange = applyEncoding;
  $("show-labels").onchange = applyEncoding;
  $("fit").onclick = () => view.fit();
  $("reheat").onclick = () => view.reheat();
  for (const b of document.querySelectorAll(".tab")) b.onclick = () => switchTab(b.dataset.tab);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") { view.highlight(null); onNodeClick(null); } });

  const hash = new URLSearchParams(location.hash.slice(1));
  if (hash.get("dataset") && datasets.some((d) => d.key === hash.get("dataset"))) $("dataset").value = hash.get("dataset");
  renderParams();
  await loadGraph();
  if (hash.get("tab") && TABS[hash.get("tab")]) switchTab(hash.get("tab"));
}

init().catch((err) => {
  document.querySelector(".panel").innerHTML = `<div class="tab-body"><p class="error">⚠ Failed to start: ${esc(err.message)}</p></div>`;
});
