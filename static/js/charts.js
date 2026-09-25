// Small D3 chart kit: line (with crosshair tooltip), bar, and animated scatter.
import { showTip, moveTip, hideTip, tipRows, fmt, esc } from "./util.js";

const M = { top: 10, right: 16, bottom: 34, left: 44 };

function frame(el, height, margin = M) {
  const root = d3.select(el).html("").classed("chart", true);
  const width = Math.max(el.clientWidth || 380, 240);
  const svg = root.append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("height", height);
  const iw = width - margin.left - margin.right;
  const ih = height - margin.top - margin.bottom;
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
  return { root, svg, g, iw, ih, width, margin };
}

function axes(g, x, y, iw, ih, { xLabel, yLabel, xFormat, yFormat, xTicks = 5, yTicks = 4 }) {
  g.append("g").attr("class", "grid")
    .call(d3.axisLeft(y).ticks(yTicks).tickSize(-iw).tickFormat(""))
    .call((s) => s.select(".domain").remove());
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`)
    .call(d3.axisBottom(x).ticks(xTicks).tickSizeOuter(0).tickFormat(xFormat || null));
  g.append("g").attr("class", "axis")
    .call(d3.axisLeft(y).ticks(yTicks).tickSize(0).tickPadding(6).tickFormat(yFormat || null))
    .call((s) => s.select(".domain").remove());
  if (xLabel) g.append("text").attr("class", "axis-label").attr("x", iw).attr("y", ih + 29).attr("text-anchor", "end").text(xLabel);
  if (yLabel) g.append("text").attr("class", "axis-label").attr("x", -M.left + 2).attr("y", -14).attr("text-anchor", "start").text(yLabel);
}

export function legend(el, items, { dots = false } = {}) {
  d3.select(el).html(items.map((s) =>
    `<span><i class="${dots ? "dot" : ""}" style="background:${s.color}"></i>${esc(s.label)}</span>`).join(""));
}

/**
 * series: [{key, label, color, points: [[x, y], ...], width?, dash?, muted?}]
 * Muted series draw underneath and are omitted from the tooltip.
 */
export function lineChart(el, { series, height = 220, xDomain, yDomain, xLabel, yLabel, xFormat, yFormat, directLabels = true, diagonal = false, onScrub }) {
  const margin = { ...M, top: yLabel ? 24 : M.top, right: directLabels ? 92 : M.right };
  const { g, iw, ih } = frame(el, height, margin);
  const all = series.flatMap((s) => s.points);
  const x = d3.scaleLinear().domain(xDomain || d3.extent(all, (p) => p[0])).range([0, iw]);
  const y = d3.scaleLinear().domain(yDomain || [0, d3.max(all, (p) => p[1]) || 1]).nice().range([ih, 0]);
  axes(g, x, y, iw, ih, { xLabel, yLabel, xFormat, yFormat });

  if (diagonal) {
    g.append("line").attr("x1", x(0)).attr("y1", y(0)).attr("x2", x(1)).attr("y2", y(1))
      .attr("stroke", "var(--axis)").attr("stroke-dasharray", "4 4");
  }
  const line = d3.line().x((p) => x(p[0])).y((p) => y(p[1]));
  const ordered = [...series.filter((s) => s.muted), ...series.filter((s) => !s.muted)];
  g.append("g").selectAll("path").data(ordered).join("path")
    .attr("fill", "none")
    .attr("stroke", (s) => s.color)
    .attr("stroke-width", (s) => s.width || 2)
    .attr("stroke-opacity", (s) => (s.muted ? 0.45 : 1))
    .attr("stroke-dasharray", (s) => s.dash || null)
    .attr("stroke-linejoin", "round").attr("stroke-linecap", "round")
    .attr("d", (s) => line(s.points));

  if (directLabels) {
    // Label the right end of each emphasised line; nudge apart to avoid collisions.
    const labels = series.filter((s) => !s.muted && s.points.length).map((s) => {
      const last = s.points[s.points.length - 1];
      return { s, y: y(last[1]), x: x(last[0]) };
    }).sort((a, b) => a.y - b.y);
    for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + 12);
    if (labels.length) {
      g.append("g").selectAll("text").data(labels).join("text").attr("class", "series-label")
        .attr("x", iw + 6).attr("y", (l) => l.y).attr("dy", "0.32em").text((l) => l.s.shortLabel || l.s.label);
    }
  }

  const cross = g.append("line").attr("class", "crosshair").attr("y1", 0).attr("y2", ih).attr("visibility", "hidden");
  const dots = g.append("g");
  const marker = g.append("line").attr("class", "marker-line").attr("y1", 0).attr("y2", ih).attr("visibility", "hidden");
  const tipSeries = series.filter((s) => !s.muted);
  const bisect = d3.bisector((p) => p[0]).center;

  g.append("rect").attr("width", iw).attr("height", ih).attr("fill", "transparent")
    .on("mousemove", (event) => {
      const [mx] = d3.pointer(event);
      const xv = x.invert(mx);
      cross.attr("x1", mx).attr("x2", mx).attr("visibility", "visible");
      const hits = tipSeries.map((s) => ({ s, p: s.points[bisect(s.points, xv)] })).filter((h) => h.p);
      dots.selectAll("circle").data(hits).join("circle").attr("r", 4)
        .attr("cx", (h) => x(h.p[0])).attr("cy", (h) => y(h.p[1]))
        .attr("fill", (h) => h.s.color).attr("stroke", "var(--surface)").attr("stroke-width", 2);
      const head = xLabel ? `${xLabel}: ${(xFormat || fmt)(hits[0]?.p[0] ?? xv)}` : fmt(xv);
      showTip(`<div class="t">${esc(head)}</div>` + hits.map((h) =>
        `<div class="r"><span><span class="sw" style="background:${h.s.color}"></span>${esc(h.s.label)}</span><b>${esc((yFormat || fmt)(h.p[1]))}</b></div>`).join(""), event);
      onScrub?.(xv, event);
    })
    .on("mouseleave", () => { cross.attr("visibility", "hidden"); dots.selectAll("*").remove(); hideTip(); });

  return {
    setMarker(xv) {
      if (xv === null || xv === undefined) return marker.attr("visibility", "hidden");
      marker.attr("x1", x(xv)).attr("x2", x(xv)).attr("visibility", "visible");
    },
  };
}

/** data: [{x, y}] — discrete bars (e.g. a degree histogram). */
export function barChart(el, { data, height = 180, xLabel, yLabel, color = "var(--s1)", tip, logY = false }) {
  const { g, iw, ih } = frame(el, height, { ...M, top: yLabel ? 24 : M.top });
  const x = d3.scaleBand().domain(data.map((d) => d.x)).range([0, iw]).paddingInner(data.length > 60 ? 0 : 0.15);
  const maxY = d3.max(data, (d) => d.y) || 1;
  const y = logY
    ? d3.scaleSymlog().domain([0, maxY]).range([ih, 0])
    : d3.scaleLinear().domain([0, maxY]).nice().range([ih, 0]);
  const every = Math.ceil(data.length / 8);
  const xAxis = d3.scaleBand().domain(x.domain()).range(x.range());
  g.append("g").attr("class", "grid")
    .call(d3.axisLeft(y).ticks(4).tickSize(-iw).tickFormat(""))
    .call((s) => s.select(".domain").remove());
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`)
    .call(d3.axisBottom(xAxis).tickValues(x.domain().filter((_, i) => i % every === 0)).tickSizeOuter(0));
  g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4, "~s").tickSize(0).tickPadding(6)).call((s) => s.select(".domain").remove());
  if (xLabel) g.append("text").attr("class", "axis-label").attr("x", iw).attr("y", ih + 29).attr("text-anchor", "end").text(xLabel);
  if (yLabel) g.append("text").attr("class", "axis-label").attr("x", -M.left + 2).attr("y", -14).text(yLabel);

  const bw = x.bandwidth();
  const r = Math.min(4, bw / 2);
  // Rounded data-end, square baseline.
  const barPath = (d) => {
    const x0 = x(d.x), x1 = x0 + Math.max(bw - (bw > 3 ? 1 : 0), 1), y0 = ih, y1 = y(d.y);
    const rr = Math.min(r, (y0 - y1) / 2);
    return `M${x0},${y0}V${y1 + rr}Q${x0},${y1} ${x0 + rr},${y1}H${x1 - rr}Q${x1},${y1} ${x1},${y1 + rr}V${y0}Z`;
  };
  g.append("g").selectAll("path").data(data.filter((d) => d.y > 0)).join("path").attr("d", barPath).attr("fill", color);
  // Hit targets span the full column height, wider than the mark.
  g.append("g").selectAll("rect").data(data).join("rect")
    .attr("x", (d) => x(d.x)).attr("width", Math.max(bw, 2)).attr("y", 0).attr("height", ih).attr("fill", "transparent")
    .on("mouseenter mousemove", (event, d) => showTip(tip ? tip(d) : tipRows(String(d.x), [["Value", fmt(d.y)]]), event))
    .on("mouseleave", hideTip);
}

/** Scatter whose points can be re-positioned with a transition (for training animations). */
export function scatter(el, { height = 260, xDomain, yDomain }) {
  const margin = { top: 10, right: 10, bottom: 10, left: 10 };
  const { g, iw, ih } = frame(el, height, margin);
  g.append("rect").attr("width", iw).attr("height", ih).attr("fill", "none").attr("stroke", "var(--grid)").attr("rx", 6);
  const x = d3.scaleLinear().domain(xDomain).range([8, iw - 8]).clamp(true);
  const y = d3.scaleLinear().domain(yDomain).range([ih - 8, 8]).clamp(true);
  const layer = g.append("g");
  return {
    update(points, duration = 0) {
      const sel = layer.selectAll("g.pt").data(points, (p) => p.id).join((enter) => {
        const e = enter.append("g").attr("class", "pt");
        e.append("circle").attr("class", "ring");
        e.append("circle").attr("class", "dot");
        e.append("path").attr("class", "cross");
        return e;
      });
      sel.on("mouseenter mousemove", (event, p) => showTip(p.tip, event)).on("mouseleave", hideTip);
      const t = duration ? sel.transition().duration(duration).ease(d3.easeLinear) : sel;
      t.attr("transform", (p) => `translate(${x(p.x)},${y(p.y)})`);
      sel.select(".dot").attr("r", 5).attr("fill", (p) => p.color).attr("stroke", "var(--surface)").attr("stroke-width", 1.5);
      sel.select(".ring").attr("r", 8.5).attr("fill", "none").attr("stroke", "var(--ink)").attr("stroke-width", 1.5)
        .attr("visibility", (p) => (p.ring ? "visible" : "hidden"));
      sel.select(".cross").attr("d", "M-3.5,-3.5L3.5,3.5M3.5,-3.5L-3.5,3.5").attr("stroke", "var(--surface)").attr("stroke-width", 1.8)
        .attr("visibility", (p) => (p.cross ? "visible" : "hidden"));
    },
  };
}

export { moveTip };
