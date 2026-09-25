// Force-directed network view (D3 v7, SVG).

export const edgeKey = (a, b) => (String(a) < String(b) ? `${a}|${b}` : `${b}|${a}`);

export class GraphView {
  constructor(svg, { onHover, onLeave, onClick } = {}) {
    this.svg = d3.select(svg);
    this.handlers = { onHover, onLeave, onClick };
    this.viewport = this.svg.append("g");
    this.edgeLayer = this.viewport.append("g");
    this.overlayLayer = this.viewport.append("g");
    this.nodeLayer = this.viewport.append("g");
    this.zoom = d3.zoom().scaleExtent([0.05, 12]).on("zoom", (e) => {
      this.viewport.attr("transform", e.transform);
      this.nodeLayer.selectAll("text").attr("font-size", 10 / Math.sqrt(e.transform.k));
    });
    this.svg.call(this.zoom).on("dblclick.zoom", null);
    this.svg.on("click", (e) => { if (e.target === svg) this.handlers.onClick?.(null); });
    this.showLabels = false;
    this.radius = () => 6;
    this.color = () => "#888";
    this.overlay = [];
    this.focus = null;
    new ResizeObserver(() => this._resize()).observe(svg);
  }

  _size() {
    const r = this.svg.node().getBoundingClientRect();
    return [Math.max(r.width, 100), Math.max(r.height, 100)];
  }

  _resize() {
    if (!this.sim) return;
    const [w, h] = this._size();
    this.sim.force("x").x(w / 2);
    this.sim.force("y").y(h / 2);
  }

  load({ nodes, edges }, weighted) {
    this.sim?.stop();
    const [w, h] = this._size();
    const n = nodes.length;
    this.big = n > 400;
    this.radius = () => (n > 400 ? 3 : 6); // caller re-encodes right after load
    this.nodes = nodes.map((d) => ({ ...d, x: w / 2 + (Math.random() - 0.5) * w * 0.5, y: h / 2 + (Math.random() - 0.5) * h * 0.5 }));
    this.byId = new Map(this.nodes.map((d) => [d.id, d]));
    this.edges = edges.map((e) => ({ ...e, key: edgeKey(e.source, e.target) }));
    this.adj = new Map(this.nodes.map((d) => [d.id, new Set()]));
    for (const e of this.edges) {
      this.adj.get(e.source).add(e.target);
      this.adj.get(e.target).add(e.source);
    }

    const wExtent = d3.extent(this.edges, (e) => e.weight);
    const widthScale = weighted && wExtent[0] !== wExtent[1]
      ? d3.scaleSqrt().domain(wExtent).range([0.6, 4])
      : () => (this.big ? 0.6 : 1.1);

    const linkDist = n > 1000 ? 18 : n > 200 ? 26 : 42;
    const charge = n > 1000 ? -18 : n > 200 ? -40 : -140;
    this.sim = d3.forceSimulation(this.nodes)
      .force("link", d3.forceLink(this.edges).id((d) => d.id).distance(linkDist).strength(0.6))
      .force("charge", d3.forceManyBody().strength(charge).distanceMax(n > 400 ? 250 : 600).theta(0.9))
      .force("x", d3.forceX(w / 2).strength(0.06))
      .force("y", d3.forceY(h / 2).strength(0.06))
      .force("collide", d3.forceCollide((d) => this.radius(d) + 1).iterations(1))
      .alphaDecay(n > 400 ? 0.05 : 0.0228)
      .on("tick", () => this._tick());

    // Settle big graphs off-screen so the first frame is already readable.
    if (n > 150) {
      this.sim.stop();
      const ticks = n > 1000 ? 120 : 200;
      for (let i = 0; i < ticks; i++) this.sim.tick();
      this.sim.alpha(0.05).restart();
    }

    this.edgeSel = this.edgeLayer.selectAll("line").data(this.edges, (e) => e.key).join("line")
      .attr("class", "edge").attr("stroke-width", (e) => widthScale(e.weight));
    this.overlayLayer.selectAll("*").remove();
    this.overlay = [];

    const self = this;
    this.nodeSel = this.nodeLayer.selectAll("g.node").data(this.nodes, (d) => d.id).join((enter) => {
      const g = enter.append("g").attr("class", "node");
      g.append("circle");
      g.append("text").attr("dy", "0.32em");
      return g;
    });
    this.nodeSel
      .on("mouseenter", function (event, d) { self._hoverFocus(d.id); self.handlers.onHover?.(event, d); })
      .on("mousemove", (event, d) => self.handlers.onHover?.(event, d))
      .on("mouseleave", () => { self._hoverFocus(null); self.handlers.onLeave?.(); })
      .on("click", (event, d) => { event.stopPropagation(); self.handlers.onClick?.(d); })
      .call(d3.drag()
        .on("start", (event, d) => { if (!event.active) self.sim.alphaTarget(0.2).restart(); d.fx = d.x; d.fy = d.y; })
        .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
        .on("end", (event, d) => { if (!event.active) self.sim.alphaTarget(0); d.fx = null; d.fy = null; }));

    this.restyle();
    this._tick();
    requestAnimationFrame(() => this.fit(0));
    this.sim.on("end.fit", () => { this.sim.on("end.fit", null); this.fit(600); });
    setTimeout(() => this.fit(600), 1000);
  }

  _tick() {
    this.edgeSel
      .attr("x1", (e) => e.source.x).attr("y1", (e) => e.source.y)
      .attr("x2", (e) => e.target.x).attr("y2", (e) => e.target.y);
    this.nodeSel.attr("transform", (d) => `translate(${d.x},${d.y})`);
    this.overlayLayer.selectAll("path").attr("d", (o) => {
      const a = this.byId.get(o.source), b = this.byId.get(o.target);
      if (!a || !b) return null;
      const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
      const dx = b.x - a.x, dy = b.y - a.y;
      return `M${a.x},${a.y} Q${mx - dy * 0.15},${my + dx * 0.15} ${b.x},${b.y}`;
    });
  }

  restyle() {
    if (!this.nodeSel) return;
    this.nodeSel.select("circle").attr("r", (d) => this.radius(d)).attr("fill", (d) => this.color(d));
    // Without "label every node", label only the largest few so the view stays readable.
    const auto = new Set([...this.nodes].sort((a, b) => this.radius(b) - this.radius(a)).slice(0, 10).map((d) => d.id));
    this.nodeSel.select("text")
      .attr("x", (d) => this.radius(d) + 3)
      .text((d) => (this.showLabels || auto.has(d.id) ? d.id : ""));
    this.nodeSel.filter((d) => auto.has(d.id)).raise();
    this.sim?.force("collide", d3.forceCollide((d) => this.radius(d) + 1).iterations(1));
  }

  setEncoding({ radius, color, labels }) {
    if (radius) this.radius = radius;
    if (color) this.color = color;
    if (labels !== undefined) this.showLabels = labels;
    this.restyle();
  }

  /** Emphasise a node subset / edge subset; everything else dims. */
  highlight(sel) {
    this.focus = sel;
    this._applyFocus(sel);
  }

  _hoverFocus(id) {
    if (this.focus) return; // a persistent highlight wins over hover
    if (id === null || this.nodes.length > 1500) return this._applyFocus(null);
    const nodes = new Set([id, ...this.adj.get(id)]);
    const edges = new Set([...this.adj.get(id)].map((nb) => edgeKey(id, nb)));
    this._applyFocus({ nodes, edges, soft: true });
  }

  _applyFocus(sel) {
    if (!this.nodeSel) return;
    this.nodeSel.classed("dim", (d) => !!sel && !sel.nodes.has(d.id));
    this.edgeSel
      .classed("dim", (e) => !!sel && !sel.edges.has(e.key))
      .classed("hl", (e) => !!sel && !sel.soft && sel.edges.has(e.key));
    if (sel && !sel.soft) this.edgeSel.filter((e) => sel.edges.has(e.key)).raise();
  }

  setRemoved(removed) {
    if (!this.nodeSel) return;
    this.nodeSel.classed("removed", (d) => !!removed && removed.has(d.id));
    this.edgeSel.classed("removed", (e) => !!removed && (removed.has(e.source.id) || removed.has(e.target.id)));
  }

  setOverlay(links) {
    this.overlay = links || [];
    this.overlayLayer.selectAll("path").data(this.overlay, (o) => edgeKey(o.source, o.target)).join("path")
      .attr("class", (o) => `overlay-link ${o.cls || ""}`);
    this._tick();
  }

  setRings(ids) {
    if (!this.nodeSel) return;
    this.nodeSel.selectAll("circle.labelled-ring").remove();
    if (!ids) return;
    this.nodeSel.filter((d) => ids.has(d.id)).insert("circle", "text")
      .attr("class", "labelled-ring").attr("r", (d) => this.radius(d) + 3.5);
  }

  select(id) {
    this.nodeSel?.classed("selected", (d) => d.id === id);
  }

  centerOn(id) {
    const d = this.byId?.get(id);
    if (!d) return;
    const [w, h] = this._size();
    const k = Math.max(d3.zoomTransform(this.svg.node()).k, 1.5);
    this.svg.transition().duration(500).call(this.zoom.transform, d3.zoomIdentity.translate(w / 2 - k * d.x, h / 2 - k * d.y).scale(k));
  }

  fit(duration = 500) {
    if (!this.nodes?.length) return;
    const [w, h] = this._size();
    const xs = d3.extent(this.nodes, (d) => d.x), ys = d3.extent(this.nodes, (d) => d.y);
    const pad = 70;
    const k = Math.min(2.5, 0.95 / Math.max((xs[1] - xs[0] + pad) / w, (ys[1] - ys[0] + pad * 2) / h));
    const t = d3.zoomIdentity.translate(w / 2 - k * (xs[0] + xs[1]) / 2, h / 2 - k * (ys[0] + ys[1]) / 2).scale(k);
    (duration ? this.svg.transition().duration(duration) : this.svg).call(this.zoom.transform, t);
  }

  reheat() {
    this.sim?.alpha(0.8).restart();
    this.sim?.on("end.fit", () => { this.sim.on("end.fit", null); this.fit(600); });
  }
}
