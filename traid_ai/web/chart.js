"use strict";

// The viewport owns only presentation. Trading state never depends on it.
class ChartViewport {
  constructor() { this.session = undefined; this.min = 0; this.max = 0; this.start = 0; this.end = 0; this.follow = true; this.range = null; }
  update(points, session) {
    if (this.session !== session) { this.session = session; this.follow = true; this.range = null; }
    this.min = points.length ? Date.parse(points[0].time) : 0;
    this.max = points.length ? Date.parse(points[points.length - 1].time) : 0;
    if (this.follow) {
      this.end = this.max;
      this.start = this.range === null ? this.min : Math.max(this.min, this.max - this.range);
    } else this.set(this.start, this.end);
  }
  set(start, end) {
    const span = Math.min(Math.max(0, end - start), this.max - this.min);
    this.start = Math.max(this.min, Math.min(start, this.max - span));
    this.end = this.start + span;
  }
  preset(range) { this.follow = true; this.range = range; this.end = this.max; this.start = range === null ? this.min : Math.max(this.min, this.max - range); }
  zoom(factor, anchor = .5) {
    const full = this.max - this.min;
    if (full <= 0) return;
    const old = this.end - this.start, minimum = Math.min(full, Math.max(1000, full / 300));
    const span = Math.max(minimum, Math.min(full, old * factor));
    const at = this.start + old * anchor;
    this.set(at - span * anchor, at + span * (1 - anchor));
    this.range = null; this.follow = false;
  }
  pan(delta) { this.set(this.start + delta, this.end + delta); this.range = null; this.follow = false; }
  latest() { const span = this.end - this.start; this.range = span >= this.max - this.min ? null : span; this.follow = true; this.end = this.max; this.start = this.range === null ? this.min : Math.max(this.min, this.max - span); }
}

class PriceChart {
  constructor() {
    this.svg = document.getElementById("price-chart"); this.viewport = new ChartViewport(); this.points = []; this.fills = []; this.drag = null;
    this.width = 800; this.height = 300; this.left = 86; this.right = 16; this.top = 22; this.bottom = 36;
    const get = id => document.getElementById(id);
    for (const button of document.querySelectorAll("[data-chart-range]")) button.addEventListener("click", () => {
      this.viewport.preset(button.dataset.chartRange === "all" ? null : Number(button.dataset.chartRange)); this.draw();
    });
    get("chart-zoom-in").addEventListener("click", () => { this.viewport.zoom(.65); this.draw(); });
    get("chart-zoom-out").addEventListener("click", () => { this.viewport.zoom(1.5); this.draw(); });
    get("chart-live").addEventListener("click", () => { this.viewport.latest(); this.draw(); });
    get("chart-scroll").addEventListener("input", event => {
      const v = this.viewport, span = v.end - v.start, start = v.min + Number(event.target.value) / 1000 * (v.max - v.min - span);
      v.set(start, start + span); v.follow = false; v.range = null; this.draw();
    });
    this.svg.addEventListener("wheel", event => {
      if (!this.points.length) return;
      event.preventDefault(); this.viewport.zoom(event.deltaY < 0 ? .8 : 1.25, this.ratio(event.clientX)); this.draw();
    }, {passive: false});
    this.svg.addEventListener("pointerdown", event => {
      if (event.button !== 0 || !this.points.length) return;
      this.svg.focus({preventScroll:true}); this.svg.setPointerCapture(event.pointerId);
      this.drag = {id:event.pointerId, x:event.clientX, start:this.viewport.start, end:this.viewport.end};
      this.svg.classList.add("dragging");
    });
    this.svg.addEventListener("pointermove", event => {
      if (this.drag) {
        const delta = -(event.clientX - this.drag.x) / this.svg.getBoundingClientRect().width * this.width / (this.width - this.left - this.right) * (this.drag.end - this.drag.start);
        this.viewport.set(this.drag.start + delta, this.drag.end + delta); this.viewport.follow = false; this.viewport.range = null; this.draw();
      }
      this.inspect(event.clientX);
    });
    const finish = () => { this.drag = null; this.svg.classList.remove("dragging"); };
    this.svg.addEventListener("pointerup", finish); this.svg.addEventListener("pointercancel", finish); this.svg.addEventListener("lostpointercapture", finish);
    this.svg.addEventListener("pointerleave", () => { if (!this.drag) this.clearFocus(); });
    this.svg.addEventListener("dblclick", () => { this.viewport.preset(null); this.draw(); });
    this.svg.addEventListener("keydown", event => {
      const v = this.viewport;
      if (["+", "=", "-", "ArrowLeft", "ArrowRight", "Home", "End", "Escape"].includes(event.key)) event.preventDefault();
      else return;
      if (["+", "="].includes(event.key)) v.zoom(.65);
      if (event.key === "-") v.zoom(1.5);
      if (event.key === "ArrowLeft") v.pan(-(v.end - v.start) * .2);
      if (event.key === "ArrowRight") v.pan((v.end - v.start) * .2);
      if (event.key === "Home") v.preset(null);
      if (event.key === "End") v.latest();
      this.draw();
    });
    this.resize = new ResizeObserver(() => this.draw());
    this.resize.observe(this.svg);
  }
  node(tag, attributes, text) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    return node;
  }
  ratio(clientX) { const rect = this.svg.getBoundingClientRect(); return Math.max(0, Math.min(1, ((clientX - rect.left) / rect.width * this.width - this.left) / (this.width - this.left - this.right))); }
  date(value) { return new Date(value).toLocaleString("ru-RU", {day:"2-digit",month:"2-digit",hour:"2-digit",minute:"2-digit",second:"2-digit"}); }
  money(value) { return Number(value).toLocaleString("ru-RU", {minimumFractionDigits:2,maximumFractionDigits:2}) + " $"; }
  update(data) {
    const signature = JSON.stringify([data.selected_id, data.points, data.fills]);
    if (signature === this.signature) return;
    // Keep pointer capture and the initial drag coordinates across polling updates.
    if (data.selected_id !== this.viewport.session) this.drag = null;
    this.signature = signature; this.points = data.points; this.fills = data.fills;
    this.viewport.update(this.points, data.selected_id); this.draw();
  }
  clearFocus() {
    if (this.focus) this.focus.setAttribute("visibility", "hidden");
    document.getElementById("chart-readout").textContent = this.points.length ? "Наведи на график, чтобы увидеть время и цену." : "Наблюдений пока нет.";
  }
  inspect(clientX) {
    if (!this.points.length || !this.focus) return;
    const target = this.viewport.start + this.ratio(clientX) * (this.viewport.end - this.viewport.start);
    const candidates = this.points.filter(p => Date.parse(p.time) >= this.viewport.start && Date.parse(p.time) <= this.viewport.end);
    if (!candidates.length) { this.clearFocus(); return; }
    const nearest = candidates.reduce((a, b) => Math.abs(Date.parse(a.time) - target) < Math.abs(Date.parse(b.time) - target) ? a : b);
    const x = this.x(Date.parse(nearest.time)), y = this.y(Number(nearest.price));
    this.focus.setAttribute("visibility", "visible"); this.focus.replaceChildren(
      this.node("line", {x1:x,x2:x,y1:this.top,y2:this.height-this.bottom,class:"crosshair"}),
      this.node("circle", {cx:x,cy:y,r:4,fill:"#e8eff6"}));
    const trades = this.fills.filter(f => Math.abs(this.x(Date.parse(f.time)) - this.x(target)) < 10 && Date.parse(f.time) >= this.viewport.start && Date.parse(f.time) <= this.viewport.end);
    document.getElementById("chart-readout").textContent = `${this.date(nearest.time)} · ${this.money(nearest.price)}` + trades.map(f => ` · ${f.side === "BUY" ? "Покупка" : "Продажа"} ${f.quantity} BTC по ${this.money(f.price)} (${this.date(f.time)})`).join("");
  }
  draw() {
    const get = id => document.getElementById(id), v = this.viewport, span = v.end - v.start, full = v.max - v.min;
    const rect = this.svg.getBoundingClientRect();
    this.width = Math.max(280, rect.width); this.height = Math.max(200, rect.height);
    this.left = this.width < 500 ? 74 : 86;
    this.svg.setAttribute("viewBox", `0 0 ${this.width} ${this.height}`);
    this.svg.replaceChildren(); this.focus = null; this.clearFocus();
    for (const button of document.querySelectorAll("[data-chart-range]")) {
      button.disabled = !this.points.length;
      button.setAttribute("aria-pressed", String(v.follow && (button.dataset.chartRange === "all" ? v.range === null : v.range === Number(button.dataset.chartRange))));
    }
    get("chart-live").setAttribute("aria-pressed", String(v.follow));
    get("chart-live").disabled = !this.points.length;
    get("chart-zoom-in").disabled = full <= 0 || span <= Math.min(full, Math.max(1000, full / 300));
    get("chart-zoom-out").disabled = span >= full;
    get("chart-scroll").disabled = span >= full;
    get("chart-scroll").value = full > span ? Math.round((v.start - v.min) / (full - span) * 1000) : 1000;
    get("chart-period").textContent = this.points.length ? `${this.date(v.start)} — ${this.date(v.end)} · ${v.follow ? "следим за последними данными" : "просмотр истории"} · до 600 точек за сессию, при длинной истории — выборка` : "";
    if (!this.points.length) return;
    const inside = this.points.filter(p => Date.parse(p.time) >= v.start && Date.parse(p.time) <= v.end);
    const before = this.points.findLast(p => Date.parse(p.time) < v.start), after = this.points.find(p => Date.parse(p.time) > v.end);
    const points = [...(before ? [before] : []), ...inside, ...(after ? [after] : [])];
    const fills = this.fills.filter(f => Date.parse(f.time) >= v.start && Date.parse(f.time) <= v.end);
    const values = [...points.map(p => Number(p.price)), ...fills.map(f => Number(f.price))];
    let low = Math.min(...values), high = Math.max(...values);
    const pad = Math.max((high - low) * .14, high * .0001, .01); low -= pad; high += pad;
    const width = this.width, height = this.height, left = this.left, right = this.right, top = this.top, bottom = this.bottom;
    this.x = value => left + (value - v.start) / Math.max(1, span) * (width - left - right);
    this.y = value => top + (high - value) / (high - low) * (height - top - bottom);
    for (let i = 0; i < 4; i++) {
      const value = low + (high - low) * i / 3, y = this.y(value);
      this.svg.append(this.node("line", {x1:left,x2:width-right,y1:y,y2:y,class:"grid"}), this.node("text", {x:left-8,y:y+3,"text-anchor":"end"}, this.money(value).replace(" $", "")));
    }
    const defs = this.node("defs", {}), clip = this.node("clipPath", {id:"price-clip"});
    clip.append(this.node("rect", {x:left,y:top,width:width-left-right,height:height-top-bottom})); defs.append(clip); this.svg.append(defs);
    const plot = this.node("g", {"clip-path":"url(#price-clip)"});
    const path = "M" + points.map(p => `${this.x(Date.parse(p.time))},${this.y(Number(p.price))}`).join(" L");
    plot.append(this.node("path", {d:`${path} L${this.x(Date.parse(points[points.length-1].time))},${height-bottom} L${this.x(Date.parse(points[0].time))},${height-bottom} Z`,class:"area"}), this.node("path", {d:path,class:"line"}));
    if (points.length === 1) plot.append(this.node("circle", {cx:left,cy:this.y(values[0]),r:3,fill:"#7ce3c1"}));
    for (const fill of fills) {
      const circle = this.node("circle", {cx:this.x(Date.parse(fill.time)),cy:this.y(Number(fill.price)),r:5,fill:fill.side === "BUY" ? "#7ce3c1" : "#edc080",stroke:"#141d26","stroke-width":2});
      circle.append(this.node("title", {}, `${fill.side === "BUY" ? "Покупка" : "Продажа"} ${this.date(fill.time)} · ${this.money(fill.price)}`)); plot.append(circle);
    }
    this.svg.append(plot);
    const tick = value => width < 550 ? new Date(value).toLocaleTimeString("ru-RU", {hour:"2-digit",minute:"2-digit"}) : this.date(value);
    this.svg.append(this.node("text", {x:left,y:height-10}, tick(v.start)), this.node("text", {x:width-right,y:height-10,"text-anchor":"end"}, tick(v.end)));
    this.focus = this.node("g", {visibility:"hidden","pointer-events":"none"}); this.svg.append(this.focus);
  }
}
