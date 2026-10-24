// Bo ve bieu do dung chung cho bao cao (khung.html) va slide (slide.html).
// scripts/70_xuat_bao_cao.py chen nguyen van file nay vao cho `/*__VE__*/` BEN TRONG IIFE
// cua moi trang, nen cac ham duoi day nam trong pham vi cua trang do, khong ra global.
  const $ = (s) => document.querySelector(s);
  const f = (x, d = 3) => Number(x).toFixed(d);
  const vn = (n) => Number(n).toLocaleString("vi-VN");
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs = {}, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  };
  const TEN_TP = { mp_sagittal: "chỉ Sagittal", mp_axial: "chỉ Axial", mp_coronal: "chỉ Coronal",
                   chung_s224: "cả 3 mặt phẳng, ViT-S", chung_b336: "cả 3 mặt phẳng, ViT-B" };

  // ---------------- tooltip
  const tip = $("#goi-y");
  const hien = (html, ev) => {
    tip.innerHTML = html; tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let x = ev.clientX + 14, y = ev.clientY + 14;
    if (x + w > innerWidth - 8) x = ev.clientX - w - 14;
    if (y + h > innerHeight - 8) y = ev.clientY - h - 14;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  };
  const an = () => { tip.hidden = true; };
  const dongTip = (mau, ten, gt) => `<div class="dg"><span><i style="background:${mau}"></i>${ten}</span><span class="so">${gt}</span></div>`;

  // ---------------- bảng số kèm theo biểu đồ
  const bangSo = (host, dau, dong) => {
    const d = document.createElement("details"); d.className = "bang-so";
    d.innerHTML = `<summary>Xem bảng số</summary><div class="cuon"><table><thead><tr>${dau.map((h, i) => `<th class="${i ? "so" : ""}">${h}</th>`).join("")}</tr></thead><tbody>${dong.map(r => `<tr>${r.map((c, i) => `<td class="${i ? "so" : ""}">${c}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
    host.after(d);
  };

  // ---------------- thanh ngang (HTML)
  const thanh = (host, rows, { fmt = vn, mau = "var(--chinh)", tipFn } = {}) => {
    const max = Math.max(...rows.map(r => r.v));
    for (const r of rows) {
      const d = document.createElement("div"); d.className = "dong";
      d.innerHTML = `<span class="ten">${r.ten}</span><span class="ray"><span class="dac" style="width:${(r.v / max) * 100}%;background:${mau}"></span></span><span class="gt">${fmt(r.v)}</span>`;
      if (tipFn) { d.addEventListener("mousemove", e => hien(tipFn(r), e)); d.addEventListener("mouseleave", an); }
      host.appendChild(d);
    }
  };

  // ---------------- biểu đồ đường
  const duong = (host, o) => {
    const W = o.w || 720, H = o.h || 280, m = { l: 46, r: o.r ?? 18, t: 14, b: 38 };
    const n = o.n, [y0, y1] = o.y;
    const X = (e) => m.l + ((e - 1) / (n - 1)) * (W - m.l - m.r);
    const Y = (v) => m.t + (1 - (v - y0) / (y1 - y0)) * (H - m.t - m.b);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.aria });
    // lưới + trục y
    for (let v = o.yTick[0]; v <= y1 + 1e-9; v += o.yTick[1]) {
      el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), style: "stroke:var(--ke);stroke-width:1" }, svg);
      el("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end", class: "chu-so" }, svg).textContent = f(v, 2);
    }
    el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, style: "stroke:var(--truc);stroke-width:1" }, svg);
    for (const e of o.xTicks) {
      el("text", { x: X(e), y: H - m.b + 18, "text-anchor": "middle", class: "chu-so" }, svg).textContent = e;
    }
    el("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle", class: "chu" }, svg).textContent = "epoch";
    // dải / vạch dọc
    for (const v of o.vach || []) {
      el("line", { x1: X(v.e), x2: X(v.e), y1: m.t, y2: H - m.b, style: `stroke:${v.mau};stroke-width:1.5;stroke-dasharray:4 4` }, svg);
      el("text", { x: X(v.e) + 6, y: m.t + 12, class: "chu-phu" }, svg).textContent = v.nhan;
    }
    // đường
    for (const s of o.series) {
      const d = s.v.map((v, i) => `${i ? "L" : "M"}${X(i + 1).toFixed(1)},${Y(v).toFixed(1)}`).join("");
      el("path", { d, fill: "none", style: `stroke:${s.mau};stroke-width:${s.day || 2};opacity:${s.mo || 1};stroke-linejoin:round;stroke-linecap:round` }, svg);
      if (s.nhan) {
        const last = s.v.length;
        el("text", { x: X(last) + 6, y: Y(s.v[last - 1]) + 4, class: "chu-phu" }, svg).textContent = s.nhan;
      }
    }
    // điểm nhấn
    for (const pt of o.diem || []) {
      el("circle", { cx: X(pt.e), cy: Y(pt.v), r: pt.r || 5, style: `fill:${pt.mau};stroke:var(--mat);stroke-width:2` }, svg);
      if (pt.nhan) el("text", { x: X(pt.e) + (pt.dx ?? 8), y: Y(pt.v) + (pt.dy ?? -8), "text-anchor": pt.anchor || "start", class: pt.dam ? "chu-dam" : "chu-phu" }, svg).textContent = pt.nhan;
    }
    // lớp rê chuột
    const cross = el("line", { y1: m.t, y2: H - m.b, style: "stroke:var(--muc-3);stroke-width:1", visibility: "hidden" }, svg);
    const vung = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
    vung.addEventListener("mousemove", (ev) => {
      const r = svg.getBoundingClientRect();
      const px = (ev.clientX - r.left) * (W / r.width);
      const e = Math.min(n, Math.max(1, Math.round(1 + (px - m.l) / (W - m.l - m.r) * (n - 1))));
      cross.setAttribute("x1", X(e)); cross.setAttribute("x2", X(e)); cross.setAttribute("visibility", "visible");
      const rows = o.series.filter(s => s.v[e - 1] != null && (s.tip ?? true))
        .map(s => dongTip(s.mau, s.ten, f(s.v[e - 1], 3))).join("");
      hien(`<b>Epoch ${e}</b>${o.tipThem ? o.tipThem(e) : ""}${rows}`, ev);
    });
    vung.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); an(); });
    host.appendChild(svg);
  };

  // ---------------- chấm + khoảng tin cậy (ngang)
  const chamKtc = (host, rows, o = {}) => {
    const W = o.w || 520, hang = o.hang || 26, m = { l: o.l || 128, r: 52, t: 6, b: 30 };
    const H = m.t + m.b + rows.length * hang, [x0, x1] = o.x;
    const X = (v) => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.aria });
    for (let v = o.tick[0]; v <= x1 + 1e-9; v += o.tick[1]) {
      el("line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, style: "stroke:var(--ke);stroke-width:1" }, svg);
      el("text", { x: X(v), y: H - m.b + 18, "text-anchor": "middle", class: "chu-so" }, svg).textContent = f(v, 1);
    }
    if (o.moc != null) el("line", { x1: X(o.moc), x2: X(o.moc), y1: m.t, y2: H - m.b, style: "stroke:var(--muc-3);stroke-width:1;stroke-dasharray:3 3" }, svg);
    rows.forEach((r, i) => {
      const y = m.t + hang * i + hang / 2;
      const g = el("g", {}, svg);
      el("rect", { x: 0, y: y - hang / 2, width: W, height: hang, fill: "transparent" }, g);
      el("text", { x: m.l - 10, y: y + 4, "text-anchor": "end", class: "chu-phu" }, g).textContent = r.ten;
      el("line", { x1: X(r.ci[0]), x2: X(r.ci[1]), y1: y, y2: y, style: "stroke:var(--truc);stroke-width:3;stroke-linecap:round" }, g);
      el("circle", { cx: X(r.v), cy: y, r: 5, style: `fill:${r.mau || "var(--chinh)"};stroke:var(--mat);stroke-width:2` }, g);
      el("text", { x: W - m.r + 8, y: y + 4, class: "chu-so" }, g).textContent = f(r.v, 3);
      g.addEventListener("mousemove", e => hien(o.tip(r), e)); g.addEventListener("mouseleave", an);
    });
    host.appendChild(svg);
  };

  // ---------------- cột dọc nhỏ (so sánh vài giá trị)
  const cot = (host, rows, o) => {
    const W = o.w || 520, H = o.h || 220, m = { l: 46, r: 12, t: 26, b: o.b || 46 };
    const [y0, y1] = o.y, n = rows.length;
    const bw = Math.min(o.bw || 96, (W - m.l - m.r) / n * 0.56);
    const X = (i) => m.l + (W - m.l - m.r) * (i + 0.5) / n;
    const Y = (v) => m.t + (1 - (v - y0) / (y1 - y0)) * (H - m.t - m.b);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.aria });
    for (let v = o.tick[0]; v <= y1 + 1e-9; v += o.tick[1]) {
      el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), style: "stroke:var(--ke);stroke-width:1" }, svg);
      el("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end", class: "chu-so" }, svg).textContent = f(v, 2);
    }
    rows.forEach((r, i) => {
      const g = el("g", {}, svg), x = X(i) - bw / 2, top = Y(r.v), bot = Y(y0);
      const rad = 4;
      el("path", { d: `M${x},${bot}V${top + rad}Q${x},${top} ${x + rad},${top}H${x + bw - rad}Q${x + bw},${top} ${x + bw},${top + rad}V${bot}Z`, style: `fill:${r.mau}` }, g);
      if (r.ci) {
        el("line", { x1: X(i), x2: X(i), y1: Y(r.ci[0]), y2: Y(r.ci[1]), style: "stroke:var(--muc);stroke-width:1.5" }, g);
        for (const c of r.ci) el("line", { x1: X(i) - 6, x2: X(i) + 6, y1: Y(c), y2: Y(c), style: "stroke:var(--muc);stroke-width:1.5" }, g);
      }
      el("text", { x: X(i), y: (r.ci ? Math.min(Y(r.ci[1]), top) : top) - 8, "text-anchor": "middle", class: "chu-dam" }, g).textContent = f(r.v, o.d ?? 4);
      const lines = r.ten.split("\n");
      lines.forEach((t, k) => { el("text", { x: X(i), y: H - m.b + 18 + k * 15, "text-anchor": "middle", class: k ? "chu" : "chu-phu" }, g).textContent = t; });
      el("rect", { x: X(i) - (W - m.l - m.r) / n / 2, y: m.t, width: (W - m.l - m.r) / n, height: H - m.t - m.b, fill: "transparent" }, g);
      if (o.tip) { g.addEventListener("mousemove", e => hien(o.tip(r), e)); g.addEventListener("mouseleave", an); }
    });
    el("line", { x1: m.l, x2: W - m.r, y1: Y(y0), y2: Y(y0), style: "stroke:var(--truc);stroke-width:1" }, svg);
    host.appendChild(svg);
  };

