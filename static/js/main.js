/* ArtSeek project page: demo replay, charts and small interactions. */
(() => {
  "use strict";

  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const esc = (t) => String(t ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  // Minimal markdown for model outputs: escape, then **bold**.
  const md = (t) => esc(t).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/^#+\s*/gm, "");
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ── theme ─────────────────────────────────────────────────────────── */
  const root = document.documentElement;
  try {
    const saved = localStorage.getItem("artseek-theme");
    if (saved) root.dataset.theme = saved;
  } catch (e) { /* storage unavailable */ }
  $("#themeToggle").addEventListener("click", () => {
    const dark = root.dataset.theme
      ? root.dataset.theme === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("artseek-theme", root.dataset.theme); } catch (e) { /* ignore */ }
    drawCharts();
  });

  /* ── hero pipeline ─────────────────────────────────────────────────── */
  const pnodes = $$("#pipeline .pnode"), parrows = $$("#pipeline .parrow");
  let pstep = 0;
  function tickPipeline() {
    pnodes.forEach((n, i) => n.classList.toggle("on", i <= pstep));
    parrows.forEach((a, i) => a.classList.toggle("on", i < pstep));
    pstep = (pstep + 1) % (pnodes.length + 2);
  }
  tickPipeline();
  if (!reduceMotion) setInterval(tickPipeline, 900);
  else { pstep = pnodes.length; tickPipeline(); }

  /* ── counters ──────────────────────────────────────────────────────── */
  function fmtCount(el, v) {
    const dec = +(el.dataset.dec || 0);
    let s = dec ? v.toFixed(dec) : Math.round(v).toString();
    if (el.dataset.sep) s = Math.round(v).toLocaleString("en-US");
    el.innerHTML = (el.dataset.prefix || "") + s + (el.dataset.suffix || "");
  }
  const counterObs = new IntersectionObserver((entries) => {
    entries.forEach((en) => {
      if (!en.isIntersecting) return;
      const el = en.target, target = +el.dataset.count, t0 = performance.now(), dur = reduceMotion ? 0 : 1400;
      counterObs.unobserve(el);
      const step = (t) => {
        const p = dur ? Math.min(1, (t - t0) / dur) : 1;
        fmtCount(el, target * (1 - Math.pow(1 - p, 3)));
        if (p < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
  }, { threshold: 0.4 });
  $$("[data-count]").forEach((el) => counterObs.observe(el));

  /* ── tabs ──────────────────────────────────────────────────────────── */
  $$("[data-tabs]").forEach((bar) => {
    const panels = $(`[data-panels="${bar.dataset.tabs}"]`);
    $$(".tab", bar).forEach((btn) => btn.addEventListener("click", () => {
      $$(".tab", bar).forEach((b) => b.classList.toggle("active", b === btn));
      $$(".tab-panel", panels).forEach((p) => p.classList.toggle("active", p.id === btn.dataset.tab));
      drawCharts();
    }));
  });

  /* ── retrieval animation (method section) ──────────────────────────── */
  const grid1 = $("#dotgrid"), grid2 = $("#dotgrid2"), ranked = $("#ranked");
  grid1.innerHTML = "<i></i>".repeat(30 * 10);
  grid2.innerHTML = "<i></i>".repeat(10 * 10);
  ranked.innerHTML = Array.from({ length: 10 }, (_, i) => `<i style="width:${100 - i * 7}%"></i>`).join("");
  async function retrievalLoop() {
    const d1 = $$("i", grid1), d2 = $$("i", grid2), r = $$("i", ranked);
    for (;;) {
      [...d1, ...d2].forEach((d) => d.classList.remove("lit")); r.forEach((d) => d.classList.remove("on"));
      await sleep(600);
      const picks = new Set(); while (picks.size < 18) picks.add(Math.floor(Math.random() * d1.length));
      for (const i of picks) { d1[i].classList.add("lit"); await sleep(45); }
      for (let i = 0; i < d2.length; i++) { d2[i].classList.add("lit"); if (i % 10 === 9) await sleep(60); }
      await sleep(300);
      for (const x of r) { x.classList.add("on"); await sleep(110); }
      await sleep(2200);
    }
  }
  if (!reduceMotion) retrievalLoop(); else $$("i", ranked).forEach((x) => x.classList.add("on"));

  /* ── demo ──────────────────────────────────────────────────────────── */
  const EX = window.EXAMPLES || [];
  // Fragments whose role is documented in our qualitative analysis.
  const KEY_DOCS = {
    "14954_visual": { idx: 4082149, label: "the fragment that answered it" },
    "10640_visual": { idx: 5819078, label: "the painting's own page" },
    "193_visual": { idx: 5768413, label: "the fragment that answered it" },
    "8181_contextual": { idx: 155929, label: "misleading: another Bellotto, in Moscow" },
  };
  const chat = $("#chat"), gallery = $("#gallery");
  let runId = 0, skipping = false;

  gallery.innerHTML = EX.map((e, i) => `
    <button class="thumb" data-i="${i}" aria-label="${esc(e.title)}">
      <span class="tag ${e.kind}">${e.kind === "success" ? "success" : "failure"}</span>
      <img src="${e.image}" alt="" loading="lazy">
      <span>${esc(e.title)}</span>
    </button>`).join("");
  $$(".thumb", gallery).forEach((b) => b.addEventListener("click", () => play(+b.dataset.i)));
  $("#replayBtn").addEventListener("click", () => play(current));
  $("#skipBtn").addEventListener("click", () => { skipping = true; });

  const scoreBadge = (s) => s == null ? "" :
    `<span class="score s${s}">${["✗ wrong", "◐ partially correct", "✓ correct"][s]} · ${s}/2</span>`;

  function setStep(s) { $$("#steps span").forEach((x) => x.classList.toggle("on", +x.dataset.s <= s)); }
  function add(html, cls) {
    const d = document.createElement("div");
    d.className = "msg " + cls; d.innerHTML = html; chat.appendChild(d);
    chat.scrollTop = chat.scrollHeight; return d;
  }
  const wait = async (ms, id) => { if (!skipping && !reduceMotion) await sleep(ms); return id === runId; };

  let current = 0;
  async function play(i) {
    const e = EX[i], id = ++runId; skipping = false; current = i;
    $$(".thumb", gallery).forEach((b) => b.classList.toggle("active", +b.dataset.i === i));
    chat.innerHTML = ""; setStep(-1);

    // side panel
    $("#baseAnswer").innerHTML = md(e.backbone.answer || "(no answer)");
    $("#baseScore").innerHTML = scoreBadge(e.backbone.score);
    $("#refAnswer").textContent = e.reference;
    const note = $("#exampleNote");
    note.className = "side-card note " + e.kind;
    note.innerHTML = `<h4>${e.kind === "success" ? "Why it works" : "What went wrong"}</h4><p>${esc(e.note)}</p>`;

    // 0. question
    setStep(0);
    add(`<div class="who">User</div><img src="${e.image}" alt="${esc(e.title)}"><div><i>${esc(e.title)}</i>, ${esc(e.artist)}</div><div style="margin-top:.35rem"><b>${esc(e.question)}</b></div>`, "user");
    if (!await wait(900, id)) return;

    // 1. artwork card
    setStep(1);
    const c = e.artseek.card;
    const rows = ["artist", "genre", "style", "media", "tag"].filter((t) => c[t] && c[t].length).map((t) =>
      `<div class="card-row"><span class="task">${t}</span><div class="bars">${c[t].map(([l, p]) =>
        `<div class="bar"><div class="fill" data-w="${Math.round(p * 100)}"></div><div class="lbl"><span>${esc(l)}</span><span>${Math.round(p * 100)}%</span></div></div>`).join("")}</div></div>`).join("");
    const card = add(`<div class="who"><i class="fa-solid fa-tags"></i> Artwork card · LICN</div>${rows}<div class="card-note">Predicted from the image alone. Top media and tags shown.</div>`, "card");
    requestAnimationFrame(() => $$(".fill", card).forEach((f) => { f.style.width = f.dataset.w + "%"; }));
    if (!await wait(1400, id)) return;

    // 2. retrieval
    setStep(2);
    const tool = add(`<div class="who"><i class="fa-solid fa-magnifying-glass"></i> get_relevant_documents</div>Searching WikiFragments with ColQwen2 late interaction… <span class="muted">(${e.artseek.tool_calls} call${e.artseek.tool_calls === 1 ? "" : "s"})</span><div class="search-anim">${"<i></i>".repeat(200)}</div>`, "tool");
    const dots = $$(".search-anim i", tool);
    if (!skipping && !reduceMotion) {
      for (let k = 0; k < 26; k++) {
        dots[Math.floor(Math.random() * dots.length)].classList.add("lit");
        if (!await wait(40, id)) return;
      }
    }
    for (let k = 0; k < 6; k++) dots[Math.floor(Math.random() * dots.length)].classList.add("hit");
    const docsEl = document.createElement("div"); docsEl.className = "docs"; chat.appendChild(docsEl);
    const key = KEY_DOCS[e.id];
    for (const [r, d] of e.artseek.docs.entries()) {
      const isKey = key && key.idx === d.idx;
      const el = document.createElement("div");
      el.className = "doc" + (d.image ? "" : " noimg") + (isKey ? " relevant" : "");
      el.innerHTML = `<span class="rank">#${r + 1}</span>${d.image ? `<img src="${d.image}" alt="" loading="lazy">` : ""}
        <div><div class="dtitle">${d.url ? `<a href="${esc(d.url)}" target="_blank" rel="noopener">${esc(d.title)}</a>` : esc(d.title)}${isKey ? ` <span class="score s${e.kind === "success" ? 2 : 0}" style="margin:0 0 0 .3rem">${esc(key.label)}</span>` : ""}</div>
        <div class="dtext">${esc(d.text)}</div></div>`;
      docsEl.appendChild(el); chat.scrollTop = chat.scrollHeight;
      if (!await wait(380, id)) return;
    }
    if (!await wait(700, id)) return;

    // 3. reasoning
    setStep(3);
    const reasoning = (e.artseek.reasoning || "").trim();
    if (reasoning && reasoning !== (e.artseek.answer || "").trim()) {
      add(`<details open><summary><i class="fa-solid fa-brain"></i> Reasoning</summary><div style="white-space:pre-line;margin-top:.35rem">${md(reasoning)}</div></details>`, "think");
      if (!await wait(1500, id)) return;
    }

    // 4. answer
    setStep(4);
    const ans = add(`<div class="who"><i class="fa-solid fa-palette"></i> ArtSeek</div><div class="atext caret" style="white-space:pre-line"></div>`, "answer");
    const target = $(".atext", ans), text = e.artseek.answer || "";
    if (skipping || reduceMotion) target.textContent = text;
    else {
      for (let k = 0; k < text.length; k += 3) {
        target.textContent = text.slice(0, k + 3);
        chat.scrollTop = chat.scrollHeight;
        if (!await wait(12, id)) return;
      }
    }
    target.classList.remove("caret");
    target.innerHTML = md(text);
    ans.insertAdjacentHTML("beforeend", scoreBadge(e.artseek.score));
    chat.scrollTop = chat.scrollHeight;
  }

  // Start the demo when it scrolls into view.
  const demoObs = new IntersectionObserver((en) => {
    if (en[0].isIntersecting) { demoObs.disconnect(); play(0); }
  }, { threshold: 0.25 });
  if (EX.length) demoObs.observe($("#demo"));

  /* ── charts (plain SVG) ────────────────────────────────────────────── */
  const R = window.RESULTS;
  const tip = document.createElement("div"); tip.className = "tooltip"; document.body.appendChild(tip);
  function bindTips(svg) {
    $$("[data-tip]", svg).forEach((el) => {
      el.addEventListener("mousemove", (ev) => {
        tip.textContent = el.dataset.tip; tip.style.opacity = 1;
        tip.style.left = ev.clientX + 12 + "px"; tip.style.top = ev.clientY - 30 + "px";
      });
      el.addEventListener("mouseleave", () => { tip.style.opacity = 0; });
    });
  }
  const cssVar = (n) => getComputedStyle(root).getPropertyValue(n).trim();
  const f3 = (x) => (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(3);

  function barGroups(el, { groups, series, max, height = 300, fmt = (v) => v.toFixed(3), showValues = false }) {
    if (!el || !el.offsetParent) return;
    const W = Math.max(el.clientWidth, 320), H = height, m = { t: 12, r: 10, b: 46, l: 40 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b, gw = iw / groups.length, bw = Math.min(34, (gw * 0.75) / series.length);
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img">`;
    for (let k = 0; k <= 4; k++) {
      const v = (max * k) / 4, y = m.t + ih - (v / max) * ih;
      s += `<line class="axis" x1="${m.l}" x2="${W - m.r}" y1="${y}" y2="${y}"/><text x="${m.l - 6}" y="${y + 4}" text-anchor="end">${max < 3 ? v.toFixed(2) : +v.toFixed(1)}</text>`;
    }
    groups.forEach((g, gi) => {
      const gx = m.l + gi * gw + (gw - bw * series.length) / 2;
      series.forEach((se, si) => {
        const v = g.values[si]; if (v == null) return;
        const h = (v / max) * ih, x = gx + si * bw, y = m.t + ih - h;
        s += `<rect x="${x + 2}" y="${y}" width="${bw - 4}" height="${h}" rx="4" fill="${se.color}" data-tip="${g.label} · ${se.name}: ${fmt(v)}"><animate attributeName="height" from="0" to="${h}" dur=".7s" fill="freeze"/><animate attributeName="y" from="${m.t + ih}" to="${y}" dur=".7s" fill="freeze"/></rect>`;
        if (showValues) s += `<text x="${x + bw / 2}" y="${y - 6}" text-anchor="middle" style="font-weight:600;fill:${cssVar("--ink")}">${fmt(v).split(" ")[0]}</text>`;
        if (g.best === si) s += `<text x="${x + bw / 2}" y="${y - 4}" text-anchor="middle" style="font-weight:600;fill:${se.color}">★</text>`;
      });
      const lines = String(g.label).split("\n");
      lines.forEach((ln, li) => { s += `<text x="${m.l + gi * gw + gw / 2}" y="${H - m.b + 18 + li * 14}" text-anchor="middle" style="fill:${cssVar("--ink")};font-weight:${li ? 400 : 500}">${esc(ln)}</text>`; });
    });
    el.innerHTML = s + "</svg>"; bindTips(el);
  }

  function ciPlot(el, { rows, min, max, height }) {
    if (!el || !el.offsetParent) return;
    const W = Math.max(el.clientWidth, 320), rowH = 34, lw = 120, H = height || rows.length * rowH + 40, m = { t: 10, r: 20, b: 26 };
    const x = (v) => lw + ((v - min) / (max - min)) * (W - lw - m.r);
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img">`;
    s += `<line class="zero" x1="${x(0)}" x2="${x(0)}" y1="${m.t}" y2="${H - m.b}"/>`;
    for (let k = 0; k <= 4; k++) { const v = min + ((max - min) * k) / 4; s += `<text x="${x(v)}" y="${H - 8}" text-anchor="middle">${f3(v).replace("000", "0")}</text>`; }
    rows.forEach((r, i) => {
      const y = m.t + i * rowH + rowH / 2, sig = r.lo > 0 || r.hi < 0, col = sig ? cssVar("--good") : cssVar("--neutral");
      s += `<text x="${lw - 8}" y="${y + 4}" text-anchor="end" style="fill:${cssVar("--ink")}">${esc(r.label)}</text>`;
      s += `<g data-tip="${r.label}: ${f3(r.d)} [${f3(r.lo)}, ${f3(r.hi)}]"><line x1="${x(r.lo)}" x2="${x(r.hi)}" y1="${y}" y2="${y}" stroke="${col}" stroke-width="3" stroke-linecap="round"/><circle cx="${x(r.d)}" cy="${y}" r="6" fill="${col}"/><rect x="${lw}" y="${y - rowH / 2}" width="${W - lw}" height="${rowH}" fill="transparent"/></g>`;
    });
    el.innerHTML = s + "</svg>"; bindTips(el);
  }

  function coverageScatter(el) {
    if (!el || !el.offsetParent) return;
    const W = Math.max(el.clientWidth, 320), H = 360, m = { t: 20, r: 30, b: 46, l: 56 };
    const x = (v) => m.l + (v / 100) * (W - m.l - m.r), y = (v) => m.t + ((0.3 - v) / 0.45) * (H - m.t - m.b);
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img">`;
    for (let v = 0; v <= 100; v += 20) s += `<line class="axis" x1="${x(v)}" x2="${x(v)}" y1="${m.t}" y2="${H - m.b}"/><text x="${x(v)}" y="${H - m.b + 16}" text-anchor="middle">${v}%</text>`;
    for (const v of [-0.1, 0, 0.1, 0.2, 0.3]) s += `<text x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end">${f3(v).replace(/0+$/, "").replace(/\.$/, "")}</text>`;
    s += `<line class="zero" x1="${m.l}" x2="${W - m.r}" y1="${y(0)}" y2="${y(0)}"/>`;
    s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle" style="fill:${cssVar("--ink")}">knowledge-base coverage of the benchmark's paintings</text>`;
    s += `<text transform="translate(14 ${(m.t + H - m.b) / 2}) rotate(-90)" text-anchor="middle" style="fill:${cssVar("--ink")}">ArtSeek − backbone</text>`;
    const low = R.vqa.filter((r) => r.cov < 30).sort((a, b) => b.d - a.d);
    R.vqa.forEach((r) => {
      const sig = r.lo > 0, col = sig ? cssVar("--good") : cssVar("--neutral");
      s += `<g data-tip="${r.bench}: coverage ${r.cov}%, Δ ${f3(r.d)} [${f3(r.lo)}, ${f3(r.hi)}]">
        <line x1="${x(r.cov)}" x2="${x(r.cov)}" y1="${y(r.lo)}" y2="${y(r.hi)}" stroke="${col}" stroke-width="3" stroke-linecap="round"/>
        <circle cx="${x(r.cov)}" cy="${y(r.d)}" r="7" fill="${col}" stroke="${cssVar("--surface")}" stroke-width="2"/></g>`;
      const li = low.indexOf(r);
      if (li >= 0) {
        const lx = x(22), ly = y(0.075) + li * 18;
        s += `<line x1="${x(r.cov) + 6}" y1="${y(r.d)}" x2="${lx - 4}" y2="${ly - 4}" stroke="${cssVar("--line")}"/>
          <text x="${lx}" y="${ly}" style="fill:${cssVar("--ink")};font-weight:500">${r.bench}</text>`;
      } else {
        s += `<text x="${x(r.cov) + (r.cov < 60 ? 12 : -12)}" y="${y(r.d) + 4}" text-anchor="${r.cov < 60 ? "start" : "end"}" style="fill:${cssVar("--ink")};font-weight:500">${r.bench}</text>`;
      }
    });
    el.innerHTML = s + "</svg>"; bindTips(el);
  }

  function drawCharts() {
    const C = { base: cssVar("--c-base"), nocls: cssVar("--c-nocls"), full: cssVar("--c-full"), gpt: cssVar("--c-gpt") };
    $("#vqaLegend").innerHTML = `<span style="--c:${C.base}">Qwen2.5-VL-32B</span><span style="--c:${C.nocls}">ArtSeek (w/o class.)</span><span style="--c:${C.full}">ArtSeek</span><span class="muted">★ best in row</span>`;
    barGroups($("#vqaChart"), {
      max: 1.6, height: 320,
      series: [{ name: "Qwen2.5-VL-32B", color: C.base }, { name: "ArtSeek (w/o class.)", color: C.nocls }, { name: "ArtSeek", color: C.full }],
      groups: R.vqa.map((r) => {
        const v = [r.base, r.nocls, r.full];
        return { label: `${r.bench}\nΔ ${f3(r.d)} · cov. ${r.cov}%`, values: v, best: v.indexOf(Math.max(...v)) };
      }),
    });
    coverageScatter($("#covChart"));
    barGroups($("#bbChart"), {
      max: 1, height: 280,
      series: [{ name: "backbone alone", color: C.base }, { name: "with ArtSeek retrieval", color: C.full }],
      groups: R.backbones.map((b) => ({ label: `${b.name}\n${f3(b.d)} [${f3(b.lo)}, ${f3(b.hi)}]`, values: [b.base, b.full] })),
    });
    ciPlot($("#policyChart"), { rows: R.policy.map((p) => ({ label: p.bench, ...p })), min: -0.05, max: 0.25 });
    barGroups($("#latChart"), {
      max: 40, height: 260, fmt: (v) => v.toFixed(1) + " s",
      series: [{ name: "median", color: C.full }, { name: "p95", color: C.base }],
      groups: R.latency.map((l) => ({ label: l.name.replace(", ", "\n").replace(" (w/o class.)", "\n(w/o class.)"), values: [l.median, l.p95] })),
    });
    const tier = $("#tierSeg .active").dataset.tier, h = R.human[tier];
    barGroups($("#humanChart"), {
      max: 6, height: 260, fmt: (v) => v.toFixed(2) + " / 6", showValues: true,
      series: [{ name: "Σ-correctness", color: C.full }],
      groups: Object.entries(h).map(([k, v]) => ({ label: k, values: [v] })),
    });
    $$("#humanChart rect").forEach((r, i) => r.setAttribute("fill", [C.base, C.full, C.gpt][i]));
  }
  $$("#tierSeg button").forEach((b) => b.addEventListener("click", () => {
    $$("#tierSeg button").forEach((x) => x.classList.toggle("active", x === b)); drawCharts();
  }));
  let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(drawCharts, 150); });
  drawCharts();

  /* ── blind test ────────────────────────────────────────────────────── */
  const HU = window.HUMAN || [];
  const NAMES = { base: "Qwen2.5-VL-32B", artseek_multiquery: "ArtSeek", gpt: "GPT-5.5" };
  let hi = 0;
  $("#blindSeg").innerHTML = HU.map((h, i) => `<button class="${i ? "" : "active"}" data-i="${i}">Painting ${i + 1}</button>`).join("");
  $$("#blindSeg button").forEach((b) => b.addEventListener("click", () => {
    $$("#blindSeg button").forEach((x) => x.classList.toggle("active", x === b)); showBlind(+b.dataset.i);
  }));
  function showBlind(i) {
    hi = i; const h = HU[i];
    $("#blindImg").src = h.image; $("#blindImg").alt = h.title;
    $("#blindTitle").innerHTML = `<i>${esc(h.title)}</i><br><span class="muted">${esc(h.artist)}, ${esc(h.year)}</span>`;
    const order = i % 2 ? ["gpt", "artseek_multiquery", "base"] : ["base", "gpt", "artseek_multiquery"];
    $("#blindOptions").innerHTML = order.map((sys, k) =>
      `<button class="bopt" data-sys="${sys}"><span class="blabel">${"ABC"[k]}</span>${md(h.answers[sys])}</button>`).join("");
    $("#blindResult").innerHTML = `<span class="muted">Read the three descriptions and click the one you think ArtSeek wrote.</span>`;
    $$("#blindOptions .bopt").forEach((b) => b.addEventListener("click", () => reveal(b)));
  }
  function reveal(chosen) {
    const h = HU[hi];
    $$("#blindOptions .bopt").forEach((b) => {
      const sys = b.dataset.sys;
      b.classList.add("reveal");
      if (sys === "artseek_multiquery") b.classList.add("is-artseek");
      else if (b === chosen) b.classList.add("wrong");
      if (!b.querySelector(".bsys")) b.insertAdjacentHTML("afterbegin", `<span class="bsys">${NAMES[sys]} · raters' Σ ${h.sigma[sys].toFixed(2)} / 6</span>`);
      b.scrollTop = 0;
    });
    const ok = chosen.dataset.sys === "artseek_multiquery";
    $("#blindResult").innerHTML = `${ok ? "✓ Correct." : "✗ Not this one."} ${esc(h.note)}`;
  }
  if (HU.length) showBlind(0);

  /* ── BibTeX ────────────────────────────────────────────────────────── */
  $("#copyBib").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText($("#bibText").textContent); $("#copyBib").textContent = "Copied!"; }
    catch (e) {
      const r = document.createRange(); r.selectNodeContents($("#bibText"));
      const sel = getSelection(); sel.removeAllRanges(); sel.addRange(r); $("#copyBib").textContent = "Selected";
    }
    setTimeout(() => { $("#copyBib").textContent = "Copy"; }, 1500);
  });
})();
