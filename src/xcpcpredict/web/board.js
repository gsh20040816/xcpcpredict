// Board overlay: fetches the contest's public data, runs XP.run in the page and draws p_ac on every pending
// cell plus medal odds on every team. Supports board.xcpcio.com and the CSG-OJ outrank at rank.iccpc.cn.
// Uses XP (core.js) and XP_MODEL.
(() => {
  "use strict";

  const REFRESH_MS = 60000;
  const SIMS = 5000;
  const SEP = "\u0001";

  const texts = (v) => (v && typeof v === "object" && v.texts ? Object.values(v.texts).map(String) : v == null ? [] : [String(v)]);
  const norm = (s) => s.replace(/\s+/g, " ").trim();
  const pct = (p) => `${Math.round(p * 100)}%`;
  const clock = (ms) => { const m = Math.floor(ms / 60000); return `${Math.floor(m / 60)}:${String(m % 60).padStart(2, "0")}`; };
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

  async function getJSON(url) {
    const r = await fetch(url, { cache: "no-cache" });
    if (!r.ok) throw new Error(`${url.split("/").pop()}: HTTP ${r.status}`);
    return r.json();
  }

  // --------------------------------------------------------------------- sites
  // A site knows where a contest's data lives and where its rows and cells are on the page:
  //   key()     identifies the contest page; a change means navigation to another contest
  //   init(s)   false when the page is not a contest board; throws when loading fails
  //   load(s)   fresh {contestId, cfg, teams, runs} for XP.run
  //   rows(s)   rendered rows: [{team, teamEl, cells: [{problem, el, pending}]}]
  //   css       makes room in a cell for the percentage chip

  // board.xcpcio.com: rows carry no team id, so a row is matched by team name, and by organization
  // name when several teams share a name. Problem columns follow the header.
  const XCPCIO = {
    host: "board.xcpcio.com",
    key: () => location.pathname.replace(/\/+$/, ""),
    async init(s) {
      const get = (f) => getJSON(/^https?:\/\//.test(f) ? f : `${location.origin}/data${s.key}/${f}`);
      let cfg;
      try {
        cfg = await get("config.json");
      } catch {
        return false;
      }
      if (!cfg || !cfg.start_time || !Array.isArray(cfg.problems)) return false;
      const orgFile = cfg.organizations?.url;
      const [teams, orgs] = await Promise.all([get("team.json"), orgFile ? get(orgFile) : null]);
      const orgName = new Map((orgs || []).map((o) => [String(o.id), texts(o.name)]));
      const list = Array.isArray(teams) ? teams : Object.entries(teams).map(([k, v]) => ({ ...v, team_id: v.team_id ?? k }));
      const nameIndex = new Map();
      for (const t of list) {
        const id = String(t.team_id ?? t.id);
        const org = orgName.get(String(t.organization_id)) || texts(t.organization);
        for (const name of texts(t.name)) (nameIndex.get(norm(name)) || nameIndex.set(norm(name), []).get(norm(name))).push({ id, org });
      }
      Object.assign(s, { get, cfg, teams, nameIndex, problemIds: cfg.problems.map((p, i) => String(p.id ?? i)) });
      return true;
    },
    async load(s) {
      return { contestId: s.key.slice(1), cfg: s.cfg, teams: s.teams, runs: await s.get("run.json") };
    },
    rows(s) {
      const table = document.querySelector("table.standings");
      const ths = table?.tHead ? [...table.tHead.rows[0].cells] : [];
      const teamCol = ths.findIndex((th) => th.classList.contains("team-column"));
      const placeCol = ths.findIndex((th) => th.classList.contains("place-column"));
      const probCols = ths.flatMap((th, i) => (th.classList.contains("problem-column") ? [i] : []));
      if (teamCol < 0 || probCols.length !== s.problemIds.length) return [];
      const orgCol = teamCol - placeCol === 2 ? placeCol + 1 : -1;
      const identify = (teamTd, orgTd) => {
        const span = [...teamTd.querySelectorAll("span")].find((x) => x.textContent.trim());
        const cands = s.nameIndex.get(norm((span || teamTd).textContent));
        if (!cands) return null;
        if (cands.length === 1) return cands[0].id;
        const ctx = `${orgTd ? orgTd.textContent : ""} ${teamTd.textContent}`;
        const hit = cands.filter((c) => c.org.some((o) => o && ctx.includes(o)));
        return hit.length === 1 ? hit[0].id : null;
      };
      return [...table.querySelectorAll("tbody > tr")]
        .filter((tr) => tr.cells.length === ths.length) // rows off screen are not rendered
        .map((tr) => {
          const tds = tr.cells;
          return {
            team: identify(tds[teamCol], orgCol >= 0 ? tds[orgCol] : null),
            teamEl: tds[teamCol],
            cells: probCols.map((col, k) => ({ problem: s.problemIds[k], el: tds[col], pending: tds[col].classList.contains("pending") })),
          };
        });
    },
    css: `[data-xp] > div:first-child > div:first-child { align-self: flex-start; padding-left: 3px; font-size: 11px; }`,
  };

  // rank.iccpc.cn: the page boots with OR_BOOT (Prefetch is the rank.json snapshot). Rows carry the
  // team id, cells the problem letter.
  const CSG = {
    host: "rank.iccpc.cn",
    key: () => location.pathname,
    async init(s) {
      const boot = window.OR_BOOT;
      if (!/^\/r\//.test(location.pathname) || !boot || !boot.Prefetch) return false;
      s.boot = boot;
      return true;
    },
    async load(s) {
      const d = XP.fromCsg(await getJSON(new URL(s.boot.Prefetch, location.origin).href), s.boot.Outrank || {});
      s.problemOf = new Map(d.cfg.problems.map((p) => [p.label, p.id]));
      return d;
    },
    rows(s) {
      return [...document.querySelectorAll("[data-row-id]")]
        .filter((row) => (row.getAttribute("data-rank-mode") || "team") === "team") // not school or roll mode
        .map((row) => ({
          team: row.getAttribute("data-row-id"),
          teamEl: row.querySelector(".team-info-section"),
          cells: [...row.querySelectorAll("[d-pro-idx]")].map((el) => ({
            problem: s.problemOf?.get(el.getAttribute("d-pro-idx")), el, pending: el.classList.contains("pro-pending"),
          })),
        }));
    },
    // the site's own rules are more specific, hence !important
    css: `[data-xp] .problem-content, [data-xp] .rank-plain-problem-lines { padding-right: 19px !important; }
      [data-xp] .time-brief { font-size: .85em; }`,
  };

  const site = [XCPCIO, CSG].find((x) => x.host === location.hostname);
  if (!site) return;

  let st = null; // per-contest state, replaced on navigation
  let show = true;

  const css = `
    :root { --xp-fill: rgba(22, 163, 74, .38); --xp-text: #065f46; --xp-chip: rgba(255, 255, 255, .85); --xp-bg: #fff; --xp-fg: #111827; --xp-mute: #6b7280; --xp-line: #e5e7eb; }
    html.dark, html[data-theme="dark"] { --xp-fill: rgba(34, 197, 94, .35); --xp-text: #86efac; --xp-chip: rgba(17, 24, 39, .85); --xp-bg: #1f2937; --xp-fg: #f3f4f6; --xp-mute: #9ca3af; --xp-line: #374151; }
    [data-xp] { position: relative; }
    .xp-p { position: absolute; top: 1px; right: 1px; padding: 1px 2px; border-radius: 3px; background: var(--xp-chip);
      font: 700 10px/1 ui-monospace, monospace; color: var(--xp-text); pointer-events: none; }
    .xp-bar { position: absolute; left: 0; right: 0; bottom: 0; height: 3px; display: flex; pointer-events: none; }
    .xp-bar i { display: block; height: 100%; }
    .xp-bar .g { background: #eab308; } .xp-bar .s { background: #94a3b8; } .xp-bar .b { background: #b45309; }
    .xp-panel { position: fixed; right: 12px; bottom: 12px; z-index: 2147483000; width: 300px; max-width: calc(100vw - 24px);
      background: var(--xp-bg); color: var(--xp-fg); border: 1px solid var(--xp-line); border-radius: 8px;
      box-shadow: 0 4px 16px rgba(0,0,0,.15); font: 12px/1.5 system-ui, sans-serif; }
    .xp-head { display: flex; align-items: center; gap: 6px; padding: 6px 10px; border-bottom: 1px solid var(--xp-line); }
    .xp-head b { flex: 1; }
    .xp-head button { font: inherit; padding: 0 6px; border: 1px solid var(--xp-line); border-radius: 4px; background: transparent; color: inherit; cursor: pointer; }
    .xp-body { padding: 6px 10px; }
    .xp-panel.xp-min .xp-body { display: none; }
    .xp-mute { color: var(--xp-mute); }
    .xp-err { color: #dc2626; }
    .xp-body table { border-collapse: collapse; width: 100%; margin-top: 4px; }
    .xp-body td { padding: 1px 4px 1px 0; vertical-align: top; }
    ${site.css}
  `;

  // ------------------------------------------------------------------ panel
  function mountPanel() {
    let el = document.querySelector(".xp-panel");
    if (!el) {
      el = document.createElement("div");
      el.className = "xp-panel";
      el.innerHTML = `<div class="xp-head"><b>封榜预测</b><button data-a="toggle"></button><button data-a="refresh">刷新</button><button data-a="min">–</button></div><div class="xp-body"></div>`;
      el.addEventListener("click", (e) => {
        const a = e.target.closest("button")?.dataset.a;
        if (a === "toggle") { show = !show; renderPanel(); applyBoard(); }
        if (a === "refresh" && st) refresh(st);
        if (a === "min") el.classList.toggle("xp-min");
      });
      document.body.appendChild(el);
    }
    return el;
  }

  function renderPanel() {
    if (!st) return;
    const el = mountPanel();
    el.querySelector('[data-a="toggle"]').textContent = show ? "隐藏标注" : "显示标注";
    const body = el.querySelector(".xp-body");
    const r = st.result;
    const parts = [];
    if (st.error) parts.push(`<div class="xp-err">${esc(st.error)}</div>`);
    if (!r) parts.push(`<div class="xp-mute">${st.error ? "" : "加载中…"}</div>`);
    else if (r.status === "no_freeze") parts.push(`<div class="xp-mute">这场比赛没有封榜。</div>`);
    else if (r.status === "before_freeze") parts.push(`<div class="xp-mute">还没封榜，${new Date(r.freezeAtMs).toLocaleTimeString()} 封榜后开始预测。</div>`);
    else {
      parts.push(`<div>${r.status === "finished" ? "比赛已结束" : `封榜中，比赛时间 ${clock(r.observationMs)}`} · ${r.cells.length} 个冻结格子</div>`);
      parts.push(`<div class="xp-mute">模型 ${esc(XP_MODEL.model_version)} · Mode ${r.mode} · ${st.updatedAt.toLocaleTimeString()} 更新 · 计算 ${Math.round(st.elapsedMs)} ms</div>`);
      const byGroup = new Map();
      for (const l of r.lines) if (l.probability >= 0.05) (byGroup.get(l.group) || byGroup.set(l.group, []).get(l.group)).push(l);
      const names = { gold: "金", silver: "银", bronze: "铜" };
      for (const [g, ls] of byGroup) {
        const rows = ["gold", "silver", "bronze"].map((tier) => {
          const items = ls.filter((l) => l.tier === tier).sort((a, b) => b.probability - a.probability)
            .map((l) => `${l.solved} 题 ${pct(l.probability)} <span class="xp-mute">罚时 ${Math.round(l.penaltyMedian)}</span>`);
          return items.length ? `<tr><td>${names[tier]}</td><td>${items.join("<br>")}</td></tr>` : "";
        }).join("");
        parts.push(`<div style="margin-top:6px">奖牌线${g === "_ratio" ? "" : `（${esc(g)}）`}</div><table>${rows}</table>`);
      }
    }
    body.innerHTML = parts.join("");
  }

  // ------------------------------------------------------------- board cells
  function decorateCell(el, id, problem, pending) {
    const pred = show && id && pending ? st.cellPred.get(id + SEP + problem) : undefined;
    const sig = pred ? `${id}|${problem}|${pred.p.toFixed(4)}` : "";
    if ((el.dataset.xp || "") === sig) return;
    let badge = el.querySelector(":scope > .xp-p");
    if (!pred) {
      badge?.remove();
      el.style.removeProperty("background-image");
      el.removeAttribute("title");
      delete el.dataset.xp;
      return;
    }
    el.dataset.xp = sig;
    const v = Math.round(pred.p * 100);
    el.style.backgroundImage = `linear-gradient(to top, var(--xp-fill) ${v}%, transparent ${v}%)`;
    if (!badge) { badge = document.createElement("div"); badge.className = "xp-p"; el.appendChild(badge); }
    badge.textContent = `${v}%`;
    el.title = `封榜后 AC 概率 ${v}%（隐藏提交 ${pred.hidden} 次）`;
  }

  function decorateTeam(el, id) {
    const s = show && id ? st.teamPred.get(id) : undefined;
    const sig = s ? `${id}|${s.expectedSolved.toFixed(3)}|${s.rankP50}|${s.pMedal}` : "";
    if ((el.dataset.xpTeam || "") === sig) return;
    let bar = el.querySelector(":scope > .xp-bar");
    if (!s) { bar?.remove(); el.removeAttribute("title"); delete el.dataset.xpTeam; return; }
    el.dataset.xpTeam = sig;
    let tip = `预测：期望 ${s.expectedSolved.toFixed(2)} 题，排名中位数 ${Math.round(s.rankP50)}（90% 区间 ${Math.round(s.rankP05)}–${Math.round(s.rankP95)}）`;
    if (s.pMedal !== null) {
      tip += `\n金 ${pct(s.pGold)} · 银 ${pct(s.pSilver)} · 铜 ${pct(s.pBronze)}`;
      if (!bar) { bar = document.createElement("div"); bar.className = "xp-bar"; bar.innerHTML = '<i class="g"></i><i class="s"></i><i class="b"></i>'; el.appendChild(bar); }
      const [g, sv, b] = bar.children;
      g.style.width = pct(s.pGold); sv.style.width = pct(s.pSilver); b.style.width = pct(s.pBronze);
    } else {
      bar?.remove();
    }
    el.title = tip;
  }

  function applyBoard() {
    if (!st || !st.cellPred) return;
    for (const r of site.rows(st)) {
      if (r.teamEl) decorateTeam(r.teamEl, r.team);
      for (const c of r.cells) decorateCell(c.el, r.team, c.problem, c.pending);
    }
  }

  // ---------------------------------------------------------------- refresh
  async function refresh(s) {
    if (s.busy) return;
    s.busy = true;
    clearTimeout(s.timer);
    let next = REFRESH_MS;
    try {
      const t = s.times, now = Date.now();
      let result;
      if (t && t.hasFreeze && now - t.startMs < t.freezeStartMs) {
        result = { status: "before_freeze", freezeAtMs: t.startMs + t.freezeStartMs };
      } else {
        const d = await site.load(s);
        if (s !== st) return;
        s.times = XP.contestTimes(d.cfg);
        const t0 = performance.now();
        result = XP.run(XP_MODEL, d.contestId, d.cfg, d.teams, d.runs, Date.now(), { sims: SIMS });
        s.elapsedMs = performance.now() - t0;
        s.cellPred = new Map((result.cells || []).map((c) => [c.team + SEP + c.problem, c]));
        s.teamPred = new Map((result.teams || []).map((x) => [x.team, x]));
      }
      s.result = result;
      s.updatedAt = new Date();
      s.error = null;
      if (result.status === "before_freeze") next = Math.min(10 * REFRESH_MS, Math.max(5000, result.freezeAtMs - Date.now() + 5000));
      // after the end the frozen board no longer changes; one more pass picks up late-arriving runs
      if (result.status === "no_freeze" || (result.status === "finished" && s.finalPass)) next = null;
      if (result.status === "finished") s.finalPass = true;
    } catch (e) {
      s.error = `预测失败：${e.message}`;
    } finally {
      s.busy = false;
    }
    if (s !== st) return;
    renderPanel();
    applyBoard();
    if (next !== null) s.timer = setTimeout(() => (document.hidden ? (s.stale = true) : refresh(s)), next);
  }

  async function enter(key) {
    if (st) clearTimeout(st.timer);
    document.querySelector(".xp-panel")?.remove();
    const s = { key, busy: false };
    st = s;
    try {
      if (!(await site.init(s)) || s !== st) return; // not a contest page
    } catch (e) {
      if (s !== st) return;
      s.error = `加载失败：${e.message}`;
      renderPanel();
      return;
    }
    renderPanel();
    refresh(s);
  }

  const style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);

  let raf = 0;
  new MutationObserver(() => {
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; applyBoard(); });
  }).observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["class"] });
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && st?.stale) { st.stale = false; refresh(st); }
  });
  // single-page apps: follow navigation between contests
  const follow = () => { if (!st || site.key() !== st.key) enter(site.key()); };
  follow();
  setInterval(follow, 1000);
})();
