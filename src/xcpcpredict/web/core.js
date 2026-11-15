// xcpcpredict in the browser: XCPCIO raw data -> public snapshot -> features -> p_ac -> standings.
// A port of normalize.py, replay.py, features.py, the calibrated CatBoost model and standings.py.
// Kept close to the Python names so the two can be compared side by side; tests/test_userscript.py
// checks that both give the same features and p_ac.
const XP = (() => {
  "use strict";

  // ------------------------------------------------------------- normalize.py
  const AC = "AC", REJ = "REJECTED", NOPEN = "REJECTED_NO_PENALTY", UNRES = "UNRESOLVED";
  const VERDICTS = [
    [AC, ["ACCEPTED", "CORRECT", "AC"]],
    [REJ, ["REJECTED", "WRONG_ANSWER", "INCORRECT", "TIME_LIMIT_EXCEEDED", "RUNTIME_ERROR",
      "MEMORY_LIMIT_EXCEEDED", "OUTPUT_LIMIT_EXCEEDED", "NO_OUTPUT", "PRESENTATION_ERROR",
      "IDLENESS_LIMIT_EXCEEDED", "HACKED", "PARTIALLY_CORRECT"]],
    [NOPEN, ["REJECTED_NO_PENALTY", "COMPILATION_ERROR", "SYSTEM_ERROR", "JUDGEMENT_FAILED",
      "CONFIGURATION_ERROR", "FILE_ERROR", "CANCELED", "SKIPPED", "SECURITY_VIOLATION", "DENIAL_OF_JUDGEMENT"]],
    [UNRES, ["UNRESOLVED", "PENDING", "FROZEN", "WAITING", "PREPARING", "COMPILING", "RUNNING", "JUDGING",
      "UNKNOWN", "UNDEFINED"]],
  ];
  const VERDICT_OF = new Map(VERDICTS.flatMap(([v, names]) => names.map((n) => [n, v])));

  function mapVerdict(raw) {
    const v = VERDICT_OF.get(String(raw).trim().toUpperCase());
    if (v === undefined) throw new Error(`unknown verdict ${raw}`);
    return v;
  }

  function absTimeMs(value) {
    const v = Math.trunc(Number(value));
    if (v >= 1e11 && v < 1e14) return v;
    if (v >= 1e8 && v < 1e11) return v * 1000;
    throw new Error(`cannot infer unit of absolute time ${value}`);
  }

  function frozenMs(value, durationMs) {
    const f = Math.trunc(Number(value));
    if (f * 1000 <= durationMs) return f * 1000;
    if (f <= durationMs) return f;
    throw new Error(`frozen_time ${value} longer than contest`);
  }

  function text(value) {
    if (value && typeof value === "object" && value.texts) {
      const texts = value.texts;
      return String(texts[value.fallback_lang] || Object.values(texts)[0] || "");
    }
    return String(value);
  }

  function teamGroups(t) {
    const groups = [];
    if (Array.isArray(t.group)) groups.push(...t.group.map(String));
    else if (typeof t.group === "string") groups.push(t.group);
    for (const k of ["official", "unofficial", "girl", "undergraduate", "vocational"]) {
      if ([true, 1, "1", "true"].includes(t[k])) groups.push(k);
    }
    return [...new Set(groups)];
  }

  function classifyContest(contestId, name) {
    const cid = contestId.toLowerCase(), low = name.toLowerCase();
    if (["online", "qualification", "preliminary"].some((m) => cid.includes(m))) return "online";
    if (["world-finals", "ec-final", "ecfinal"].some((m) => cid.includes(m)) || cid.endsWith("/final")) return "final";
    if (["girl", "vocational", "gaozhi"].some((m) => cid.includes(m))) return "special";
    if (cid.startsWith("provincial-contest/")) return "provincial";
    if (cid.includes("invitational") || name.includes("邀请赛") || low.includes("invitational")) return "invitational";
    if (cid.startsWith("icpc/") || cid.startsWith("ccpc/")) return "regional";
    return "special";
  }

  function parseMedalRule(raw) {
    if (typeof raw === "string") return raw.toLowerCase() === "ccpc" ? { ratio: [0.1, 0.2, 0.3] } : null;
    if (raw && raw.csg) return { csg: raw.csg }; // set by fromCsg
    if (raw && typeof raw === "object") {
      const counts = {};
      for (const [g, v] of Object.entries(raw)) {
        if (v && typeof v === "object") counts[g] = [v.gold || 0, v.silver || 0, v.bronze || 0].map(Number);
      }
      return Object.keys(counts).length ? { counts } : null;
    }
    return null;
  }

  // CSG-OJ award lines (port of RankToolGetAwardRank): cumulative ranks [gold, silver, bronze] for `base`
  // eligible teams. Percent mode rounds each cumulative line up in integer arithmetic; a value >= 100 there
  // means (value - 100) teams. Quantity mode takes the three values as team counts.
  function csgCuts(base, ratios, qty) {
    const B = Math.max(0, Math.floor(base));
    if (B <= 0) return [0, 0, 0];
    const [g, s, b] = ratios.map((v) => Math.max(0, Math.floor(v)));
    if (qty) {
      const nG = Math.min(g, B), nS = Math.min(s, B - nG), nB = Math.min(b, B - nG - nS);
      return [nG, nG + nS, nG + nS + nB];
    }
    let acc = 0, line = 0;
    return [g, s, b].map((v) => {
      if (v >= 100) { line = Math.min(B, line + v - 100); acc = line * 100; }
      else { acc += B * v; line = Math.min(B, Math.floor((acc + 99) / 100)); }
      return line;
    });
  }

  // Cumulative medal lines of one group; nValid = members with at least one solve.
  function medalCuts(rule, group, nValid) {
    if (rule.counts) { const [a, b, c] = rule.counts[group]; return [a, a + b, a + b + c]; }
    if (rule.csg) return csgCuts(nValid, rule.csg.ratios, rule.csg.qty);
    const [a, b, c] = rule.ratio.map((f) => Math.ceil(f * nValid));
    return [a, a + b, a + b + c];
  }

  // Start, duration and freeze from config.json alone (no runs needed).
  function contestTimes(cfg) {
    const startMs = absTimeMs(cfg.start_time);
    const durationMs = absTimeMs(cfg.end_time) - startMs;
    if (durationMs <= 0) throw new Error("non-positive duration");
    const freezeLen = frozenMs(cfg.frozen_time || 0, durationMs);
    return { startMs, durationMs, freezeStartMs: durationMs - freezeLen, hasFreeze: freezeLen > 0 };
  }

  function normalizeContest(contestId, cfg, rawTeams, rawRuns) {
    const { startMs, durationMs, freezeStartMs, hasFreeze } = contestTimes(cfg);
    const opts = cfg.options || {};
    const unit = opts.submission_timestamp_unit;
    const mult = unit === "millisecond" ? 1 : (unit === "second" || unit == null) ? 1000 : NaN;
    if (Number.isNaN(mult)) throw new Error(`unknown submission_timestamp_unit ${unit}`);
    const maxTs = rawRuns.reduce((m, r) => Math.max(m, Number(r.timestamp)), 0);
    if (maxTs * mult > durationMs * 1.05) throw new Error("run timestamps exceed the contest duration");

    const problemIds = cfg.problems.map((p, i) => String(p.id ?? i));
    const teamList = Array.isArray(rawTeams)
      ? rawTeams : Object.entries(rawTeams).map(([k, v]) => ({ ...v, team_id: v.team_id ?? k }));
    const teams = new Map();
    for (const t of teamList) teams.set(String(t.team_id ?? t.id), teamGroups(t));

    const pids = new Set(problemIds);
    const subs = [];
    for (const r of rawRuns) {
      const team = String(r.team_id), prob = String(r.problem_id);
      if (!pids.has(prob)) continue;
      if (!teams.has(team)) teams.set(team, ["unknown"]);
      const t = Math.trunc(Number(r.timestamp)) * mult;
      if (t < 0 || t > durationMs) continue;
      const sid = r.submission_id ?? r.id;
      subs.push({ team, prob, t, v: mapVerdict(r.status), sid: sid == null ? "" : String(sid) });
    }
    subs.sort((a, b) => a.t - b.t || (a.sid < b.sid ? -1 : a.sid > b.sid ? 1 : 0));

    const name = text(cfg.contest_name ?? contestId);
    return {
      contestId, startMs, durationMs, freezeStartMs, hasFreeze,
      penaltyMs: Number(cfg.penalty ?? 1200) * 1000,
      penaltyRule: opts.calculation_of_penalty || "in_minutes",
      problemIds, teams, subs,
      medalRule: parseMedalRule(cfg.medal),
      contestType: classifyContest(contestId, name),
    };
  }

  // ---------------------------------------------------------------- replay.py
  // Hidden submissions (at/after the freeze) keep no verdict; in Mode B they keep no time either.
  function makeSnapshot(c, obsMs, mode) {
    const subs = [];
    for (const s of c.subs) {
      if (s.t > obsMs) continue;
      if (s.t < c.freezeStartMs) {
        if (s.v !== NOPEN) subs.push({ team: s.team, prob: s.prob, t: s.t, v: s.v });
      } else {
        subs.push({ team: s.team, prob: s.prob, t: mode === "A" ? s.t : null, v: null });
      }
    }
    return {
      fz: c.freezeStartMs, obs: obsMs, dur: c.durationMs, penaltyMs: c.penaltyMs, penaltyRule: c.penaltyRule,
      problems: c.problemIds, teamIds: [...c.teams.keys()], groups: c.teams, mode, subs,
      medalRule: c.medalRule, contestType: c.contestType,
    };
  }

  // -------------------------------------------------------------- features.py
  const MIN = 60000, SMOOTH_K = 3, SHORT_GAP_MIN = 5, RECENT_HIDDEN_MIN = 15, SILENT_MIN = 20, LAST_MINUTES = 5;
  const STAR_MARKERS = ["unofficial", "star", "打星"], GIRL_MARKERS = ["girl", "female", "女"];
  const TIME_COLUMNS = [
    "first_hidden_time", "last_hidden_time", "last_hidden_age", "hidden_span", "min_gap", "max_gap", "mean_gap",
    "last_gap", "short_gap_count", "recent_hidden_count", "gap_pre_to_first_hidden",
    "other_subs_after_last", "other_problems_after_last", "other_subs_before_first_hidden", "team_last_activity_age",
    "after_last_harder_problems", "after_last_easier_problems", "after_last_min_ac_frac",
    "crowd_silent_frac", "crowd_moved_on_frac", "crowd_recent_teams",
    "last_hidden_before_end", "first_hidden_before_end", "hidden_last5_count",
  ];

  const sum = (a) => a.reduce((x, y) => x + y, 0);
  const mean = (a) => sum(a) / a.length;
  const pstd = (a) => { const m = mean(a); return Math.sqrt(mean(a.map((x) => (x - m) ** 2))); };
  const allEqual = (a) => a.every((x) => x === a[0]);
  const smooth = (num, den, prior) => (num + SMOOTH_K * prior) / (den + SMOOTH_K);
  const cmpKey = (a, b) => a[0] - b[0] || a[1] - b[1];
  const hasMarker = (groups, markers) => { const g = groups.join(" ").toLowerCase(); return markers.some((m) => g.includes(m)); };

  function bisectLeft(sorted, key) {
    let lo = 0, hi = sorted.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (cmpKey(sorted[mid], key) < 0) lo = mid + 1; else hi = mid; }
    return lo;
  }

  function median(sorted) {
    const n = sorted.length, h = n >> 1;
    return n % 2 ? sorted[h] : (sorted[h - 1] + sorted[h]) / 2;
  }

  function corr(x, y) {
    const mx = mean(x), my = mean(y);
    let sxy = 0, sxx = 0, syy = 0;
    for (let i = 0; i < x.length; i++) {
      const dx = x[i] - mx, dy = y[i] - my;
      sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
    }
    return Math.max(-1, Math.min(1, sxy / Math.sqrt(sxx * syy)));
  }

  function tierOf(sortedKeys, cuts, key) {
    const pos = bisectLeft(sortedKeys, key);
    for (let i = 0; i < cuts.length; i++) if (pos < cuts[i]) return i;
    return 3;
  }

  function getOr(map, key, init) {
    let v = map.get(key);
    if (v === undefined) { v = init(); map.set(key, v); }
    return v;
  }

  function extractFeatures(snap) {
    const { fz, obs, dur } = snap;
    const modeA = snap.mode === "A";
    const cells = new Map();
    const T = new Map();
    const team = (id) => getOr(T, id, () => ({
      solved: [], penMs: 0, wa: 0, firstTry: [], acTimes: [], preTimes: [], hidden: [], lastByProb: new Map(),
    }));
    for (const s of snap.subs) {
      const c = getOr(cells, `${s.team}\u0001${s.prob}`, () => ({ team: s.team, prob: s.prob, pre: [], ht: [], hn: 0 }));
      if (s.v === null) {
        c.hn += 1;
        if (s.t !== null) c.ht.push(s.t);
        team(s.team).hidden.push([s.t, s.prob]);
      } else {
        c.pre.push([s.t, s.v]);
        team(s.team).preTimes.push(s.t);
      }
    }
    for (const c of cells.values()) {
      c.pre.sort((a, b) => a[0] - b[0]);
      c.ht.sort((a, b) => a - b);
      c.judged = c.pre.filter((x) => x[1] !== UNRES);
      c.acIdx = c.pre.findIndex((x) => x[1] === AC);
    }

    const problems = snap.problems, nProblems = problems.length;
    const activeTeams = [...new Set([...cells.values()].map((c) => c.team))].sort();
    const activeSet = new Set(activeTeams);
    const nActive = Math.max(1, activeTeams.length);
    const P = new Map();
    const prob = (id) => getOr(P, id, () => ({
      acTeams: [], attempt: 0, firstTry: [], wrong: 0, acTimes: [], hiddenTeams: 0, hiddenSubs: 0, firstAttempt: [],
    }));
    problems.forEach(prob);

    // pre-freeze scoreboard
    for (const c of cells.values()) {
      if (!c.judged.length) continue;
      const tr = team(c.team), pr = prob(c.prob);
      pr.attempt += 1;
      const ft = c.judged[0][1] === AC ? 1 : 0;
      tr.firstTry.push(ft); pr.firstTry.push(ft);
      pr.firstAttempt.push(c.judged[0][0]);
      if (c.acIdx >= 0) {
        const acMs = c.pre[c.acIdx][0];
        let wrong = 0;
        for (let i = 0; i < c.acIdx; i++) if (c.pre[i][1] === REJ) wrong += 1;
        tr.solved.push(c.prob); tr.wa += wrong; tr.acTimes.push(acMs);
        tr.penMs += (snap.penaltyRule === "in_minutes" ? Math.floor(acMs / 60000) * 60000 : acMs) + wrong * snap.penaltyMs;
        pr.acTeams.push(c.team); pr.wrong += wrong; pr.acTimes.push(acMs);
      }
    }
    const nSolved = (t) => (T.has(t) ? T.get(t).solved.length : 0);
    const penalty = new Map(activeTeams.map((t) => [t, Math.floor(team(t).penMs / MIN)]));
    const order = [...activeTeams].sort((a, b) => nSolved(b) - nSolved(a) || penalty.get(a) - penalty.get(b));
    const rank = new Map();
    let prevKey = null, prevRank = 0;
    order.forEach((t, i) => {
      const k = `${nSolved(t)}:${penalty.get(t)}`;
      if (k !== prevKey) { prevKey = k; prevRank = i + 1; }
      rank.set(t, prevRank);
    });
    const rankPct = (t) => (rank.get(t) - 1) / Math.max(1, nActive - 1);

    let totalAttempt = 0, totalAc = 0, gWrongN = 0;
    const allFirst = [];
    for (const pr of P.values()) {
      totalAttempt += pr.attempt; totalAc += pr.acTeams.length; gWrongN += pr.wrong; allFirst.push(...pr.firstTry);
    }
    const gSolve = totalAttempt ? totalAc / totalAttempt : 0.5;
    const gDirt = totalAc ? gWrongN / (gWrongN + totalAc) : 0.5;
    const gFirst = allFirst.length ? sum(allFirst) / allFirst.length : 0.5;

    const acN = (p) => prob(p).acTeams.length;
    const acFrac = new Map(problems.map((p) => [p, acN(p) / nActive]));
    const easeRank = new Map([...problems].sort((a, b) => acN(b) - acN(a)).map((p, i) => [p, i / Math.max(1, nProblems - 1)]));
    const strong = activeTeams.filter((t) => rankPct(t) <= 0.25 && nSolved(t) > 0);

    for (const c of cells.values()) {
      if (c.hn && c.acIdx < 0) { const pr = prob(c.prob); pr.hiddenTeams += 1; pr.hiddenSubs += c.hn; }
    }
    const recentWindow = fz - (dur - fz);

    // team groups
    const labelled = activeTeams.some((t) => (snap.groups.get(t) || []).length > 0);
    const flag = (t, markers) => (labelled ? +hasMarker(snap.groups.get(t) || [], markers) : NaN);

    // medal distance
    const boardTeams = [...new Set(snap.teamIds)].sort();
    const keyOf = (t) => [-nSolved(t), penalty.has(t) ? penalty.get(t) : 0];
    const medal = new Map();
    const rule = snap.medalRule;
    if (rule) {
      const inGroup = (t, g) => (snap.groups.get(t) || []).includes(g);
      const groups = rule.counts
        ? Object.keys(rule.counts).map((g) => [g, g === "all" ? boardTeams : boardTeams.filter((t) => inGroup(t, g))])
        : rule.csg ? [["official", boardTeams.filter((t) => inGroup(t, "official"))]]
          : [["_ratio", boardTeams.filter((t) => !(activeSet.has(t) && flag(t, STAR_MARKERS) === 1))]];
      for (const [g, members] of groups) {
        if (!members.length) continue;
        const cuts = medalCuts(rule, g, members.filter((t) => keyOf(t)[0] < 0).length);
        const sk = members.map(keyOf).sort(cmpKey);
        for (const t of members) if (!medal.has(t)) medal.set(t, [sk, cuts]);
      }
    }
    const allKeys = activeTeams.map(keyOf).sort(cmpKey);

    // contest structure
    const solvedArr = activeTeams.map(nSolved);
    const discrimination = new Map();
    for (const p of problems) {
      const x = activeTeams.map((t) => (team(t).solved.includes(p) ? 1 : 0));
      const rest = solvedArr.map((s, i) => s - x[i]);
      discrimination.set(p, x.length && !allEqual(x) && !allEqual(rest) ? corr(x, rest) : NaN);
    }
    const discVals = [...discrimination.values()].filter((v) => !Number.isNaN(v));
    const ct = snap.contestType;
    const structure = {};
    for (const k of ["regional", "invitational", "provincial", "online", "final"]) structure[`ctype_${k}`] = ct ? +(ct === k) : NaN;
    Object.assign(structure, {
      contest_mean_discrimination: discVals.length ? mean(discVals) : NaN,
      contest_solved_cv: solvedArr.length && mean(solvedArr) > 0 ? pstd(solvedArr) / mean(solvedArr) : NaN,
      contest_zero_frac: solvedArr.length ? solvedArr.filter((s) => s === 0).length / solvedArr.length : NaN,
      contest_levels_frac: new Set(solvedArr).size / (nProblems + 1),
      contest_problem_spread: pstd(problems.map((p) => acFrac.get(p))),
      contest_unsolved_problems: problems.filter((p) => acFrac.get(p) === 0).length,
    });

    // crowd behaviour on each problem; each row subtracts its own cell
    for (const c of cells.values()) if (c.ht.length) team(c.team).lastByProb.set(c.prob, c.ht[c.ht.length - 1]);
    const movedOn = (t, p, lastH) => [...team(t).lastByProb].some(([q, tm]) => q !== p && tm > lastH);
    const crowd = new Map(), cellInd = new Map();
    for (const [key, c] of cells) {
      if (!c.hn || c.acIdx >= 0) continue;
      const v = [1, c.hn, +(c.hn >= 2), NaN, NaN, NaN];
      if (modeA) {
        const lastH = c.ht[c.ht.length - 1];
        v[3] = +(obs - lastH >= SILENT_MIN * MIN);
        v[4] = +movedOn(c.team, c.prob, lastH);
        v[5] = +(obs - lastH <= RECENT_HIDDEN_MIN * MIN);
      }
      cellInd.set(key, v);
      const acc = getOr(crowd, c.prob, () => [0, 0, 0, 0, 0, 0]);
      v.forEach((x, i) => { acc[i] += x; });
    }

    const rows = [];
    for (const [key, c] of cells) {
      if (!c.hn || c.acIdx >= 0) continue;
      const tr = team(c.team), pr = prob(c.prob);
      const solved = tr.solved, ns = solved.length, wa = tr.wa, pAc = pr.acTeams.length, pWrong = pr.wrong;
      const acTimes = [...pr.acTimes].sort((a, b) => a - b);
      const fails = c.judged.filter((x) => x[1] !== AC).map((x) => x[0]);
      const r = {
        team_id: c.team,
        problem_id: c.prob,
        // M1
        problem_ac_count: pAc,
        problem_attempt_count: pr.attempt,
        problem_solve_rate: smooth(pAc, pr.attempt, gSolve),
        problem_ac_frac: acFrac.get(c.prob),
        solved_count: ns,
        rank_pct: rankPct(c.team),
        n_problems: nProblems,
        n_active_teams: nActive,
        // M2
        penalty: penalty.get(c.team),
        penalty_wa_count: wa,
        team_dirt: smooth(wa, wa + ns, gDirt),
        team_dirt_raw: ns ? wa / (wa + ns) : NaN,
        team_first_ac_rate: smooth(sum(tr.firstTry), tr.firstTry.length, gFirst),
        attempted_count: tr.firstTry.length,
        recent_ac_count: tr.acTimes.filter((t) => t >= recentWindow).length,
        recent_submission_count: tr.preTimes.filter((t) => t >= recentWindow).length,
        weighted_solved: sum(solved.map((p) => 1 - acFrac.get(p))),
        minutes_since_last_ac: ns ? (fz - Math.max(...tr.acTimes)) / MIN : NaN,
        problem_dirt: smooth(pWrong, pWrong + pAc, gDirt),
        problem_first_ac_rate: smooth(sum(pr.firstTry), pr.firstTry.length, gFirst),
        problem_first_ac_n: pr.firstTry.length,
        first_ac_time: acTimes.length ? acTimes[0] / MIN : NaN,
        median_ac_time: acTimes.length ? median(acTimes) / MIN : NaN,
        strong_team_solve_rate: smooth(strong.filter((t) => team(t).solved.includes(c.prob)).length, strong.length, gSolve),
        strong_team_n: strong.length,
        problem_ease_rank: easeRank.get(c.prob),
        pre_wa_count: c.pre.filter((x) => x[1] === REJ).length,
        has_pre_attempt: +(c.judged.length > 0),
        pre_first_time: c.pre.length ? c.pre[0][0] / MIN : NaN,
        pre_last_fail_age: fails.length ? (fz - fails[fails.length - 1]) / MIN : NaN,
        pre_span: c.pre.length ? (c.pre[c.pre.length - 1][0] - c.pre[0][0]) / MIN : NaN,
        ability_gap: acFrac.get(c.prob) - rankPct(c.team),
        n_harder_solved: solved.filter((p) => acN(p) < pAc).length,
        n_easier_unsolved: problems.filter((p) => !solved.includes(p) && acN(p) > pAc).length,
      };

      // M3: counts (Mode A and B)
      const th = tr.hidden;
      Object.assign(r, {
        hidden_count: c.hn,
        team_hidden_total: th.length,
        team_hidden_problems: new Set(th.map((x) => x[1])).size,
        other_hidden_count: th.length - c.hn,
        problem_hidden_teams: pr.hiddenTeams,
        problem_hidden_subs: pr.hiddenSubs,
        obs_since_freeze: (obs - fz) / MIN,
        obs_frac: (obs - fz) / Math.max(1, dur - fz),
        time_remaining: (dur - obs) / MIN,
      });

      // M6, count-based part
      const own = cellInd.get(key);
      const tot = crowd.get(c.prob).map((x, i) => x - own[i]);
      const nOther = tot[0];
      const otherProbs = [...new Set(th.map((x) => x[1]).filter((p) => p !== c.prob))];
      const recentAc = acTimes.filter((t) => t >= recentWindow).length;
      Object.assign(r, {
        other_hidden_harder_problems: otherProbs.filter((p) => acN(p) < pAc).length,
        other_hidden_easier_problems: otherProbs.filter((p) => acN(p) > pAc).length,
        crowd_mean_hidden: nOther ? tot[1] / nOther : NaN,
        crowd_multi_frac: nOther ? tot[2] / nOther : NaN,
        problem_recent_ac: recentAc,
        problem_recent_ac_share: pAc ? recentAc / pAc : NaN,
        problem_recent_new_attempts: pr.firstAttempt.filter((t) => t >= recentWindow).length,
        problem_last_ac_age: acTimes.length ? (fz - acTimes[acTimes.length - 1]) / MIN : NaN,
        team_unofficial: flag(c.team, STAR_MARKERS),
        team_girl: flag(c.team, GIRL_MARKERS),
      }, structure);
      r.problem_discrimination = discrimination.get(c.prob);

      const myKey = keyOf(c.team);
      const keyIfAc = [myKey[0] - 1, myKey[1] + Math.floor(fz / MIN) + r.pre_wa_count * snap.penaltyMs / MIN];
      r.rank_gain_if_ac = (bisectLeft(allKeys, myKey) - bisectLeft(allKeys, keyIfAc)) / nActive;
      if (!rule) {
        Object.assign(r, { medal_eligible: NaN, medal_tier_now: NaN, solved_gap_next_tier: NaN, rank_gap_next_tier: NaN, tier_gain_if_ac: NaN });
      } else if (!medal.has(c.team)) {
        Object.assign(r, { medal_eligible: 0, medal_tier_now: NaN, solved_gap_next_tier: NaN, rank_gap_next_tier: NaN, tier_gain_if_ac: NaN });
      } else {
        const [sk, cuts] = medal.get(c.team);
        const tier = tierOf(sk, cuts, myKey);
        const pos = bisectLeft(sk, myKey);
        let solvedGap = NaN, rankGap = NaN;
        if (tier > 0 && cuts[tier - 1] > 0) {
          const cutoffKey = sk[Math.min(cuts[tier - 1], sk.length) - 1];
          solvedGap = myKey[0] - cutoffKey[0];
          rankGap = pos + 1 - cuts[tier - 1];
        }
        Object.assign(r, {
          medal_eligible: 1, medal_tier_now: tier, solved_gap_next_tier: solvedGap, rank_gap_next_tier: rankGap,
          tier_gain_if_ac: tier - tierOf(sk, cuts, keyIfAc),
        });
      }

      // M4/M5 and time-based M6: need hidden times
      if (modeA) {
        const ht = c.ht;
        const gaps = ht.slice(1).map((t, i) => (t - ht[i]) / MIN);
        const lastPre = c.pre.length ? c.pre[c.pre.length - 1][0] : null;
        const firstH = ht[0], lastH = ht[ht.length - 1];
        const others = th.filter((x) => x[1] !== c.prob);
        const after = others.filter((x) => x[0] > lastH);
        const afterProbs = [...new Set(after.map((x) => x[1]))];
        const teamTimes = th.map((x) => x[0]).concat(tr.preTimes);
        Object.assign(r, {
          first_hidden_time: (firstH - fz) / MIN,
          last_hidden_time: (lastH - fz) / MIN,
          last_hidden_age: (obs - lastH) / MIN,
          hidden_span: (lastH - firstH) / MIN,
          min_gap: gaps.length ? Math.min(...gaps) : NaN,
          max_gap: gaps.length ? Math.max(...gaps) : NaN,
          mean_gap: gaps.length ? mean(gaps) : NaN,
          last_gap: gaps.length ? gaps[gaps.length - 1] : NaN,
          short_gap_count: gaps.filter((g) => g < SHORT_GAP_MIN).length,
          recent_hidden_count: ht.filter((t) => obs - t <= RECENT_HIDDEN_MIN * MIN).length,
          gap_pre_to_first_hidden: lastPre !== null ? (firstH - lastPre) / MIN : NaN,
          other_subs_after_last: after.length,
          other_problems_after_last: afterProbs.length,
          other_subs_before_first_hidden: others.filter((x) => x[0] < firstH).length,
          team_last_activity_age: (obs - Math.max(...teamTimes)) / MIN,
          after_last_harder_problems: afterProbs.filter((p) => acN(p) < pAc).length,
          after_last_easier_problems: afterProbs.filter((p) => acN(p) > pAc).length,
          after_last_min_ac_frac: after.length ? Math.min(...after.map((x) => acFrac.get(x[1]))) : NaN,
          crowd_silent_frac: nOther ? tot[3] / nOther : NaN,
          crowd_moved_on_frac: nOther ? tot[4] / nOther : NaN,
          crowd_recent_teams: tot[5],
          last_hidden_before_end: (dur - lastH) / MIN,
          first_hidden_before_end: (dur - firstH) / MIN,
          hidden_last5_count: ht.filter((t) => dur - t <= LAST_MINUTES * MIN).length,
        });
      } else {
        for (const k of TIME_COLUMNS) r[k] = NaN;
      }
      rows.push(r);
    }
    return rows;
  }

  // ------------------------------------------- calibrated CatBoost (models.py, calibration.py)
  // model: {columns, trees: [[featureIdx[], borders[], leafValues[]]], scale, bias, platt: [coef, intercept]}
  // Split i sets bit i of the leaf index; NaN never exceeds a border, matching CatBoost's AsFalse/AsIs.
  function predictProba(model, rows) {
    const sigmoid = (z) => 1 / (1 + Math.exp(-z));
    return rows.map((r) => {
      const x = model.columns.map((c) => (c in r ? r[c] : NaN));
      let s = 0;
      for (const [f, b, leaves] of model.trees) {
        let idx = 0;
        for (let i = 0; i < f.length; i++) if (x[f[i]] > b[i]) idx |= 1 << i;
        s += leaves[idx];
      }
      const p = Math.min(Math.max(sigmoid(model.scale * s + model.bias), 1e-4), 1 - 1e-4);
      return sigmoid(model.platt[0] * Math.log(p / (1 - p)) + model.platt[1]);
    });
  }

  // ------------------------------------------------------------- standings.py
  const TIERS = ["gold", "silver", "bronze"];

  // Mulberry32: seeded so that the same board gives the same numbers on every refresh.
  function rng(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // numpy's default (linear) percentile over the samples described by a histogram of integer values.
  function percentileFromHist(hist, offset, length, n, q) {
    const h = (n - 1) * q / 100, lo = Math.floor(h);
    let seen = 0, vLo = null;
    for (let v = 0; v < length; v++) {
      seen += hist[offset + v];
      if (vLo === null && seen > lo) vLo = v;
      if (seen > lo + 1 || (seen > lo && h === lo)) return vLo + (h - lo) * (v - vLo);
    }
    return vLo;
  }

  function percentile(values, q) {
    const s = Float64Array.from(values).sort();
    const h = (s.length - 1) * q / 100, lo = Math.floor(h), hi = Math.min(lo + 1, s.length - 1);
    return s[lo] + (h - lo) * (s[hi] - s[lo]);
  }

  // cells: [{team, problem, p}] for the frozen cells of `snap`.
  function simulateStandings(snap, cells, nSims = 10000, seed = 1) {
    const teams = [...new Set(snap.teamIds)].sort();
    const idx = new Map(teams.map((t, i) => [t, i]));
    const n = teams.length;
    // visible scoreboard at the freeze
    const byCell = new Map();
    for (const s of snap.subs) getOr(byCell, `${s.team}\u0001${s.prob}`, () => ({ team: s.team, vis: [], hidden: [] }))[s.v === null ? "hidden" : "vis"].push(s);
    const solved0 = new Float64Array(n), pen0 = new Float64Array(n), penMs = new Float64Array(n);
    for (const c of byCell.values()) {
      const vis = [...c.vis].sort((a, b) => a.t - b.t);
      let wrong = 0;
      for (const s of vis) {
        if (s.v === AC) {
          const i = idx.get(c.team);
          solved0[i] += 1;
          penMs[i] += (snap.penaltyRule === "in_minutes" ? Math.floor(s.t / 60000) * 60000 : s.t) + wrong * snap.penaltyMs;
          break;
        }
        if (s.v === REJ) wrong += 1;
      }
    }
    for (let i = 0; i < n; i++) pen0[i] = Math.floor(penMs[i] / 60000);
    // an AC frozen cell is taken at its last hidden submission (Mode B: middle of [freeze, observation])
    const mid = Math.floor((snap.fz + snap.obs) / 2), penMin = snap.penaltyMs / 60000;
    const cellTeam = new Int32Array(cells.length), cellP = new Float64Array(cells.length), cellPen = new Float64Array(cells.length);
    cells.forEach((cell, k) => {
      const c = byCell.get(`${cell.team}\u0001${cell.problem}`);
      const times = c.hidden.map((s) => s.t);
      const tAc = snap.mode === "A" ? Math.max(...times) : mid;
      const tPart = snap.penaltyRule === "in_minutes" ? Math.floor(tAc / 60000) : tAc / 60000;
      const preWa = c.vis.filter((s) => s.v === REJ).length;
      cellTeam[k] = idx.get(cell.team); cellP[k] = cell.p; cellPen[k] = tPart + penMin * (preWa + times.length - 1);
    });

    // medal groups
    const groupsOf = (t) => snap.groups.get(t) || [];
    const rule = snap.medalRule;
    const membersOf = (g) => teams.flatMap((t, i) => (g === "all" || groupsOf(t).includes(g) ? [i] : []));
    const medalGroups = !rule ? [] : rule.counts ? Object.keys(rule.counts).map((g) => [g, membersOf(g)])
      : rule.csg ? [["official", membersOf("official")]]
        : [["_ratio", teams.flatMap((t, i) => (hasMarker(groupsOf(t), STAR_MARKERS) ? [] : [i]))]];
    const groups = medalGroups.filter(([, m]) => m.length).map(([name, members]) => {
      const memberOf = new Int32Array(n).fill(-1);
      members.forEach((i, j) => { memberOf[i] = j; });
      return {
        name, members: Int32Array.from(members), memberOf, tierCount: new Float64Array(members.length * 4),
        lines: TIERS.map(() => new Map()), cuts: null, seen: 0, prevScore: NaN, prevRank: 0,
      };
    });

    // Scores are integers, so score * M + team index sorts by score and keeps the team recoverable.
    const M = 2 ** Math.ceil(Math.log2(n + 1));
    const random = rng(seed);
    const solved = new Float64Array(n), pen = new Float64Array(n), keys = new Float64Array(n);
    const rankHist = new Float64Array(n * (n + 1)), rankSum = new Float64Array(n), solvedSum = new Float64Array(n), solvedSq = new Float64Array(n);
    for (let sim = 0; sim < nSims; sim++) {
      solved.set(solved0); pen.set(pen0);
      for (let k = 0; k < cellP.length; k++) {
        if (random() < cellP[k]) { solved[cellTeam[k]] += 1; pen[cellTeam[k]] += cellPen[k]; }
      }
      for (let i = 0; i < n; i++) keys[i] = (solved[i] * 1e6 - Math.floor(pen[i])) * M + i;
      keys.sort();
      for (const g of groups) {
        let nValid = 0;
        for (const i of g.members) if (solved[i] > 0) nValid += 1;
        g.cuts = medalCuts(rule, g.name, nValid);
        g.seen = 0; g.prevScore = NaN;
      }
      // walk from the best score down; rank = 1 + number of strictly better scores
      let prevScore = NaN, prevRank = 0;
      for (let q = n - 1; q >= 0; q--) {
        const score = Math.floor(keys[q] / M), i = keys[q] - score * M;
        const r = score === prevScore ? prevRank : n - q;
        prevScore = score; prevRank = r;
        rankHist[i * (n + 1) + r] += 1; rankSum[i] += r; solvedSum[i] += solved[i]; solvedSq[i] += solved[i] * solved[i];
        for (let gi = 0; gi < groups.length; gi++) {
          const g = groups[gi], j = g.memberOf[i];
          if (j < 0) continue;
          const gr = score === g.prevScore ? g.prevRank : g.seen + 1;
          g.prevScore = score; g.prevRank = gr; g.seen += 1;
          const cuts = g.cuts;
          const tier = solved[i] === 0 ? 3 : gr <= cuts[0] ? 0 : gr <= cuts[1] ? 1 : gr <= cuts[2] ? 2 : 3;
          g.tierCount[j * 4 + tier] += 1;
          for (let k = 0; k < 3; k++) {
            // the last team inside the tier
            if (cuts[k] > 0 && g.seen === Math.min(cuts[k], g.members.length)) {
              const s = Math.ceil(score / 1e6);
              getOr(g.lines[k], s, () => []).push(s * 1e6 - score);
            }
          }
        }
      }
    }

    const cellsPerTeam = new Int32Array(n);
    cellTeam.forEach((i) => { cellsPerTeam[i] += 1; });
    const out = teams.map((t, i) => ({
      team: t, solved0: solved0[i], pen0: pen0[i], frozenCells: cellsPerTeam[i],
      expectedSolved: solvedSum[i] / nSims,
      solvedStd: Math.sqrt(Math.max(0, solvedSq[i] / nSims - (solvedSum[i] / nSims) ** 2)),
      rankMean: rankSum[i] / nSims,
      rankP05: percentileFromHist(rankHist, i * (n + 1), n + 1, nSims, 5),
      rankP50: percentileFromHist(rankHist, i * (n + 1), n + 1, nSims, 50),
      rankP95: percentileFromHist(rankHist, i * (n + 1), n + 1, nSims, 95),
      pGold: null, pSilver: null, pBronze: null, pMedal: null,
    }));
    const lines = [];
    for (const g of groups) {
      g.members.forEach((i, j) => {
        const c = (k) => g.tierCount[j * 4 + k] / nSims;
        Object.assign(out[i], { pGold: c(0), pSilver: c(1), pBronze: c(2), pMedal: c(0) + c(1) + c(2) });
      });
      TIERS.forEach((tier, k) => {
        for (const [s, pens] of [...g.lines[k]].sort((a, b) => a[0] - b[0])) {
          lines.push({
            group: g.name, tier, solved: s, probability: pens.length / nSims,
            penaltyP10: percentile(pens, 10), penaltyMedian: percentile(pens, 50), penaltyP90: percentile(pens, 90),
          });
        }
      });
    }
    out.sort((a, b) => a.rankMean - b.rankMean);
    return { teams: out, lines };
  }

  // ---------------------------------------------- CSG-OJ outrank (rank.iccpc.cn)
  // rank.json of a CSG-OJ public board -> {contestId, cfg, teams, runs} in the XCPCIO shape used above.
  // meta is the page's OR_BOOT.Outrank: {uuid, title, ckind, level, start_time (ISO, UTC)}.
  const CSG_STATUS = {
    4: "ACCEPTED", 5: "PRESENTATION_ERROR", 6: "WRONG_ANSWER", 7: "TIME_LIMIT_EXCEEDED",
    8: "MEMORY_LIMIT_EXCEEDED", 9: "OUTPUT_LIMIT_EXCEEDED", 10: "RUNTIME_ERROR",
  };

  // An id that classifyContest maps to the right contest type.
  function csgContestId(meta, contest) {
    const kind = meta.ckind === "icpc" ? "icpc" : "ccpc", key = meta.uuid || contest.contest_id;
    const title = meta.title || contest.title || "", level = meta.level || "national";
    if (/网络|online/i.test(title)) return `${kind}/online/${key}`;
    if (/女生|girl/i.test(title)) return `${kind}/girl/${key}`;
    if (/总决赛|final/i.test(title)) return `${kind}/${key}/final`;
    if (level.startsWith("provincial")) return `provincial-contest/${key}`;
    if (level === "national_invite") return `${kind}/invitational/${key}`;
    if (level === "national") return `${kind}/${key}`;
    return `csg/${key}`;
  }

  function fromCsg(rank, meta = {}) {
    const c = rank.contest;
    // times are wall-clock strings in the contest's time zone; only differences are needed, plus the
    // absolute start (from the page, default UTC+8)
    const wall = (v) => Date.parse(`${String(v).trim().replace(" ", "T")}Z`);
    const start = wall(c.start_time);
    const offset = meta.start_time ? Date.parse(meta.start_time) - start : -8 * 3600 * 1000;
    const teams = rank.team.map((t) => {
      const group = [Number(t.tkind) === 2 ? "unofficial" : "official"];
      if (Number(t.tkind) === 1 || Number(t.flg_girls) === 1) group.push("girl");
      return { id: String(t.team_id), name: t.name, organization: t.school, group };
    });
    // what the board shows: CE and still-judging runs are dropped, runs after a visible AC are ignored
    const sols = [...rank.solution].sort((a, b) => wall(a.in_date) - wall(b.in_date) || a.solution_id - b.solution_id);
    const solved = new Set(), runs = [];
    for (const s of sols) {
      const r = Number(s.result), cell = `${s.team_id}\u0001${s.problem_id}`;
      if (!(r < 0 || r in CSG_STATUS) || solved.has(cell)) continue;
      if (r === 4) solved.add(cell);
      runs.push({
        id: String(s.solution_id), team_id: String(s.team_id), problem_id: String(s.problem_id),
        timestamp: wall(s.in_date) - start, status: r < 0 ? "FROZEN" : CSG_STATUS[r],
      });
    }
    // medals: one award setting for official teams; separate settings per group are not modelled
    let medal = null;
    if (!Number(rank.is_multi_group)) {
      const code = Number(c.award_ratio) || 0;
      const ratios = code ? [code % 1000, Math.floor(code / 1000) % 1000, Math.floor(code / 1e6) % 1000] : [10, 20, 30];
      const qty = Number(c.flg_award_qty_mode) === 1;
      const base = Number(c.flg_award_base) || 0; // 0: teams with a solve, 1: teams with a submission, 2: all
      if (base === 0) {
        medal = { csg: { ratios, qty } }; // the base moves as frozen cells resolve
      } else {
        const submitted = new Set(rank.solution.map((s) => String(s.team_id)));
        const official = teams.filter((t) => t.group.includes("official") && (base === 2 || submitted.has(t.id)));
        const [g, sv, b] = csgCuts(official.length, ratios, qty);
        medal = { official: { gold: g, silver: sv - g, bronze: b - sv } };
      }
    }
    const cfg = {
      contest_name: meta.title || c.title,
      start_time: start + offset, end_time: wall(c.end_time) + offset,
      frozen_time: (Number(c.frozen_minute) || 0) * 60, penalty: 1200,
      problems: [...rank.problem].sort((a, b) => a.num - b.num)
        .map((p) => ({ id: String(p.problem_id), label: String.fromCharCode(65 + Number(p.num)) })),
      medal,
      status_time_display: { pending: true }, // every hidden run's time is listed in the cell popover
      options: { submission_timestamp_unit: "millisecond", calculation_of_penalty: "accumulate_in_seconds_and_finally_to_the_minute" },
    };
    return { contestId: csgContestId(meta, c), cfg, teams, runs };
  }

  // --------------------------------------------------------------- one call
  // Mode A when the board shows the time of pending runs (XCPCIO's default), otherwise Mode B.
  function boardMode(cfg) {
    const d = cfg.status_time_display;
    return d == null || d.pending ? "A" : "B";
  }

  function run(bundle, contestId, cfg, rawTeams, rawRuns, nowMs, opts = {}) {
    const times = contestTimes(cfg);
    if (!times.hasFreeze) return { status: "no_freeze" };
    const obs = Math.min(times.durationMs, nowMs - times.startMs);
    if (obs < times.freezeStartMs) return { status: "before_freeze", freezeAtMs: times.startMs + times.freezeStartMs };
    const mode = opts.mode || boardMode(cfg);
    const contest = normalizeContest(contestId, cfg, rawTeams, rawRuns);
    // once final verdicts of post-freeze runs are public, the board is revealed and nothing is left to predict
    if (contest.subs.some((s) => s.t >= times.freezeStartMs && (s.v === AC || s.v === REJ))) return { status: "revealed" };
    const snap = makeSnapshot(contest, obs, mode);
    const rows = extractFeatures(snap);
    const ps = predictProba(bundle.models[mode], rows);
    const cells = rows.map((r, i) => ({ team: r.team_id, problem: r.problem_id, hidden: r.hidden_count, p: ps[i] }));
    const st = opts.sims === 0 ? { teams: [], lines: [] } : simulateStandings(snap, cells, opts.sims ?? 10000, opts.seed ?? 1);
    return {
      status: obs >= times.durationMs ? "finished" : "frozen", mode, observationMs: obs, ...times,
      cells, rows, teams: st.teams, lines: st.lines,
    };
  }

  return { normalizeContest, makeSnapshot, extractFeatures, predictProba, simulateStandings, contestTimes, boardMode, run, fromCsg };
})();

if (typeof module !== "undefined" && module.exports) module.exports = XP;
