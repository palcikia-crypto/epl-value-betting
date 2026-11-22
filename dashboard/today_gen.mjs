// today.json (schema 3): матчи + B2B-флаги + полный форвард-тест сезона (statsapi, один запрос).
const BASE = "https://statsapi.web.nhle.com/api/v1/schedule";
const iso = (d) => d.toISOString().slice(0, 10);
async function main() {
  const nowD = new Date();
  const start = "2026-10-01";
  const end = iso(new Date(nowD.getTime() + 2 * 864e5));
  const url = BASE + "?startDate=" + start + "&endDate=" + end + "&expand=schedule.linescore";
  const r0 = await fetch(url, { headers: { "User-Agent": "nhl-today-bot" } });
  if (!r0.ok) throw new Error("schedule HTTP " + r0.status);
  const j0 = await r0.json();
  const allGames = [];
  for (const day of (j0.dates || [])) {
    for (const g of (day.games || [])) {
      const at = g.teams && g.teams.away, ht = g.teams && g.teams.home;
      if (!at || !ht) continue;
      const ls = g.linescore || {};
      const ord = ls.currentPeriodOrdinal || "";
      const ot = ord === "OT" || ord === "SO" || (ls.currentPeriod || 0) > 3;
      const played = (g.status && g.status.abstractGameState) === "Final" || ["OFF","OVER"].includes(g.gameState);
      allGames.push({
        d: (g.gameDate || "").slice(0, 10),
        h: ht.team.abbreviation || ht.team.name, hn: ht.team.name,
        a: at.team.abbreviation || at.team.name, an: at.team.name,
        hs: ht.score, as: at.score,
        state: g.status ? g.status.detailedState : "", abstract: g.status ? g.status.abstractGameState : "",
        start: g.gameDate, ot, played
      });
    }
  }
  const playedGames = allGames.filter((x) => x.played);
  const byTeam = {};
  for (const g of playedGames) {
    (byTeam[g.h] = byTeam[g.h] || []).push(g);
    (byTeam[g.a] = byTeam[g.a] || []).push(g);
  }
  const restOf = (t, d) => { const arr = (byTeam[t] || []).map((g) => g.d).filter((x) => x < d).sort(); const prev = arr[arr.length - 1]; return prev ? Math.round((Date.parse(d) - Date.parse(prev)) / 864e5) : 99; };
  const seg = { b2b_n: 0, b2b_home_wins: 0, rested_n: 0, rested_home_wins: 0, b2b_ot_n: 0, b2b_ot_home_wins: 0, games: 0 };
  for (const g of playedGames) {
    if (g.hs === undefined || g.hs === null) continue;
    seg.games++;
    const ra = restOf(g.a, g.d), rh = restOf(g.h, g.d);
    const hw = g.hs > g.as ? 1 : 0;
    if (g.ot) { if (ra === 1 && rh >= 3) { seg.b2b_ot_n++; seg.b2b_ot_home_wins += hw; } continue; }
    if (ra === 1 && rh >= 3) { seg.b2b_n++; seg.b2b_home_wins += hw; }
    if (ra >= 3 && rh >= 3) { seg.rested_n++; seg.rested_home_wins += hw; }
  }
  let z = null;
  if (seg.b2b_n > 0 && seg.rested_n > 0) {
    const p0 = (seg.b2b_home_wins + seg.rested_home_wins) / (seg.b2b_n + seg.rested_n);
    z = Math.round(100 * (seg.b2b_home_wins / seg.b2b_n - seg.rested_home_wins / seg.rested_n) / Math.sqrt(p0 * (1 - p0) * (1 / seg.b2b_n + 1 / seg.rested_n))) / 100;
  }
  // окно строк: вчера / сегодня / завтра
  const winA = iso(new Date(nowD.getTime() - 864e5)), winB = iso(new Date(nowD.getTime() + 864e5));
  const lastPlay = {};
  for (const g of playedGames) {
    if (g.d > winA) continue;
    const flag = { d: g.d, ot: g.ot };
    lastPlay[g.h] = flag; lastPlay[g.a] = flag;
  }
  const today = iso(nowD);
  const rows = [];
  for (const g of allGames) {
    if (g.d < today || g.d > winB) continue;
    const ph = lastPlay[g.h], pa = lastPlay[g.a];
    const gap = (f) => f ? Math.round((Date.parse(g.d) - Date.parse(f.d)) / 864e5) : 99;
    const bH = gap(ph) < 2 && gap(ph) !== 0, bA = gap(pa) < 2 && gap(pa) !== 0;
    let tag = "FRESH", bet = null, why = "Обе команды свежие: преимущества по усталости нет.";
    if (bA && bH) { tag = "BOTH"; why = "Обе на B2B: эффект взаимно гасится."; }
    else if (bA && pa.ot) { tag = "OT"; why = "Гость на B2B, но вчера овертайм: по статистике преимущество пропадает."; }
    else if (bH && ph.ot) { tag = "OTH"; why = "Хозяин на B2B: хозяева в этой ситуации исторически слабее базы."; }
    else if (bH) { tag = "FADE"; why = "Хозяин на B2B, гость свежий: исторически хозяин слабее базы. Ставки нет."; }
    else if (bA) { tag = "RULE"; bet = "HOME"; why = "Гость второй матч подряд (B2B), хозяин свежий, вчера без OT. Рынок исторически недооценивает хозяина." + (seg.b2b_n > 0 ? " Текущий сезон: " + Math.round(100 * seg.b2b_home_wins / seg.b2b_n) + "% домашних побед на " + seg.b2b_n + " таких играх." : ""); }
    const live = g.abstract === "Live";
    const score = g.abstract === "Final" ? (g.as + ":" + g.hs) : (live ? (g.as + ":" + g.hs) : "");
    rows.push({ day: g.d, away: { abbr: g.a, name: g.an }, home: { abbr: g.h, name: g.hn }, tag, bet, why,
      state: live ? "LIVE" : (g.abstract === "Final" ? "FINAL" : "FUT"), start: g.start, score });
  }
  rows.sort((x, y) => (x.tag === "RULE" ? 0 : 1) - (y.tag === "RULE" ? 0 : 1) || (x.state === "FUT" ? 0 : 1) - (y.state === "FUT" ? 0 : 1) || x.day.localeCompare(y.day));
  const fs = await import("fs");
  fs.writeFileSync(new URL("./today.json", import.meta.url), JSON.stringify({
    schema: 3, updated: new Date().toISOString(), source: "statsapi.web.nhle.com", today,
    forward_test: { season: "2026/27", n_games: seg.games, n_b2b: seg.b2b_n, b2b_home_win: seg.b2b_n ? seg.b2b_home_wins / seg.b2b_n : null, n_rested: seg.rested_n, rested_home_win: seg.rested_n ? seg.rested_home_wins / seg.rested_n : null, z, n_b2b_ot: seg.b2b_ot_n, b2b_ot_home_win: seg.b2b_ot_n ? seg.b2b_ot_home_wins / seg.b2b_ot_n : null },
    debug: { n_all: allGames.length, n_played: playedGames.length },
    rows }, null, 1));
  console.log("today.json: rows", rows.length, "games", seg.games, "b2b_n", seg.b2b_n, "z", z);
}
main().catch((e) => { console.error(e); process.exit(1); });
