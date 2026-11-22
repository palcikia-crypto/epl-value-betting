// today.json (schema 3): расписание + B2B-флаги + ПОЛНЫЙ форвард-тест сезона 2026/27.
const API = "https://api-web.nhle.com/v1/score/";
function seasonDates() { const out = []; const d = new Date("2026-10-01T00:00:00Z"); const now = new Date(); while (d < now) { out.push(d.toISOString().slice(0, 10)); d.setUTCDate(d.getUTCDate() + 1); } return out; }
async function main() {
  const nowJson = await (await fetch(API + "now")).json();
  const near = [nowJson.prevDate, nowJson.currentDate, nowJson.nextDate].filter(Boolean);
  const allDays = {};
  const days = seasonDates();
  for (let s = 0; s < days.length; s += 12) {
    await Promise.all(days.slice(s, s + 12).map(async (d) => {
      try { allDays[d] = await (await fetch(API + d)).json(); } catch (e) { allDays[d] = null; }
    }));
  }
  const games = [];
  for (const d of Object.keys(allDays).sort()) {
    const j = allDays[d]; if (!j) continue;
    for (const g of (j.games || [])) {
      const at = g.awayTeam, ht = g.homeTeam;
      if (!at || !ht) continue;
      const ot = !!(g.periodDescriptor && g.periodDescriptor.periodType && g.periodDescriptor.periodType !== "REG");
      const played = ["FINAL","OFF","OVER"].includes(g.gameState);
      games.push({ d, h: ht.abbrev, a: at.abbrev, hs: ht.score, as: at.score, ot, played });
    }
  }
  const byTeam = {};
  for (const g of games.filter((x) => x.played)) {
    (byTeam[g.h] = byTeam[g.h] || []).push(g);
    (byTeam[g.a] = byTeam[g.a] || []).push(g);
  }
  const restOf = (t, d) => { const arr = (byTeam[t] || []).map((g) => g.d).filter((x) => x < d).sort(); const prev = arr[arr.length - 1]; return prev ? Math.round((Date.parse(d) - Date.parse(prev)) / 864e5) : 99; };
  const seg = { b2b_n: 0, b2b_home_wins: 0, rested_n: 0, rested_home_wins: 0, b2b_ot_n: 0, b2b_ot_home_wins: 0, games: 0 };
  for (const g of games.filter((x) => x.played && x.hs !== undefined && x.hs !== null)) {
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
  const lastPlay = {};
  for (const d of near) {
    const j = allDays[d] || {};
    for (const g of (j.games || [])) {
      const ot = !!(g.periodDescriptor && g.periodDescriptor.periodType && g.periodDescriptor.periodType !== "REG");
      const flag = { d: j.currentDate || d, ot };
      if (g.homeTeam) lastPlay[g.homeTeam.abbrev] = flag;
      if (g.awayTeam) lastPlay[g.awayTeam.abbrev] = flag;
    }
  }
  const rows = [];
  for (const d of near) {
    const j = allDays[d] || {};
    const day = j.currentDate || d;
    for (const g of (j.games || [])) {
      const at = g.awayTeam, ht = g.homeTeam;
      if (!at || !ht) continue;
      const ph = lastPlay[ht.abbrev], pa = lastPlay[at.abbrev];
      const gap = (f) => f ? (Date.parse(day) - Date.parse(f.d)) / 864e5 : 99;
      const bH = gap(ph) < 2, bA = gap(pa) < 2;
      let tag = "FRESH", bet = null, why = "Обе команды свежие: преимущества по усталости нет.";
      if (bA && bH) { tag = "BOTH"; why = "Обе на B2B: эффект взаимно гасится."; }
      else if (bA && pa.ot) { tag = "OT"; why = "Гость на B2B, но вчера овертайм: по статистике преимущество пропадает."; }
      else if (bH && ph.ot) { tag = "OTH"; why = "Хозяин на B2B: в этой ситуации хозяева исторически слабее базы."; }
      else if (bH) { tag = "FADE"; why = "Хозяин на B2B, гость свежий: исторически хозяин слабее базы. Ставки нет."; }
      else if (bA) { tag = "RULE"; bet = "HOME"; why = "Гость второй матч подряд (B2B), хозяин свежий, вчера без OT. Рынок исторически недооценивает хозяина." + (seg.b2b_n > 0 ? " Текущий сезон: " + Math.round(100 * seg.b2b_home_wins / seg.b2b_n) + "% домашних побед на " + seg.b2b_n + " таких играх." : ""); }
      const score = (ht.score !== undefined && g.gameState !== "FUT" && g.gameState !== "PRE") ? (at.score + ":" + ht.score) : "";
      rows.push({ day, away: { abbr: at.abbrev, name: (at.name && at.name.default) || at.abbrev }, home: { abbr: ht.abbrev, name: (ht.name && ht.name.default) || ht.abbrev }, tag, bet, why, state: g.gameState, start: g.startTimeUTC, score });
    }
  }
  rows.sort((x, y) => (x.tag === "RULE" ? 0 : 1) - (y.tag === "RULE" ? 0 : 1) || (x.state === "FUT" ? 0 : 1) - (y.state === "FUT" ? 0 : 1) || x.day.localeCompare(y.day));
  const fs = await import("fs");
  fs.writeFileSync(new URL("./today.json", import.meta.url), JSON.stringify({
    schema: 3, updated: new Date().toISOString(), source: "api-web.nhle.com",
    forward_test: { season: "2026/27", n_games: seg.games, n_b2b: seg.b2b_n, b2b_home_win: seg.b2b_n ? seg.b2b_home_wins / seg.b2b_n : null, n_rested: seg.rested_n, rested_home_win: seg.rested_n ? seg.rested_home_wins / seg.rested_n : null, z, n_b2b_ot: seg.b2b_ot_n, b2b_ot_home_win: seg.b2b_ot_n ? seg.b2b_ot_home_wins / seg.b2b_ot_n : null },
    rows }, null, 1));
  console.log("today.json: rows", rows.length, "games", seg.games, "b2b_n", seg.b2b_n, "z", z);
}
main().catch((e) => { console.error(e); process.exit(1); });
