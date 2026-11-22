// Генератор dashboard/today.json (schema 2): NHL расписание + B2B/OT-флаги и вердикт ставки.
const API = "https://api-web.nhle.com/v1/score/";
async function main() {
  const now = await (await fetch(API + "now")).json();
  const dates = [now.prevDate, now.currentDate, now.nextDate].filter(Boolean);
  const all = await Promise.all(dates.map((d) => fetch(API + d).then((r) => r.json())));
  const lastPlay = {};
  for (let i = 0; i < 2 && i < all.length; i++) {
    const j = all[i];
    for (const g of (j.games || [])) {
      const ot = !!(g.periodDescriptor && g.periodDescriptor.periodType && g.periodDescriptor.periodType !== "REG");
      const flag = { d: j.currentDate || dates[i], ot };
      if (g.homeTeam) lastPlay[g.homeTeam.abbrev] = flag;
      if (g.awayTeam) lastPlay[g.awayTeam.abbrev] = flag;
    }
  }
  const rows = [];
  for (let i = 1; i < all.length; i++) {
    const j = all[i];
    const day = j.currentDate || dates[i];
    for (const g of (j.games || [])) {
      const at = g.awayTeam, ht = g.homeTeam;
      if (!at || !ht) continue;
      const ph = lastPlay[ht.abbrev], pa = lastPlay[at.abbrev];
      const gap = (f) => f ? (Date.parse(day) - Date.parse(f.d)) / 3.6e6 : 999;
      const bH = gap(ph) < 48, bA = gap(pa) < 48;
      let tag = "FRESH", bet = null, why = "Обе команды свежие: преимущества по усталости нет.";
      if (bA && bH) { tag = "BOTH"; why = "Обе на B2B: эффект взаимно гасится."; }
      else if (bA && pa.ot) { tag = "OT"; why = "Гость на B2B, но вчера овертайм: преимущество пропадает."; }
      else if (bH && ph.ot) { tag = "OTH"; why = "Хозяин на B2B: лучше не ставить на хозяина."; }
      else if (bA) { tag = "RULE"; bet = "HOME"; why = "Гость на B2B (второй матч подряд), хозяин свежий, вчера без OT: рынок исторически недооценивает хозяина (+6.4% yield в бэктесте)."; }
      const score = (ht.score !== undefined && g.gameState !== "FUT") ? (at.score + ":" + ht.score) : "";
      rows.push({ day,
        away: { abbr: at.abbrev, name: (at.name && at.name.default) || at.abbrev },
        home: { abbr: ht.abbrev, name: (ht.name && ht.name.default) || ht.abbrev },
        tag, bet, why, state: g.gameState, start: g.startTimeUTC, score });
    }
  }
  rows.sort((x, y) => (x.tag === "RULE" ? 0 : 1) - (y.tag === "RULE" ? 0 : 1) || (x.state === "FUT" ? 0 : 1) - (y.state === "FUT" ? 0 : 1) || x.day.localeCompare(y.day));
  const fs = await import("fs");
  fs.writeFileSync(new URL("./today.json", import.meta.url), JSON.stringify({ schema: 2, updated: new Date().toISOString(), source: "api-web.nhle.com", rows }, null, 1));
  console.log("today.json written:", rows.length, "games");
}
main().catch((e) => { console.error(e); process.exit(1); });