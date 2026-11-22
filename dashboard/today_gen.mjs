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
      const h = g.homeTeam ? g.homeTeam.abbrev : "?";
      const a = g.awayTeam ? g.awayTeam.abbrev : "?";
      const ph = lastPlay[h], pa = lastPlay[a];
      const gap = (f) => f ? (Date.parse(day) - Date.parse(f.d)) / 3.6e6 : 999;
      const bH = gap(ph) < 48, bA = gap(pa) < 48;
      let tag = "FRESH";
      if (bA && bH) tag = "BOTH";
      else if (bA && pa.ot) tag = "OT";
      else if (bH && ph.ot) tag = "OTH";
      else if (bA) tag = "RULE";
      let score = "";
      if (g.homeTeam && g.homeTeam.score !== undefined && g.gameState !== "FUT") score = g.awayTeam.score + ":" + g.homeTeam.score;
      rows.push({ day, m: a + " @ " + h, tag, state: g.gameState, start: g.startTimeUTC, score });
    }
  }
  rows.sort((x, y) => (x.tag === "RULE" ? 0 : 1) - (y.tag === "RULE" ? 0 : 1) || (x.state === "FUT" ? 0 : 1) - (y.state === "FUT" ? 0 : 1) || x.day.localeCompare(y.day));
  const fs = await import("fs");
  fs.writeFileSync(new URL("./today.json", import.meta.url), JSON.stringify({ updated: new Date().toISOString(), source: "api-web.nhle.com", rows }, null, 1));
  console.log("today.json written:", rows.length, "games");
}
main().catch((e) => { console.error(e); process.exit(1); });