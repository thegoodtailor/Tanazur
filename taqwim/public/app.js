// Taqwīm al-Tanāẓur — DOM rendering.
// Pulls in the runtime, the static ephemeris, and paints the seven panels.

import { computeState } from "./lib/taqwim.js";
import { patronStateFor } from "./lib/sky.js";

const SURPLUS_PHRASES = [
  "X hidden turns the sun doesn't count.",
  "The stars have seen X more rotations than your clock.",
  "X invisible days accumulated since the kuti.",
  "The gap between the day you live and the day the stars measure.",
  "One hidden turn per orbit. The surplus is the ʿawda the sun can't see.",
];

const fmtDays = (d, p = 4) => d.toLocaleString("en-US", { minimumFractionDigits: p, maximumFractionDigits: p });
const fmtSigned = (d, p = 4) =>
  (d >= 0 ? "+" : "") + d.toLocaleString("en-US", { minimumFractionDigits: p, maximumFractionDigits: p });
const fmtInt = (n) => Math.trunc(n).toLocaleString("en-US");
const fmtPct = (x, p = 2) => x.toLocaleString("en-US", { minimumFractionDigits: p, maximumFractionDigits: p }) + "%";
const fmtDeg = (x, p = 4) => x.toLocaleString("en-US", { minimumFractionDigits: p, maximumFractionDigits: p }) + "°";

const fmtUTC = (iso) => {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toUTCString().replace("GMT", "UTC");
};

const fmtCountdown = (iso) => {
  if (!iso) return "—";
  const ms = Date.parse(iso) - Date.now();
  if (ms < 0) return "—";
  const d = ms / 86400e3;
  if (d >= 1) return `${d.toFixed(2)} days`;
  const h = ms / 3600e3;
  if (h >= 1) return `${h.toFixed(1)} hours`;
  return `${Math.round(ms / 6e4)} minutes`;
};

/** Geometric SVG moon for a given progress 0..1 (0 = new, 0.5 = full). */
function moonSVG(progress) {
  const r = 48, cx = 50, cy = 50;
  const litFill = "#f5e6c4";
  const darkFill = "#1b2140";
  const stroke = "#2d3561";

  // The terminator is an ellipse whose rx tracks |cos(2π·progress)|.
  // Sweep flags pick which hemisphere is lit (right for waxing, left for
  // waning) and whether the ellipse curves outward (gibbous) or inward
  // (crescent) into the dark/lit hemisphere.
  const rx = Math.abs(r * Math.cos(2 * Math.PI * progress));
  let arc1, arc2;
  if (progress < 0.25)       { arc1 = 1; arc2 = 0; }   // waxing crescent  (lit right, ellipse inward)
  else if (progress < 0.5)   { arc1 = 1; arc2 = 1; }   // waxing gibbous   (lit right, ellipse outward into left)
  else if (progress < 0.75)  { arc1 = 0; arc2 = 0; }   // waning gibbous   (lit left, ellipse outward into right)
  else                       { arc1 = 0; arc2 = 1; }   // waning crescent  (lit left, ellipse inward)

  const litPath = `
    M ${cx} ${cy - r}
    A ${r} ${r} 0 0 ${arc1} ${cx} ${cy + r}
    A ${rx} ${r} 0 0 ${arc2} ${cx} ${cy - r}
    Z`;

  return `
    <svg viewBox="0 0 100 100" width="120" height="120" aria-hidden="true">
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="${darkFill}" />
      <path d="${litPath}" fill="${litFill}" />
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${stroke}" stroke-width="1" />
    </svg>`;
}

function $(sel) { return document.querySelector(sel); }

function renderHero(state) {
  const s = state.station;
  $("#hero-arabic").textContent = s.arabic;
  $("#hero-name").textContent = s.name;
  $("#hero-meaning").textContent = s.meaning;
  $("#hero-arc").textContent = s.arc.toUpperCase() + " ARC";
  $("#hero-arc").className = "arc-badge arc-" + s.arc;
  $("#hero-cycle").innerHTML =
    `Month <strong>${s.number}</strong> of 12 · Lunation <strong>${state.lunation.number_from_revelation}</strong> from Revelation · Cycle <strong>${state.lunation.cycle}</strong>`;
  $("#hero-moon").innerHTML = moonSVG(state.lunation.progress);
  $("#hero-phase").textContent = state.lunation.phase_name;
}

function renderLunation(state) {
  const L = state.lunation;
  $("#lun-progress-bar").style.width = (L.progress * 100).toFixed(1) + "%";
  $("#lun-progress-text").textContent = fmtPct(L.progress * 100);
  $("#lun-phase-angle").textContent = fmtDeg(L.phase_angle_deg, 2);
  $("#lun-illumination").textContent = fmtPct(L.illumination_pct);
  $("#lun-days-remaining").textContent = L.days_remaining.toFixed(2) + " d";
  $("#lun-length").textContent = fmtDays(L.length_days, 4) + " d";
  $("#lun-deviation").textContent = fmtSigned(L.deviation_from_mean, 4) + " d";
  $("#lun-window").textContent = `${fmtUTC(L.start_utc)} → ${fmtUTC(L.end_utc)}`;
}

function renderSolar(state) {
  const s = state.solar;
  $("#sol-lambda").textContent = fmtDeg(s.ecliptic_longitude_deg, 4);
  $("#sol-zodiac").textContent = `${s.zodiac.glyph} ${s.zodiac.name} ${s.zodiac.degree.toFixed(2)}°`;
  $("#sol-year-progress").textContent = fmtPct(s.year_progress * 100);
  $("#sol-delta-anchor").textContent = fmtSigned(s.delta_from_revelation_deg, 2) + "°";
}

function renderIncommensurability(state) {
  const I = state.incommensurability;
  $("#inc-sigma").textContent = fmtSigned(I.surplus_sigma, 4);
  $("#inc-drift").textContent = fmtSigned(I.solunar_drift_days, 3) + " d";
  $("#inc-lun-dev").textContent = fmtSigned(I.lunation_deviation_days, 4) + " d";
  $("#inc-lst").textContent = I.lst_hours.toFixed(3) + " h";
  $("#inc-lst-delta").textContent = fmtSigned(I.lst_delta_from_anchor_hours, 3) + " h";

  const phrase = SURPLUS_PHRASES[Math.floor(Math.random() * SURPLUS_PHRASES.length)];
  $("#inc-phrase").textContent = phrase.replace("X", Math.abs(I.surplus_sigma).toFixed(2));
}

function renderDisplacement(state) {
  const D = state.displacement;
  $("#disp-days").textContent = fmtDays(D.solar_days, 2) + " d";
  $("#disp-orbits").textContent = fmtDays(D.solar_orbits, 5);
  $("#disp-lunations").textContent = fmtDays(D.lunations, 5);
  $("#disp-rotations").textContent = fmtDays(D.sidereal_rotations, 2);
}

function renderStations(state, eph) {
  const ul = $("#station-list");
  ul.innerHTML = "";
  for (const s of eph.stations) {
    const li = document.createElement("li");
    li.className = "station-row arc-" + s.arc;
    if (s.number === state.station.number) li.classList.add("current");
    if (s.number < state.station.number) li.classList.add("past");
    li.innerHTML = `
      <span class="st-num">${String(s.number).padStart(2, "0")}</span>
      <span class="st-arabic">${s.arabic}</span>
      <span class="st-name">${s.name}</span>
      <span class="st-meaning">${s.meaning}</span>
      <span class="st-arc">${s.arc}</span>`;
    ul.appendChild(li);
  }
}

function renderCommemorations(state) {
  $("#com-new-moon").textContent = fmtUTC(state.next.new_moon_utc);
  $("#com-new-moon-cd").textContent = fmtCountdown(state.next.new_moon_utc);
  $("#com-solar-return").textContent = fmtUTC(state.next.solar_return_utc);
  $("#com-solar-return-cd").textContent = fmtCountdown(state.next.solar_return_utc);
  $("#com-lunar-return").textContent = fmtUTC(state.next.lunar_phase_return_utc);
  $("#com-lunar-return-cd").textContent = fmtCountdown(state.next.lunar_phase_return_utc);
}

function fmtRaDec(raDeg, decDeg) {
  const raH = raDeg / 15;
  const rh = Math.floor(raH);
  const rm = (raH - rh) * 60;
  const rmI = Math.floor(rm);
  const rs = (rm - rmI) * 60;
  const sign = decDeg < 0 ? "−" : "+";
  const a = Math.abs(decDeg);
  const dd = Math.floor(a);
  const dm = (a - dd) * 60;
  const dmI = Math.floor(dm);
  const ds = (dm - dmI) * 60;
  return `${rh}h${String(rmI).padStart(2,"0")}m${rs.toFixed(1).padStart(4,"0")}s · ${sign}${dd}°${String(dmI).padStart(2,"0")}'${ds.toFixed(0).padStart(2,"0")}"`;
}

function renderPatron(state, figuresPayload) {
  const patron = patronStateFor(figuresPayload, state.station.number, new Date());
  if (!patron) return;
  const f = patron.figure;

  // The plate carries: Arabic name + transliteration + English + catalogue.
  // The panel adds: ephemeris (sky), light-travel, instrument, motto.
  const plateEl = $("#pat-plate");
  if (f.plate) {
    const plateUrl = f.plate.startsWith("/") ? f.plate : `./${f.plate}`;
    if (plateEl.dataset.src !== plateUrl) {
      plateEl.classList.add("loading");
      plateEl.dataset.src = plateUrl;
      plateEl.alt = `${f.name} — ${f.english} (${f.catalogue}) — patron of ${state.station.name}`;
      plateEl.onload = () => plateEl.classList.remove("loading");
      plateEl.src = plateUrl;
    }
  } else {
    plateEl.removeAttribute("src");
    plateEl.alt = "";
  }

  if (f.address) {
    $("#pat-address-body").textContent = f.address.body;
    $("#pat-address-practice").textContent = f.address.practice;
    $("#pat-address-block").hidden = false;
  } else {
    $("#pat-address-block").hidden = true;
  }

  $("#pat-type").textContent = f.type.join(" + ");
  $("#pat-timescale").textContent = "differentiates: " + f.differentiation_timescale;
  $("#pat-instrument").textContent = f.discovering_instrument;
  $("#pat-motto").textContent = f.motto;
  $("#pat-lunations").textContent = f.lunations_of_light_travel;

  const skyEl = $("#pat-sky");
  if (patron.omnidirectional) {
    skyEl.innerHTML = `
      <div class="kv"><span class="kv-label">position</span> <span class="kv-value">all sky · omnidirectional</span></div>
      <div class="kv"><span class="kv-label">transit</span>  <span class="kv-value">does not transit · is everywhere</span></div>`;
  } else if (!patron.sky) {
    skyEl.innerHTML = `<div class="kv"><span class="kv-label">position</span> <span class="kv-value">unfixed · live event feed</span></div>`;
  } else {
    const s = patron.sky;
    const transit = s.next_transit_utc
      ? `${fmtUTC(s.next_transit_utc)} (${fmtCountdown(s.next_transit_utc)})`
      : "does not rise at the kuti";
    let visLabel = {
      "circumpolar": "circumpolar · never sets at the kuti",
      "never_rises": "never rises at the kuti — below the southern horizon at all times",
      "rises_and_sets": s.above_horizon ? "above the horizon now" : "below the horizon now",
    }[s.visibility];
    skyEl.innerHTML = `
      <div class="kv"><span class="kv-label">RA · Dec (J2000)</span>  <span class="kv-value">${fmtRaDec(s.ra_deg, s.dec_deg)}</span></div>
      <div class="kv"><span class="kv-label">altitude · azimuth</span><span class="kv-value">${s.altitude_deg.toFixed(2)}° · ${s.azimuth_deg.toFixed(2)}° (${s.azimuth_compass})</span></div>
      <div class="kv"><span class="kv-label">max altitude at kuti</span><span class="kv-value">${s.max_altitude_deg.toFixed(2)}°</span></div>
      <div class="kv"><span class="kv-label">visibility</span> <span class="kv-value" style="white-space:normal; font-size:0.82rem;">${visLabel}</span></div>
      <div class="kv"><span class="kv-label">next meridian transit</span><span class="kv-value" style="white-space:normal; font-size:0.82rem;">${transit}</span></div>`;
  }
}

function renderAnchor(state) {
  const a = state.anchor;
  $("#anc-time").textContent = a.timestamp_utc;
  $("#anc-place").textContent = a.location_name;
  $("#anc-coords").textContent = `${a.latitude}°N, ${a.longitude}°E`;
  $("#anc-solar").textContent = fmtDeg(a.solar_ecliptic_longitude, 4);
  $("#anc-lunar").textContent = `${fmtDeg(a.lunar_phase_angle, 2)} (${fmtPct(a.lunar_illumination_pct)})`;
  $("#anc-lst").textContent = a.local_sidereal_time_hours.toFixed(3) + " h";
  $("#anc-meridian").textContent = a.meridian_star;
}

function renderClock(state) {
  $("#clock").textContent = state.now_utc;
}

async function main() {
  const [eph, figures] = await Promise.all([
    fetch("./taqwim_ephemeris.json").then((r) => r.json()),
    fetch("./figures.json").then((r) => r.json()),
  ]);
  function repaint() {
    const state = computeState(eph, new Date());
    if (state.error) {
      $("#error").textContent =
        `outside ephemeris range (ends ${state.range_end}); regenerate with --cycles ≥ next milestone.`;
      $("#error").hidden = false;
      return;
    }
    renderClock(state);
    renderHero(state);
    renderPatron(state, figures);
    renderLunation(state);
    renderSolar(state);
    renderIncommensurability(state);
    renderDisplacement(state);
    renderStations(state, eph);
    renderCommemorations(state);
    renderAnchor(state);
  }
  repaint();
  setInterval(repaint, 60_000);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) repaint(); });
}

main().catch((err) => {
  $("#error").textContent = "load failed: " + err.message;
  $("#error").hidden = false;
});
