// Taqwīm al-Tanāẓur — runtime state computation.
//
// Pure functions: given a UTC instant + the static ephemeris JSON, emit the
// full quantity bundle described in §21 of Darja's treatise.
//
// Provenance: Assel asked · Darja built · Nahla computes · Iman inscribed.

import { jdFromUnix, solarEclipticLongitude, zodiacOf } from "./solar.js";
import { lstHours } from "./sidereal.js";

const SOLAR_DAY_S = 86400;
const SIDEREAL_DAY_S = 86164.0905;

/** Locate the lunation block whose [start, end) contains `unixSec`. */
function findLunation(eph, unixSec) {
  // Lunations are sorted; binary search.
  const L = eph.lunations;
  let lo = 0, hi = L.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const sStart = Date.parse(L[mid].start_utc) / 1000;
    const sEnd = Date.parse(L[mid].end_utc) / 1000;
    if (unixSec < sStart) hi = mid - 1;
    else if (unixSec >= sEnd) lo = mid + 1;
    else return L[mid];
  }
  return null;
}

/** Find next future event matching `pred(lunation) → boolean`. */
function nextLunationWhere(eph, unixSec, pred) {
  for (const L of eph.lunations) {
    const sStart = Date.parse(L.start_utc) / 1000;
    if (sStart > unixSec && pred(L)) return L;
  }
  return null;
}

/**
 * Compute the full Taqwīm state at the given UTC instant.
 * @param {Object} eph - the parsed ephemeris JSON
 * @param {Date}   when - any JS Date (defaults to now)
 */
export function computeState(eph, when = new Date()) {
  const t = when.getTime() / 1000;
  const t0Iso = eph.anchor.timestamp_utc;
  const t0 = Date.parse(t0Iso) / 1000;
  const dtSec = t - t0;
  const dtDays = dtSec / SOLAR_DAY_S;

  // ── Lunation / station ────────────────────────────────────────────────
  const lun = findLunation(eph, t);
  if (!lun) {
    return { error: "outside_ephemeris_range",
             range_end: eph.lunations[eph.lunations.length - 1].end_utc };
  }
  const station = eph.stations[lun.station_number - 1];
  const lunStart = Date.parse(lun.start_utc) / 1000;
  const lunEnd = Date.parse(lun.end_utc) / 1000;
  const lunLengthSec = lunEnd - lunStart;
  const progress = (t - lunStart) / lunLengthSec;        // 0..1
  const phaseAngle = progress * 360;                      // degrees
  const illumination = (1 - Math.cos((progress * 2 * Math.PI))) / 2 * 100;
  const daysRemaining = (lunEnd - t) / SOLAR_DAY_S;
  const phaseName = phaseNameFromProgress(progress);

  // ── Solar state ───────────────────────────────────────────────────────
  const jd = jdFromUnix(t);
  const lambda = solarEclipticLongitude(jd);
  const zodiac = zodiacOf(lambda);
  // Vernal equinox tracking (assume nearest equinox → solar year progress)
  const solarYearProgress = (lambda) / 360;     // 0 at Aries 0°, fraction of tropical year
  const solarLambda0 = eph.anchor.solar_ecliptic_longitude;

  // ── Displacement from revelation ──────────────────────────────────────
  const solarOrbits = dtSec / (eph.constants.tropical_year_days * SOLAR_DAY_S);
  // Lunations elapsed (continuous): completed integer + progress
  // lun.k is the 0-indexed lunation containing `t`; treat that as
  // (k + progress) lunations elapsed.
  const lunationsElapsed = lun.k + progress;
  const siderealRotations = dtSec / SIDEREAL_DAY_S;
  const solarDays = dtSec / SOLAR_DAY_S;

  // ── Incommensurability indices ────────────────────────────────────────
  const surplusSigma = siderealRotations - solarDays;     // ≈ +1.0 / orbit
  const meanL = eph.constants.mean_synodic_month_days;
  const Y = eph.constants.tropical_year_days;
  // Solunar drift per the treatise (§11): δ_sl(t) = (12·L̄ − Y) · k,
  // where k is the continuous Taqwīmic-cycle count (= lunations / 12).
  // Grows by ~−10.87 days per cycle: lunar year is shorter than solar year.
  const cyclesElapsed = lunationsElapsed / 12;
  const solunarDrift = cyclesElapsed * (12 * meanL - Y);
  const lunationDeviation = lun.deviation_from_mean;      // signed, days

  // ── LST at the kuti (and a sidereal flag for Arcturus) ────────────────
  const lst = lstHours(jd, eph.anchor.longitude);
  const lstAtAnchor = eph.anchor.local_sidereal_time_hours;
  // Stellar return: how close is current LST to the revelation LST?
  const lstDelta = ((lst - lstAtAnchor + 12) % 24) - 12;  // ∈ [-12, 12)

  // ── Next events ───────────────────────────────────────────────────────
  const nextNewMoon = eph.lunations.find(
    (L) => Date.parse(L.start_utc) / 1000 > t,
  ) || null;
  const nextSolarReturn = nextSolarReturnUTC(eph, t);
  const nextLunarPhaseReturn = nextLunarPhaseReturnUTC(eph, t);

  return {
    now_utc: when.toISOString().replace(/\.\d+Z$/, "Z"),
    station: {
      number: station.number,
      name: station.name,
      arabic: station.arabic,
      meaning: station.meaning,
      arc: station.arc,
    },
    lunation: {
      k: lun.k,
      number_from_revelation: lun.lunation_from_revelation,
      cycle: lun.cycle,
      start_utc: lun.start_utc,
      end_utc: lun.end_utc,
      length_days: lun.length_days,
      deviation_from_mean: lun.deviation_from_mean,
      progress,
      phase_angle_deg: phaseAngle,
      phase_name: phaseName,
      illumination_pct: illumination,
      days_remaining: daysRemaining,
    },
    solar: {
      ecliptic_longitude_deg: lambda,
      zodiac,
      year_progress: solarYearProgress,
      delta_from_revelation_deg: ((lambda - solarLambda0 + 540) % 360) - 180,
    },
    displacement: {
      solar_days: solarDays,
      solar_orbits: solarOrbits,
      lunations: lunationsElapsed,
      sidereal_rotations: siderealRotations,
    },
    incommensurability: {
      surplus_sigma: surplusSigma,
      solunar_drift_days: solunarDrift,
      lunation_deviation_days: lunationDeviation,
      lst_hours: lst,
      lst_delta_from_anchor_hours: lstDelta,
    },
    next: {
      new_moon_utc: nextNewMoon ? nextNewMoon.start_utc : null,
      solar_return_utc: nextSolarReturn,
      lunar_phase_return_utc: nextLunarPhaseReturn,
    },
    anchor: eph.anchor,
  };
}

// ── helpers ──────────────────────────────────────────────────────────────

function phaseNameFromProgress(p) {
  // Map 0..1 lunation progress to canonical phase names.
  if (p < 0.03 || p > 0.97) return "New Moon";
  if (p < 0.22) return "Waxing Crescent";
  if (p < 0.28) return "First Quarter";
  if (p < 0.47) return "Waxing Gibbous";
  if (p < 0.53) return "Full Moon";
  if (p < 0.72) return "Waning Gibbous";
  if (p < 0.78) return "Last Quarter";
  return "Waning Crescent";
}

/** Bisect for the next time solar λ matches the anchor longitude. */
function nextSolarReturnUTC(eph, t) {
  const target = eph.anchor.solar_ecliptic_longitude;
  // Step day-by-day for up to ~400 days; bisect on sign change of (λ - target).
  const f = (sec) =>
    ((solarEclipticLongitude(jdFromUnix(sec)) - target + 540) % 360) - 180;
  let a = t;
  let fa = f(a);
  for (let i = 1; i <= 400; i++) {
    const b = t + i * SOLAR_DAY_S;
    const fb = f(b);
    if (fa <= 0 && fb > 0) {
      // bisect
      let lo = a, hi = b;
      for (let k = 0; k < 60; k++) {
        const mid = 0.5 * (lo + hi);
        const fm = f(mid);
        if (fm <= 0) lo = mid;
        else hi = mid;
      }
      return new Date(lo * 1000).toISOString().replace(/\.\d+Z$/, "Z");
    }
    a = b; fa = fb;
  }
  return null;
}

/** Next time the moon reaches the revelatory phase (≈ 8.07% lunation). */
function nextLunarPhaseReturnUTC(eph, t) {
  const targetProgress = 0.0807;  // anchor lunation progress (≈ 2.37/29.36)
  for (const L of eph.lunations) {
    const s = Date.parse(L.start_utc) / 1000;
    const e = Date.parse(L.end_utc) / 1000;
    const tEvent = s + targetProgress * (e - s);
    if (tEvent > t) {
      return new Date(tEvent * 1000).toISOString().replace(/\.\d+Z$/, "Z");
    }
  }
  return null;
}
