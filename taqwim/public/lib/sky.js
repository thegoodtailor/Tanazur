// Equatorial → Horizontal coordinate transformation at the kuti.
// All angles in radians internally; degrees on the boundary.
// Sufficient precision for transit-time and altitude display (sub-arcminute).

import { lstHours } from "./sidereal.js";
import { jdFromUnix } from "./solar.js";

const D = (x) => (x * Math.PI) / 180;
const Dinv = (x) => (x * 180) / Math.PI;
const wrap360 = (x) => ((x % 360) + 360) % 360;

/**
 * Altitude / azimuth of an equatorial-coordinate target (J2000) from
 * (latitudeDeg, longitudeDeg) at Julian Date jd.
 * Azimuth measured from north, clockwise (0=N, 90=E, 180=S, 270=W).
 */
export function altAzFromKuti(raDeg, decDeg, jd, latDeg, lonDeg) {
  const lstDeg = lstHours(jd, lonDeg) * 15;
  const haDeg = ((lstDeg - raDeg + 540) % 360) - 180; // (-180, 180]
  const ha = D(haDeg);
  const dec = D(decDeg);
  const lat = D(latDeg);

  const sinAlt =
    Math.sin(dec) * Math.sin(lat) + Math.cos(dec) * Math.cos(lat) * Math.cos(ha);
  const alt = Math.asin(sinAlt);

  const cosAz =
    (Math.sin(dec) - Math.sin(alt) * Math.sin(lat)) /
    (Math.cos(alt) * Math.cos(lat) || 1e-12);
  let az = Math.acos(Math.max(-1, Math.min(1, cosAz)));
  if (Math.sin(ha) > 0) az = 2 * Math.PI - az;

  return {
    altitude_deg: Dinv(alt),
    azimuth_deg: Dinv(az),
    hour_angle_deg: haDeg,
  };
}

/** N/E/S/W cardinal label closest to azimuth. */
export function compassOf(azimuthDeg) {
  const dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];
  return dirs[Math.round(wrap360(azimuthDeg) / 22.5) % 16];
}

/**
 * Visibility classification at observer latitude.
 *   "circumpolar" — declination always above horizon
 *   "never_rises" — declination always below horizon
 *   "rises_and_sets" — normal
 */
export function visibilityClass(decDeg, latDeg) {
  if (latDeg > 0 && decDeg > 90 - latDeg) return "circumpolar";
  if (latDeg < 0 && decDeg < -90 - latDeg) return "circumpolar";
  if (latDeg > 0 && decDeg < -(90 - latDeg)) return "never_rises";
  if (latDeg < 0 && decDeg > 90 + latDeg) return "never_rises";
  return "rises_and_sets";
}

/** Maximum altitude reached at upper transit (south of zenith for δ < φ). */
export function maxAltitudeDeg(decDeg, latDeg) {
  // For an upper transit: alt = 90° - |φ - δ|
  return 90 - Math.abs(latDeg - decDeg);
}

/**
 * Time of next upper meridian transit.
 * Returns ISO UTC string. The mean rate of LST is 360.98564736629°/solar day,
 * so the time delta is (raDeg - lstNowDeg) mod 360 / that rate.
 */
export function nextTransitUTC(raDeg, unixSec, lonDeg) {
  const jd = jdFromUnix(unixSec);
  const lstDeg = lstHours(jd, lonDeg) * 15;
  const deltaDeg = ((raDeg - lstDeg) % 360 + 360) % 360;
  const ratePerSec = 360.98564736629 / 86400; // deg / second
  const secondsAhead = deltaDeg / ratePerSec;
  return new Date((unixSec + secondsAhead) * 1000)
    .toISOString()
    .replace(/\.\d+Z$/, "Z");
}

/**
 * Compose patron-figure runtime data for a station, given the figures.json
 * payload, the current state's station number, and "now" timestamp.
 */
export function patronStateFor(figuresPayload, stationNumber, now) {
  const figure = figuresPayload.figures.find(
    (f) => f.station_number === stationNumber,
  );
  if (!figure) return null;

  const out = { figure };
  const isOmni = figure.position === "omnidirectional" ||
                 figure.position === "omnidirectional / structural";
  out.omnidirectional = isOmni;

  if (isOmni || typeof figure.ra_deg !== "number") {
    out.sky = null;
    return out;
  }

  const lat = figuresPayload.kuti.latitude_deg;
  const lon = figuresPayload.kuti.longitude_deg;
  const t = now.getTime() / 1000;
  const jd = jdFromUnix(t);

  const aa = altAzFromKuti(figure.ra_deg, figure.dec_deg, jd, lat, lon);
  const visibility = visibilityClass(figure.dec_deg, lat);
  const maxAlt = maxAltitudeDeg(figure.dec_deg, lat);
  const transit = visibility === "never_rises"
    ? null
    : nextTransitUTC(figure.ra_deg, t, lon);

  out.sky = {
    ra_deg: figure.ra_deg,
    dec_deg: figure.dec_deg,
    altitude_deg: aa.altitude_deg,
    azimuth_deg: aa.azimuth_deg,
    azimuth_compass: compassOf(aa.azimuth_deg),
    visibility,
    max_altitude_deg: maxAlt,
    next_transit_utc: transit,
    above_horizon: aa.altitude_deg > 0,
  };
  return out;
}
