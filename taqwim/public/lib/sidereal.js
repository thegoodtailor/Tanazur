// Greenwich Mean Sidereal Time → Local Sidereal Time
// Meeus, Astronomical Algorithms, Ch 12 (Eq. 12.4).

const J2000 = 2451545.0;

/** GMST in hours (0–24), given Julian Date (UT). */
export function gmstHours(jd) {
  const T = (jd - J2000) / 36525;
  const gmstDeg =
    280.46061837 +
    360.98564736629 * (jd - J2000) +
    0.000387933 * T * T -
    (T * T * T) / 38710000;
  const wrapped = ((gmstDeg % 360) + 360) % 360;
  return wrapped / 15;
}

/** Local Sidereal Time at east-longitude (degrees), in hours (0–24). */
export function lstHours(jd, longitudeEastDeg) {
  return ((gmstHours(jd) + longitudeEastDeg / 15) % 24 + 24) % 24;
}
