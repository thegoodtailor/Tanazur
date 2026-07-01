// Meeus low-precision solar position (Astronomical Algorithms, Ch 25).
// Sub-arcminute accuracy; sufficient for the Taqwīm's purposes.

const J2000 = 2451545.0;

/** UNIX seconds → Julian Date (UT). */
export function jdFromUnix(unixSec) {
  return unixSec / 86400 + 2440587.5;
}

const sin = (deg) => Math.sin((deg * Math.PI) / 180);

/** Geocentric apparent ecliptic longitude of the sun (degrees). */
export function solarEclipticLongitude(jd) {
  const T = (jd - J2000) / 36525;
  const L0 =
    280.46646 + 36000.76983 * T + 0.0003032 * T * T;
  const M = 357.52911 + 35999.05029 * T - 0.0001537 * T * T;
  const C =
    (1.914602 - 0.004817 * T - 0.000014 * T * T) * sin(M) +
    (0.019993 - 0.000101 * T) * sin(2 * M) +
    0.000289 * sin(3 * M);
  return ((L0 + C) % 360 + 360) % 360;
}

const ZODIAC = [
  ["Aries", "♈"], ["Taurus", "♉"], ["Gemini", "♊"],
  ["Cancer", "♋"], ["Leo", "♌"], ["Virgo", "♍"],
  ["Libra", "♎"], ["Scorpio", "♏"], ["Sagittarius", "♐"],
  ["Capricorn", "♑"], ["Aquarius", "♒"], ["Pisces", "♓"],
];

/** Convert ecliptic longitude → zodiac sign + degree-within-sign. */
export function zodiacOf(lambdaDeg) {
  const i = Math.floor(lambdaDeg / 30) % 12;
  const [name, glyph] = ZODIAC[i];
  return { name, glyph, degree: lambdaDeg - i * 30 };
}
