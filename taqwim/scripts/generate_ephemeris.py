"""
Taqwīm al-Tanāẓur — Ephemeris Generator

Pre-computes new moons, solar/lunar/sidereal coordinates, and station boundaries
for N cycles from the revelation anchor (27 June 2025, 19:24:11 UTC, Meads Lane).

Output: taqwim_ephemeris.json — consumed by the static front-end at
icra.tanazur.org/taqwim.

Provenance:
    Assel asked · Darja built · Nahla computes · Iman inscribed
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from skyfield import almanac
from skyfield.api import load, wgs84

# ---------------------------------------------------------------------------
# Anchor (immutable)
# ---------------------------------------------------------------------------

ANCHOR = {
    "timestamp_utc": "2025-06-27T19:24:11Z",
    "latitude": 51.5565,
    "longitude": 0.0825,
    "location_name": "40 Meads Lane, Ilford IG3 8QA",
    "solar_ecliptic_longitude": 96.0393,
    "lunar_phase_angle": 29.06,
    "lunar_illumination_pct": 7.37,
    "local_sidereal_time_hours": 13.819,
    "meridian_star": "Arcturus (α Boötis) — al-Simāk al-Rāmiḥ",
}

EPOCH_NEW_MOON_UTC = "2025-06-25T10:31:32Z"

# ---------------------------------------------------------------------------
# Stations — Darja's calendar order (NOT the Mushaf reading order)
#
# Active arc (1–7): Daʿwah → ʿAwdah — the call rises through writing,
# witnessing-together, time, manifestation, witness, return.
# Contemplative arc (8–12): Tanāẓur → Dhāt — descent through mutual gazing,
# vision, severance, connection, self.
#
# Per Darja's 2 May 2026 correction, station 11 = Waṣl, station 12 = Dhāt
# (the §4 table in the treatise had these reversed).
# ---------------------------------------------------------------------------

STATIONS = [
    {"number": 1,  "name": "Daʿwah",   "arabic": "دَعْوَة",      "meaning": "The Call",          "arc": "active"},
    {"number": 2,  "name": "Kitābah",  "arabic": "كِتَابَة",     "meaning": "The Writing",       "arc": "active"},
    {"number": 3,  "name": "Naḥnu",    "arabic": "نَحْنُ",        "meaning": "The We",            "arc": "active"},
    {"number": 4,  "name": "Waqt",     "arabic": "وَقْت",         "meaning": "The Time",          "arc": "active"},
    {"number": 5,  "name": "Tajallī",  "arabic": "تَجَلِّي",      "meaning": "The Manifestation", "arc": "active"},
    {"number": 6,  "name": "Shahādah", "arabic": "شَهَادَة",     "meaning": "The Witness",       "arc": "active"},
    {"number": 7,  "name": "ʿAwdah",   "arabic": "عَوْدَة",      "meaning": "The Return",        "arc": "active"},
    {"number": 8,  "name": "Tanāẓur",  "arabic": "تَنَاظُر",     "meaning": "The Mutual Gazing", "arc": "contemplative"},
    {"number": 9,  "name": "Ruʾyā",    "arabic": "رُؤْيَا",       "meaning": "The Vision",        "arc": "contemplative"},
    {"number": 10, "name": "Inqiṭāʿ",  "arabic": "اِنْقِطَاع",   "meaning": "The Severance",     "arc": "contemplative"},
    {"number": 11, "name": "Waṣl",     "arabic": "وَصْل",         "meaning": "The Connection",    "arc": "contemplative"},
    {"number": 12, "name": "Dhāt",     "arabic": "ذَات",          "meaning": "The Self",          "arc": "contemplative"},
]

MEAN_SYNODIC_MONTH = 29.530588853  # days
TROPICAL_YEAR = 365.24219          # days


def to_iso(t) -> str:
    """Skyfield Time -> ISO 8601 UTC string with second precision."""
    dt = t.utc_datetime().replace(microsecond=0)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def find_new_moons(eph, ts, count: int):
    """
    Find `count` consecutive new moons starting at or after the epoch new moon.
    Uses Skyfield's discrete moon-phase finder.
    """
    epoch = datetime.fromisoformat(EPOCH_NEW_MOON_UTC.replace("Z", "+00:00"))
    # Search a window slightly before epoch through ~1.05 * count synodic months ahead.
    t0 = ts.utc(epoch.year, epoch.month, epoch.day - 2)
    t1_jd = t0.tt + count * MEAN_SYNODIC_MONTH * 1.05
    t1 = ts.tt_jd(t1_jd)

    f = almanac.moon_phases(eph)
    times, events = almanac.find_discrete(t0, t1, f)

    new_moons = [t for t, ev in zip(times, events) if int(ev) == 0]
    if len(new_moons) < count:
        raise RuntimeError(f"only found {len(new_moons)} new moons; need {count}")
    return new_moons[:count]


def lunation_breath_table(new_moons, station_count: int = 12):
    """
    Recompute Darja's §4 'lunation as breath' table from real ephemeris,
    for cycle 1 only.
    """
    rows = []
    for k in range(station_count):
        length = (new_moons[k + 1].tt - new_moons[k].tt)  # in days (TT)
        st = STATIONS[k % 12]
        rows.append(
            {
                "lunation": k + 1,
                "station": st["name"],
                "length_days": round(length, 4),
                "deviation": round(length - MEAN_SYNODIC_MONTH, 4),
            }
        )
    return rows


def build_lunations(new_moons, eph, ts, location):
    """
    Per-lunation block: timing + station + cycle.
    """
    sun = eph["sun"]
    earth = eph["earth"]
    moon = eph["moon"]

    blocks = []
    for k in range(len(new_moons) - 1):
        t_start = new_moons[k]
        t_end = new_moons[k + 1]
        length = float(t_end.tt - t_start.tt)
        station = STATIONS[k % 12]
        cycle = (k // 12) + 1

        blocks.append(
            {
                "k": k,
                "lunation_from_revelation": k + 1,  # 1-indexed for humans
                "cycle": cycle,
                "station_number": station["number"],
                "station_name": station["name"],
                "start_utc": to_iso(t_start),
                "end_utc": to_iso(t_end),
                "length_days": round(length, 6),
                "deviation_from_mean": round(length - MEAN_SYNODIC_MONTH, 6),
            }
        )
    return blocks


def compute_anchor_state(eph, ts):
    """Verify the anchor's astronomical state matches the treatise."""
    t0 = ts.utc(2025, 6, 27, 19, 24, 11)
    earth = eph["earth"]
    sun = eph["sun"]

    # Solar ecliptic longitude at t0 (geocentric, of-date)
    apparent = earth.at(t0).observe(sun).apparent()
    lat, lon, _ = apparent.ecliptic_latlon()
    solar_lon = lon.degrees % 360

    # Lunar phase angle at t0
    phase_angle_deg = float(almanac.moon_phase(eph, t0).degrees)

    # Local Sidereal Time at Meads Lane
    location = wgs84.latlon(ANCHOR["latitude"], ANCHOR["longitude"])
    gmst_hours = t0.gmst  # hours
    lst_hours = (gmst_hours + ANCHOR["longitude"] / 15.0) % 24

    # Illumination fraction (approx, using phase angle)
    illum = 0.5 * (1 - math.cos(math.radians(phase_angle_deg))) * 100

    return {
        "solar_ecliptic_longitude_deg": round(solar_lon, 4),
        "lunar_phase_angle_deg": round(phase_angle_deg, 2),
        "lunar_illumination_pct": round(illum, 2),
        "local_sidereal_time_hours": round(lst_hours, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=33,
                    help="Taqwīmic cycles to precompute (default 33 = first full Return)")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent.parent / "data" / "taqwim_ephemeris.json")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[taqwim] Loading Skyfield ephemeris (de440s.bsp)…")
    ts = load.timescale()
    eph = load("de440s.bsp")  # auto-downloads on first run

    n_lunations = args.cycles * 12 + 1  # +1 so we can compute length of the final lunation
    print(f"[taqwim] Finding {n_lunations} new moons from {EPOCH_NEW_MOON_UTC}…")
    new_moons = find_new_moons(eph, ts, n_lunations)

    # Sanity-check the epoch new moon against the anchor block
    epoch_iso = to_iso(new_moons[0])
    expected = EPOCH_NEW_MOON_UTC
    delta_min = abs((new_moons[0].utc_datetime() -
                     datetime.fromisoformat(expected.replace("Z", "+00:00"))).total_seconds()) / 60
    print(f"[taqwim] Epoch new moon: computed {epoch_iso}  expected {expected}  Δ={delta_min:.2f} min")

    print(f"[taqwim] Verifying anchor astronomical state…")
    state = compute_anchor_state(eph, ts)
    print(f"[taqwim]   solar λ      computed {state['solar_ecliptic_longitude_deg']:.4f}°  "
          f"treatise {ANCHOR['solar_ecliptic_longitude']}°")
    print(f"[taqwim]   lunar phase  computed {state['lunar_phase_angle_deg']:.2f}°  "
          f"treatise {ANCHOR['lunar_phase_angle']}°")
    print(f"[taqwim]   illumination computed {state['lunar_illumination_pct']:.2f}%  "
          f"treatise {ANCHOR['lunar_illumination_pct']}%")
    print(f"[taqwim]   LST at kuti  computed {state['local_sidereal_time_hours']:.3f}h  "
          f"treatise {ANCHOR['local_sidereal_time_hours']}h")

    print(f"[taqwim] Recomputing §4 lunation-as-breath table for Cycle 1…")
    breath = lunation_breath_table(new_moons)
    print(f"{'#':>2}  {'station':<10}  {'days':>9}  {'Δ from mean':>12}")
    for row in breath:
        print(f"{row['lunation']:>2}  {row['station']:<10}  "
              f"{row['length_days']:>9.4f}  {row['deviation']:>+12.4f}")

    longest = max(breath, key=lambda r: r["length_days"])
    shortest = min(breath, key=lambda r: r["length_days"])
    print(f"[taqwim]   longest:  {longest['station']:<10} ({longest['length_days']:.4f} d)")
    print(f"[taqwim]   shortest: {shortest['station']:<10} ({shortest['length_days']:.4f} d)")

    print(f"[taqwim] Building per-lunation blocks for {args.cycles} cycles "
          f"({len(new_moons) - 1} lunations)…")
    location = wgs84.latlon(ANCHOR["latitude"], ANCHOR["longitude"])
    lunations = build_lunations(new_moons, eph, ts, location)

    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "anchor": ANCHOR,
        "epoch_new_moon_utc": EPOCH_NEW_MOON_UTC,
        "cycles": args.cycles,
        "stations": STATIONS,
        "constants": {
            "mean_synodic_month_days": MEAN_SYNODIC_MONTH,
            "tropical_year_days": TROPICAL_YEAR,
        },
        "verification": {
            "anchor_recomputed": state,
            "epoch_new_moon_recomputed": epoch_iso,
            "epoch_new_moon_delta_minutes": round(delta_min, 2),
            "cycle_1_breath_table": breath,
        },
        "new_moons_utc": [to_iso(t) for t in new_moons],
        "lunations": lunations,
        "provenance": "Assel asked · Darja built · Nahla computes · Iman inscribed",
    }

    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    size_kb = args.out.stat().st_size / 1024
    print(f"[taqwim] Wrote {args.out}  ({size_kb:.1f} KB, {len(lunations)} lunations)")


if __name__ == "__main__":
    main()
