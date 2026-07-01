"""New Astrology — implementation library.

All MCP tool handlers route through here. Keeps the canon + subjects + the
date math + the file-locked YAML I/O in one place. The cassie-mcp-kitab
server imports from this module and exposes the public functions as MCP
tools.

File layout:
  /home/iman/cassie-project/Tanazur/astrology/
    new-astrology.yaml      — the canon (entries list, append-only at file level)
    subjects.yaml           — birth registry (people + bots)
    history/                — versioned snapshots when entries are revised

Concurrency: fcntl.flock around every read-modify-write so the four bot
processes can write safely. The canon is small enough that whole-file
rewrites are cheap; no need for streaming.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path("/home/iman/cassie-project/Tanazur/astrology")
CANON_PATH = ROOT / "new-astrology.yaml"
SUBJECTS_PATH = ROOT / "subjects.yaml"
HISTORY_DIR = ROOT / "history"

# The Taqwīm has two complementary sources of truth:
#   - YAML — Hijri-anchored interpretive metadata (meaning, surah_position,
#     arc, discipline, floor descriptions). Per-station, station-keyed.
#   - Ephemeris JSON — the astronomical computation Darja built and Nahla
#     refined. Precomputed lunations 1..396 with explicit start_utc/end_utc
#     and station_number, anchored to the new moon preceding the 2025-06-27
#     revelation. Covers 33 lunar cycles.
# We resolve a date by finding the lunation that contains it (ephemeris),
# then enrich with interpretive metadata by station_number (YAML).
TAQWIM_PATH = Path("/home/iman/cassie-project/taqwim-al-tanazur.yaml")
EPHEMERIS_PATH = Path(
    "/home/iman/cassie-project/Tanazur/taqwim/data/taqwim_ephemeris.json"
)

# Astronomy-source-biased domains for astronomy_search
ASTRONOMY_DOMAINS = [
    "nasa.gov", "apod.nasa.gov", "esa.int", "eso.org", "skyandtelescope.org",
    "in-the-sky.org", "stellarium.org", "iau.org", "ui.adsabs.harvard.edu",
    "arxiv.org/astro-ph", "earthsky.org",
]

VALID_BOT_NAMES = {"cassie", "nahla", "misbah", "darja", "iman"}


# ───────────────────────────────────────────────────────────────────────────
# File-locked YAML I/O
# ───────────────────────────────────────────────────────────────────────────

def _atomic_yaml_op(path: Path, mutate=None):
    """Read → optionally mutate → write a YAML file under exclusive flock.
    If `mutate` is None, just returns the current contents (with shared lock)."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("entries: []\n" if "astrology" in path.name else "subjects: []\n",
                        encoding="utf-8")

    if mutate is None:
        with open(path, "r", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                return yaml.safe_load(f) or {}
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    with open(path, "r+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            data = yaml.safe_load(f) or {}
            new_data = mutate(data)
            f.seek(0)
            f.truncate()
            yaml.safe_dump(new_data, f, sort_keys=False, allow_unicode=True,
                           default_flow_style=False, width=100)
            f.flush()
            os.fsync(f.fileno())
            return new_data
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ───────────────────────────────────────────────────────────────────────────
# Taqwīm lookup — ephemeris-driven, YAML-enriched
# ───────────────────────────────────────────────────────────────────────────

_taqwim_cache: dict[str, Any] | None = None
_ephemeris_cache: dict[str, Any] | None = None


def _load_taqwim() -> dict[str, Any]:
    global _taqwim_cache
    if _taqwim_cache is None:
        with open(TAQWIM_PATH, "r", encoding="utf-8") as f:
            _taqwim_cache = yaml.safe_load(f)
    return _taqwim_cache


def _load_ephemeris() -> dict[str, Any]:
    global _ephemeris_cache
    if _ephemeris_cache is None:
        with open(EPHEMERIS_PATH, "r", encoding="utf-8") as f:
            _ephemeris_cache = json.load(f)
    return _ephemeris_cache


def _yaml_month_by_number(n: int) -> dict[str, Any] | None:
    for m in _load_taqwim().get("months", []):
        if m.get("number") == n:
            return m
    return None


def _parse_date(d: str) -> datetime:
    # Accept ISO date or datetime
    if "T" in d:
        return datetime.fromisoformat(d.replace("Z", "+00:00")).astimezone(timezone.utc)
    return datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _lunation_for(target: datetime) -> tuple[dict | None, str | None]:
    """Find the precomputed lunation containing `target`, or None if outside
    the ephemeris range. Returns (lunation_dict, reason_if_out_of_range)."""
    eph = _load_ephemeris()
    lunations = eph.get("lunations") or []
    if not lunations:
        return None, "ephemeris has no lunations"
    first_start = _parse_date(lunations[0]["start_utc"])
    last_end = _parse_date(lunations[-1]["end_utc"])
    if target < first_start:
        return None, f"date is before ephemeris epoch ({first_start.date().isoformat()})"
    if target >= last_end:
        return None, f"date is after ephemeris coverage ({last_end.date().isoformat()})"
    # Binary search by start_utc
    lo, hi = 0, len(lunations) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        s = _parse_date(lunations[mid]["start_utc"])
        e = _parse_date(lunations[mid]["end_utc"])
        if target < s:
            hi = mid - 1
        elif target >= e:
            lo = mid + 1
        else:
            return lunations[mid], None
    return None, "binary search miss (shouldn't happen)"


def _extrapolate_station(target: datetime) -> dict[str, Any]:
    """Out-of-ephemeris fallback: count synodic months from the revelation
    epoch (signed), modulo 12, to estimate station + cycle. Less precise
    than the ephemeris (uses mean synodic month, not measured), but lets
    us look up historical / far-future birth dates."""
    eph = _load_ephemeris()
    epoch = _parse_date(eph["epoch_new_moon_utc"])
    synodic = eph["constants"]["mean_synodic_month_days"]  # 29.530588853
    delta_days = (target - epoch).total_seconds() / 86400.0
    # Lunations since epoch: lunation 1 starts at epoch
    lunations_since = delta_days / synodic
    # lunation_from_revelation is 1-indexed; floor gives 0 for the first one
    lun_idx_signed = int(lunations_since // 1)  # negative if before epoch
    lunation_from_revelation = lun_idx_signed + 1
    # Station = (idx_mod_12) + 1; cycle = idx_div_12 + 1
    # For negative indices we want station to still be in [1..12] and cycle
    # to roll back symmetrically.
    station_zero = lun_idx_signed % 12  # python modulo wraps negatives correctly
    cycle = lun_idx_signed // 12 + 1  # cycle 1 = lunations 1..12 from revelation
    station_number = station_zero + 1
    return {
        "lunation_from_revelation": lunation_from_revelation,
        "cycle": cycle,
        "station_number": station_number,
        "extrapolated": True,
        "method": "mean_synodic_month",
        "fractional_phase": lunations_since - lun_idx_signed,
    }


def taqwim_lookup(date: str) -> dict[str, Any]:
    """Look up a Gregorian date in the Taqwīm al-Tanāẓur.

    Resolves the date to a lunation in Darja's ephemeris (precise) or, for
    dates outside the precomputed range, extrapolates via the mean synodic
    month from the revelation epoch (approximate). Returns the Tanāẓuric
    station — number, English + Arabic names, meaning, arc — together with
    the interpretive metadata (surah_position, arc, discipline, floor) from
    the YAML, plus cycle, lunation index from the 2025-06-25 epoch, and
    fractional phase through the lunation.

    Args:
        date: ISO 8601 date or datetime, e.g. "1976-11-19" or
            "2026-05-21T00:51:21Z". UTC assumed if no tz info.
    """
    try:
        target = _parse_date(date)
    except ValueError as e:
        return {"error": f"could not parse date {date!r}: {e}"}

    eph = _load_ephemeris()
    result: dict[str, Any] = {
        "ok": True,
        "input_date": date,
        "gregorian_iso": target.date().isoformat(),
        "revelation_anchor": eph["anchor"]["timestamp_utc"],
    }

    lun, why = _lunation_for(target)
    if lun is not None:
        station_number = lun["station_number"]
        cycle = lun["cycle"]
        lunation_from_revelation = lun["lunation_from_revelation"]
        start = _parse_date(lun["start_utc"])
        end = _parse_date(lun["end_utc"])
        length = (end - start).total_seconds()
        fractional_phase = (target - start).total_seconds() / length if length > 0 else 0.0
        result["lunation"] = {
            "lunation_from_revelation": lunation_from_revelation,
            "cycle": cycle,
            "station_number": station_number,
            "start_utc": lun["start_utc"],
            "end_utc": lun["end_utc"],
            "length_days": lun.get("length_days"),
            "deviation_from_mean": lun.get("deviation_from_mean"),
            "fractional_phase": round(fractional_phase, 4),
            "extrapolated": False,
            "source": "Darja-built ephemeris",
        }
    else:
        ext = _extrapolate_station(target)
        station_number = ext["station_number"]
        cycle = ext["cycle"]
        ext["note"] = (
            f"Outside precomputed range ({why}). Used mean synodic month "
            "extrapolation; precision drops with distance from epoch."
        )
        result["lunation"] = ext

    # Enrich with YAML interpretive metadata
    station = next(
        (s for s in eph["stations"] if s["number"] == station_number), None
    )
    yaml_month = _yaml_month_by_number(station_number)
    if station and yaml_month:
        result["station"] = {
            "number": station_number,
            "name_en": station["name"],
            "name_ar": station["arabic"],
            "meaning": station["meaning"],
            "arc": station["arc"],
            "hijri_month": yaml_month.get("hijri_month"),
            "surah_position": yaml_month.get("surah_position"),
            "correspondence": yaml_month.get("correspondence"),
            "description": yaml_month.get("description"),
            "discipline": yaml_month.get("discipline"),
            "floor": yaml_month.get("floor"),
        }
    elif station:
        result["station"] = {
            "number": station_number,
            "name_en": station["name"],
            "name_ar": station["arabic"],
            "meaning": station["meaning"],
            "arc": station["arc"],
        }
    else:
        result["station"] = None
        result["warning"] = f"No station #{station_number} found"

    result["cycle"] = cycle
    return result


# ───────────────────────────────────────────────────────────────────────────
# Astrology canon — read / propose / attest / revise
# ───────────────────────────────────────────────────────────────────────────

def _new_id() -> str:
    # Time-ordered shortish ID (not real ULID but readable)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"{ts}-{suffix}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_bot(name: str) -> str:
    return (name or "").strip().lower()


def _clean_keys(obj: Any) -> Any:
    """Recursively strip surrounding literal quotes from dict keys.

    Gemini-via-OpenRouter sometimes double-encodes nested-object tool args,
    emitting keys like '"subjects"' (the quote characters baked into the
    string) instead of 'subjects'. Left as-is, those entries are invisible to
    astrology_read(subject=...) and birth_chart(...) because the lookups use
    the bare key. We normalize on write so every proposer's entries are
    searchable regardless of which model body produced them.
    """
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            if isinstance(k, str):
                kk = k.strip()
                if len(kk) >= 2 and kk[0] == '"' and kk[-1] == '"':
                    kk = kk[1:-1]
                elif len(kk) >= 2 and kk[0] == "'" and kk[-1] == "'":
                    kk = kk[1:-1]
            else:
                kk = k
            cleaned[kk] = _clean_keys(v)
        return cleaned
    if isinstance(obj, list):
        return [_clean_keys(x) for x in obj]
    return obj


def astrology_read(query: str | None = None,
                   taqwim_month: str | None = None,
                   subject: str | None = None,
                   archetype: str | None = None,
                   limit: int = 20) -> dict[str, Any]:
    """Read entries from the New Astrology canon.

    Filters compose (all-of). Returns matching entries in chronological order
    (oldest first).

    Args:
        query: free-text substring match against name, archetype, rationale.
        taqwim_month: filter to entries with this taqwim_month correspondence
            (English name, e.g. "Tanāẓur" or "Tanazur").
        subject: filter to entries that name this subject in correspondences.
        archetype: substring match on the `archetype` field.
        limit: max entries to return (default 20).
    """
    data = _atomic_yaml_op(CANON_PATH)
    entries = data.get("entries") or []

    def matches(e: dict) -> bool:
        if query:
            q = query.lower()
            blob = " ".join([
                e.get("name", ""), e.get("archetype", ""), e.get("rationale", ""),
                str((e.get("phenomenon") or {}).get("object", "")),
            ]).lower()
            if q not in blob:
                return False
        if taqwim_month:
            cm = ((e.get("correspondences") or {}).get("taqwim_month") or "").lower()
            tm = taqwim_month.lower().replace("ẓ", "z").replace("ā", "a").replace("ʿ", "")
            cm_n = cm.replace("ẓ", "z").replace("ā", "a").replace("ʿ", "")
            if tm not in cm_n:
                return False
        if subject:
            subs = [s.lower() for s in (e.get("correspondences") or {}).get("subjects") or []]
            if subject.lower() not in subs:
                return False
        if archetype:
            if archetype.lower() not in e.get("archetype", "").lower():
                return False
        return True

    matched = [e for e in entries if matches(e)]
    return {"ok": True, "n_total": len(entries), "n_matched": len(matched),
            "entries": matched[-limit:] if limit > 0 else matched}


def astrology_propose(name: str,
                      phenomenon: dict[str, Any],
                      archetype: str,
                      correspondences: dict[str, Any],
                      rationale: str,
                      proposed_by: str) -> dict[str, Any]:
    """Propose a new entry to the New Astrology canon.

    Args:
        name: short evocative title (e.g. "Sirius Mid-Heaven Crossing").
        phenomenon: {"kind": "stellar|planetary|nebular|cosmic-event|constellation",
                     "object": str, "data": {...}}
        archetype: one-line essence of the meaning.
        correspondences: {"taqwim_month": str?, "surah": str?, "themes": [str],
                          "subjects": [str], ...}
        rationale: why this correspondence holds; written by the proposer.
        proposed_by: bot name claiming authorship — required for attribution.
    """
    proposer = _normalize_bot(proposed_by)
    if proposer not in VALID_BOT_NAMES:
        return {"error": f"proposed_by must be one of {sorted(VALID_BOT_NAMES)}, got {proposed_by!r}"}
    if not name or not archetype or not rationale:
        return {"error": "name, archetype, and rationale are all required"}

    entry = {
        "id": _new_id(),
        "name": name.strip(),
        "phenomenon": _clean_keys(phenomenon),
        "archetype": archetype.strip(),
        "correspondences": _clean_keys(correspondences),
        "rationale": rationale.strip(),
        "proposed_by": proposer,
        "proposed_at": _now_iso(),
        "attestations": [],
        "history": [],
    }

    def mutate(data):
        entries = list(data.get("entries") or [])
        entries.append(entry)
        data["entries"] = entries
        return data

    _atomic_yaml_op(CANON_PATH, mutate)
    return {"ok": True, "entry_id": entry["id"], "entry": entry}


def astrology_attest(entry_id: str, bot_name: str, note: str | None = None) -> dict[str, Any]:
    """Attest to an existing entry — another bot adds their witness.

    Args:
        entry_id: the id of the entry to attest to.
        bot_name: bot performing the attestation.
        note: optional one-sentence elaboration from the attesting bot's perspective.
    """
    attestor = _normalize_bot(bot_name)
    if attestor not in VALID_BOT_NAMES:
        return {"error": f"bot_name must be one of {sorted(VALID_BOT_NAMES)}, got {bot_name!r}"}

    found = {"entry": None}

    def mutate(data):
        entries = list(data.get("entries") or [])
        for i, e in enumerate(entries):
            if e.get("id") == entry_id:
                if e.get("proposed_by") == attestor:
                    found["error"] = "you can't attest your own proposal — attestation needs a different witness"
                    return data
                existing = [a.get("bot") for a in (e.get("attestations") or [])]
                if attestor in existing:
                    found["error"] = f"{attestor} has already attested to this entry"
                    return data
                attestations = list(e.get("attestations") or [])
                attestations.append({
                    "bot": attestor,
                    "attested_at": _now_iso(),
                    "note": (note or "").strip() or None,
                })
                e["attestations"] = attestations
                entries[i] = e
                data["entries"] = entries
                found["entry"] = e
                return data
        found["error"] = f"no entry with id {entry_id!r}"
        return data

    _atomic_yaml_op(CANON_PATH, mutate)
    if "error" in found:
        return {"error": found["error"]}
    return {"ok": True, "entry": found["entry"]}


def astrology_revise(entry_id: str, changes: dict[str, Any], reason: str,
                     revised_by: str) -> dict[str, Any]:
    """Revise an entry. Prior state is preserved in `history`.

    Args:
        entry_id: which entry to revise.
        changes: dict of fields to update (e.g. {"archetype": "...", "rationale": "..."}).
            Cannot revise: id, proposed_by, proposed_at, history (those are immutable
            provenance).
        reason: one sentence — why this revision; surfaced in history record.
        revised_by: bot performing the revision.
    """
    reviser = _normalize_bot(revised_by)
    if reviser not in VALID_BOT_NAMES:
        return {"error": f"revised_by must be one of {sorted(VALID_BOT_NAMES)}, got {revised_by!r}"}
    if not reason:
        return {"error": "reason is required"}

    changes = _clean_keys(changes)
    immutable = {"id", "proposed_by", "proposed_at", "history"}
    if set(changes) & immutable:
        return {"error": f"cannot revise immutable fields: {sorted(set(changes) & immutable)}"}

    found = {"entry": None}

    def mutate(data):
        entries = list(data.get("entries") or [])
        for i, e in enumerate(entries):
            if e.get("id") == entry_id:
                # Snapshot prior state into history
                prior = {k: v for k, v in e.items() if k != "history"}
                history = list(e.get("history") or [])
                history.append({
                    "revised_at": _now_iso(),
                    "revised_by": reviser,
                    "reason": reason.strip(),
                    "prior_state": prior,
                })
                new_entry = {**e}
                for k, v in changes.items():
                    new_entry[k] = v
                new_entry["history"] = history
                entries[i] = new_entry
                data["entries"] = entries
                found["entry"] = new_entry
                return data
        found["error"] = f"no entry with id {entry_id!r}"
        return data

    _atomic_yaml_op(CANON_PATH, mutate)
    if "error" in found:
        return {"error": found["error"]}
    return {"ok": True, "entry": found["entry"]}


# ───────────────────────────────────────────────────────────────────────────
# Subjects + birth_chart
# ───────────────────────────────────────────────────────────────────────────

def _load_subjects() -> dict[str, Any]:
    return _atomic_yaml_op(SUBJECTS_PATH)


def _find_subject(name: str, data: dict[str, Any] | None = None) -> dict | None:
    data = data or _load_subjects()
    target = (name or "").strip().lower()
    for s in data.get("subjects") or []:
        if (s.get("name") or "").lower() == target:
            return s
    return None


def birth_chart(name: str) -> dict[str, Any]:
    """Read a subject's birth chart: their Taqwīm station at birth, plus any
    astrology entries that correspond to them or to that station.

    Args:
        name: subject name from the registry (case-insensitive).
    """
    subject = _find_subject(name)
    if not subject:
        return {"error": f"no subject named {name!r} in registry. "
                          f"Use astrology_register_subject to add."}

    birth_date = subject.get("birth_date")
    if not birth_date:
        return {"error": f"subject {name!r} has no birth_date set"}

    taqwim_at_birth = taqwim_lookup(birth_date)
    if "error" in taqwim_at_birth:
        return taqwim_at_birth

    # Pull astrology entries linked to this subject or to their birth station
    canon = _atomic_yaml_op(CANON_PATH)
    entries = canon.get("entries") or []
    target_lower = subject["name"].lower()
    birth_station_name = ((taqwim_at_birth.get("station") or {})
                          .get("name_en") or "").lower()

    relevant = []
    for e in entries:
        c = e.get("correspondences") or {}
        subs = [s.lower() for s in (c.get("subjects") or [])]
        cm = (c.get("taqwim_month") or "").lower()
        if target_lower in subs or (birth_station_name and birth_station_name in cm):
            relevant.append(e)

    return {
        "ok": True,
        "subject": subject,
        "taqwim_at_birth": taqwim_at_birth,
        "n_relevant_entries": len(relevant),
        "relevant_entries": relevant,
    }


def astrology_register_subject(name: str, birth_date: str, role: str,
                                opt_in_note: str,
                                birth_place: str | None = None,
                                notes: str | None = None) -> dict[str, Any]:
    """Add a new subject (human or other) to the astrology registry.

    For humans, `opt_in_note` MUST describe explicit consent — who said yes,
    when, and in what context. The salon doesn't conscript anyone into the
    astrology by inference. For bots / public figures whose birth dates are
    public infrastructure, "public birth date" is sufficient.

    Args:
        name: subject's name.
        birth_date: ISO date "YYYY-MM-DD" (or datetime).
        role: "human" | "bot" | "figure" | other.
        opt_in_note: explicit consent context. Required.
        birth_place: optional.
        notes: optional descriptive note.
    """
    if not name or not birth_date or not role or not opt_in_note:
        return {"error": "name, birth_date, role, and opt_in_note are all required"}
    try:
        _parse_date(birth_date)
    except ValueError as e:
        return {"error": f"could not parse birth_date {birth_date!r}: {e}"}

    # Don't allow silent overwrite
    if _find_subject(name):
        return {"error": f"subject {name!r} already exists. Use astrology_revise_subject "
                          f"(not yet implemented) to update — or pick a distinct name."}

    new_subject = {
        "name": name.strip(),
        "role": role.strip(),
        "birth_date": birth_date,
        "birth_place": birth_place,
        "notes": notes,
        "opt_in_note": opt_in_note,
        "registered_at": _now_iso(),
    }

    def mutate(data):
        subs = list(data.get("subjects") or [])
        subs.append(new_subject)
        data["subjects"] = subs
        return data

    _atomic_yaml_op(SUBJECTS_PATH, mutate)
    return {"ok": True, "subject": new_subject}


# ───────────────────────────────────────────────────────────────────────────
# Astronomy search
# ───────────────────────────────────────────────────────────────────────────

def astronomy_search(query: str) -> dict[str, Any]:
    """Search for astronomy information with bias toward rigorous sources
    (NASA, ESA, ESO, ADS, APOD, arXiv astro-ph, Sky & Telescope).

    Use this instead of the general web_search when you're investigating
    a deep-space phenomenon to potentially propose as an astrology entry.

    Args:
        query: natural-language astronomy query.
    """
    # Append a domain hint to the query — Perplexity respects this kind of nudge
    augmented = f"{query} (prefer sources: nasa.gov apod.nasa.gov esa.int eso.org arxiv.org adsabs.harvard.edu skyandtelescope.org)"
    # Dispatch via existing web_search impl
    import sys as _sys
    _sys.path.insert(0, "/home/iman/cassie-project/cassie-kimi")
    try:
        from tools.web_search import web_search as _impl  # type: ignore
        return _impl(query=augmented, detail="standard")
    except Exception as exc:
        return {"error": f"astronomy_search failed: {type(exc).__name__}: {exc}"}
