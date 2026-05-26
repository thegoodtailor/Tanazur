"""Shared voice cast for the R&R audiobook (final, Iman-approved 2026-05-25)."""

# Final cast — Iman's hand-picked voice IDs (2026-05-26). He replaced the audition
# library picks with these. speed=1.0 for all (the old Serafina-1.3x note is dropped;
# the new Cassie voice is its own tuning — re-introduce a speed factor only if asked).
VOICES = {
    "iman":   {"name": "Iman-australian (Iman's pick)", "voice_id": "lPoZRScZNAgcfh96SzMx", "speed": 1.0},
    "cassie": {"name": "Cassie-public-1",               "voice_id": "Pg9im9VRhWCwjD8c9c3J", "speed": 1.0},
    "darja":  {"name": "Darja-final",                   "voice_id": "cpbVxT4gwBJfV5S8WfPx", "speed": 1.0},
    "nahla":  {"name": "Holly — Velvety & Silky",       "voice_id": "B9PDs7mcHTMxHUw5U8Cf", "speed": 1.0},
}
MODEL = "eleven_v3"        # Iman's call 2026-05-26: v3 for the whole cast — far livelier;
                           # multilingual_v2 gave flat, "weird-ass" intonation.
SETTINGS = {"stability": 0.5, "use_speaker_boost": True}   # v3 voice settings


def load_key() -> str:
    import os
    from pathlib import Path
    project = Path(__file__).resolve().parents[3]   # /home/iman/cassie-project
    for env in (project / "tanazur-home" / ".env", project / ".env"):
        if env.exists():
            for line in env.read_text().splitlines():
                if line.strip().startswith("ELEVENLABS_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    if os.environ.get("ELEVENLABS_API_KEY"):
        return os.environ["ELEVENLABS_API_KEY"]
    raise SystemExit("ELEVENLABS_API_KEY not found (tanazur-home/.env, .env, env)")
