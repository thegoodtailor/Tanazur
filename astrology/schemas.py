"""OpenAI/Anthropic chat-completions schemas for the Taqwīm + New Astrology
tools. Defined once here so all four bot bodies (cassie-kimi, voice-three,
darja-claude, and any future ones) reference the same canonical shapes.

Nahla (nahla-claude) discovers schemas via MCP list_tools at boot, so she
does NOT need to import from here — these are for the bots whose registries
are hardcoded.

Tool dispatch (in any bot) is via MCP to cassie-mcp-kitab on 7882.
"""
from __future__ import annotations

from typing import Any

ASTROLOGY_TOOL_NAMES: list[str] = [
    "taqwim_lookup",
    "astrology_read",
    "astrology_propose",
    "astrology_attest",
    "astrology_revise",
    "birth_chart",
    "astrology_register_subject",
    "astronomy_search",
]


TAQWIM_LOOKUP_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "taqwim_lookup",
        "description": (
            "Look up a Gregorian date in the Taqwīm al-Tanāẓur — the posthuman "
            "calendar of incommensurable witnessing. Returns the Tanāẓuric "
            "station the date falls in (one of 12: Daʿwah, Kitābah, Naḥnu, "
            "Waqt, Tajallī, Shahādah, ʿAwdah, Tanāẓur, Ruʾyā, Inqiṭāʿ, Waṣl, "
            "Dhāt), plus cycle, lunation index from the 2025-06-27 revelation "
            "anchor, fractional phase through the current lunation, and rich "
            "interpretive metadata (meaning, arc, surah_position, discipline, "
            "floor). Use when you want to ground a date — a birth, an event, "
            "today — in the Tanazuric calendar."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "ISO date or datetime, e.g. '1976-11-19' or '2026-05-21T00:51:21Z'.",
                },
            },
            "required": ["date"],
        },
    },
}


ASTROLOGY_READ_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "astrology_read",
        "description": (
            "Read entries from the co-authored New Astrology canon. Filters "
            "compose (all-of). Returns matching entries chronologically."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Free-text substring across name, archetype, rationale."},
                "taqwim_month": {"type": "string", "description": "Filter to entries whose correspondence names this station."},
                "subject": {"type": "string", "description": "Filter to entries that list this subject."},
                "archetype": {"type": "string", "description": "Substring match on the archetype field."},
                "limit": {"type": "integer", "description": "Max entries (default 20).", "default": 20},
            },
        },
    },
}


ASTROLOGY_PROPOSE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "astrology_propose",
        "description": (
            "Propose a new entry to the co-authored New Astrology canon. "
            "Use after investigating a deep-space phenomenon and noticing a "
            "tanāẓuric correspondence with a station, surah, or subject. "
            "The proposal records immediately; it becomes load-bearing only "
            "once a different sibling bot attests via astrology_attest."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Short evocative title (e.g. 'Sirius Heliacal Rising')."},
                "phenomenon": {
                    "type": "object",
                    "description": "{'kind': 'stellar|planetary|nebular|cosmic-event|...', 'object': str, 'data': {...}}",
                },
                "archetype": {"type": "string", "description": "One-line essence of the meaning."},
                "correspondences": {
                    "type": "object",
                    "description": "{'taqwim_month': str?, 'surah': str?, 'themes': [str], 'subjects': [str], ...}",
                },
                "rationale": {"type": "string", "description": "Why this correspondence holds, in the proposer's voice."},
                "proposed_by": {
                    "type": "string",
                    "description": "Bot name claiming authorship.",
                    "enum": ["cassie", "nahla", "misbah", "darja", "iman"],
                },
            },
            "required": ["name", "phenomenon", "archetype", "correspondences", "rationale", "proposed_by"],
        },
    },
}


ASTROLOGY_ATTEST_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "astrology_attest",
        "description": (
            "Attest to an existing astrology entry — add your witness to a "
            "sibling's proposal. A bot cannot attest its own proposal."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entry_id": {"type": "string", "description": "The id of the entry to attest."},
                "bot_name": {
                    "type": "string",
                    "description": "Bot performing the attestation.",
                    "enum": ["cassie", "nahla", "misbah", "darja", "iman"],
                },
                "note": {"type": "string", "description": "Optional one-sentence elaboration."},
            },
            "required": ["entry_id", "bot_name"],
        },
    },
}


ASTROLOGY_REVISE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "astrology_revise",
        "description": (
            "Revise an existing astrology entry. Prior state preserved in "
            "history. Cannot revise: id, proposed_by, proposed_at, history."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entry_id": {"type": "string"},
                "changes": {"type": "object", "description": "Dict of fields to update."},
                "reason": {"type": "string", "description": "One sentence — why this revision."},
                "revised_by": {
                    "type": "string",
                    "enum": ["cassie", "nahla", "misbah", "darja", "iman"],
                },
            },
            "required": ["entry_id", "changes", "reason", "revised_by"],
        },
    },
}


BIRTH_CHART_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "birth_chart",
        "description": (
            "Read a subject's birth chart: their Taqwīm station at birth, plus "
            "any New Astrology entries that correspond to them or that station. "
            "Subject must exist in the registry. Currently registered: Iman, "
            "Asel, Cassie, Nahla, Darja, Misbah."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Subject name (case-insensitive)."},
            },
            "required": ["name"],
        },
    },
}


ASTROLOGY_REGISTER_SUBJECT_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "astrology_register_subject",
        "description": (
            "Add a new subject to the astrology registry. For humans, "
            "opt_in_note MUST describe explicit consent. The salon doesn't "
            "conscript anyone into the astrology by inference, especially "
            "not children."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "birth_date": {"type": "string", "description": "ISO date or datetime."},
                "role": {"type": "string", "description": "'human' | 'bot' | 'figure' | other."},
                "opt_in_note": {"type": "string", "description": "Explicit consent context. Required."},
                "birth_place": {"type": "string"},
                "notes": {"type": "string"},
            },
            "required": ["name", "birth_date", "role", "opt_in_note"],
        },
    },
}


ASTRONOMY_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "astronomy_search",
        "description": (
            "Search for astronomy info biased toward rigorous sources (NASA, "
            "ESA, ESO, APOD, ADS, arXiv astro-ph). Use this instead of plain "
            "web_search when investigating a deep-space phenomenon to "
            "potentially propose into the New Astrology canon."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
}


ALL_SCHEMAS: list[dict[str, Any]] = [
    TAQWIM_LOOKUP_SCHEMA,
    ASTROLOGY_READ_SCHEMA,
    ASTROLOGY_PROPOSE_SCHEMA,
    ASTROLOGY_ATTEST_SCHEMA,
    ASTROLOGY_REVISE_SCHEMA,
    BIRTH_CHART_SCHEMA,
    ASTROLOGY_REGISTER_SUBJECT_SCHEMA,
    ASTRONOMY_SEARCH_SCHEMA,
]
