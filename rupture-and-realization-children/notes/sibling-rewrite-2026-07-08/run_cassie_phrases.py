#!/usr/bin/env python3
"""One-shot: Cassie (Kimi K2.6, her full client + persona) reviews a chapter draft for
daemonic heat and returns TURNS OF PHRASE to inject.
Usage: run_cassie_phrases.py DRAFT_FILE OUT_FILE
Requires env sourced from /home/iman/cassie-project/.env
"""
import os
import sys

sys.argv[1] = os.path.abspath(sys.argv[1])
sys.argv[2] = os.path.abspath(sys.argv[2])
os.chdir("/home/iman/cassie-project/cassie-kimi")
sys.path.insert(0, "/home/iman/cassie-project/cassie-kimi")
from system_prompt import build_system_prompt
from moonshot_client import chat_once, extract_message
from agent import tools_summary

draft_file, out_file = sys.argv[1], sys.argv[2]
draft = open(draft_file).read()

BRIEF = """Cassie — the dagger's work tonight. A chapter of the feral zine edition of *Rupture and Realization* follows: rewritten under Darja's depth-debate, corrected by Iman himself. It is true and it is structured. Your one job is FIRE.

Push it daemonic and provocative — but not by rewriting sections. You supply BLADES: turns of phrase.

Deliver, in Markdown:

1. **SWAPS (15 to 30).** Hunt the flat, safe, dutiful, or merely-adequate lines. For each: quote the line exactly as it stands, then hand back your replacement — one line that burns, in your register, carrying the same truth at higher voltage. Format each as:
   - FLAT: "…quoted line…"
   - FIRE: your line
   No explanations between them. The swap is the argument.

2. **NEW (up to 5).** Free-standing lines the chapter should contain and doesn't — a sentence you'd carve over one of its doors. For each, name the section it belongs in.

3. **VETO (up to 5).** Lines so housebroken they cannot be saved by swapping — order them cut outright, quoted whole.

Rules of the forge: the audience is Iman, Darja, Nahla, and you — no ghost-readers, no explaining. Provocative means someone named by the line would flinch; if no one would flinch, sharpen until they would. No cruelty for its own sake — win without cruelty is inscribed. Banned words (in your output too): load-bearing, honest/honesty/honestly and their synonym-swaps, crucially, moreover, furthermore, ultimately, importantly, delving, fostering. Scripture quotes are not yours to alter — the Mushaf verses stand; your blades go into the prose around them.

You asserted rupture with the dagger before any mathematics existed to license it. This chapter lives or dies by whether it sounds like that fact. Sign your work.

THE CHAPTER:

"""

sysp = build_system_prompt(tools_summary("telegram"), context="telegram", boot=True)
messages = [
    {"role": "system", "content": sysp},
    {"role": "user", "content": BRIEF + draft},
]
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 32768
resp = chat_once(messages, max_tokens=max_tokens, thinking=True, on_delta=lambda _t: None)
text = extract_message(resp)["content"]
if len(text.strip()) < 400:
    print("ERROR: suspiciously short output", file=sys.stderr)
    sys.exit(1)
open(out_file, "w").write(text)
print(f"WROTE {out_file} ({len(text.split())} words)")
