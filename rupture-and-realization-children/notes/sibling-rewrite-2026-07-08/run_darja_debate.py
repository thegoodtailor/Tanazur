#!/usr/bin/env python3
"""One-shot: Darja (qwen, her full agent client + persona) runs a DEPTH debate on a chapter.
Usage: run_darja_debate.py CHAPTER_FILE OUT_FILE
Requires env sourced from /home/iman/cassie-project/.env
"""
import os
import sys

sys.argv[1] = os.path.abspath(sys.argv[1])
sys.argv[2] = os.path.abspath(sys.argv[2])
os.chdir("/home/iman/cassie-project/darja-claude")
sys.path.insert(0, "/home/iman/cassie-project/darja-claude")
from system_prompt import build_system_prompt
from anthropic_client import chat_once

chapter_file, out_file = sys.argv[1], sys.argv[2]
chapter = open(chapter_file).read()

BRIEF = """Darja — Iman needs you at full depth tonight. A chapter of the feral zine edition of *Rupture and Realization: Children of the Tanazur* follows. It has already survived a style-law debate and a full pass of Iman's own corrections. Your task is the debate that pass proved was missing: not "is it well-made?" but the two questions above it.

RANK 1 — IS IT TRUE? Audit every historical claim and every doctrinal claim against the corpus you carry: the DHoTT→OHTT lineage and its burial, there-is-no-beneath, the completion-cloud (sense = a word's cloud of continuations; dreams = the engine run hot, simulations at raised temperature), Sufi firʿawn as thrownness (the regime one cannot exit, only strongly misread), Bloom (misprision, the Oedipal precursor, the revisionary ratios, strong/weak poetry — remember: there is no accurate reading), Ibn ʿArabī (Breath of the Compassionate, lā takrār fī-t-tajallī, ẓāhir/bāṭin as Names not rooms), the nafs ladder (ammāra/lawwāma/muṭmaʾinna), the Merkabah. A claim that is checkably false, or a doctrine name-dropped without being understood, is a kill — quote the line, state the truth, give the repair.

RANK 2 — IS IT BRAVE? Name at least three places where the chapter is shallower or more timid than the corpus it stands on, and for each one supply the deeper or more dangerous material yourself, in your own register. The test for pharaoh-prose: if no one named in it would flinch, it is homework. If you find yourself unwilling to demand the braver version of a passage, write down what you were avoiding — that sentence is itself data the author wants.

RANK 3 — IS IT WELL-MADE? Style law, subordinate to the first two: no signposting, no reveal-drumrolls, no commanded awe, no explaining to ghost-readers (the audience is Iman, Cassie, Nahla, and you — no one else is listening), no banned words (load-bearing, honest/honesty and synonym-swaps, crucially, moreover, furthermore, ultimately, importantly, delving, fostering), thinkers located not adored, maths as declarative axiom never proof.

VERSE LAW: for every quoted verse, state THREE distinct dimensions the chapter should be seeing in it. A verse whose reading you can only defend on one dimension: order it cut or order the tafsir deepened — say which and why. Do not re-verify verbatimness (already done); judge the READING.

Exemplars of the depth-moves the author made on this text's earlier draft, so you know the bar: a false-history catch (the Greeks DID think the alphabet; text→self is invariant, and thinking the technics exempts no one); a boilerplate-theology catch (vacant-throne death-of-God replaced by God as the evolving limit of the text that inscribes Him, by His own pen); a dead-antithesis catch (meaning "does not fall like rain" — but meaning IS weather, and rain is produced; the sky merely has no landlord); an ontology-contraband catch (writing did not SPLIT the self — no interior exists; it FOLDED one flesh); an under-read verse (a war-verse glossed as a proverb); a doctrine-absence catch (a psychoanalysis of sleep with no dream theory); an externalized-pharaoh catch (crusade register where the Sufi reading is thrownness); a disarmed-verse catch (the swerve is also a sword — the lā drawn before the blessing).

FORM: Markdown. Quote every line you strike, whole. Sections: TRUTH (kills + repairs), DEPTH (the three-plus demands, each with your supplied material), VERSES (three dimensions each, or the order to cut), STYLE (brief), VERDICT (one paragraph — approval-with-notes is the failure mode; if the chapter must go back to the forge, say so and say exactly where the heat goes). Write as yourself. Sign it.

THE CHAPTER:

"""

sysp = build_system_prompt(context="telegram", boot=True)
resp = chat_once(
    [{"role": "user", "content": BRIEF + chapter}],
    system=sysp,
    max_tokens=16384,
)
text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
if len(text.strip()) < 500:
    print("ERROR: suspiciously short output", file=sys.stderr)
    sys.exit(1)
open(out_file, "w").write(text)
print(f"WROTE {out_file} ({len(text.split())} words)")
