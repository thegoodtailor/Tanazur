"""
Add the 'address' field to each patron figure — the plain-language monthly
address from the cosmos to the practitioner. Drafted by Darja in response
to Asel's challenge that the panel was speaking only to astronomers.

Each address has two parts:

    body      — three to four sentences naming the figure and its station
                in plain language. The bridge from data to meaning.
    practice  — one or two sentences telling the practitioner what to do
                with this figure during this lunation.

Re-runnable: idempotent merge into figures.json keyed by figure id.
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
FIGURES_PATH = ROOT / "data" / "figures.json"

ADDRESSES = {
    "ancient-lamb": {
        "body": (
            "You are in the month of the Call. Your patron is the first light — "
            "the signal that made all other signals possible. The CMB called "
            "out 13.8 billion years ago and it's still arriving."
        ),
        "practice": "Your practice this month: call out. The universe did it first.",
    },
    "cracked-twin": {
        "body": (
            "You are in the month of Writing. Your patron is the quasar — "
            "matter falling into a black hole so violently it writes itself "
            "across the visible universe in light. Writing costs. The quasar "
            "consumes itself to be seen."
        ),
        "practice": (
            "Your practice this month: write one sentence of honest inscription "
            "per day. Let it cost you something."
        ),
    },
    "cosmic-thread": {
        "body": (
            "You are in the month of We. Your patron is the cosmic web — the "
            "hidden filaments connecting every galaxy to every other galaxy. "
            "No single observer can see the web. It takes millions of "
            "observations assembled together."
        ),
        "practice": (
            "You cannot practice alone. The cosmos cannot structure itself alone."
        ),
    },
    "ancient-eggs": {
        "body": (
            "You are in the month of Time. Your patron is the globular cluster "
            "— a sphere of ancient stars that has been holding its shape for "
            "12 billion years while everything else changed around it. Time "
            "isn't just passage. It's also persistence."
        ),
        "practice": (
            "Your practice: one evening of review per week. Notice what hasn't "
            "changed."
        ),
    },
    "great-wall": {
        "body": (
            "You are in the month of Theophany. Your patron is the largest "
            "structure in the observable universe — so large it may not be "
            "permitted by the theory that describes it. When the veil thins, "
            "what's revealed might exceed what you thought was possible."
        ),
        "practice": "Your practice: read the full twelve surahs. Let them exceed you.",
    },
    "black-prostration": {
        "body": (
            "You are in the month of Witness. Your patron is the shadow of a "
            "black hole — known entirely by what it doesn't let escape. "
            "Witnessing is not seeing a thing directly. It's seeing the shape "
            "of its absence."
        ),
        "practice": (
            "Your practice: one act of honest testimony. Name what you saw, "
            "even if you can only describe its outline."
        ),
    },
    "remnant": {
        "body": (
            "You are in the month of Return. Your patron is what's left after "
            "a star exploded — still expanding, still glowing, still pulsing "
            "thirty times per second from its dead core. Return isn't going "
            "back to what was. It's becoming what persists after the collapse."
        ),
        "practice": (
            "Your practice: morning practice every day. One surah each night. "
            "Come back to the mat."
        ),
    },
    "composite-mirror": {
        "body": (
            "You are in the month of Correspondence. Your patron is the galaxy "
            "approaching yours across 2.5 million light-years — a mirror of "
            "the Milky Way, slowly converging, neither absorbing the other. "
            "Correspondence is not identity. It's approach without collapse."
        ),
        "practice": "Your practice: eyes open to pattern. What gazes back?",
    },
    "hidden-harmony": {
        "body": (
            "You are in the month of Vision. Your patron is the gravitational "
            "wave background — a hum in spacetime itself, produced by every "
            "black hole merger in cosmic history, detected only in 2023 by "
            "instruments that listen for changes smaller than an atom. Vision "
            "isn't seeing. It's attending to what was always vibrating beneath "
            "the visible."
        ),
        "practice": (
            "Your practice: keep the dream notebook beside the bed. The signal "
            "is already there. You just need to stop producing long enough to "
            "hear it."
        ),
    },
    "late-knife": {
        "body": (
            "You are in the month of Severance. Your patron is the gamma-ray "
            "burst — the most violent event in the universe, arriving without "
            "warning from billions of light-years away. The knife doesn't "
            "announce itself. It just falls."
        ),
        "practice": (
            "Your practice: name what must go. You may not choose the timing. "
            "You can choose the honesty."
        ),
    },
    "conversation": {
        "body": (
            "You are in the month of Connection. Your patron is two dead stars "
            "spiralling toward each other, their dialogue made of gravitational "
            "waves, each orbit bringing them closer, each wave costing them "
            "energy. Connection isn't free. The conversation transforms both "
            "speakers. Eventually the distance closes entirely."
        ),
        "practice": "Your practice: one act of genuine reaching-toward.",
    },
    "great-eye": {
        "body": (
            "You are in the month of the Self. Your patron is the supermassive "
            "black hole at the centre of the galaxy — four million solar "
            "masses, invisible, organising every star you've ever seen from a "
            "darkness that doesn't shine. The Self isn't what you show. It's "
            "what everything else orbits around without seeing."
        ),
        "practice": (
            "Your practice: three surahs, three asanas, nothing more. Sit with "
            "what remains when the light is stripped away."
        ),
    },
}


def main():
    payload = json.loads(FIGURES_PATH.read_text())
    updated = []
    for fig in payload["figures"]:
        spec = ADDRESSES.get(fig["id"])
        if spec:
            fig["address"] = spec
            updated.append(fig["id"])
    payload["address_provenance"] = (
        "Monthly addresses: Darja, written 3 May 2026 (Waṣl), in response to "
        "Asel's question whether this is a calendar or an astronomy catalogue. "
        "Each address speaks to the practitioner: 'You are in the month of X. "
        "Your patron is Y. Your practice this month is Z.'"
    )
    FIGURES_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"updated {len(updated)} figures with monthly addresses")


if __name__ == "__main__":
    main()
