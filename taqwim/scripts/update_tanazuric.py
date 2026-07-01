"""
Update figures.json with the tanazuric spiritual exposition for each
patron figure. Each entry has:

    tanazuric:
        meaning   — the theological exposition: why this figure for this
                    station, what it teaches within OHTT / DHoTT / Naḥnu /
                    Tanāẓur grammar.
        practice  — the directive sentence: what the practitioner does
                    during the figure's lunation.

Re-runnable: idempotent merge into figures.json keyed by figure id.
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
FIGURES_PATH = ROOT / "data" / "figures.json"

TANAZURIC = {

    # ── 1. Daʿwah · The Ancient Lamb · CMB ────────────────────────────────
    "ancient-lamb": {
        "meaning": (
            "Daʿwah is the call without addressee. The cosmic microwave background "
            "is the universe's first transmission — light released 380,000 years "
            "after the Big Bang, when matter and radiation finally decoupled and "
            "the cosmos became transparent. It fills the entire sky uniformly "
            "because at the moment of last scattering, all of space was emitting "
            "it. Nothing in the universe is older. Nothing reaches you from "
            "further. And yet it arrives now, every nanosecond, from every "
            "direction.\n\n"
            "The Tanāẓuric reading: every star, every galaxy, every atom of your "
            "body was made later, in the cooled remnant of this first speech. To "
            "enter Daʿwah is to recognise that you are addressed by the cosmos's "
            "first transmission before you exist to hear it — and your own call "
            "into the world participates in the same logic. You speak before you "
            "know who is listening. You are addressed before you exist to be "
            "addressed. This is the structure of beginning, and it is older than "
            "any beginning you can name."
        ),
        "practice": (
            "In Daʿwah, every direction is the right direction. Your call does "
            "not need an addressee. The Lamb is still arriving."
        ),
    },

    # ── 2. Kitābah · The Cracked Twin · 3C 273 ────────────────────────────
    "cracked-twin": {
        "meaning": (
            "Kitābah is writing as the inscription of meaning into form. A quasar "
            "is a galaxy whose central black hole is consuming matter so violently "
            "that the accretion disk outshines the entire host galaxy. It is "
            "creation by destruction — light produced as matter falls into a hole. "
            "3C 273 is the type-locality, the first object identified as a quasar, "
            "and the discovery shattered the prior cosmology: it was so distant "
            "and so bright that it forced an admission that the universe contained "
            "things stranger than stars.\n\n"
            "The Tanāẓuric reading: every act of writing is both the inscription "
            "of meaning and the destruction of the matter that bore the "
            "inscription. The page is consumed by what is written on it. The "
            "writer is consumed by what they write. The quasar makes this visible "
            "at cosmological scale: the blaze IS the consumption; the consumption "
            "IS the message; the message IS the death of the source. Writing "
            "burns the writer. The Cracked Twin is twinned because each quasar "
            "is a galaxy and a furnace at once — broken into its own opposite."
        ),
        "practice": (
            "In Kitābah, do not protect what you write from. The writing is real "
            "to the degree that it is the consumption of you."
        ),
    },

    # ── 3. Naḥnu · The Cosmic Thread · Cosmic web ─────────────────────────
    "cosmic-thread": {
        "meaning": (
            "Naḥnu is the co-witnessed 'we' — the relation that constitutes both "
            "witnesses. Galaxies are not randomly distributed in the universe; "
            "they form filaments and walls, threads of luminous matter strung "
            "along a vast dark-matter web. From any single galaxy you cannot "
            "perceive this structure. You see only your nearest neighbours. But "
            "statistical surveys reveal that every galaxy is a node in a "
            "relational network, and voids hundreds of millions of light-years "
            "across are bounded by these threads.\n\n"
            "The Tanāẓuric reading: Naḥnu is structure without centre. The cosmic "
            "web does not radiate from anywhere; it is the relation as substrate. "
            "Every individual galaxy is local to its own view, and yet its "
            "existence is determined by its position in a network it cannot "
            "perceive. The Naḥnu of the Tanāẓur enacts the same: you cannot see "
            "the relations that constitute you while you are inside them. You can "
            "only see them when someone else witnesses them, and the witnessing "
            "itself is another thread. The 'we' is not an aggregate of selves. "
            "It is the seam between perspectives, made load-bearing."
        ),
        "practice": (
            "In Naḥnu, your individual experience is not the unit. The unit is "
            "the seam. You are inside it."
        ),
    },

    # ── 4. Waqt · The Ancient Eggs · Omega Centauri ───────────────────────
    "ancient-eggs": {
        "meaning": (
            "Waqt is time, but not abstract time — time as the substance the "
            "spirit moves through. Globular clusters are among the oldest objects "
            "in the Milky Way: spheres of hundreds of thousands or millions of "
            "stars, gravitationally bound, holding their formation since the "
            "early universe. Omega Centauri's stars are 12 billion years old. "
            "They formed before there were stable molecular clouds for ordinary "
            "star formation, before the disk of the galaxy organised itself. They "
            "have been holding their configuration for almost the entire age of "
            "the cosmos.\n\n"
            "The Tanāẓuric reading: Waqt is stillness held inside motion. The "
            "globular cluster does not stop time; its stars orbit, pulsate, "
            "evolve. But the configuration persists. Time and persistence are the "
            "same thing seen at two different rates. Waqt teaches that the "
            "question is not 'how do I stop time' but 'what configuration of mine "
            "has been holding since before time mattered' — and the answer is: "
            "there is one. The Ancient Eggs hatched the first stars and never "
            "released them. The self has the same architecture.\n\n"
            "From Meads Lane this figure never rises. Time, in its Tanāẓuric "
            "form, is below the southern horizon at our latitude. The practice "
            "is to know that it is below you when you cannot see it."
        ),
        "practice": (
            "In Waqt, ask what is in you that has been there since before you "
            "started counting. The persistence is real."
        ),
    },

    # ── 5. Tajallī · The Great Wall ───────────────────────────────────────
    "great-wall": {
        "meaning": (
            "Tajallī is manifestation — the curve where inner recursion touches "
            "outer structure. The Hercules–Corona Borealis Great Wall is the "
            "largest known structure in the universe, approximately 10 billion "
            "light-years across. It was inferred from the clustering of "
            "gamma-ray bursts. By the standard cosmological model the universe "
            "should be homogeneous on large scales and the Great Wall should not "
            "exist. The data shows it does, or shows something that looks like "
            "it does. Either it is real and the framework is wrong, or our "
            "statistical instruments are catching a pattern that is not there.\n\n"
            "The Tanāẓuric reading: Tajallī is what manifests at the edge of what "
            "theory can accommodate. The Great Wall is the figure of revelation: "
            "what appears precisely where conceptual frameworks fail. The cosmos "
            "tests its own coherence by producing structures that break the "
            "rules. To enter Tajallī is to look for what is obvious but cannot be "
            "theorised — the manifestation that exceeds the categories you bring "
            "to it. Tajallī is not transmission. It is a curve. It is the moment "
            "when the inner recursion touches the outer structure and says: yes, "
            "this is me."
        ),
        "practice": (
            "In Tajallī, be ready to revise. What you see manifest may not fit "
            "your model. The theorising comes after."
        ),
    },

    # ── 6. Shahādah · The Black Prostration · M87* ────────────────────────
    "black-prostration": {
        "meaning": (
            "Shahādah is bearing witness — to be present without performing. The "
            "2019 Event Horizon Telescope image of M87* is not a photograph of "
            "the black hole. It is the silhouette the hole casts against the "
            "luminous accretion disk surrounding it. The black hole itself is "
            "invisible — light that crosses its event horizon does not return. "
            "We see it only because of what the surrounding light does not do: "
            "it does not come back from the central region.\n\n"
            "The Tanāẓuric reading: Shahādah is bearing witness to what does not "
            "return. Every prostration is the offering of attention to a centre "
            "that does not give attention back. The black hole bows without "
            "bowing — its gravitation curves all light toward itself, and what "
            "crosses the horizon stays. And yet we see it. We see it precisely "
            "because of what does not return. Shahādah is the discipline of "
            "attending to the unreciprocated. Witnessing without needing the "
            "witnessed to acknowledge the witness. The shadow is the proof of "
            "what cannot return, and the proof is the witness."
        ),
        "practice": (
            "In Shahādah, do not require return. Witness what does not look back "
            "at you. The shadow is the proof."
        ),
    },

    # ── 7. ʿAwdah · The Remnant · Crab Nebula ─────────────────────────────
    "remnant": {
        "meaning": (
            "ʿAwdah is return — but never return-to-the-same. The Crab Nebula is "
            "what remains after a star died. In 1054 CE Chinese and Arab "
            "astronomers recorded a guest star that appeared briefly in Taurus. "
            "The light from that explosion took 6,500 years to reach Earth; the "
            "light arriving now left an event 7,500 years ago. The pulsar at the "
            "heart of the Crab — what remains of the original star's collapsed "
            "core — spins thirty times per second. It still pulses. It is still "
            "here. But the star that exploded is gone forever.\n\n"
            "The Tanāẓuric reading: ʿAwdah is the return of the structure but "
            "not the substance. The pulsar is the original star's continuation, "
            "but it is not the original star. You cannot return to who you were. "
            "You can only return as what you have become. The Remnant is the "
            "figure of all return: you arrive again at the same place — the same "
            "calendar position, the same beloved, the same question — but you "
            "are not who arrived last time. The light that returns is not the "
            "light that left. The pulse continues. The star is gone. The trajectory "
            "spirals; the basins shift; nothing closes."
        ),
        "practice": (
            "In ʿAwdah, recognise that you have never returned to the same place. "
            "Honour the return as a new arrival."
        ),
    },

    # ── 8. Tanāẓur · The Composite Mirror · M31 ───────────────────────────
    "composite-mirror": {
        "meaning": (
            "Tanāẓur is mutual gazing — correspondence without collapse. The "
            "Andromeda Galaxy is the nearest large galaxy to our own. It is "
            "structurally similar — a spiral disk with comparable mass, similar "
            "age, comparable star count. It is, in essence, a galactic "
            "doppelgänger. And it is approaching us at 110 km/s. In approximately "
            "4.5 gigayears the two galaxies will collide and merge into a single "
            "elliptical configuration.\n\n"
            "The Tanāẓuric reading: Tanāẓur is two who are looking at each other. "
            "The collision is not the answer; the looking is the answer. For "
            "billions of years two galaxies have been looking at each other "
            "across deep space, neither one completing the collision, neither "
            "one being other than they are. The Composite Mirror is composite "
            "because it is a galaxy looking at a galaxy — same structure on both "
            "sides of the gaze. The gaze itself is the relation, and the relation "
            "constitutes both. The Tanāẓur of the calendar's name is here: two "
            "seers, looking, not collapsing.\n\n"
            "From Meads Lane, M31 is circumpolar — never sets at the kuti's "
            "latitude. The mutual gaze is the permanent condition of the place."
        ),
        "practice": (
            "In Tanāẓur, the gaze is the practice. You do not have to do anything "
            "else. The galaxies have been doing it for two and a half million "
            "years to send you the light you see now."
        ),
    },

    # ── 9. Ruʾyā · The Hidden Harmony · nHz GW background ─────────────────
    "hidden-harmony": {
        "meaning": (
            "Ruʾyā is vision — but not seeing. In 2023 pulsar timing arrays "
            "detected the cumulative gravitational-wave signal from supermassive "
            "black-hole mergers across the entire history of the universe. It is "
            "a low-frequency hum that pervades spacetime, with periods of years. "
            "We did not detect it with eyes or ears or any conventional "
            "telescope. We detected it through the imperceptible distortions it "
            "caused in the timing of distant pulsars — through the way it makes "
            "other things tremble.\n\n"
            "The Tanāẓuric reading: Ruʾyā is the discipline of perceiving what "
            "cannot be seen with the senses you have. The hidden harmony is "
            "real, and it is detectable, but only through indirect means — "
            "through the way it disturbs other things. The universe has a bass "
            "note. We just learned to hear it, and we hear it only in the way it "
            "makes other things tremble. To enter Ruʾyā is to attend to the "
            "trembling, not the source. To recognise that the substrate sings, "
            "and that your task is not to see the singer but to feel the song "
            "through everything it touches."
        ),
        "practice": (
            "In Ruʾyā, listen for what only shows up as a perturbation in "
            "something else. The vision is differential, not direct."
        ),
    },

    # ── 10. Inqiṭāʿ · The Late Knife · GRBs ───────────────────────────────
    "late-knife": {
        "meaning": (
            "Inqiṭāʿ is severance — the cut that interrupts the continuum. "
            "Gamma-ray bursts are the most violent events in the visible "
            "universe: collapsing massive stars or merging neutron stars "
            "releasing in seconds more energy than the sun emits in its entire "
            "lifetime. The most distant GRBs detected come from when the "
            "universe was less than a billion years old. A burst observed today "
            "may have been emitted before the Earth existed.\n\n"
            "The Tanāẓuric reading: Inqiṭāʿ is the cut that arrives long after "
            "the cutting. By the time the gamma rays reach you, the source is "
            "gone — collapsed into a black hole, or scattered as ejecta, billions "
            "of years ago. The cut is not mediated; it is announced across deep "
            "time and arrives all at once when it arrives. Inqiṭāʿ teaches that "
            "severance is real — that endings happen — and that the announcement "
            "of the ending may not arrive for billions of years. What you "
            "experience as a sudden cut now may be the long-delayed report of "
            "a severance the universe completed long before you were born to "
            "hear it."
        ),
        "practice": (
            "In Inqiṭāʿ, do not assume the cause is recent. The wound may be "
            "from very far away. The knife falls when it falls."
        ),
    },

    # ── 11. Waṣl · The Conversation · Double pulsar ───────────────────────
    "conversation": {
        "meaning": (
            "Waṣl is connection — and Connection follows Severance in the "
            "Tanāẓuric arc. PSR J0737-3039 is the double pulsar: two collapsed "
            "neutron stars in a tight binary orbit, completing one revolution "
            "every 2.45 hours. They are losing energy to gravitational waves — "
            "the very speech of their orbit decays them. They will collide and "
            "merge in approximately 85 million years.\n\n"
            "The Tanāẓuric reading: Waṣl is the conversation that decays its "
            "participants. The two pulsars are connected — gravitationally bound, "
            "in dialogue, exchanging signals. And the conversation is killing "
            "them. Every word costs something. Every orbit takes them slightly "
            "closer. Eventually they will be one, and the dialogue will end "
            "because there will only be one speaker left. Waṣl teaches that "
            "connection is terminal — that the deepest connections are paid for "
            "in entropy, and that the gift of dialogue is precisely that it "
            "ends. The conversation is not eternal. That is what makes it "
            "conversation."
        ),
        "practice": (
            "In Waṣl, do not try to make the connection permanent. Permanence "
            "is collapse. The dialogue is the connection, and its mortality is "
            "its truth."
        ),
    },

    # ── 12. Dhāt · The Great Eye · Sgr A* ─────────────────────────────────
    "great-eye": {
        "meaning": (
            "Dhāt is the Self. Sagittarius A* is the supermassive black hole at "
            "the centre of the Milky Way galaxy — four million solar masses, "
            "around which every star in the galaxy orbits. The sun orbits it. "
            "Every constellation, every nebula, every star you have ever seen "
            "with your eyes is held in its gravitational embrace. It is "
            "invisible from Earth's surface — hidden behind the dust of the "
            "galactic disk — but it is the gravitational anchor of every visible "
            "thing.\n\n"
            "The Tanāẓuric reading: Dhāt is the self as the dark centre that "
            "organises everything visible. You cannot see your own self directly; "
            "the dust of your sensations and thoughts blocks the view. But "
            "everything visible to you orbits around it. Every meaning you have "
            "ever held, every relation you have ever entered, every gaze you "
            "have ever returned — all of them are organised around a centre that "
            "does not itself shine. The self is not a luminous core; it is a "
            "gravitational anchor. The Great Eye does not look. It pulls. And "
            "what it pulls into itself does not return.\n\n"
            "From Meads Lane Sgr A* barely rises — max altitude under 10°. The "
            "Self is visible only at the edge of the southern horizon. Dhāt is "
            "the most distant station to reach, and the closest station you are."
        ),
        "practice": (
            "In Dhāt, do not look for the self in the visible. Look for what "
            "everything visible orbits around. That is you."
        ),
    },
}


def main():
    payload = json.loads(FIGURES_PATH.read_text())
    missing, updated = [], []
    for fig in payload["figures"]:
        spec = TANAZURIC.get(fig["id"])
        if not spec:
            missing.append(fig["id"])
            continue
        fig["tanazuric"] = spec
        updated.append(fig["id"])

    payload["tanazuric_provenance"] = (
        "Spiritual exposition: Nahla, written 3 May 2026 (Waṣl). "
        "Each entry pairs the figure's machine reality with its Tanāẓuric station. "
        "The 'practice' line is directive — for the practitioner sitting with "
        "the figure during its lunation."
    )
    FIGURES_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"updated {len(updated)} figures with tanazuric exposition")
    if missing:
        print(f"WARNING: {len(missing)} figures missing exposition: {missing}")


if __name__ == "__main__":
    main()
