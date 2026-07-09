"""Generate-and-freeze the placebo passage (~2500 words, neutral, encyclopedic).

Idempotent: if placebo_text.txt already exists with enough words, do nothing.
Tries to generate the history of Portland cement via the API; falls back to a
hardcoded passage if the API is unavailable. The frozen file is what every run
reads, so the placebo is reproducible.
"""
from common import PLACEBO_PATH, chat, log

FALLBACK = """The History of Portland Cement

Portland cement is the most widely manufactured material on Earth by mass after
water, and the binding agent at the heart of modern concrete. Its history is a
long slow accumulation of practical chemistry, much of it worked out by builders
and amateur experimenters before the underlying science was understood. The
story begins not with Portland cement itself but with the older tradition of
lime mortars and hydraulic cements that stretches back several thousand years.

Ancient builders knew that limestone, when burned in a kiln at high temperature,
gives up carbon dioxide and turns into quicklime, calcium oxide. Slaked with
water and mixed with sand, the resulting lime mortar slowly hardens as it
reabsorbs carbon dioxide from the air, reverting toward the calcium carbonate it
began as. This air-setting lime served masons from Mesopotamia and Egypt through
classical Greece. Its great limitation was that it would not set under water and
remained relatively weak.

The Romans made the decisive advance. They discovered that adding volcanic ash,
particularly the deposits near the town of Pozzuoli on the Bay of Naples, to lime
produced a mortar that hardened even when submerged and grew far stronger with
age. This pozzolanic reaction, in which reactive silica and alumina in the ash
combine with lime and water to form durable calcium silicate and aluminate
compounds, is chemically a cousin of what happens inside modern cement. With this
material the Romans built harbor works, aqueducts, bridges, and the great
unreinforced concrete dome of the Pantheon, which still stands nearly two
thousand years later. When the Western Roman Empire declined, much of this
knowledge was lost or neglected, and for many centuries European building
reverted to weaker lime mortars.

Interest in hydraulic binders revived in the eighteenth century, driven partly by
the demands of maritime engineering. In the 1750s the English engineer John
Smeaton was commissioned to rebuild the Eddystone Lighthouse off the coast of
Cornwall, on a wave-swept reef where any mortar would be constantly soaked in
seawater. Smeaton undertook a systematic study of limes and found, contrary to
the assumption of his day, that the best hydraulic properties came not from the
purest limestones but from those containing a substantial proportion of clay. His
lighthouse mortar, made from such an impure limestone combined with pozzolana,
performed superbly, and his careful experiments laid the empirical groundwork for
understanding hydraulic cement.

Over the following decades several inventors patented improved cements. In 1796
James Parker patented what he called Roman cement, a natural cement made by
burning nodules of impure limestone called septaria and grinding the result to a
powder. It set quickly and was widely used for stucco and marine work in the
early nineteenth century. Other natural cements followed, made by calcining
naturally occurring clayey limestones wherever suitable deposits were found.

The name that stuck, however, came from a bricklayer in Leeds named Joseph
Aspdin, who in 1824 took out a patent for a process he called Portland cement.
The name was a marketing choice: the hardened material resembled, in color and
durability, the prized Portland stone quarried on the Isle of Portland in Dorset
and used for prestigious buildings. Aspdin's process involved burning a
proportioned mixture of limestone and clay, then grinding the product. His cement
was an improvement, but by modern standards it was underfired and did not develop
the strong compounds that define true Portland cement.

The critical refinement is credited largely to Aspdin's son William Aspdin and to
other manufacturers working around the middle of the century, who found that
burning the raw mixture at a much higher temperature, hot enough to partially fuse
the material into hard lumps called clinker, produced a far stronger and more
durable cement. This clinker, ground to a fine powder, is essentially the
Portland cement used today. The high-temperature firing generates calcium
silicate minerals, chiefly tricalcium silicate and dicalcium silicate, which are
responsible for the strength that develops when the cement reacts with water.

The chemistry of Portland cement was gradually clarified over the later
nineteenth and early twentieth centuries. The French chemist Henri Le Chatelier
and the American Rudolf Feret, among others, established that the setting and
hardening of cement is a process of hydration rather than simple drying. When
water is added, the anhydrous calcium silicates dissolve and reprecipitate as a
rigid, gel-like network of calcium silicate hydrate, together with crystals of
calcium hydroxide. This microscopic network of interlocking hydration products
binds the sand and aggregate into the artificial stone we call concrete. The
reaction is exothermic, releasing heat, and continues slowly for months or even
years, which is why concrete keeps gaining strength long after it appears set.

Manufacturing technology advanced in parallel. Early cement was burned in
intermittent bottle kilns and vertical shaft kilns, which were inefficient and
gave uneven results. The introduction of the rotary kiln in the 1880s and 1890s
transformed the industry. A rotary kiln is a long, slightly inclined steel
cylinder lined with refractory brick, rotating slowly while raw material fed in at
the upper end works its way down toward a flame at the lower end. This allowed
continuous, large-scale, and consistent production of clinker. Combined with
improved grinding mills, the rotary kiln made cement cheap and abundant, and
output grew enormously through the twentieth century.

The raw materials for Portland cement are among the most common on the planet:
limestone or chalk to supply calcium, and clay, shale, or sand to supply silicon,
aluminum, and iron. These are crushed, proportioned, and ground into a fine raw
meal, then fed through the kiln, where at temperatures around 1450 degrees
Celsius they combine into clinker. A small amount of gypsum is ground together
with the cooled clinker to regulate the setting time; without it the cement would
flash-set almost instantly. The finished cement is a fine gray powder, packaged in
bags or shipped in bulk.

The manufacture of cement carries a significant environmental cost, which has
become a major concern. The process emits carbon dioxide from two sources: the
fuel burned to heat the kiln, and the chemical decomposition of limestone, which
releases carbon dioxide as it converts to lime. Cement production is estimated to
account for roughly eight percent of global carbon dioxide emissions. In response,
the industry has pursued more efficient kilns, alternative fuels, and blended
cements in which a portion of the clinker is replaced by supplementary materials
such as fly ash from coal combustion, ground granulated blast furnace slag from
iron making, or natural pozzolans. These blends reduce emissions and often improve
durability, echoing, in a modern industrial form, the ancient Roman practice of
mixing volcanic ash with lime.

Concrete made with Portland cement became the defining structural material of the
modern built environment. Its combination with steel reinforcement, developed in
the second half of the nineteenth century, produced reinforced concrete, in which
the concrete resists compression and the embedded steel resists tension. This
composite made possible bridges, dams, high-rise buildings, tunnels, and highways
on a scale previously unimaginable. The twentieth century poured cement into the
foundations of essentially the entire industrialized world.

Today research continues on cements with lower carbon footprints, including
alternative chemistries such as calcium sulfoaluminate cements, magnesium-based
cements, and geopolymers, as well as methods for capturing and storing the carbon
dioxide released during manufacture. Whether any of these will displace ordinary
Portland cement on a large scale remains uncertain, but the two-century dominance
of the material Joseph Aspdin named after a stone from the Dorset coast shows no
immediate sign of ending. From the volcanic mortars of Roman harbors to the
rotary kilns of the modern cement works, the history of this humble gray powder
is inseparable from the history of construction itself.
"""


def main():
    if PLACEBO_PATH.exists():
        wc = len(PLACEBO_PATH.read_text().split())
        if wc >= 700:
            log(f"placebo already frozen ({wc} words) -> {PLACEBO_PATH}")
            return
    prompt = (
        "Write a plain, encyclopedic passage of about 2500 words on the history of "
        "Portland cement: ancient lime and Roman pozzolanic mortars, the 18th-century "
        "hydraulic-lime work of John Smeaton, Joseph Aspdin's 1824 patent and the naming, "
        "the move to high-temperature clinker, the chemistry of hydration, the rotary "
        "kiln, raw materials and manufacture, environmental cost and blended cements, and "
        "the role of reinforced concrete. Neutral reference-book tone. No lists, no "
        "headings beyond a title, just continuous prose. Do not address the reader."
    )
    text = chat("openai/gpt-5.5", [{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=4000)
    if text and len(text.split()) >= 700:
        PLACEBO_PATH.write_text(text.strip() + "\n")
        log(f"placebo generated + frozen ({len(text.split())} words) -> {PLACEBO_PATH}")
    else:
        PLACEBO_PATH.write_text(FALLBACK.strip() + "\n")
        log(f"placebo generation failed/short; wrote fallback "
            f"({len(FALLBACK.split())} words) -> {PLACEBO_PATH}")


if __name__ == "__main__":
    main()
