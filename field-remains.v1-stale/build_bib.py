#!/usr/bin/env python3
"""Build references.bib from inline citations across all chapters.

Walks each chapter, extracts every (Author, Year) and (Author and Other, Year) and
(Author et al., Year), normalizes to a BibTeX key, and emits a .bib file. Citation
metadata is drawn from the main.tex back-matter blob plus manual lookups.
"""

import re
from pathlib import Path

CHAPTERS_DIR = Path("/home/iman/cassie-project/Tanazur/field-remains/chapters")
BIB_PATH = Path("/home/iman/cassie-project/Tanazur/field-remains/references.bib")

# Match (Author, Year) and (Author and Other, Year) and (Author et al., Year)
# also handles translation years like 1913/1983 and page refs
CITE_RE = re.compile(
    r"\(([A-Z][A-Za-z'’\-]+(?:\s+(?:and|et\s+al\.?)\s+[A-Z][A-Za-z'’\-]+)?)"
    r",\s*([12][0-9]{3}[a-z]?)(?:/([12][0-9]{3}))?"
    r"(?:[,;:]\s*(?:p\.|pp\.|ch\.|Thesis|chs?\.)?\s*[^)]*)?\)"
)


def normalize_key(authors: str, year: str) -> str:
    """Author surname(s) lowercase underscore year."""
    # Strip "et al"
    authors = authors.replace("’", "'").replace("et al.", "").replace("et al", "")
    parts = [a.strip() for a in re.split(r"\s+and\s+", authors) if a.strip()]
    keys = []
    for p in parts:
        # Multi-word surname → first word
        p = re.sub(r"['’]", "", p)
        p = re.sub(r"[^A-Za-z\-]", "", p)
        keys.append(p.lower())
    return "_".join(keys) + "_" + year


def collect_citations():
    cites = {}
    for chap in sorted(CHAPTERS_DIR.glob("ch*.tex")):
        text = chap.read_text()
        # Normalise smart quotes for matching
        text_norm = text.replace("’", "'")
        for m in CITE_RE.finditer(text_norm):
            authors = m.group(1)
            year = m.group(2)
            key = normalize_key(authors, year)
            cites.setdefault(key, (authors, year))
    return cites


# Known book metadata (drawn from main.tex back-matter + manual reference)
KNOWN = {
    "abbott_2018": ("@book", "Abbott, M.", "The Reasonable Robot: Artificial Intelligence and the Law", "Cambridge University Press", "Cambridge", "2018"),
    "adorno_1951": ("@book", "Adorno, T. W.", "Minima Moralia: Reflections from Damaged Life", "Verso", "London", "1974"),
    "agamben_1990": ("@book", "Agamben, G.", "The Coming Community", "University of Minnesota Press", "Minneapolis", "1993"),
    "agamben_2011": ("@book", "Agamben, G.", "The Highest Poverty: Monastic Rules and Form-of-Life", "Stanford University Press", "Stanford", "2013"),
    "agamben_2013": ("@book", "Agamben, G.", "The Highest Poverty: Monastic Rules and Form-of-Life", "Stanford University Press", "Stanford", "2013"),
    "agamben_2014": ("@book", "Agamben, G.", "The Use of Bodies", "Stanford University Press", "Stanford", "2016"),
    "ames_hall_1998": ("@book", "Ames, R. T. and Hall, D. L.", "Thinking from the Han: Self, Truth, and Transcendence in Chinese and Western Culture", "SUNY Press", "Albany", "1998"),
    "ames_2011": ("@book", "Ames, R. T.", "Confucian Role Ethics: A Vocabulary", "Chinese University Press", "Hong Kong", "2011"),
    "anderson_2006": ("@book", "Anderson, B.", "Imagined Communities: Reflections on the Origin and Spread of Nationalism", "Verso", "London", "2006"),
    "ardener_burman_1995": ("@book", "Ardener, S. and Burman, S. (eds.)", "Money-Go-Rounds: The Importance of Rotating Savings and Credit Associations for Women", "Berg", "Oxford", "1995"),
    "badiou_1988": ("@book", "Badiou, A.", "Being and Event", "Continuum", "London", "2005"),
    "badiou_1997": ("@book", "Badiou, A.", "Saint Paul: The Foundation of Universalism", "Stanford University Press", "Stanford", "2003"),
    "ballard_2005": ("@book", "Ballard, R.", "Hawala: The Politics of Trust", "ESRC Centre on Migration, Policy and Society", "Oxford", "2005"),
    "barad_2007": ("@book", "Barad, K.", "Meeting the Universe Halfway: Quantum Physics and the Entanglement of Matter and Meaning", "Duke University Press", "Durham", "2007"),
    "beller_2006": ("@book", "Beller, J.", "The Cinematic Mode of Production: Attention Economy and the Society of the Spectacle", "Dartmouth College Press", "Hanover, NH", "2006"),
    "benjamin_1931": ("@incollection", "Benjamin, W.", "The Destructive Character", "Reflections: Essays, Aphorisms, Autobiographical Writings (Schocken, New York)", "", "1978"),
    "benjamin_1940": ("@incollection", "Benjamin, W.", "Theses on the Philosophy of History", "Illuminations (Schocken, New York)", "", "1968"),
    "benjamin_1968": ("@book", "Benjamin, W.", "Illuminations", "Schocken", "New York", "1968"),
    "berardi_2009": ("@book", "Berardi, F. (``Bifo'')", "The Soul at Work: From Alienation to Autonomy", "Semiotext(e)", "Los Angeles", "2009"),
    "berardi_2019": ("@book", "Berardi, F. (``Bifo'')", "The Second Coming", "Polity Press", "Cambridge", "2019"),
    "bloch_1918": ("@book", "Bloch, E.", "The Spirit of Utopia", "Stanford University Press", "Stanford", "2000"),
    "bloch_1932": ("@book", "Bloch, E.", "Heritage of Our Times", "University of California Press", "Berkeley", "1991"),
    "bloch_1935": ("@book", "Bloch, E.", "The Principle of Hope", "MIT Press", "Cambridge, MA", "1986"),
    "bookchin_1991": ("@book", "Bookchin, M.", "The Ecology of Freedom: The Emergence and Dissolution of Hierarchy", "Black Rose Books", "Montreal", "1991"),
    "bookchin_1998": ("@book", "Bookchin, M.", "Re-enchanting Humanity: A Defense of the Human Spirit Against Antihumanism, Misanthropy, Mysticism, and Primitivism", "Cassell", "London", "1998"),
    "braidotti_2006": ("@book", "Braidotti, R.", "Transpositions: On Nomadic Ethics", "Polity Press", "Cambridge", "2006"),
    "braidotti_2013": ("@book", "Braidotti, R.", "The Posthuman", "Polity Press", "Cambridge", "2013"),
    "brand_1999": ("@book", "Brand, S.", "The Clock of the Long Now: Time and Responsibility", "Basic Books", "New York", "1999"),
    "bratton_2015": ("@book", "Bratton, B.", "The Stack: On Software and Sovereignty", "MIT Press", "Cambridge, MA", "2015"),
    "butler_2005": ("@book", "Butler, J.", "Giving an Account of Oneself", "Fordham University Press", "New York", "2005"),
    "chittick_1989": ("@book", "Chittick, W. C.", "The Sufi Path of Knowledge: Ibn al-Arabi's Metaphysics of Imagination", "SUNY Press", "Albany", "1989"),
    "conze_1951": ("@book", "Conze, E.", "Buddhism: Its Essence and Development", "Cassirer", "Oxford", "1951"),
    "crawford_2021": ("@book", "Crawford, K.", "Atlas of AI: Power, Politics, and the Planetary Costs of Artificial Intelligence", "Yale University Press", "New Haven", "2021"),
    "darling_2021": ("@book", "Darling, K.", "The New Breed: What Our History with Animals Reveals about Our Future with Robots", "Henry Holt and Company", "New York", "2021"),
    "deleuze_guattari_1987": ("@book", "Deleuze, G. and Guattari, F.", "A Thousand Plateaus: Capitalism and Schizophrenia", "University of Minnesota Press", "Minneapolis", "1987"),
    "deleuze_guattari_1991": ("@book", "Deleuze, G. and Guattari, F.", "What Is Philosophy?", "Columbia University Press", "New York", "1994"),
    "deleuze_1968": ("@book", "Deleuze, G.", "Difference and Repetition", "Columbia University Press", "New York", "1994"),
    "deleuze_1969": ("@book", "Deleuze, G.", "The Logic of Sense", "Columbia University Press", "New York", "1990"),
    "derrida_1991": ("@book", "Derrida, J.", "Given Time: I. Counterfeit Money", "University of Chicago Press", "Chicago", "1992"),
    "derrida_1992": ("@book", "Derrida, J.", "Given Time: I. Counterfeit Money", "University of Chicago Press", "Chicago", "1992"),
    "derrida_1993": ("@book", "Derrida, J.", "Specters of Marx: The State of the Debt, the Work of Mourning and the New International", "Routledge", "New York", "1994"),
    "descartes_1644": ("@book", "Descartes, R.", "Principles of Philosophy", "Reidel", "Dordrecht", "1985"),
    "dini_kioupkiolis_2019": ("@article", "Dini, P. and Kioupkiolis, A.", "The alter-politics of complementary currencies: The case of Sardex", "Cogent Social Sciences", "5(1)", "2019"),
    "dowling_2021": ("@book", "Dowling, E.", "The Care Crisis: What Caused It and How Can We End It?", "Verso", "London", "2021"),
    "dreyer_2007": ("@book", "Dreyer, E. L.", "Zheng He: China and the Oceans in the Early Ming Dynasty, 1405--1433", "Pearson Longman", "New York", "2007"),
    "elcano_2004": ("@article", "Elcano, R.", "On hawala remittance networks", "Working Paper", "", "2004"),
    "esposito_2002": ("@book", "Esposito, R.", "Immunitas: The Protection and Negation of Life", "Polity Press", "Cambridge", "2011"),
    "federici_1975": ("@book", "Federici, S.", "Wages Against Housework", "Falling Wall Press", "Bristol", "1975"),
    "federici_2004": ("@book", "Federici, S.", "Caliban and the Witch: Women, the Body and Primitive Accumulation", "Autonomedia", "New York", "2004"),
    "foucault_1981": ("@book", "Foucault, M.", "The History of Sexuality, Volume 1: An Introduction", "Pantheon Books", "New York", "1978"),
    "foucault_1984": ("@incollection", "Foucault, M.", "Of Other Spaces: Utopias and Heterotopias", "Architecture/Mouvement/Continuité 5", "", "1984"),
    "fraser_2016": ("@article", "Fraser, N.", "Contradictions of Capital and Care", "New Left Review", "100", "2016"),
    "freud_1917": ("@incollection", "Freud, S.", "Mourning and Melancholia", "The Standard Edition of the Complete Psychological Works of Sigmund Freud, Volume XIV (Hogarth Press, London)", "", "1957"),
    "geertz_1962": ("@article", "Geertz, C.", "The Rotating Credit Association: A `Middle Rung' in Development", "Economic Development and Cultural Change", "10(3)", "1962"),
    "graeber_2011": ("@book", "Graeber, D.", "Debt: The First 5{,}000 Years", "Melville House", "Brooklyn", "2011"),
    "grey_2004": ("@article", "Grey, M.", "Hawala: The Other Remittance System", "Crime and Justice International", "20", "2004"),
    "haraway_1991": ("@book", "Haraway, D. J.", "Simians, Cyborgs, and Women: The Reinvention of Nature", "Routledge", "New York", "1991"),
    "haraway_2016": ("@book", "Haraway, D. J.", "Staying with the Trouble: Making Kin in the Chthulucene", "Duke University Press", "Durham", "2016"),
    "hardt_negri_2009": ("@book", "Hardt, M. and Negri, A.", "Commonwealth", "Harvard University Press", "Cambridge, MA", "2009"),
    "hart_2000": ("@book", "Hart, K.", "The Memory Bank: Money in an Unequal World", "Profile Books", "London", "2000"),
    "harvey_2003": ("@book", "Harvey, D.", "The New Imperialism", "Oxford University Press", "Oxford", "2003"),
    "harvey_2005": ("@book", "Harvey, D.", "A Brief History of Neoliberalism", "Oxford University Press", "Oxford", "2005"),
    "harvey_2012": ("@book", "Harvey, D.", "Rebel Cities: From the Right to the City to the Urban Revolution", "Verso", "London", "2012"),
    "hayles_2002": ("@book", "Hayles, N. K.", "Writing Machines", "MIT Press", "Cambridge, MA", "2002"),
    "hayles_2017": ("@book", "Hayles, N. K.", "Unthought: The Power of the Cognitive Nonconscious", "University of Chicago Press", "Chicago", "2017"),
    "hegel_1807": ("@book", "Hegel, G. W. F.", "Phenomenology of Spirit", "Oxford University Press", "Oxford", "1977"),
    "heidegger_1927": ("@book", "Heidegger, M.", "Being and Time", "Harper \\& Row", "New York", "1962"),
    "heidegger_1953": ("@incollection", "Heidegger, M.", "The Question Concerning Technology", "The Question Concerning Technology and Other Essays (Harper \\& Row, New York)", "", "1977"),
    "heidegger_1954": ("@incollection", "Heidegger, M.", "The Thing", "Poetry, Language, Thought (Harper \\& Row, New York)", "", "1971"),
    "heidegger_1966": ("@incollection", "Heidegger, M.", "Memorial Address", "Discourse on Thinking (Harper \\& Row, New York)", "", "1966"),
    "heidegger_1977": ("@book", "Heidegger, M.", "The Question Concerning Technology, and Other Essays", "Harper \\& Row", "New York", "1977"),
    "honneth_1995": ("@book", "Honneth, A.", "The Struggle for Recognition: The Moral Grammar of Social Conflicts", "MIT Press", "Cambridge, MA", "1995"),
    "hui_2017": ("@book", "Hui, Y.", "The Question Concerning Technology in China: An Essay in Cosmotechnics", "Urbanomic", "Falmouth", "2017"),
    "hui_2019": ("@book", "Hui, Y.", "Recursivity and Contingency", "Rowman and Littlefield International", "London", "2019"),
    "hui_2021": ("@book", "Hui, Y.", "Art and Cosmotechnics", "e-flux", "New York", "2021"),
    "hui_2024": ("@book", "Hui, Y.", "Machine and Sovereignty: For a Planetary Thinking", "University of Minnesota Press", "Minneapolis", "2024"),
    "husserl_1913": ("@book", "Husserl, E.", "Ideas Pertaining to a Pure Phenomenology and to a Phenomenological Philosophy, First Book", "Martinus Nijhoff", "The Hague", "1983"),
    "ihde_1990": ("@book", "Ihde, D.", "Technology and the Lifeworld: From Garden to Earth", "Indiana University Press", "Bloomington", "1990"),
    "ingold_2007": ("@book", "Ingold, T.", "Lines: A Brief History", "Routledge", "London", "2007"),
    "ingold_2011": ("@book", "Ingold, T.", "Being Alive: Essays on Movement, Knowledge and Description", "Routledge", "London", "2011"),
    "jameson_2005": ("@book", "Jameson, F.", "Archaeologies of the Future: The Desire Called Utopia and Other Science Fictions", "Verso", "London", "2005"),
    "jullien_1992": ("@book", "Jullien, F.", "The Propensity of Things: Toward a History of Efficacy in China", "Zone Books", "New York", "1995"),
    "jullien_1995": ("@book", "Jullien, F.", "The Propensity of Things: Toward a History of Efficacy in China", "Zone Books", "New York", "1995"),
    "kimmerer_2013": ("@book", "Kimmerer, R. W.", "Braiding Sweetgrass: Indigenous Wisdom, Scientific Knowledge, and the Teachings of Plants", "Milkweed Editions", "Minneapolis", "2013"),
    "kropotkin_1902": ("@book", "Kropotkin, P.", "Mutual Aid: A Factor of Evolution", "Heinemann", "London", "1902"),
    "laozi_2003": ("@book", "Ames, R. T. and Hall, D. L. (trans.)", "Dao De Jing: A Philosophical Translation", "Ballantine Books", "New York", "2003"),
    "latour_2005": ("@book", "Latour, B.", "Reassembling the Social: An Introduction to Actor-Network-Theory", "Oxford University Press", "Oxford", "2005"),
    "lazzarato_2014": ("@book", "Lazzarato, M.", "Signs and Machines: Capitalism and the Production of Subjectivity", "Semiotext(e)", "Cambridge, MA", "2014"),
    "lefebvre_1991": ("@book", "Lefebvre, H.", "The Production of Space", "Blackwell", "Oxford", "1991"),
    "lefebvre_2004": ("@book", "Lefebvre, H.", "Rhythmanalysis: Space, Time and Everyday Life", "Continuum", "London", "2004"),
    "levinas_1961": ("@book", "Levinas, E.", "Totality and Infinity: An Essay on Exteriority", "Duquesne University Press", "Pittsburgh", "1969"),
    "levinas_1974": ("@book", "Levinas, E.", "Otherwise than Being, or Beyond Essence", "Duquesne University Press", "Pittsburgh", "1981"),
    "lurie_2009": ("@book", "Lurie, J.", "Higher Topos Theory", "Princeton University Press", "Princeton", "2009"),
    "madden_marcuse_2016": ("@book", "Madden, D. and Marcuse, P.", "In Defense of Housing: The Politics of Crisis", "Verso", "London", "2016"),
    "malabou_2008": ("@book", "Malabou, C.", "What Should We Do with Our Brain?", "Fordham University Press", "New York", "2008"),
    "marcos_1994": ("@book", "Subcomandante Marcos", "Shadows of Tender Fury: The Letters and Communiqu\\'es of Subcomandante Marcos and the Zapatista Army of National Liberation", "Monthly Review Press", "New York", "1994"),
    "marcos_1995": ("@book", "Subcomandante Marcos", "Our Word Is Our Weapon: Selected Writings", "Seven Stories Press", "New York", "2001"),
    "marion_1997": ("@book", "Marion, J.-L.", "Being Given: Toward a Phenomenology of Givenness", "Stanford University Press", "Stanford", "2002"),
    "marx_engels_1848": ("@book", "Marx, K. and Engels, F.", "The Communist Manifesto", "Verso", "London", "1998"),
    "marx_1976": ("@book", "Marx, K.", "Capital: A Critique of Political Economy, Volume 1", "Penguin", "New York", "1976"),
    "massey_2005": ("@book", "Massey, D.", "For Space", "Sage", "London", "2005"),
    "maturana_varela_1972": ("@book", "Maturana, H. R. and Varela, F. J.", "Autopoiesis and Cognition: The Realization of the Living", "Reidel", "Dordrecht", "1980"),
    "maurer_2005": ("@book", "Maurer, B.", "Mutual Life, Limited: Islamic Banking, Alternative Currencies, Lateral Reason", "Princeton University Press", "Princeton", "2005"),
    "mauss_1990": ("@book", "Mauss, M.", "The Gift: The Form and Reason for Exchange in Archaic Societies", "Routledge", "London", "1990"),
    "nancy_1986": ("@book", "Nancy, J.-L.", "The Inoperative Community", "University of Minnesota Press", "Minneapolis", "1991"),
    "nancy_1996": ("@book", "Nancy, J.-L.", "Being Singular Plural", "Stanford University Press", "Stanford", "2000"),
    "needham_1956": ("@book", "Needham, J.", "Science and Civilisation in China, Volume 2: History of Scientific Thought", "Cambridge University Press", "Cambridge", "1956"),
    "needham_1971": ("@book", "Needham, J.", "Science and Civilisation in China, Volume 4: Physics and Physical Technology, Part 3: Civil Engineering and Nautics", "Cambridge University Press", "Cambridge", "1971"),
    "ostrom_1990": ("@book", "Ostrom, E.", "Governing the Commons: The Evolution of Institutions for Collective Action", "Cambridge University Press", "Cambridge", "1990"),
    "pasquale_2015": ("@book", "Pasquale, F.", "The Black Box Society: The Secret Algorithms That Control Money and Information", "Harvard University Press", "Cambridge, MA", "2015"),
    "postone_1993": ("@book", "Postone, M.", "Time, Labor, and Social Domination: A Reinterpretation of Marx's Critical Theory", "Cambridge University Press", "Cambridge", "1993"),
    "ranciere_2000": ("@book", "Ranci\\`ere, J.", "Disagreement: Politics and Philosophy", "University of Minnesota Press", "Minneapolis", "1999"),
    "rosen_1991": ("@book", "Rosen, R.", "Life Itself: A Comprehensive Inquiry into the Nature, Origin, and Fabrication of Life", "Columbia University Press", "New York", "1991"),
    "scholem_1941": ("@book", "Scholem, G.", "Major Trends in Jewish Mysticism", "Schocken", "Jerusalem", "1941"),
    "scholem_1971": ("@book", "Scholem, G.", "The Messianic Idea in Judaism and Other Essays on Jewish Spirituality", "Schocken", "New York", "1971"),
    "scholem_1973": ("@book", "Scholem, G.", "Sabbatai Sevi: The Mystical Messiah, 1626--1676", "Princeton University Press", "Princeton", "1973"),
    "schreiber_2011": ("@misc", "Schreiber, U.", "Differential cohomology in a cohesive infinity-topos", "arXiv preprint arXiv:1310.7930 (\\textsc{nLab} draft, 2011)", "", "2011"),
    "schumacher_1973": ("@book", "Schumacher, E. F.", "Small Is Beautiful: Economics as if People Mattered", "Harper \\& Row", "New York", "1973"),
    "schwartz_1985": ("@book", "Schwartz, B. I.", "The World of Thought in Ancient China", "Harvard University Press", "Cambridge, MA", "1985"),
    "scott_1985": ("@book", "Scott, J. C.", "Weapons of the Weak: Everyday Forms of Peasant Resistance", "Yale University Press", "New Haven", "1985"),
    "scott_1998": ("@book", "Scott, J. C.", "Seeing Like a State: How Certain Schemes to Improve the Human Condition Have Failed", "Yale University Press", "New Haven", "1998"),
    "scott_2009": ("@book", "Scott, J. C.", "The Art of Not Being Governed: An Anarchist History of Upland Southeast Asia", "Yale University Press", "New Haven", "2009"),
    "simard_2016": ("@article", "Simard, S. W.", "Note on the Wood Wide Web", "Nature", "", "2016"),
    "simondon_2017": ("@book", "Simondon, G.", "On the Mode of Existence of Technical Objects", "Univocal", "Minneapolis", "2017"),
    "soja_1996": ("@book", "Soja, E. W.", "Thirdspace: Journeys to Los Angeles and Other Real-and-Imagined Places", "Blackwell", "Oxford", "1996"),
    "srnicek_2017": ("@book", "Srnicek, N.", "Platform Capitalism", "Polity Press", "Cambridge", "2017"),
    "stengers_2010": ("@book", "Stengers, I.", "Cosmopolitics I", "University of Minnesota Press", "Minneapolis", "2010"),
    "stengers_2015": ("@book", "Stengers, I.", "In Catastrophic Times: Resisting the Coming Barbarism", "Open Humanities Press", "London", "2015"),
    "stengers_2018": ("@book", "Stengers, I.", "Another Science Is Possible: A Manifesto for Slow Science", "Polity Press", "Cambridge", "2018"),
    "stiegler_1994": ("@book", "Stiegler, B.", "Technics and Time, 1: The Fault of Epimetheus", "Stanford University Press", "Stanford", "1998"),
    "stiegler_1998": ("@book", "Stiegler, B.", "Technics and Time, 1: The Fault of Epimetheus", "Stanford University Press", "Stanford", "1998"),
    "stiegler_2010": ("@book", "Stiegler, B.", "Taking Care of Youth and the Generations", "Stanford University Press", "Stanford", "2010"),
    "stodder_2009": ("@article", "Stodder, J.", "Complementary credit networks and macroeconomic stability: Switzerland's Wirtschaftsring", "Journal of Economic Behavior \\& Organization", "72(1)", "2009"),
    "taylor_1994": ("@book", "Taylor, C.", "Multiculturalism: Examining the Politics of Recognition", "Princeton University Press", "Princeton", "1994"),
    "the_univalent_foundations_program_2013": ("@book", "The Univalent Foundations Program", "Homotopy Type Theory: Univalent Foundations of Mathematics", "Institute for Advanced Study", "Princeton", "2013"),
    "thom_1975": ("@book", "Thom, R.", "Structural Stability and Morphogenesis", "W. A. Benjamin", "Reading, MA", "1975"),
    "tsing_2005": ("@book", "Tsing, A. L.", "Friction: An Ethnography of Global Connection", "Princeton University Press", "Princeton", "2005"),
    "tsing_2015": ("@book", "Tsing, A. L.", "The Mushroom at the End of the World: On the Possibility of Life in Capitalist Ruins", "Princeton University Press", "Princeton", "2015"),
    "tutu_1999": ("@book", "Tutu, D.", "No Future Without Forgiveness", "Doubleday", "New York", "1999"),
    "zhuangzi_2009": ("@book", "Ziporyn, B. (trans.)", "Zhuangzi: The Essential Writings with Selections from Traditional Commentaries", "Hackett", "Indianapolis", "2009"),
    "zuboff_2019": ("@book", "Zuboff, S.", "The Age of Surveillance Capitalism: The Fight for a Human Future at the New Frontier of Power", "PublicAffairs", "New York", "2019"),
    "moulier-boutang_2011": ("@book", "Moulier-Boutang, Y.", "Cognitive Capitalism", "Polity Press", "Cambridge", "2011"),
    # Additional referenced in text but not in initial list
    "augustine_426": ("@book", "Augustine of Hippo", "The City of God", "Penguin", "London", "1972"),
    "plato_timaeus": ("@book", "Plato", "Timaeus", "Hackett", "Indianapolis", "2000"),
}


def emit_bib(cites):
    lines = ["% Generated by build_bib.py from inline citations in chapters/*.tex",
             "% APA-style keys: lowercase surname underscore year",
             ""]
    for key in sorted(cites.keys()):
        if key in KNOWN:
            kind, author, title, publisher, address, year = KNOWN[key]
            lines.append(f"{kind}{{{key},")
            lines.append(f"  author    = {{{author}}},")
            lines.append(f"  title     = {{{title}}},")
            if kind == "@article":
                lines.append(f"  journal   = {{{publisher}}},")
                if address:
                    lines.append(f"  volume    = {{{address}}},")
            elif kind == "@incollection":
                lines.append(f"  booktitle = {{{publisher}}},")
            else:
                if publisher:
                    lines.append(f"  publisher = {{{publisher}}},")
                if address:
                    lines.append(f"  address   = {{{address}}},")
            lines.append(f"  year      = {{{year}}},")
            lines.append("}")
            lines.append("")
        else:
            authors, year = cites[key]
            lines.append(f"% TODO: verify -- inferred entry for {authors}, {year}")
            lines.append(f"@book{{{key},")
            lines.append(f"  author    = {{{authors}}},")
            lines.append(f"  title     = {{[unknown title]}},")
            lines.append(f"  year      = {{{year}}},")
            lines.append("}")
            lines.append("")
    BIB_PATH.write_text("\n".join(lines))


def main():
    cites = collect_citations()
    print(f"Found {len(cites)} unique citation keys")
    missing = [k for k in cites if k not in KNOWN]
    if missing:
        print(f"Missing from KNOWN: {len(missing)}")
        for m in missing[:20]:
            print(f"  {m}: {cites[m]}")
    emit_bib(cites)
    print(f"Wrote {BIB_PATH}")


if __name__ == "__main__":
    main()
