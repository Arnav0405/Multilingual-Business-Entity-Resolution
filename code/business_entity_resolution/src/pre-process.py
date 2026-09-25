"""
Pre-process the Business Entity Resolution data.

Pipeline (per spec):
  1. Drop rows with any empty field in the three source listings.
     Exception: the ground-truth file is NEVER dropped — an empty
     `source1_entity_id` there is replaced with the literal token `<empty>`
     (empty `matched_entity_ids` get the same token).
  2. String normalization: unicode NFKC fold, lowercase, strip accents,
     collapse punctuation/whitespace, drop legal-entity suffix noise
     ("ltd", "llc", "pvt", ...). Keeps Devanagari (Hindi) characters so
     source2's Indian names survive normalization.
  3. Character n-grams (n=3, full-string fallback for very short strings)
     over normalized name and address.
  4. Canopy clustering blocking: each record is assigned to canopies defined
     by a small family of cheap blocking keys (name prefix+length bucket,
     token prefixes, sampled n-gram buckets, character-order skeleton).
     NO country blocking — the test data contains countries not in train.
  5. Everything is written as ONE csv file with preprocessed fields, the
     n-grams, canopy ids, and (for source1 rows) the ground-truth matches.

Run:  python pre-process.py               # train sources + ground truth
      python pre-process.py --split test  # test sources (no gt merge)
      python pre-process.py --start 0 --end 500000   # rows [0, 500000) of
                                                     # EACH source file
Output: <repo>/output/preprocessed_train.csv (or _test.csv; a row range
        adds _rows<start>-<end>, so chunks don't overwrite each other)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import unicodedata
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
DATA_DIR = REPO_ROOT / "data" / "student_resource" / "dataset"
OUTPUT_DIR = REPO_ROOT / "output"

SOURCE_FILES = {
    "S1": "train_source1.tsv",
    "S2": "train_source2.tsv",
    "S3": "train_source3.tsv",
}
GROUND_TRUTH_FILE = "train_ground_truth.tsv"

EMPTY_TOKEN = "<empty>"

# ---------------------------------------------------------------------------
# String normalization
# ---------------------------------------------------------------------------

# Legal / corporate suffixes stripped from names. Raw forms; the set actually
# used (_SUFFIXES, below normalize_text) is these passed through
# normalize_text, so they match exactly what the pipeline produces (dots
# dropped, zero-width joiners removed, NFKC/NFC applied).
_RAW_SUFFIXES = [
    # English / generic Latin
    "ltd", "limited", "llc", "llp", "lp", "inc", "incorporated", "corp",
    "corporation", "co", "company", "pvt", "private", "pcl", "plc",
    "gmbh", "srl", "sa", "ag", "bv", "nv", "pty",
    # French legal forms (test split adds France); "cie" = "& Co"
    "sarl", "sas", "sasu", "eurl", "sci", "snc", "selarl", "cie",
    # "private" / "limited" / "pvt." / "ltd." / "LLP" in every Indic script
    # present in source2/3 (tokens copied from the data itself)
    "प्राइवेट", "लिमिटेड", "प्रा.", "लि.", "एलएलपी",                 # Devanagari
    "பிரைவேட்", "லிமிடெட்", "எல்எல்பி",                             # Tamil
    "ప్రైవేట్", "లిమిటెడ్", "ఎల్\u200cఎల్\u200cపీ",                  # Telugu
    "ಪ್ರೈವೇಟ್", "ಲಿಮಿಟೆಡ್", "ಎಲ್ಎಲ್\u200cಪಿ",                        # Kannada
    "প্রাইভেট", "লিমিটেড", "এলএলপি",                                # Bengali
    "પ્રાઇવેટ", "લિમિટેડ", "પ્રા.", "લિ.", "એલએલપી",                 # Gujarati
    "പ്രൈവറ്റ്", "ലിമിറ്റഡ്", "എൽഎൽപി",                             # Malayalam
    "ପ୍ରାଇଭେଟ୍", "ଲିମିଟେଡ୍", "ଏଲ୍\u200cଏଲ୍\u200cପି",                  # Oriya
    "ਪ੍ਰਾਈਵੇਟ", "ਲਿਮਟਿਡ", "ਪ੍ਰਾ.", "ਲਿ.", "ਐਲਐਲਪੀ",                  # Gurmukhi
]

# dotted single-letter acronyms: "s.a.r.l." / "l.l.c." / "a.p" -> "sarl" /
# "llc" / "ap". Without this, punctuation removal yields "s a r l" and the
# suffix is never recognised.
_DOTTED_ACRONYM_RE = re.compile(r"\b[a-z](?:\.[a-z])+\b\.?")

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")
# NOTE: matras are category Mc, which [^\w\s] kills. They are semantic in
# Devanagari, so normalize_text must keep every mark that is not an ASCII
# (Latin) accent. See the mark-preservation step inside normalize_text.

_PLACEHOLDER_ADDRESSES = {"na", "n a", "n", "none", "null", "--", "-"}


# zero-width / invisible chars that corrupt Indic text and tokenization
_INVISIBLE_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff\u00ad]")
# UTF-8 punctuation mis-decoded as Latin-1 ("\u00e2\x80\x93" = en dash,
# "\u00c2\x80\x99" = apostrophe): C1 control bytes never occur legitimately
_MOJIBAKE_RE = re.compile(r"[\u00c2\u00e2]?[\u0080-\u009f]+")


def normalize_text(text: str) -> str:
    """NFKC fold, lowercase, strip accents/diacritics, kill punctuation,
    collapse whitespace. Preserves word characters of any script (so Devanagari
    and other non-Latin names are kept, not emptied)."""
    if not text or text == EMPTY_TOKEN:
        return ""
    # strip zero-width/invisible chars FIRST — they corrupt Indic scripts
    # and tokenization if left in (e.g. ZWJ inside Telugu names)
    text = _INVISIBLE_RE.sub("", text)
    text = _MOJIBAKE_RE.sub(" ", text)
    s = unicodedata.normalize("NFKC", text).lower()
    # decompose, then strip accents from Latin letters only — removing
    # combining marks from Devanagari would destroy matras
    s = unicodedata.normalize("NFD", s)
    chars = []
    for ch in s:
        # Latin diacritics live exclusively in U+0300-036F — drop those.
        # Devanagari matras/virama are elsewhere (Mc/Mn but not in that
        # range) and stay, so Hindi names are untouched.
        if 0x0300 <= ord(ch) <= 0x036F:
            continue
        cat = unicodedata.category(ch)
        if cat == "Cn" or ord(ch) < 0x20:  # unassigned / control junk
            continue
        chars.append(ch)
    s = "".join(chars)
    # recompose so accented-Latin bases are normal single chars again
    s = unicodedata.normalize("NFC", s)
    s = _DOTTED_ACRONYM_RE.sub(lambda m: m.group(0).replace(".", ""), s)
    # punctuation removal — but matras/Halant are category Mc/Mn, not P*, and
    # must survive: they spell Devanagari words
    def _is_punct(ch: str) -> bool:
        cat = unicodedata.category(ch)
        if cat in ("Mc", "Mn", "Me"):
            return False
        return not (ch.isalnum() or ch.isspace() or cat.startswith("N"))
    s = "".join(" " if _is_punct(ch) else ch for ch in s)
    return _WS_RE.sub(" ", s).strip()


_SUFFIXES = {normalize_text(s) for s in _RAW_SUFFIXES}


def normalize_name(text: str) -> str:
    """normalize_text + drop legal-entity suffix tokens."""
    norm = normalize_text(text)
    if not norm:
        return ""
    toks = [t for t in norm.split() if t not in _SUFFIXES]
    # keep at least one token so the record is never fully emptied
    return " ".join(toks) if toks else norm


def normalize_address(text: str) -> str:
    """normalize_text + map placeholder addresses to '' (handled downstream)."""
    norm = normalize_text(text)
    if not norm or norm in _PLACEHOLDER_ADDRESSES:
        return ""
    return norm


# ---------------------------------------------------------------------------
# Address parsing: pincode / country / state / city
# ---------------------------------------------------------------------------

# US states: full names and USPS abbreviations (lowercase)
_US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut",
    "de": "delaware", "fl": "florida", "ga": "georgia", "hi": "hawaii",
    "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine",
    "md": "maryland", "ma": "massachusetts", "mi": "michigan",
    "mn": "minnesota", "ms": "mississippi", "mo": "missouri",
    "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire",
    "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio",
    "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota",
    "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont",
    "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
    "alabama": "alabama", "alaska": "alaska", "arizona": "arizona",
    "arkansas": "arkansas", "california": "california", "colorado": "colorado",
    "connecticut": "connecticut", "delaware": "delaware", "florida": "florida",
    "georgia": "georgia", "hawaii": "hawaii", "idaho": "idaho",
    "illinois": "illinois", "indiana": "indiana", "iowa": "iowa",
    "kansas": "kansas", "kentucky": "kentucky", "louisiana": "louisiana",
    "maine": "maine", "maryland": "maryland", "massachusetts":
    "massachusetts", "michigan": "michigan", "minnesota": "minnesota",
    "mississippi": "mississippi", "missouri": "missouri", "montana":
    "montana", "nebraska": "nebraska", "nevada": "nevada",
    "new hampshire": "new hampshire", "new jersey": "new jersey",
    "new mexico": "new mexico", "new york": "new york", "north carolina":
    "north carolina", "north dakota": "north dakota", "ohio": "ohio",
    "oklahoma": "oklahoma", "oregon": "oregon", "pennsylvania":
    "pennsylvania", "rhode island": "rhode island", "south carolina":
    "south carolina", "south dakota": "south dakota", "tennessee":
    "tennessee", "texas": "texas", "utah": "utah", "vermont": "vermont",
    "virginia": "virginia", "washington": "washington", "west virginia":
    "west virginia", "wisconsin": "wisconsin", "wyoming": "wyoming",
    "district of columbia": "district of columbia",
}

# Indian states / UTs — English names as commonly written in addresses,
# incl. common short forms.
_INDIA_STATES = {
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
    "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka",
    "kerala", "madhya pradesh", "maharashtra", "manipur", "meghalaya",
    "mizoram", "nagaland", "odisha", "orissa", "punjab", "rajasthan",
    "sikkim", "tamil nadu", "telangana", "tripura", "uttar pradesh",
    "uttarakhand", "uttaranchal", "west bengal", "delhi", "new delhi",
    "nct of delhi", "jammu and kashmir", "ladakh", "puducherry", "pondicherry",
    "chandigarh", "andaman and nicobar", "andaman nicobar",
    "dadra and nagar haveli", "daman and diu", "lakshadweep",
}

# Native-script state names -> English. Source2/3 write the state as a
# whole component in the regional script; these forms are every non-ASCII
# state component in train+test (~2M rows). Mapping to English keeps the
# state column comparable across sources.
_INDIA_STATES_NATIVE = {
    "महाराष्ट्र": "maharashtra", "दिल्ली": "delhi", "नई दिल्ली": "new delhi",
    "उत्तर प्रदेश": "uttar pradesh", "हरियाणा": "haryana",
    "राजस्थान": "rajasthan", "बिहार": "bihar", "मध्य प्रदेश": "madhya pradesh",
    "ಕರ್ನಾಟಕ": "karnataka", "தமிழ்நாடு": "tamil nadu", "ગુજરાત": "gujarat",
    "পশ্চিমবঙ্গ": "west bengal", "తెలంగాణ": "telangana", "കേരളം": "kerala",
    "ఆంధ్రప్రదేశ్": "andhra pradesh", "ਪੰਜਾਬ": "punjab", "ଓଡ଼ିଶା": "odisha",
}

# alias -> canonical state (delhi district ambiguity, renamed states, and
# the vehicle-code abbreviations ~20% of India rows use as the state
# component: "..., Pune, MH"). Only consulted for India rows, so "ka"/"ga"
# never collide with the US table.
_STATE_ALIASES = {"new delhi": "delhi", "pondicherry": "puducherry",
                  "orissa": "odisha", "uttaranchal": "uttarakhand",
                  "keralam": "kerala",
                  "mh": "maharashtra", "up": "uttar pradesh", "ka": "karnataka",
                  "tn": "tamil nadu", "gj": "gujarat", "wb": "west bengal",
                  "tg": "telangana", "ts": "telangana", "hr": "haryana",
                  "rj": "rajasthan", "kl": "kerala", "br": "bihar",
                  "mp": "madhya pradesh", "ap": "andhra pradesh",
                  "pb": "punjab", "od": "odisha", "dl": "delhi",
                  "ga": "goa", "jh": "jharkhand", "cg": "chhattisgarh",
                  "hp": "himachal pradesh", "jk": "jammu and kashmir",
                  "uk": "uttarakhand"}
_INDIA_STATES |= _STATE_ALIASES.keys()


# French regions (state column for France) and the departments that show up
# instead of / alongside them; a department maps to its region so S1's
# "Pays de la Loire" and S2's "Loire-Atlantique" agree. "Paris" and "Vienne"
# are left out on purpose: they are also city names.
_FR_REGIONS_RAW = [
    "Hauts-de-France", "Nouvelle-Aquitaine", "Pays de la Loire",
    "Île-de-France", "Auvergne-Rhône-Alpes", "Provence-Alpes-Côte d'Azur",
    "Occitanie", "Grand Est", "Bretagne", "Normandie",
    "Bourgogne-Franche-Comté", "Centre-Val de Loire", "Corse",
]
_FR_DEPARTMENTS_RAW = {
    "Hauts-de-France": ["Nord", "Pas-de-Calais", "Somme", "Aisne", "Oise"],
    "Nouvelle-Aquitaine": ["Gironde", "Landes", "Pyrénées-Atlantiques",
                           "Charente", "Charente-Maritime", "Dordogne",
                           "Lot-et-Garonne", "Haute-Vienne", "Deux-Sèvres",
                           "Corrèze", "Creuse"],
    "Pays de la Loire": ["Loire-Atlantique", "Maine-et-Loire", "Mayenne",
                         "Sarthe", "Vendée"],
    "Île-de-France": ["Hauts-de-Seine", "Seine-Saint-Denis", "Val-de-Marne",
                      "Essonne", "Yvelines", "Val-d'Oise", "Seine-et-Marne"],
    "Auvergne-Rhône-Alpes": ["Rhône", "Isère", "Savoie", "Haute-Savoie"],
    "Provence-Alpes-Côte d'Azur": ["Bouches-du-Rhône", "Alpes-Maritimes",
                                   "Var", "Vaucluse"],
    "Occitanie": ["Haute-Garonne", "Hérault", "Gard"],
    "Grand Est": ["Bas-Rhin", "Haut-Rhin", "Moselle", "Marne"],
}

# leading tokens that make a component a street / building / unit line, not
# a city (French puts the type first: "rue ...", "av fleming", "bat d")
_STREET_LEAD = {
    "rue", "r", "avenue", "av", "ave", "bd", "boulevard", "blvd", "place",
    "pl", "allee", "chemin", "ch", "impasse", "imp", "quai", "cours",
    "route", "rte", "square", "sq", "passage", "bat", "batiment", "residence",
    "res", "complexe", "salle", "zone", "za", "zi", "centre", "lotissement",
    "cite", "appt", "apt", "unit", "suite", "fl", "floor", "bldg", "building",
    "room", "po", "box", "c", "no", "plot", "flat", "door", "house", "kh",
    "khasra", "sector", "sec", "shop", "office",
}
# trailing tokens that make a component a street (US / India put the type
# last: "mack rd", "west street")
_STREET_TAIL = {
    "street", "st", "road", "rd", "avenue", "ave", "av", "lane", "ln",
    "drive", "dr", "court", "ct", "boulevard", "blvd", "way", "place", "pl",
    "circle", "cir", "trail", "trl", "terrace", "ter", "parkway", "pkwy",
    "highway", "hwy", "path", "loop", "square", "sq", "marg",
}
_COUNTRY_WORDS = {"us", "usa", "united states", "united states of america",
                  "india", "bharat", "france", "fr"}
_CITY_PREFIX = {"st": "saint", "ste": "sainte"}

_COUNTRY_CANON = {
    "us": "us", "usa": "us", "united states": "us",
    "india": "india", "in": "india",
    "france": "france", "fr": "france",
}

# Indian PINs appear in the data only as "600 102" (never 6 contiguous
# digits); accept both. US ZIP: 5 digits (+4) that do NOT lead their
# component — the data has no ZIPs, but 5-digit house numbers ("17560 Ellis
# Road") are everywhere. France: any standalone 5-digit code (street numbers
# are short, and "75002 Paris" puts the code first).
_IN_PIN_RE = re.compile(r"(?<!\d)(\d{3}) ?(\d{3})(?!\d)")
_US_ZIP_RE = re.compile(r"(?:(?<=\S )|^)(\d{5})(?: \d{4})?$")
# ("bp 40413" / "cs82228" are PO-box numbers, not postcodes)
_FR_CP_RE = re.compile(r"(?<!\d)(?<!bp)(?<!bp )(?<!cs)(?<!cs )(\d{5})(?!\d)")

_STATE_LOOKUP = {
    "us": _US_STATES,  # keys already normalized lowercase
    "india": {
        **{normalize_text(s): _STATE_ALIASES.get(s, s) for s in _INDIA_STATES},
        **{normalize_text(n): _STATE_ALIASES.get(e, e)
           for n, e in _INDIA_STATES_NATIVE.items()},
    },
    "france": {
        **{normalize_text(r): normalize_text(r) for r in _FR_REGIONS_RAW},
        **{normalize_text(d): normalize_text(r)
           for r, ds in _FR_DEPARTMENTS_RAW.items() for d in ds},
    },
}


def _extract_pincode(comp: str, country: str) -> tuple[str, str]:
    """(pincode, component with the pincode cut out) for one component."""
    if country == "us":
        # anchored at the component end: "ma 01104" / "01104" match,
        # "17560 ellis road" (house number) does not
        m = _US_ZIP_RE.search(comp)
        if m:
            return m.group(1), comp[: m.start()].strip()
        return "", comp
    regex = {"india": _IN_PIN_RE, "france": _FR_CP_RE}.get(country)
    found = list(regex.finditer(comp)) if regex else []
    if not found:
        return "", comp
    m = found[-1]
    pin = "".join(g for g in m.groups() if g)
    return pin, (comp[: m.start()] + " " + comp[m.end():]).strip()


def _has_street_word(comp: str) -> bool:
    toks = comp.split()
    return toks[0] in _STREET_LEAD or toks[-1] in _STREET_TAIL


def parse_address(address_raw: str, country: str = "") -> dict:
    """Split a RAW address into structured columns: pincode, state, city.

    Works on the comma-separated components of the raw string, because the
    component ORDER is shuffled across sources ("Riverhead, NY, 189 Trout
    Brook Lane" / "Hauts-de-France, Dunkerque, 71 Rue de la Paix") — once
    normalization drops the commas, street words, states and cities blur.
    The record's country column picks the rules; a country with no tables
    gets pincode-free city extraction only and an empty state.

      pincode: India "600 102" / 600102; US 5(+4) digits not leading a
               street component; France standalone 5-digit code
      state:   a WHOLE component equal to a state (US: prefer the 2-letter
               abbreviation, so "Washington, ..., DC" -> DC); the last such
               component wins; French departments map to their region
      city:    the last component that is not the state, a country, or a
               numbered street/unit line, preferring one without a street
               word ("rue", "rd", "av", ...)
    """
    out = {"pincode": "", "state": "", "city": ""}
    if not address_raw:
        return out
    ctry = _COUNTRY_CANON.get((country or "").strip().lower(), "")
    table = _STATE_LOOKUP.get(ctry, {})

    comps: list[str] = []
    for part in address_raw.split(","):
        comp = normalize_text(part)
        if not comp or comp in _PLACEHOLDER_ADDRESSES:
            continue
        pin, comp = _extract_pincode(comp, ctry)
        if pin:
            out["pincode"] = pin
        if comp and comp not in _COUNTRY_WORDS:
            comps.append(comp)

    hits = [i for i, c in enumerate(comps) if c in table]
    if ctry == "us":
        # "New York, NY" / "Washington, ..., WA": the abbreviation is the
        # state, the full name next to it is the city
        hits = [i for i in hits if len(comps[i]) == 2] or hits
    elif ctry == "india":
        # "New Delhi, Delhi": the canonical name is the state, the alias
        # ("new delhi") next to it is the city
        hits = [i for i in hits if comps[i] == table[comps[i]]] or hits
    if not hits and ctry == "india":
        # state glued onto a longer component: "8 new paneri udaipur rajasthan"
        for i in range(len(comps) - 1, -1, -1):
            words = comps[i].split()
            for n in (3, 2, 1):
                tail = " ".join(words[-n:])
                if len(words) > n and len(tail) >= 4 and tail in table:
                    comps.insert(i + 1, tail)
                    comps[i] = " ".join(words[:-n])
                    hits = [i + 1]
                    break
            if hits:
                break
    state_idx = hits[-1] if hits else -1
    if hits:
        out["state"] = table[comps[state_idx]]

    def _not_city(i: int, c: str) -> bool:
        if i == state_idx or any(ch.isdigit() for ch in c):
            return True
        if ctry == "france":
            return c in table          # any region / department
        if ctry == "india":
            return c == out["state"]   # repeated "..., delhi, delhi"
        return False

    candidates = [(i, c) for i, c in enumerate(comps) if not _not_city(i, c)]
    # prefer a component with no street word; "Pike Road, AL" and
    # "Mountlake Terrace, WA" are real cities, so fall back to any
    city = next((c for _, c in reversed(candidates)
                 if not _has_street_word(c)), "") or         (candidates[-1][1] if candidates else "")
    if city:
        toks = city.split()
        toks[0] = _CITY_PREFIX.get(toks[0], toks[0])
        out["city"] = " ".join(toks)
    return out



# ---------------------------------------------------------------------------
# Character n-grams
# ---------------------------------------------------------------------------

NGRAM_SIZE = 3
FALLBACK_NGRAM_SIZE = 2   # for very short strings
MAX_NGRAMS = 24           # cap per string to keep the csv bounded


def char_ngrams(text: str, n: int = NGRAM_SIZE,
                max_ngrams: int = MAX_NGRAMS) -> list[str]:
    """Character n-grams over the space-padded string (padded form is the
    record-linkage standard: 'ab cd' -> ' ab','abc','bc ',' cd')."""
    if not text:
        return []
    padded = f" {text} "
    if len(padded) < 2 * n - 1:
        n = FALLBACK_NGRAM_SIZE
    grams = [padded[i: i + n] for i in range(len(padded) - n + 1)]
    seen: set[str] = set()
    out: list[str] = []
    for g in grams:
        if g not in seen:
            seen.add(g)
            out.append(g)
        if len(out) >= max_ngrams:
            break
    return out


# ---------------------------------------------------------------------------
# Phonetic keys
# ---------------------------------------------------------------------------

try:
    import jellyfish

    def _metaphone_key(word: str) -> str:
        return jellyfish.metaphone(word) or ""

    def _soundex_key(word: str) -> str:
        try:
            return jellyfish.soundex(word) or ""
        except Exception:
            return ""
except ImportError:  # stdlib fallback so the pipeline never breaks
    def _metaphone_key(word: str) -> str:
        return _stdlib_soundex(word)

    def _soundex_key(word: str) -> str:
        return _stdlib_soundex(word)


def _stdlib_soundex(word: str) -> str:
    """Classic American Soundex — used only if jellyfish is unavailable."""
    if not word:
        return ""
    codes = {**{c: "1" for c in "bfpv"}, **{c: "2" for c in "cgjkqsxz"},
             "d": "3", "t": "3", "l": "4", "m": "5",
             "n": "5", "r": "6"}
    w = word.upper()
    first = w[0]
    digits = []
    prev_code = None
    for ch in w[1:]:
        code = codes.get(ch)
        if code is None:
            prev_code = None if ch in "hw" else prev_code
            continue
        if code != prev_code:
            digits.append(code)
        prev_code = code
    return (first + "".join(digits) + "000")[:4]

##
## Phonetic-key design note: metaphone alone conflates many French/English
## names; we combine metaphone (primary) + soundex (secondary) and bucket by
## leading chars so canopy families stay cheap. Both operate ONLY on
## ASCII-romanized text — jellyfish is not unicode-script aware.
##


def phonetic_key(name_norm: str) -> str:
    """phonetic skeleton of a normalized name: metaphone of the first token,
    plus soundex of the last (distinguishing) token. Empty when no tokens.
    Primary key for the name_phonetic column."""
    toks = name_norm.split() if name_norm else []
    if not toks:
        return ""
    first_mp = _metaphone_key(toks[0])
    last_sdx = _soundex_key(toks[-1]) if len(toks) > 1 else ""
    return (first_mp + ("_" + last_sdx if last_sdx else "")).strip("_")


def phonetic_buckets(name_norm: str) -> set[str]:
    """ALL phonetic buckets of a name: {metaphone, soundex} x {first token,
    last token}. Two variant spellings of the same name usually share at
    least one bucket (metaphone catches vowel drift, soundex catches
    initial-consonant-cluster differences like Sh-/S-)."""
    toks = name_norm.split() if name_norm else []
    buckets: set[str] = set()
    if not toks:
        return buckets
    mp_first = _metaphone_key(toks[0])
    sdx_first = _soundex_key(toks[0])
    mp_last = _metaphone_key(toks[-1])
    sdx_last = _soundex_key(toks[-1])
    for k in (mp_first, sdx_first, mp_last, sdx_last):
        if k:
            buckets.add(k)
    return buckets


def phonetic_variants(name_norm: str) -> list[str]:
    """Additional blocking variants: metaphone of the LAST token and the
    concatenated name skeleton — catches swapped first/last token orders
    and pooled multi-token name spellings."""
    toks = name_norm.split() if name_norm else []
    variants: list[str] = []
    if len(toks) >= 2:
        mp_last = _metaphone_key(toks[-1])
        if mp_last:
            variants.append(mp_last)
    if toks:
        skel = "".join(_metaphone_key(t) for t in toks[:3])
        if skel:
            variants.append(skel[:10])
    return variants


def _canopy_id(tag: str, key: str) -> str:
    """Stable short canopy id: <tag><first 10 hex of md5(tag|key)>."""
    digest = hashlib.md5(f"{tag}|{key}".encode("utf-8")).hexdigest()[:10]
    return f"{tag}{digest}"


def canopy_keys(name_norm: str, address_norm: str) -> list[str]:
    """Generate canopy blocking keys for one record.

    Canopy-key families (name-first — the ground truth matches are driven by
    name similarity; address only assists when the name is bare):
      N1  name-prefix(4) + length bucket
      N2  first-two-chars of first two tokens
      N3  first char of first token + first 3 chars of last token
      N4  sampled name 3-gram buckets
      N5  character-order skeleton (collapses transpositions/reorders)
      N7  phonetic keys (metaphone/soundex of romanized name) — catches
          same-sound different-spelling pairs across French/English/Indic
      N6  address-assisted key, only when the name has <=1 token
      A1  address-prefix key when the record has NO usable name
    No country key: test data contains countries unseen in train on purpose.
    """
    keys: list[str] = []
    toks = name_norm.split() if name_norm else []

    if toks:
        first = toks[0]
        len_bucket = min(len(name_norm) // 6, 8)
        keys.append(_canopy_id("N1", f"{first[:4]}#{len_bucket}"))
        if len(toks) >= 2:
            keys.append(_canopy_id("N2", f"{first[:2]}_{toks[1][:2]}"))
            keys.append(_canopy_id("N3", f"{first[:1]}_{toks[-1][:3]}"))
        if len(toks) <= 1 and address_norm:
            keys.append(_canopy_id(
                "N6", f"{first[:1]}_{address_norm.split()[0][:3]}"))
        grams = char_ngrams(name_norm, NGRAM_SIZE, MAX_NGRAMS)
        for g in grams[::4][:5]:
            keys.append(_canopy_id("N4", g))
        skeleton = "".join(dict.fromkeys(name_norm.replace(" ", "")))[:12]
        if skeleton:
            keys.append(_canopy_id("N5", f"{skeleton[:2]}{len(skeleton)}"))
        # N7: phonetic canopies — only meaningful for romanized tokens, so
        # skip tokens containing non-ASCII chars (jellyfish treats any
        # non-ASCII as unknown and would emit empty/garbage keys)
        ascii_toks = [t for t in toks if t.isascii()]
        if ascii_toks:
            for bucket in phonetic_buckets(" ".join(ascii_toks)):
                keys.append(_canopy_id("N7", bucket))
    elif address_norm:
        atoks = address_norm.split()
        keys.append(_canopy_id(
            "A1", f"{atoks[0][:4]}#{min(len(address_norm) // 6, 8)}"))
    return keys


# ---------------------------------------------------------------------------
# Loading + cleaning
# ---------------------------------------------------------------------------

def load_tsv(path: Path, start: int = 0,
             end: int | None = None) -> pd.DataFrame:
    """Read data rows [start, end) (0-based, header excluded). Rows outside
    the range are skipped by the parser, never materialized. QUOTE_NONE
    means one line == one row, so the indices are exact line offsets."""
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        quoting=csv.QUOTE_NONE,
        engine="c",
        skiprows=range(1, start + 1) if start else None,
        nrows=None if end is None else end - start,
    )


def clean_source(df: pd.DataFrame, source_tag: str) -> pd.DataFrame:
    """Drop rows with an empty name or address; rename to raw columns."""
    before = len(df)
    mask = (df["business_name"].str.strip() == "") | \
           (df["business_address"].str.strip() == "")
    df = df[~mask].copy()
    df = df.rename(columns={
        "entity_id": "entity_id",
        "business_name": "name_raw",
        "business_address": "address_raw",
    })
    df["source"] = source_tag
    for col in ("entity_id", "name_raw", "address_raw", "country"):
        df[col] = df[col].str.strip()
    df = df.drop_duplicates(subset=["entity_id"])
    print(f"  source {source_tag}: kept {len(df):,} / {before:,} rows "
          f"(dropped {before - len(df):,})")
    return df


def load_ground_truth_df(gt: pd.DataFrame) -> pd.DataFrame:
    """Ground truth rows are NEVER dropped. Empty ids are tokenized
    to <empty> instead."""
    before = len(gt)
    gt["source1_entity_id"] = gt["source1_entity_id"].str.strip()
    gt["matched_entity_ids"] = gt["matched_entity_ids"].str.strip()
    gt["source1_entity_id"] = gt["source1_entity_id"].replace("", EMPTY_TOKEN)
    gt["matched_entity_ids"] = gt["matched_entity_ids"].replace("", EMPTY_TOKEN)
    n_tok = int((gt["source1_entity_id"] == EMPTY_TOKEN).sum())
    print(f"  ground truth: kept all {before:,} rows "
          f"(tokenized {n_tok:,} empty source1 ids to {EMPTY_TOKEN!r})")
    return gt


def load_ground_truth(path: Path) -> pd.DataFrame:
    return load_ground_truth_df(load_tsv(path))


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess(sources: dict[str, pd.DataFrame],
               gt: pd.DataFrame | None) -> pd.DataFrame:
    """Clean + normalize + n-grams + canopies; return ONE combined frame.

    Output columns:
      entity_id, source, name_raw, address_raw, name_norm, address_norm,
      name_ngrams, address_ngrams, name_ngram_count, address_ngram_count,
      name_phonetic, canopy_ids, country, ground_truth
    """
    frames = []
    for tag, df in sources.items():
        print(f"Processing source {tag}...")
        df = clean_source(df, tag)
        df["name_norm"] = df["name_raw"].map(normalize_name)
        df["address_norm"] = df["address_raw"].map(normalize_address)

        # structured address columns — parsed from the RAW address (needs
        # the comma components); country never re-derived from text, it
        # stays the record's own country column
        parsed = [parse_address(a, c) for a, c in
                  zip(df["address_raw"], df["country"])]
        df["pincode"] = [p["pincode"] for p in parsed]
        df["state"] = [p["state"] for p in parsed]
        df["city"] = [p["city"] for p in parsed]

        name_grams = [char_ngrams(s) for s in df["name_norm"]]
        addr_grams = [char_ngrams(s) for s in df["address_norm"]]
        df["name_ngrams"] = ["|".join(g) for g in name_grams]
        df["address_ngrams"] = ["|".join(g) for g in addr_grams]
        df["name_ngram_count"] = [len(g) for g in name_grams]
        df["address_ngram_count"] = [len(g) for g in addr_grams]

        canopy_keys_list = [canopy_keys(n, a)
                            for n, a in zip(df["name_norm"], df["address_norm"])]
        df["canopy_ids"] = [" ".join(k) for k in canopy_keys_list]
        # phonetic key column: metaphone skeleton of ASCII tokens only
        df["name_phonetic"] = [
            phonetic_key(" ".join(t for t in s.split() if t.isascii()))
            for s in df["name_norm"]
        ]
        frames.append(df[[
            "entity_id", "source", "name_raw", "address_raw", "name_norm",
            "address_norm", "pincode", "state", "city",
            "name_ngrams", "address_ngrams",
            "name_ngram_count", "address_ngram_count", "name_phonetic",
            "canopy_ids", "country"]])

    combined = pd.concat(frames, ignore_index=True)

    # Merge ground truth so the final csv is self-contained; only source1
    # rows carry matches. Empty gt ids were already tokenized to <empty>,
    # and those tokens are never joined against real entity ids.
    if gt is not None:
        gt_map = dict(zip(gt["source1_entity_id"], gt["matched_entity_ids"]))
        s1_mask = combined["source"] == "S1"
        combined["ground_truth"] = ""
        combined.loc[s1_mask, "ground_truth"] = \
            combined.loc[s1_mask, "entity_id"].map(gt_map).fillna("")
    else:
        combined["ground_truth"] = ""

    cols = ["entity_id", "source", "name_raw", "address_raw", "name_norm",
            "address_norm", "pincode", "state", "city",
            "name_ngrams", "address_ngrams",
            "name_ngram_count", "address_ngram_count", "name_phonetic",
            "canopy_ids", "country", "ground_truth"]
    return combined[cols]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "test"], default="train",
                        help="preprocess the train or test sources")
    parser.add_argument("--out", default=None,
                        help="output csv path "
                             "(default output/preprocessed_<split>.csv, or "
                             "preprocessed_<split>_rows<start>-<end>.csv "
                             "when a row range is given)")
    parser.add_argument("--start", type=int, default=0,
                        help="first data row to process in EACH source file "
                             "(0-based, header excluded; default 0)")
    parser.add_argument("--end", type=int, default=None,
                        help="stop before this data row in EACH source file "
                             "(exclusive; default: end of file)")
    args = parser.parse_args()
    if args.start < 0:
        parser.error("--start must be >= 0")
    if args.end is not None and args.end <= args.start:
        parser.error("--end must be greater than --start")

    split_dir = DATA_DIR / args.split
    src = dict(SOURCE_FILES)
    gt_path: Path | None = None
    if args.split == "test":
        src = {k: f.replace("train_", "test_") for k, f in src.items()}
    else:
        gt_path = split_dir / GROUND_TRUTH_FILE

    ranged = args.start > 0 or args.end is not None
    print(f"Data dir: {split_dir}")
    if ranged:
        print(f"Row range: [{args.start:,}, "
              f"{'EOF' if args.end is None else f'{args.end:,}'}) per source")
    sources = {tag: load_tsv(split_dir / fname, args.start, args.end)
               for tag, fname in src.items()}
    # ground truth is a lookup keyed by source1 id, so it is always loaded
    # in full — the S1 rows in range still get their matches
    gt = load_ground_truth(gt_path) if gt_path is not None else None

    combined = preprocess(sources, gt)

    suffix = f"_rows{args.start}-{'end' if args.end is None else args.end}" \
        if ranged else ""
    out_path = Path(args.out) if args.out else \
        OUTPUT_DIR / f"preprocessed_{args.split}{suffix}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {out_path} ...")
    combined.to_csv(out_path, index=False, encoding="utf-8")
    print(f"rows={len(combined):,}  cols={list(combined.columns)}")

    s1 = combined[combined["source"] == "S1"]
    has_grams = (s1["name_ngram_count"] > 0).mean()
    print(f"verification: source1 rows={len(s1):,}  "
          f"with n-grams={has_grams:.1%}  "
          f"gt cell non-empty={(s1['ground_truth'] != '').sum():,}")
    print("Done.")


if __name__ == "__main__":
    main()
