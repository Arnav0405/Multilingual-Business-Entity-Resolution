"""
Failure-case gallery for the "Where it fails" tab.

Transcribed verbatim from report.md §22.2 (false positives / false negatives)
and §23 (error distribution). Values are the modelD test-split scores at its
val-tuned threshold 0.701. No model calls — this is static report data.

Each FAILURES entry:
    category      one of "domain-compression" | "noise-string" | "same-core-words"
    s1_name       query-side business name
    candidate_name  the (mis)matched candidate name
    script_pair   e.g. "Latin -> Latin"
    score         modelD cosine
    reason        one-sentence why it fails
"""

from __future__ import annotations

FAILURE_INTRO = (
    "A demo that only shows wins is not honest. These are the model's real "
    "errors on the held-out test split (modelD, threshold 0.701): false "
    "positives, where two genuinely different businesses score high, and false "
    "negatives, where the same business is missed. Almost every miss is a "
    "domain-style compression of a name into a URL or social handle, or a "
    "near-random noise string — the population a cross-encoder reranker over "
    "the bi-encoder's top-K (report §31, future work) would target."
)

# --- categories, in the order they should render -----------------------------
CATEGORIES = ("domain-compression", "noise-string", "same-core-words")

CATEGORY_TITLES = {
    "domain-compression": "Domain / social-handle compression",
    "noise-string": "Short / no-overlap noise strings",
    "same-core-words": "Same core words, different business",
}


FAILURES: list[dict] = [
    # ===== False positives (report §22.2) — model says match, truth says no =
    {
        "category": "same-core-words",
        "s1_name": "International Projects Private Limited",
        "candidate_name": "इंटरनेशनल प्रोजेक्ट्स प्राइवेट लिमिटेड",
        "script_pair": "Latin -> Devanagari",
        "score": 0.9070,
        "reason": "Exact transliteration recorded under a different entity_id — "
                  "arguably a human-labelling failure more than a model one (report §22.2).",
    },
    {
        "category": "same-core-words",
        "s1_name": "Division of Housing",
        "candidate_name": "Division Division Health of",
        "script_pair": "Latin -> Latin",
        "score": 0.8355,
        "reason": "Shared high-frequency tokens (Division/of) dominate when the "
                  "name carries no distinctive token.",
    },
    {
        "category": "same-core-words",
        "s1_name": "Maa Divine Ltd",
        "candidate_name": "Maa LLP Center",
        "script_pair": "Latin -> Latin",
        "score": 0.8355,
        "reason": "One shared rare token (Maa) plus generic suffixes reads as a "
                  "match to a bi-encoder with no reranker.",
    },
    {
        "category": "same-core-words",
        "s1_name": "Sun Infra",
        "candidate_name": "Sunrise Infra",
        "script_pair": "Latin -> Latin",
        "score": 0.8183,
        "reason": "Subword overlap (Sun/Sunrise) + identical suffix; distinct "
                  "businesses the model cannot separate at name level.",
    },
    {
        "category": "domain-compression",
        "s1_name": "Lotus Finance",
        "candidate_name": "lotustrading.com",
        "script_pair": "Latin -> Latin",
        "score": 0.7675,
        "reason": "Candidate is a compressed domain form — no shared token "
                  "beyond the subword Lotus.",
    },
    {
        "category": "same-core-words",
        "s1_name": "New Delhi Scientifics",
        "candidate_name": "New Delhi Solar [Ltd]",
        "script_pair": "Latin -> Latin",
        "score": 0.7651,
        "reason": "Shared prefix (New Delhi) is a location, not the business "
                  "identity; the tail tokens differ entirely.",
    },
    {
        "category": "same-core-words",
        "s1_name": "White Media LLP",
        "candidate_name": "White Pvt Ltd Center",
        "script_pair": "Latin -> Latin",
        "score": 0.7233,
        "reason": "Common leading token (White) plus legal-form boilerplate "
                  "pushes the score over threshold.",
    },
    {
        "category": "same-core-words",
        "s1_name": "Pediatric Pacific Specialists Inc",
        "candidate_name": "PEDIATRIC CLINIC LLC PARTNERS",
        "script_pair": "Latin -> Latin",
        "score": 0.7125,
        "reason": "Domain-generic medical token (Pediatric) + legal forms; the "
                  "distinctive tokens (Pacific vs Clinic) don't align.",
    },
    {
        "category": "same-core-words",
        "s1_name": "Pacific Council PC",
        "candidate_name": "Pacific Multimedia",
        "script_pair": "Latin -> Latin",
        "score": 0.7985,
        "reason": "Shared leading token (Pacific) with unrelated second token.",
    },
    {
        "category": "same-core-words",
        "s1_name": "Dermatology Clinic LLC",
        "candidate_name": "Dermatology Group",
        "script_pair": "Latin -> Latin",
        "score": 0.8645,
        "reason": "The canonical hard-negative case: one shared domain token, "
                  "different businesses (report §23, failure mode 2).",
    },

    # ===== False negatives (report §22.2) — model says no, truth says match =
    {
        "category": "same-core-words",
        "s1_name": "Hotel Engineering Private Limited",
        "candidate_name": "HOTEL PRIVATE LIMITED SERVICE",
        "script_pair": "Latin -> Latin",
        "score": 0.6880,
        "reason": "Word-order change: the shared tokens are present but the "
                  "sequence is reordered below threshold.",
    },
    {
        "category": "domain-compression",
        "s1_name": "Allahabad Pharma Private Limited",
        "candidate_name": "pharmaprivate.c0m",
        "script_pair": "Latin -> Latin",
        "score": 0.6553,
        "reason": "Domain-style compression (spaces removed, .com, 'o'->'0') "
                  "the bi-encoder cannot bridge.",
    },
    {
        "category": "domain-compression",
        "s1_name": "Laughlin Supreme Infrastructure LLC",
        "candidate_name": "@laughlinsupreme",
        "script_pair": "Latin -> Latin",
        "score": 0.6436,
        "reason": "Social-handle contraction: only one token survives and it is "
                  "buried in a handle.",
    },
    {
        "category": "domain-compression",
        "s1_name": "Sai Builders Private Limited",
        "candidate_name": "buildersprivate.com",
        "script_pair": "Latin -> Latin",
        "score": 0.6011,
        "reason": "Domain-style compression plus reorder of the shared tokens.",
    },
    {
        "category": "noise-string",
        "s1_name": "Calderon, Rivera and Vega",
        "candidate_name": "Tavoevoonyx",
        "script_pair": "Latin -> Latin",
        "score": 0.5749,
        "reason": "Near-random character sequence with no token overlap — "
                  "unresolvable without further context.",
    },
    {
        "category": "domain-compression",
        "s1_name": "Chrissy Harvey, DMD",
        "candidate_name": "hcdmd.com",
        "script_pair": "Latin -> Latin",
        "score": 0.5690,
        "reason": "Initials-plus-credential compressed into a domain form.",
    },
    {
        "category": "noise-string",
        "s1_name": "Karveer India Limited",
        "candidate_name": "Jáxkeloiri",
        "script_pair": "Latin -> Latin",
        "score": 0.5330,
        "reason": "Noise string with accented characters and no shared tokens.",
    },
    {
        "category": "domain-compression",
        "s1_name": "Primary Care Health of Danville LLC",
        "candidate_name": "carehealth.com",
        "script_pair": "Latin -> Latin",
        "score": 0.5207,
        "reason": "Domain compression with no shared token beyond generic "
                  "health-care words.",
    },
    {
        "category": "noise-string",
        "s1_name": "Confederation Vidyalaya",
        "candidate_name": "BELOEVOYUMA",
        "script_pair": "Latin -> Latin",
        "score": 0.4927,
        "reason": "Near-random uppercase string; no lexical signal to match on.",
    },
    {
        "category": "noise-string",
        "s1_name": "Summit Empire Chs Inc",
        "candidate_name": "Rizanextavo",
        "script_pair": "Latin -> Latin",
        "score": 0.3399,
        "reason": "Noise string with essentially zero overlap — the lowest-"
                  "scoring miss.",
    },
]


# Error-type distribution (report §23) — optional summary rendered under the gallery.
ERROR_DISTRIBUTION: list[dict] = [
    {"error_type": "Short / no-overlap noise strings", "fn": 84, "fp": 2, "total": 86, "pct": 35.0},
    {"error_type": "Same-core-words different business", "fn": 29, "fp": 13, "total": 42, "pct": 17.6},
    {"error_type": "Domain / social-handle compression", "fn": 31, "fp": 6, "total": 37, "pct": 15.5},
    {"error_type": "Cross-script false positives", "fn": 0, "fp": 6, "total": 6, "pct": 2.5},
    {"error_type": "Word-order change", "fn": None, "fp": None, "total": None, "pct": None},
    {"error_type": "Transliteration ambiguity", "fn": 0, "fp": 0, "total": 0, "pct": 0.0},
]
