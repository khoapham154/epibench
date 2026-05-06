"""ILAE-compliant controlled vocabularies and rule-based consolidators (v4).

v4 changes vs v3:
- survival task dropped (replaced by status_epilepticus)
- status_epilepticus added (ILAE 2015): Convulsive SE / Non-convulsive SE /
  Refractory SE / None / Unknown
- consolidate_ez_localization rewritten with explicit precedence to handle
  bilateral, multi-lobe combinations, sublobar regions, and bare lobe words.

Used by:
- STAGE_B prompt builder (controlled vocab strings injected verbatim)
- build_v4_xlsx.py (consolidation of raw _truth strings to canonical class)
"""
from __future__ import annotations
import re

# Canonical class lists (ILAE 2017 + 2010 + 2015)
LABELS = {
    "epilepsy_type": [
        "Focal", "Generalised", "Combined Focal and Generalised", "Unknown",
    ],
    "seizure_type": [
        "Focal", "Generalised", "Unknown", "Unclassified",
    ],
    "ez_localization": [
        "Temporal", "Extratemporal", "Multifocal", "Hemispheric", "Unknown",
    ],
    "aed_response": [
        "drug-responsive", "drug-resistant", "unspecified",
    ],
    "surgery_outcome": [
        "Seizure-free", "Improved", "No improvement", "Not applicable",
    ],
    "status_epilepticus": [
        "Convulsive SE", "Non-convulsive SE", "Refractory SE", "None", "Unknown",
    ],
}


# ---------- Definitions injected into STAGE_B prompt ----------

PROMPT_DEFINITIONS = """### CONTROLLED VOCABULARY (ILAE-COMPLIANT)

For each ground truth, choose ONE value from the list. Output null if the
paper does not state enough to decide. Do NOT invent values.

**epilepsy_type** (ILAE 2017 epilepsy classification):
  - "Focal": seizures originating in networks limited to one hemisphere
  - "Generalised": seizures originating in bilaterally distributed networks
  - "Combined Focal and Generalised": same patient has both types
  - "Unknown": insufficient information to determine

**seizure_type** (ILAE 2017 seizure classification, drop the "onset" suffix):
  - "Focal": clear focal onset (motor / non-motor; aware / impaired awareness)
  - "Generalised": bilateral simultaneous onset (tonic-clonic, absence, myoclonic, atonic, etc.)
  - "Unknown": onset not observed/witnessed but evidence indicates one of the above
  - "Unclassified": insufficient information to classify even broadly

**ez_localization** (epileptogenic zone, only if paper states it):
  - "Temporal": temporal lobe (mesial or lateral), single lobe only
  - "Extratemporal": single non-temporal lobe (frontal / parietal / occipital /
    insular) or sublobar region (opercular / frontobasal / rolandic / perisylvian /
    cingulate / motor cortex)
  - "Multifocal": bilateral OR two or more non-contiguous lobes (e.g.
    fronto-temporal, temporo-parietal, bilateral temporal)
  - "Hemispheric": entire hemisphere or large hemispheric region
  - "Unknown": stated to be uncertain, non-localising, cryptogenic, or only
    side (left/right) without lobe

**aed_response** (ILAE 2010 definition of drug-resistance):
  - "drug-resistant": failed adequate trials of >=2 tolerated, appropriately chosen
    and used antiseizure medications, alone or in combination
  - "drug-responsive": seizure-free for >=12 months OR three times the longest
    pre-treatment seizure-free interval, on current ASMs
  - "unspecified": on treatment but response not reported / cannot be classified

**surgery_outcome** (Engel mapping, only if patient had surgery):
  - "Seizure-free": Engel class I (free of disabling seizures)
  - "Improved": Engel class II or III (rare disabling seizures / worthwhile improvement)
  - "No improvement": Engel class IV (no worthwhile improvement)
  - "Not applicable": no surgery performed

**status_epilepticus** (ILAE 2015, only label if paper discusses SE):
  - "Convulsive SE": generalized tonic-clonic SE, GCSE
  - "Non-convulsive SE": NCSE with/without impaired awareness, absence SE,
    focal impaired-awareness SE
  - "Refractory SE": SE continuing despite first-line benzodiazepine + one
    appropriately dosed ASM (includes super-refractory SE)
  - "None": paper explicitly states no history/episode of SE
  - "Unknown": SE status not discussed
"""


# ---------- Consolidator regexes ----------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def consolidate_epilepsy_type(text: str | None) -> str | None:
    if not text:
        return None
    t = _norm(text)
    if not t or t in {"none", "null", "n/a", "na"}:
        return None
    if t in {"unknown", "not specified", "not stated"}:
        return "Unknown"
    if re.search(r"combined.*focal.*generali[sz]ed|combined.*generali[sz]ed.*focal", t):
        return "Combined Focal and Generalised"
    has_focal = bool(re.search(r"\bfocal\b|\bpartial\b|temporal|frontal|parietal|occipital|insular|cortical dysplasia|mtle|tle|fle|ple|ole", t))
    has_gen = bool(re.search(r"\bgenerali[sz]ed\b|\bidiopathic generali[sz]ed\b|\bige\b|\babsence\b|\bjuvenile myoclonic\b|\bjme\b|\blennox-?gastaut\b|\bdravet\b|\bwest\b|\binfantile spasms?\b", t))
    if has_focal and has_gen:
        return "Combined Focal and Generalised"
    if has_focal:
        return "Focal"
    if has_gen:
        return "Generalised"
    if re.search(r"unknown|cryptogenic|undetermined|unclassified", t):
        return "Unknown"
    return None


def consolidate_seizure_type(text: str | None) -> str | None:
    if not text:
        return None
    t = _norm(text)
    if not t or t in {"none", "null", "n/a", "na"}:
        return None
    if re.search(r"unclassif", t):
        return "Unclassified"
    if re.search(r"\bunknown\b|onset not observ|witness", t):
        return "Unknown"
    has_focal = bool(re.search(r"\bfocal\b|\bpartial\b|focal onset|focal aware|focal impaired|fias|fas|complex partial|simple partial|fbtcs|focal to bilateral", t))
    has_gen = bool(re.search(r"\bgenerali[sz]ed\b|tonic.?clonic|absence|myoclonic|atonic|tonic seizure|gtcs|gtc seizure|spasms?", t))
    if has_focal and has_gen:
        return "Focal"
    if has_focal:
        return "Focal"
    if has_gen:
        return "Generalised"
    if re.search(r"status epilepticus", t):
        return None
    return None


# Lobe patterns — single-word hits indicating a lobe or sublobar region
_LOBE_NAMES = {
    "temporal", "frontal", "parietal", "occipital", "insular",
}
# Sublobar extratemporal regions (belong to frontal/insular/parietal territory)
_SUBLOBAR_EXTRA = re.compile(
    r"\boperculum\b|\bopercular\b|\bfrontobasal\b|\brolandic\b|\bperisylvian\b|"
    r"\bmotor cortex\b|\bcingulate\b|\bpre.?central\b|\bpost.?central\b|"
    r"\bsupplementary motor\b|\bpre.?motor\b|\borbitofrontal\b"
)
# Compound lobe patterns (hyphenated / fused) — always multifocal
_COMPOUND_LOBES = re.compile(
    r"fronto.?temporal|temporo.?frontal|temporo.?parietal|parieto.?temporal|"
    r"fronto.?parietal|parieto.?frontal|temporo.?occipital|occipito.?temporal|"
    r"parieto.?occipital|occipito.?parietal|fronto.?insular|insulo.?frontal|"
    r"frontoparietal|temporoparietal|parietooccipital|temporooccipital|"
    r"frontotemporal"
)


def _count_distinct_lobes(t: str) -> int:
    """How many distinct lobe names appear in this text, treating
    `extratemporal` as NOT a lobe match (the exclusion comes first)."""
    hits = set()
    for lobe in _LOBE_NAMES:
        if re.search(rf"\b{lobe}\b", t):
            hits.add(lobe)
    # "extratemporal" contains "temporal" as a substring; don't let it count as "temporal"
    if "temporal" in hits and re.search(r"\bextra.?temporal\b", t):
        # If the ONLY match for "temporal" came from "extratemporal", remove it
        # Check for a standalone "temporal" that isn't part of "extratemporal"
        if not re.search(r"(?<!extra[-\s])(?<!extra)\btemporal\b", t):
            hits.discard("temporal")
    return len(hits)


def consolidate_ez_localization(text: str | None) -> str | None:
    """ILAE-compliant EZ localization bucketing.

    Precedence (top → bottom, first match wins):
    1. `bilateral` → Multifocal
    2. Two+ distinct lobes OR compound lobe pattern → Multifocal
    3. `hemispher*` → Hemispheric
    4. Explicit `multifocal` / `multiple foc*` → Multifocal
    5. Exactly one lobe = temporal → Temporal
    6. Single non-temporal lobe or sublobar region → Extratemporal
    7. Bare focal / left / right / cryptogenic / uncertain → Unknown
    8. Else → None (consolidator gives up)
    """
    if not text:
        return None
    t = _norm(text)
    if not t or t in {"none", "null", "n/a", "na"}:
        return None

    # 1. Bilateral → Multifocal
    if re.search(r"\bbilateral\b|\bbi.?hemispheric\b", t):
        return "Multifocal"

    # 2a. Compound lobe patterns (fronto-temporal, temporo-parietal, etc.)
    if _COMPOUND_LOBES.search(t):
        return "Multifocal"

    # 2b. Two+ distinct single-word lobes (e.g., "temporal and frontal")
    if _count_distinct_lobes(t) >= 2:
        return "Multifocal"

    # 3. Hemispheric
    if re.search(r"\bhemispher", t):
        return "Hemispheric"

    # 4. Explicit multifocal
    if re.search(r"\bmultifocal\b|\bmulti.?focal\b|multiple foc", t):
        return "Multifocal"

    # 5. Temporal only (one lobe match, and it's temporal, and not extratemporal)
    has_temporal = bool(re.search(r"\btemporal\b", t)) and not re.search(r"\bextra.?temporal\b", t)
    if has_temporal and _count_distinct_lobes(t) == 1:
        return "Temporal"

    # 6. Single extratemporal lobe or sublobar region
    has_extratemporal_word = bool(re.search(r"\bextra.?temporal\b", t))
    has_single_extra_lobe = any(
        re.search(rf"\b{lobe}\b", t) for lobe in {"frontal", "parietal", "occipital", "insular"}
    )
    has_sublobar = bool(_SUBLOBAR_EXTRA.search(t))
    if has_extratemporal_word or has_single_extra_lobe or has_sublobar:
        return "Extratemporal"

    # 7. Bare unknown-ish terms
    if re.search(r"\bfocal\b|\bpartial\b|\bleft\b|\bright\b|\bcryptogenic\b|"
                 r"\buncertain\b|\bnot localis|\bnon.?localis|\bundetermined\b", t):
        return "Unknown"

    return None


def consolidate_aed_response(text: str | None) -> str | None:
    if not text:
        return None
    t = _norm(text)
    if not t or t in {"none", "null", "n/a", "na"}:
        return None
    if re.search(r"drug.?resistan|refractor|intractable|pharmacoresistan|medication.?resistan|failed.*\d.*(asm|aed|drug)|dre\b", t):
        return "drug-resistant"
    if re.search(r"drug.?responsi|seizure.?free.*(\d+\s*month|year)|controlled on|well.?controlled|good.?response|responded well|responsive\b", t):
        return "drug-responsive"
    if re.search(r"on (asm|aed|antiseizure|treatment)|treated with|taking (asm|aed)|under treatment|unspecified", t):
        return "unspecified"
    return None


def consolidate_surgery_outcome(text: str | None) -> str | None:
    if not text:
        return None
    t = _norm(text)
    if not t or t in {"none", "null", "n/a", "na"}:
        return None
    if "no surgery" in t or re.search(r"\bdid not (undergo|have) surgery\b|\bno operation\b|\bnot operated\b", t):
        return "Not applicable"
    if re.search(r"engel\s*(i\b|1\b|class i\b)|seizure.?free|free of disabling|no seizure", t):
        return "Seizure-free"
    if re.search(r"engel\s*(ii\b|2\b|iii\b|3\b|class (ii|iii)\b)|improv|reduced seizure|worthwhile|rare disabl", t):
        return "Improved"
    if re.search(r"engel\s*(iv\b|4\b|class iv\b)|no improvement|no benefit|unchanged|unsuccessful|failed surgery", t):
        return "No improvement"
    return None


def consolidate_status_epilepticus(text: str | None) -> str | None:
    """ILAE 2015 SE classification bucketing.

    Precedence: refractory → non-convulsive → convulsive → none → unknown.
    """
    if not text:
        return None
    t = _norm(text)
    if not t or t in {"none", "null", "n/a", "na"}:
        return None

    # 1. Refractory / super-refractory SE (includes RSE)
    if re.search(r"refractory status epilepticus|refractory se\b|\brse\b|super.?refractory|"
                 r"se (refractory|resistant)|persistent status epilepticus", t):
        return "Refractory SE"

    # 2. Non-convulsive SE (NCSE, absence SE, complex-partial SE, focal impaired-awareness SE)
    if re.search(r"non.?convulsive (status|se)|\bncse\b|absence status|complex partial status|"
                 r"electrographic status|subclinical status|focal impaired.?awareness status", t):
        return "Non-convulsive SE"

    # 3. Convulsive SE (GCSE, tonic-clonic SE, grand mal SE)
    if re.search(r"convulsive status|generali[sz]ed tonic.?clonic status|\bgcse\b|"
                 r"tonic.?clonic status|grand mal status|generali[sz]ed convulsive status", t):
        return "Convulsive SE"

    # 4. Explicitly "None"
    if re.search(r"no (history of |episode of |prior |reported )?status epilepticus|"
                 r"without (status epilepticus|se\b)|\bno se\b|absence of status epilepticus|"
                 r"denied (status epilepticus|se\b)", t):
        return "None"

    # 5. SE mentioned but not classified
    if re.search(r"\bstatus epilepticus\b|\bse\b", t):
        return "Unknown"

    return None


CONSOLIDATORS = {
    "epilepsy_type": consolidate_epilepsy_type,
    "seizure_type": consolidate_seizure_type,
    "ez_localization": consolidate_ez_localization,
    "aed_response": consolidate_aed_response,
    "surgery_outcome": consolidate_surgery_outcome,
    "status_epilepticus": consolidate_status_epilepticus,
}


def consolidate(task: str, raw_text: str | None) -> str | None:
    fn = CONSOLIDATORS.get(task)
    if not fn:
        raise KeyError(f"unknown task: {task}")
    return fn(raw_text)
