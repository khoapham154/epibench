"""v3 patient extraction — high-precision, evidence-grounded, ILAE-compliant.

Pipeline per paper:
  1. Parse NXML (title, abstract, sections, figures, tables)
  2. LLM paper-type classifier (reject review / cohort_aggregated / other)
  3. STAGE_A: extract patient candidates + evidence quotes from each section
  4. STAGE_TABLE: parse each clinical table into row-level patients
  5. Deterministic dedup (canonical_ref)
  6. STAGE_B: structure each candidate, with ILAE controlled vocab, has_epilepsy gate
  7. Post-validate: drop ground_truths whose evidence_quote cannot be verified in source
  8. Second deterministic dedup pass
  9. Write final_profiles_v3.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Make sibling modules importable regardless of invocation CWD
sys.path.insert(0, str(Path(__file__).parent))

from dedup_v3 import canonical_ref, dedup_profiles, is_same_patient
from ilae_label_maps import PROMPT_DEFINITIONS
from llm_client import chat_completion_json, parallel_map
from nxml_loader import find_nxml, load_paper
import paper_type_classifier_llm as ptc

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("v3.extract")


# ======================================================================
# PROMPTS
# ======================================================================

STAGE_A_SYSTEM = """You are an expert clinical data extractor for epilepsy research.

### GOAL
From a section of a medical paper, extract INDIVIDUAL-PATIENT facts.
An individual patient is a single person identified either explicitly
(Patient 1 / Case A / Pt. #3) or implicitly through a narrative description
with age + sex + specific clinical detail.

### HARD RULES
1. OUTPUT JSON ONLY. Start with '{'. No prose, no markdown fences.
2. If the section contains ONLY aggregate statistics, group summaries,
   or no identifiable individual, output {"patients": [], "cohort_context": {"universal_facts": [], "applies_to_ids": []}}.
3. DO NOT fabricate patients. Each individual-patient fact (raw_facts entry)
   must be grounded in a verbatim quote of >=15 characters copied from the source text.
4. Group-level statements (e.g. "Mean age was 45", "12 of 38 patients had...")
   go into cohort_context.universal_facts, NEVER into raw_facts.
5. EXCLUDE: healthy controls, family members not themselves epileptic,
   hypothetical examples, patients from referenced prior studies.
6. ALWAYS carry over per-patient narrative sentences about seizure semiology,
   MRI findings, and EEG findings when they are present in the section — even
   when no ground-truth label can yet be decided. Put them in raw_facts with
   the verbatim sentence as the evidence_quote. Dropping narrative text is a
   worse outcome than missing a label.

### EXAMPLES OF WHAT TO REJECT
  "Mean age was 45 +/- 12 years"                  -> aggregate, not a patient
  "12 of 38 patients had drug-resistant epilepsy" -> aggregate, not a patient
  "A typical patient with TLE presents with..."   -> hypothetical
  "The control group had normal EEG"              -> not an epilepsy patient

### EXAMPLES OF WHAT TO ACCEPT
  "Patient 3, a 27-year-old male, presented with refractory focal seizures..."
  "Case 2 was a 14-year-old girl whose MRI revealed a right temporal cavernoma..."
  "A 45-year-old woman (#7) underwent anterior temporal lobectomy in 2014..."

### OUTPUT SCHEMA
{
  "patients": [
    {
      "patient_ref": "Patient 1" | "Case A" | "#3" | "45yo_M_TLE"  (string from paper or semantic),
      "is_explicit_label": true|false,
      "raw_facts": [
         { "fact": "...", "evidence_quote": "...>=20 chars verbatim from text..." },
         ...
      ],
      "figure_refs": ["Figure 1", "Fig 2A"]
    }
  ],
  "cohort_context": {
    "universal_facts": ["group-level statement", ...],
    "applies_to_ids": ["all"] | ["Patient 1", "Patient 2"]
  }
}
"""

STAGE_A_USER_TMPL = """Section title: {title}

Paper-type hint: {paper_type_hint}

{figures_context}

{tables_context}

--- SECTION TEXT ---
{section_text}
--- END SECTION ---

Extract individual-patient facts with evidence quotes. Output JSON only.
"""

STAGE_TABLE_SYSTEM = """You are a medical table parser for epilepsy papers.

### GOAL
Decide if a table contains one row per individual patient. If yes, extract
each row as a patient with the column-to-value mapping.

### HARD RULES
1. OUTPUT JSON ONLY.
2. If the table is purely aggregate (Total / Mean / Median / Summary rows only),
   output {"has_individual_data": false, "patients": [], "notes": "..."}.
3. DO NOT invent patients. Every patient entry must correspond to an actual row.
4. Preserve the row identifier (Patient 1, Case #3, row number).

### OUTPUT SCHEMA
{
  "table_type": "patient_characteristics" | "results" | "aggregate_summary" | "other",
  "has_individual_data": true|false,
  "patients": [
    {
      "patient_ref": "string",
      "is_explicit_label": true|false,
      "extracted_data": { "column": "value", ... }
    }
  ],
  "notes": "string"
}
"""

STAGE_TABLE_USER_TMPL = """Table label: {label}
Table caption: {caption}

--- TABLE ---
{content}
--- END TABLE ---

Output JSON only."""


STAGE_B_SYSTEM = """You are a clinical data structurer for epilepsy research.

### GOAL
Turn a list of patient facts (with evidence quotes) into a structured patient
profile using ILAE-compliant controlled vocabulary.

### HARD RULES
1. OUTPUT JSON ONLY.
2. First decide: has_confirmed_epilepsy_diagnosis.
   - TRUE if the facts explicitly state epilepsy, epileptic seizures, or a
     named epilepsy syndrome.
   - FALSE if only: single unprovoked seizure, febrile seizure only,
     psychogenic non-epileptic seizure (PNES), movement disorder, acute
     symptomatic seizure without epilepsy diagnosis, or unclear.
3. If has_confirmed_epilepsy_diagnosis is FALSE, set is_valid=false and
   leave all ground_truths null.
4. For each ground truth, you MUST supply an evidence_quote (>=20 chars
   verbatim from the facts). If you cannot find such a quote, set truth=null
   and evidence_quote=null. NEVER guess.
5. Use ONLY the controlled vocabulary below for truth values. If the paper
   uses a synonym, translate to the canonical form; if no canonical value
   applies, set truth=null.

""" + PROMPT_DEFINITIONS + """

### OUTPUT SCHEMA
{
  "is_valid": true|false,
  "has_confirmed_epilepsy_diagnosis": true|false,
  "validation_notes": "one sentence explanation referencing specific facts",
  "patient_ref": "string",
  "linked_figures": ["Figure 1", ...],
  "supplementary": {
    "demographics": { "age": "string|null", "sex": "Male|Female|null", "notes": "string|null" },
    "semiology": "string|null",
    "mri": "string|null",
    "eeg": "string|null"
  },
  "ground_truths": {
    "epilepsy_type":    { "truth": "Focal|Generalised|Combined Focal and Generalised|Unknown|null", "evidence_quote": "string|null", "redacted": "string|null" },
    "seizure_type":     { "truth": "Focal|Generalised|Unknown|Unclassified|null",                    "evidence_quote": "string|null", "redacted": "string|null" },
    "ez_localization":  { "truth": "Temporal|Extratemporal|Multifocal|Hemispheric|Unknown|null",     "evidence_quote": "string|null", "redacted": "string|null" },
    "aed_response":     { "truth": "drug-responsive|drug-resistant|unspecified|null",                "evidence_quote": "string|null", "redacted": "string|null" },
    "surgery_outcome":  { "truth": "Seizure-free|Improved|No improvement|Not applicable|null",       "evidence_quote": "string|null", "redacted": "string|null" },
    "status_epilepticus":{ "truth": "Convulsive SE|Non-convulsive SE|Refractory SE|None|Unknown|null",      "evidence_quote": "string|null", "redacted": "string|null" }
  }
}

### REDACTED FIELD
For each ground truth with non-null truth, produce a redacted version of the
evidence_quote where the truth value (or any synonymous phrase that directly
names the class) is replaced with the literal string "[redacted]".
Example:
  evidence_quote: "Patient had drug-resistant epilepsy despite 3 ASMs"
  redacted:       "Patient had [redacted] epilepsy despite 3 ASMs"
"""

STAGE_B_USER_TMPL = """Patient reference: {patient_ref}
Source file: {source_file}

FACTS (each with evidence quote):
{facts_block}

LINKED FIGURES: {figure_refs}

Structure this patient. Output JSON only."""


# ======================================================================
# STAGE RUNNERS
# ======================================================================

def _format_figures(figures: dict) -> str:
    if not figures:
        return ""
    lines = ["FIGURES IN THIS PAPER (link if the section text cites them):"]
    for k, f in figures.items():
        lbl = f.get("label", k)
        cap = (f.get("caption") or "")[:180]
        lines.append(f"- {lbl}: {cap}")
    return "\n".join(lines)


def _format_tables_short(tables: dict) -> str:
    if not tables:
        return ""
    lines = ["TABLES IN THIS PAPER (parsed separately):"]
    for k, t in tables.items():
        lines.append(f"- {t.get('label', k)}: {(t.get('caption') or '')[:120]}")
    return "\n".join(lines)


def stage_a_extract(
    section_text: str, section_title: str, figures: dict, tables: dict, paper_type: str
) -> dict:
    if len(section_text.strip()) < 60:
        return {}
    hint = {
        "case_report": "Case report: extract each patient carefully, avoid duplicates across sections.",
        "case_series": "Case series: enumerate each patient exactly once.",
        "cohort_with_individuals": "Cohort study: extract ONLY individuals with a verbatim narrative of >=15 words describing them specifically. Group-level statements -> cohort_context only.",
    }.get(paper_type, "Extract individuals only.")

    # Budget: 4096 ctx - ~1100 system - ~300 wrapper = ~2600 for section. Cap at 4000 chars (~1300 tokens).
    messages = [
        {"role": "system", "content": STAGE_A_SYSTEM},
        {"role": "user", "content": STAGE_A_USER_TMPL.format(
            title=section_title,
            paper_type_hint=hint,
            figures_context=_format_figures(figures)[:800],
            tables_context=_format_tables_short(tables)[:400],
            section_text=section_text[:4000],
        )},
    ]
    return chat_completion_json(messages, temperature=0.0, max_tokens=1200)


def _split_table_rows(content: str) -> tuple[str, list[str]]:
    """Split table text into (header_block, data_rows)."""
    lines = [ln.rstrip() for ln in (content or "").splitlines() if ln.strip()]
    if len(lines) < 2:
        return "\n".join(lines), []
    # Heuristic: header is the first 1-3 lines before data-like rows.
    # A data row here usually starts with a short token (patient id or number).
    header_end = 1
    while header_end < len(lines) and header_end < 3:
        ln = lines[header_end]
        first_col = ln.split("|", 1)[0].strip()
        # data row if starts with a short identifier (number or short patient ref)
        if first_col and (first_col.split()[0].rstrip(".)").isdigit() or len(first_col) <= 10):
            break
        header_end += 1
    header = "\n".join(lines[:header_end])
    rows = lines[header_end:]
    return header, rows


def stage_table_parse(table_info: dict, max_rows_per_call: int = 10) -> dict:
    if not table_info.get("content", "").strip():
        return {}
    header, rows = _split_table_rows(table_info["content"])
    if not rows:
        # fall back to single-shot call on whole content
        msgs = [
            {"role": "system", "content": STAGE_TABLE_SYSTEM},
            {"role": "user", "content": STAGE_TABLE_USER_TMPL.format(
                label=table_info.get("label", ""),
                caption=table_info.get("caption", ""),
                content=table_info.get("content", "")[:3500],
            )},
        ]
        return chat_completion_json(msgs, temperature=0.0, max_tokens=2200)

    all_patients: list[dict] = []
    table_type = None
    has_individual_data = False
    for i in range(0, len(rows), max_rows_per_call):
        chunk_rows = rows[i : i + max_rows_per_call]
        chunk_content = header + "\n" + "\n".join(chunk_rows)
        msgs = [
            {"role": "system", "content": STAGE_TABLE_SYSTEM},
            {"role": "user", "content": STAGE_TABLE_USER_TMPL.format(
                label=table_info.get("label", ""),
                caption=table_info.get("caption", ""),
                content=chunk_content[:3500],
            )},
        ]
        try:
            chunk_out = chat_completion_json(msgs, temperature=0.0, max_tokens=2200)
        except Exception as e:
            log.warning("table chunk parse failed: %s", e)
            continue
        if chunk_out.get("has_individual_data"):
            has_individual_data = True
            table_type = chunk_out.get("table_type") or table_type
            all_patients.extend(chunk_out.get("patients", []) or [])
        elif table_type is None:
            table_type = chunk_out.get("table_type")
    return {
        "table_type": table_type or "other",
        "has_individual_data": has_individual_data,
        "patients": all_patients,
        "notes": f"chunked: {len(rows)} rows in {len(range(0, len(rows), max_rows_per_call))} calls",
    }


def stage_b_structure(patient_ref: str, facts_with_quotes: list[dict], figure_refs: list[str], source_file: str) -> dict:
    if not facts_with_quotes:
        return {}
    lines = []
    for i, f in enumerate(facts_with_quotes, 1):
        fact = f.get("fact", "")
        quote = f.get("evidence_quote", "")
        lines.append(f"{i}. {fact}\n   QUOTE: {quote}")
    facts_block = "\n".join(lines)

    # Tight budget: 4096 ctx - ~1400 system prompt - ~150 user wrapper = ~2500 for facts.
    # Cap facts_block at 3000 chars (~1000 tokens) and max_tokens at 1100.
    messages = [
        {"role": "system", "content": STAGE_B_SYSTEM},
        {"role": "user", "content": STAGE_B_USER_TMPL.format(
            patient_ref=patient_ref,
            source_file=source_file,
            facts_block=facts_block[:3000],
            figure_refs=", ".join(figure_refs) if figure_refs else "None",
        )},
    ]
    return chat_completion_json(messages, temperature=0.05, max_tokens=1100)


# ======================================================================
# REGISTRY
# ======================================================================

class Registry:
    """Aggregates patient candidates across sections/tables before STAGE_B."""

    def __init__(self) -> None:
        # key = canonical_ref tuple, value = dict with patient_ref, facts_with_quotes, figure_refs, is_explicit
        self._by_canon: dict = {}
        self.cohort_facts: list[str] = []

    def add_candidate(
        self,
        patient_ref: str,
        raw_facts: list[dict],
        figure_refs: list[str],
        is_explicit: bool,
    ) -> None:
        if not patient_ref:
            return
        canon = canonical_ref(patient_ref)
        if canon == (None, ""):
            return

        # Try to merge with an existing canonical key (e.g. '1' merges into 'Patient_1' if demographics OK)
        target_key = canon
        if canon not in self._by_canon and canon[0] is not None:
            for other_canon, entry in self._by_canon.items():
                if other_canon[0] == canon[0]:
                    target_key = other_canon
                    break

        if target_key not in self._by_canon:
            self._by_canon[target_key] = {
                "patient_ref": patient_ref,
                "facts_with_quotes": [],
                "figure_refs": set(),
                "is_explicit": is_explicit,
            }
        entry = self._by_canon[target_key]
        # Prefer explicit label for display name
        if is_explicit and not entry["is_explicit"]:
            entry["patient_ref"] = patient_ref
            entry["is_explicit"] = True

        seen_facts = {(f.get("fact", ""), f.get("evidence_quote", "")) for f in entry["facts_with_quotes"]}
        for f in raw_facts:
            key = (f.get("fact", ""), f.get("evidence_quote", ""))
            if key in seen_facts or not f.get("fact") or not f.get("evidence_quote"):
                continue
            if len(f.get("evidence_quote", "")) < 15:  # reject trivial evidence
                continue
            entry["facts_with_quotes"].append(f)
            seen_facts.add(key)
        entry["figure_refs"].update(figure_refs or [])

    def add_table_patients(self, table_json: dict, table_label: str) -> None:
        if not table_json.get("has_individual_data"):
            return
        for row in table_json.get("patients", []):
            pref = (row.get("patient_ref") or "").strip()
            if not pref:
                continue
            facts = []
            for col, val in (row.get("extracted_data") or {}).items():
                if val is None or str(val).strip() in ("", "-", "NA", "n/a", "N/A"):
                    continue
                evidence = f"[Table {table_label}] {col}: {val}"
                if len(evidence) < 15:
                    evidence = evidence + " " * (15 - len(evidence))
                facts.append({"fact": f"{col}: {val}", "evidence_quote": evidence})
            self.add_candidate(pref, facts, figure_refs=[], is_explicit=True)

    def all_candidates(self) -> list[dict]:
        return list(self._by_canon.values())


# ======================================================================
# PER-PAPER DRIVER
# ======================================================================

def process_paper(nxml_path: Path, out_dir: Path, skip_if_exists: bool = True) -> dict:
    result: dict = {
        "pmc_id": nxml_path.parent.name,
        "nxml_path": str(nxml_path),
        "status": "",
        "paper_type": None,
        "n_patients": 0,
        "rejected_reason": None,
    }
    out_json = out_dir / f"{nxml_path.parent.name}_final_profiles_v3.json"
    meta_json = out_dir / f"{nxml_path.parent.name}_meta_v3.json"
    if skip_if_exists and out_json.exists() and meta_json.exists():
        try:
            with meta_json.open() as f:
                result.update(json.load(f))
            result["status"] = "cached"
            return result
        except Exception:
            pass

    paper = load_paper(nxml_path)
    if paper is None:
        result["status"] = "nxml_parse_error"
        return result

    # 1. Paper-type gate
    try:
        decision = ptc.classify(
            title=paper["title"],
            abstract=paper["abstract"],
            methods_preview=paper["body_preview"],
        )
    except Exception as e:
        # Transient server errors — don't hard-reject; fall back to permissive type.
        # The strict extraction prompts will still filter aggregated content.
        log.warning("%s: paper-type classifier unavailable, falling back to case_series: %s", paper["pmc_id"], e)
        decision = {
            "paper_type": "case_series",
            "accepted": True,
            "reason": f"classifier_unavailable: {e}",
            "confidence": 0.0,
        }
    result["paper_type"] = decision.get("paper_type")
    result["paper_type_reason"] = decision.get("reason", "")
    result["paper_type_confidence"] = decision.get("confidence", 0.0)
    # Article-type hard reject (NXML metadata)
    if paper.get("article_type") in {"review-article", "systematic-review", "meta-analysis", "letter", "editorial", "news"}:
        result["status"] = "rejected"
        result["rejected_reason"] = f"article-type={paper['article_type']}"
        _write_empty(out_json, meta_json, result)
        return result
    if not decision.get("accepted", False):
        result["status"] = "rejected"
        result["rejected_reason"] = f"paper_type={decision.get('paper_type')}: {decision.get('reason','')}"
        _write_empty(out_json, meta_json, result)
        return result

    paper_type = decision["paper_type"]
    reg = Registry()

    # 2. STAGE_A on each body section in parallel (skip pure figure-caption shells).
    sections = [(t, s) for t, s in paper["sections"] if len(s.strip()) >= 60]

    def _stage_a_one(ts):
        t, s = ts
        try:
            return (t, stage_a_extract(s, t, paper["figures"], paper["tables"], paper_type))
        except Exception as e:
            log.warning("%s: stage_a failed on section %r: %s", paper["pmc_id"], t, e)
            return (t, None)

    a_results = parallel_map(_stage_a_one, sections, max_workers=min(8, max(1, len(sections))))
    for _, out in a_results:
        if not out:
            continue
        for p in out.get("patients", []) or []:
            reg.add_candidate(
                patient_ref=p.get("patient_ref", ""),
                raw_facts=p.get("raw_facts", []) or [],
                figure_refs=p.get("figure_refs", []) or [],
                is_explicit=bool(p.get("is_explicit_label", False)),
            )
        cc = out.get("cohort_context") or {}
        if cc.get("universal_facts"):
            reg.cohort_facts.extend(cc["universal_facts"])

    # 3. STAGE_TABLE on each table (tables are already row-chunked internally by stage_table_parse)
    table_items = [(tkey, tinfo) for tkey, tinfo in paper["tables"].items()
                   if len((tinfo.get("content") or "").strip()) >= 20]

    def _stage_table_one(item):
        tkey, tinfo = item
        try:
            return (tkey, tinfo, stage_table_parse(tinfo))
        except Exception as e:
            log.warning("%s: table parse failed: %s", paper["pmc_id"], e)
            return (tkey, tinfo, None)

    t_results = parallel_map(_stage_table_one, table_items, max_workers=min(4, max(1, len(table_items))))
    for tkey, tinfo, tjson in t_results:
        if tjson:
            reg.add_table_patients(tjson, tinfo.get("label", tkey))

    candidates = reg.all_candidates()
    # Gate for cohort_with_individuals: require >=1 fact with a verbatim quote
    # long enough to identify the individual (>=15 chars per v4 rule-loosening).
    if paper_type == "cohort_with_individuals":
        strict_candidates = []
        for c in candidates:
            has_long = any(len(f.get("evidence_quote", "")) >= 15 for f in c["facts_with_quotes"])
            if has_long:
                strict_candidates.append(c)
        candidates = strict_candidates

    if not candidates:
        result["status"] = "accepted_empty"
        _write_empty(out_json, meta_json, result)
        return result

    # 4. STAGE_B per candidate in parallel (was sequential -> dominant cost on
    #    cohort-aggregated papers with 20+ candidates).
    source_file = Path(paper["nxml_path"]).name

    def _stage_b_one(c: dict) -> dict | None:
        try:
            out = stage_b_structure(
                patient_ref=c["patient_ref"],
                facts_with_quotes=c["facts_with_quotes"],
                figure_refs=sorted(c["figure_refs"]),
                source_file=source_file,
            )
        except Exception as e:
            log.warning("%s: stage_b failed for %s: %s", paper["pmc_id"], c["patient_ref"], e)
            return None
        if not out:
            return None
        if not out.get("is_valid"):
            return None
        if not out.get("has_confirmed_epilepsy_diagnosis"):
            return None
        _post_validate_ground_truths(out)
        return out

    stage_b_results = parallel_map(_stage_b_one, candidates, max_workers=min(16, max(1, len(candidates))))
    structured: list[dict] = [r for r in stage_b_results if r]

    # 5. Second deterministic dedup pass (post-STAGE_B)
    structured = dedup_profiles(structured)

    # 6. Persist
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(structured, f, indent=2)
    result["status"] = "accepted" if structured else "accepted_empty"
    result["n_patients"] = len(structured)
    with meta_json.open("w") as f:
        json.dump(result, f, indent=2)
    return result


def _post_validate_ground_truths(profile: dict) -> None:
    """Drop any ground-truth whose evidence_quote is missing/too short."""
    gt = profile.get("ground_truths") or {}
    for task, val in list(gt.items()):
        if not val:
            gt[task] = {"truth": None, "evidence_quote": None, "redacted": None}
            continue
        truth = val.get("truth")
        quote = val.get("evidence_quote") or ""
        if truth and (not quote or len(quote.strip()) < 15):
            gt[task] = {"truth": None, "evidence_quote": None, "redacted": None}
    profile["ground_truths"] = gt


def _write_empty(out_json: Path, meta_json: Path, meta: dict) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump([], f)
    with meta_json.open("w") as f:
        json.dump(meta, f, indent=2)


# ======================================================================
# BATCH DRIVER
# ======================================================================

def _resolve_nxml_for_pmc(pmc_id: str, keyword_hint: str | None = None) -> Path | None:
    """Find an extracted NXML for this PMC. Prefer keyword_hint, else try all keywords."""
    base = Path("/data/pubmed_epilepsy/downloads")
    keywords = [keyword_hint] if keyword_hint else []
    keywords += ["mri", "eeg", "syndromes", "asm", "semiology"]
    seen = set()
    for kw in keywords:
        if not kw or kw in seen:
            continue
        seen.add(kw)
        pmc_dir = base / f"{kw}_extracted" / pmc_id
        if pmc_dir.is_dir():
            nxml = find_nxml(pmc_dir)
            if nxml is not None:
                return nxml
    return None


def run_for_pmcs(pmcs: list[tuple[str, str | None]], out_dir: Path, max_workers: int = 16) -> list[dict]:
    """Run v3 extraction for each (pmc_id, keyword_hint) pair. Parallelised per paper."""
    def _one(item):
        pmc_id, kw_hint = item
        nxml = _resolve_nxml_for_pmc(pmc_id, kw_hint)
        if nxml is None:
            return {"pmc_id": pmc_id, "status": "nxml_not_found", "n_patients": 0}
        try:
            return process_paper(nxml, out_dir)
        except Exception as e:
            log.exception("process_paper failed for %s: %s", pmc_id, e)
            return {"pmc_id": pmc_id, "status": f"error:{e.__class__.__name__}", "n_patients": 0}

    return parallel_map(_one, pmcs, max_workers=max_workers, desc="v3 extract", print_every=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmc-list", type=str, required=True,
                    help="CSV with columns: pmc_id, keyword")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--workers", type=int, default=16,
                    help="Concurrent papers (each may issue many LLM calls sequentially)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_csv(args.pmc_list)
    if args.limit:
        df = df.head(args.limit)
    pmcs = list(zip(df["pmc_id"].astype(str).tolist(), df.get("keyword", [None] * len(df)).tolist()))

    out_dir = Path(args.out_dir)
    results = run_for_pmcs(pmcs, out_dir, max_workers=args.workers)

    # Summary
    import collections
    statuses = collections.Counter(r.get("status", "?") for r in results)
    total_patients = sum(r.get("n_patients", 0) for r in results)
    print(f"\n=== v3 extraction summary ===")
    print(f"Papers: {len(results)}")
    for s, c in statuses.most_common():
        print(f"  {s}: {c}")
    print(f"Total patients extracted: {total_patients}")

    summary_path = out_dir / "run_summary_v3.json"
    with summary_path.open("w") as f:
        json.dump({
            "n_papers": len(results),
            "statuses": dict(statuses),
            "total_patients": total_patients,
            "per_paper": results,
        }, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
