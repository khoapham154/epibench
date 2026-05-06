"""LLM-based paper-type gate.

Replaces the regex classifier in pipeline_v2/prescreen_papers.py:186-209.
One Qwen call per paper on title + abstract + first 800 chars of methods.

Returns one of:
  case_report           — single or few patients, narrative-style → ACCEPT
  case_series           — small N, individual patient descriptions → ACCEPT
  cohort_with_individuals — cohort study but contains identifiable individuals → ACCEPT (strict mode)
  cohort_aggregated     — cohort with only aggregate stats → REJECT
  review                — narrative/systematic review, no original patient data → REJECT
  other                 — basic science / methods paper / not clinical → REJECT
"""
from __future__ import annotations
from llm_client import chat_completion_json

VALID_TYPES = {
    "case_report", "case_series", "cohort_with_individuals",
    "cohort_aggregated", "review", "other",
}
# Hard-reject only reviews. Everything else goes through STAGE_A which has its
# own per-section aggregate filter + evidence-quote grounding. False accepts
# are cheap (return empty patients); false rejects lose real patient data.
ACCEPTED_TYPES = {"case_report", "case_series", "cohort_with_individuals", "cohort_aggregated", "other"}

SYSTEM = """You are a careful editorial classifier of biomedical papers.

Decide whether a paper contains EXTRACTABLE INDIVIDUAL-PATIENT DATA.

Output JSON ONLY with this schema:
{
  "paper_type": "case_report" | "case_series" | "cohort_with_individuals" | "cohort_aggregated" | "review" | "other",
  "confidence": float in [0,1],
  "reason": "one short sentence quoting evidence from the abstract/methods"
}

Definitions:
- "case_report": narrative description of 1-3 patients. Title or abstract often says "case report", "we report a", "case of".
- "case_series": 4-30 patients each described individually. Methods/results enumerate patients (Patient 1, Patient 2, ...).
- "cohort_with_individuals": cohort/retrospective/prospective study (>=10 patients) that ALSO contains an individual case description embedded in the discussion or supplementary material.
- "cohort_aggregated": cohort/retrospective/prospective study reporting ONLY aggregate statistics (means, percentages, p-values). NO individual patients enumerated.
- "review": systematic review, narrative review, meta-analysis, scoping review. Reviews other people's data; no original patients.
- "other": basic science, animal study, methods paper, editorial, letter without patient data.

Strict rules:
- If the paper says "we conducted a retrospective review of N patients" and the abstract reports only mean/median/percent statistics, it is "cohort_aggregated".
- If the paper says "review" or "meta-analysis" anywhere in title/abstract -> "review".
- If you are uncertain between case_series and cohort_with_individuals, choose case_series.
- If you cannot find clinical patient data at all, choose "other".
"""


def classify(title: str, abstract: str, methods_preview: str = "") -> dict:
    """Classify a paper. Returns {paper_type, confidence, reason, accepted: bool}."""
    user = f"""TITLE:
{title.strip()[:500]}

ABSTRACT:
{abstract.strip()[:3000]}

METHODS / FIRST PARAGRAPH OF BODY:
{methods_preview.strip()[:1500]}

Classify this paper. Output JSON only."""

    out = chat_completion_json(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=400,
    )
    pt = (out.get("paper_type") or "other").strip().lower()
    if pt not in VALID_TYPES:
        pt = "other"
    out["paper_type"] = pt
    out["accepted"] = pt in ACCEPTED_TYPES
    out.setdefault("confidence", 0.0)
    out.setdefault("reason", "")
    return out
