# LLM prompts used by the extraction pipeline

Mirrors Appendix B of the paper. All five prompts run on
**Mistral-Small-3.2-24B-Instruct-2506** served via vLLM 0.19.1 across
8 instances on 8× A100-80GB, with `temperature=0.0`, `max_length=16,384`,
and bfloat16 precision.

---

## 1. Paper / document-type classification

```
You are an expert clinical-data classifier. Classify this paper into exactly
one of: case_report, case_series, cohort_with_individuals, cohort_aggregated,
review, other.

Definitions:
- case_report: describes 1-3 patients in narrative detail
- case_series: describes 4-20 patients each individually
- cohort_with_individuals: a cohort study (>20 patients) with a separable
  detailed sub-cohort describing individuals
- cohort_aggregated: reports only group statistics, no individuals
- review: synthesises prior literature, no original patient data
- other: anything else, including non-epilepsy papers

Reject the paper if it is not about epilepsy / seizures / antiseizure
medication / EEG / MRI in the context of epilepsy.

INPUT:
- Title
- Abstract

Output JSON:
  {"paper_type": "case_report" | "case_series" | "cohort_with_individuals"
                | "cohort_aggregated" | "review" | "other",
   "confidence": float in [0, 1],
   "reason": "<one-sentence justification>",
   "accepted": true | false}

Accept paper_types: case_report, case_series, cohort_with_individuals,
cohort_aggregated, other. Hard-reject only "review".
```

---

## 2. Individual-patient identification

```
You are an expert clinical data extractor for epilepsy research.

GOAL: From a section of a medical paper, extract INDIVIDUAL-PATIENT facts.
An individual patient is a single person identified either explicitly
(Patient 1 / Case A / Pt. #3) or implicitly through a narrative description
with age + sex + specific clinical detail.

HARD RULES:
1. OUTPUT JSON ONLY.
2. If the section contains ONLY aggregate statistics or group summaries,
   output {"patients": [], "cohort_context": {...}}.
3. Each individual-patient fact must be grounded in a verbatim quote of
   >=15 characters copied from the source text.
4. Group-level statements go into cohort_context.universal_facts, NEVER
   raw_facts.
5. EXCLUDE: healthy controls, family members not themselves epileptic,
   hypothetical examples, patients from referenced prior studies.
6. ALWAYS carry over per-patient narrative sentences about seizure
   semiology, MRI findings, and EEG findings when they appear in the
   section, even if no ground-truth label can yet be decided.

REJECT EXAMPLES:
  "Mean age was 45 +/- 12 years"                          aggregate
  "12 of 38 patients had drug-resistant epilepsy"         aggregate
  "A typical patient with TLE presents with..."           hypothetical
  "Her father also had childhood epilepsy"                family/excluded

ACCEPT EXAMPLES:
  "Patient 3, a 27-year-old male, presented with refractory focal
   seizures..."
  "Case 2 was a 14-year-old girl whose MRI revealed a right temporal
   cavernoma..."

OUTPUT SCHEMA:
{
  "patients": [
    {
      "patient_ref": "Patient 3",
      "raw_facts": [
        {"field": "age", "value": "27", "evidence_quote": "..."},
        {"field": "sex", "value": "M", "evidence_quote": "..."},
        {"field": "semiology", "value": "...", "evidence_quote": "..."},
        ...
      ],
      "figure_refs_from_extraction": ["Figure 1A", "Figure 2"]
    }, ...
  ],
  "cohort_context": {
    "universal_facts": [{"fact": "...", "evidence_quote": "..."}, ...]
  }
}
```

---

## 3. Structured table extraction

```
You are a medical table parser for epilepsy papers.

GOAL: Decide if a table contains one row per individual patient. If yes,
extract each row as a patient with the column-to-value mapping.

HARD RULES:
1. OUTPUT JSON ONLY.
2. If the table is purely aggregate (Total / Mean / Median / Summary rows
   only), output {"has_individual_data": false, "patients": [],
   "notes": "..."}.
3. DO NOT invent patients. Every patient entry must correspond to an
   actual row.
4. Preserve the row identifier (Patient 1, Case #3, row number).

INPUT:
  Table label: {label}
  Table caption: {caption}
  --- TABLE ---
  {row-by-row content}
  --- END TABLE ---

OUTPUT SCHEMA:
{
  "table_type": "patient_characteristics" | "results" |
                "aggregate_summary" | "other",
  "has_individual_data": true | false,
  "patients": [
    {
      "patient_ref": "string",
      "is_explicit_label": true | false,
      "extracted_data": {"<column>": "<value>", ...}
    }, ...
  ],
  "notes": "string"
}
```

---

## 4. Patient linking within a paper

```
You are linking patient mentions across one paper.

Given multiple candidate patient mentions found in different sections,
paragraphs, and tables of a single paper, decide which mentions refer to
the same individual.

SIGNALS to use (in order of strength):
- Identifier equivalence: P1, Pt. 1, Patient 1, Case 1, #1 are typically
  equivalent.
- Age (treat +/-2 years as compatible).
- Sex (M/F first letter).
- Diagnosis or syndrome name.
- Surgical history or treatment timeline.

EXAMPLE INPUT MENTIONS:
  Section A: "Patient 1, 45F, with TLE..."
  Table 1 row 1: P1, age 45, sex F, surgery Engel I
  Section C: "Case 1 had recurrent seizures despite surgery..."

Output JSON:
  [{"canonical_id": "Patient_1",
    "evidence_refs": ["Section A", "Table 1 row 1", "Section C"]},
   ...]
```

---

## 5. Six-task ground-truth extraction

```
You are a clinical data structurer for epilepsy research.

GOAL: Turn a list of patient facts (with evidence quotes) into a
structured patient profile using ILAE-compliant controlled vocabulary.

HARD RULES:
1. OUTPUT JSON ONLY.
2. First decide: has_confirmed_epilepsy_diagnosis.
   TRUE  if the facts explicitly state epilepsy, epileptic seizures, or a
         named epilepsy syndrome.
   FALSE if only: single unprovoked seizure, febrile seizure only,
         psychogenic non-epileptic seizure (PNES), movement disorder,
         acute symptomatic seizure without epilepsy diagnosis, or unclear.
3. If has_confirmed_epilepsy_diagnosis is FALSE, set is_valid=false and
   leave all ground_truths null.
4. For each ground_truth: supply an evidence_quote (verbatim, >=15 chars
   copied from the facts). If you cannot find one, set truth=null and
   evidence_quote=null. Never guess.
5. Use ONLY the controlled vocabulary in docs/controlled_vocabulary.md.
   If the paper uses a synonym (e.g. "Generalized" vs "Generalised";
   "Extra-temporal" vs "Extratemporal"), translate to the canonical form;
   otherwise set truth=null.
6. The redacted field is the evidence_quote with the truth value (or any
   synonym) replaced by [redacted], so that the redacted text can be
   used as input to a downstream model without leaking the label.

OUTPUT SCHEMA: see docs/schema.md.
```
