# Output JSON schema and field invariants

Mirrors Appendix C of the paper. Every patient profile produced by
`pipeline/extraction/extract_patients.py` conforms to this schema.

## Patient profile

```json
{
  "is_valid": "bool",
  "has_confirmed_epilepsy_diagnosis": "bool",
  "validation_notes": "<one-sentence explanation when invalid>",
  "patient_ref": "string",                     // canonical Patient_N
  "linked_figures": ["Figure 1", "Figure 2A"],
  "supplementary": {
    "demographics": {
      "age": "int | null",
      "sex": "M | F | null",
      "notes": "string | null"
    },
    "semiology":  "string | null",             // narrative text
    "mri":        "string | null",             // MRI report text
    "eeg":        "string | null"              // EEG report text
  },
  "ground_truths": {
    "epilepsy_type":      {"truth": "...", "evidence_quote": "...", "redacted": "..."},
    "seizure_type":       {"truth": "...", "evidence_quote": "...", "redacted": "..."},
    "ez_localization":    {"truth": "...", "evidence_quote": "...", "redacted": "..."},
    "aed_response":       {"truth": "...", "evidence_quote": "...", "redacted": "..."},
    "surgery_outcome":    {"truth": "...", "evidence_quote": "...", "redacted": "..."},
    "status_epilepticus": {"truth": "...", "evidence_quote": "...", "redacted": "..."}
  }
}
```

## Field invariants enforced post-extraction

- `is_valid = true` requires `has_confirmed_epilepsy_diagnosis = true`.
- Every non-null `truth` requires a non-null `evidence_quote` of ≥15
  characters that is verbatim-anchored in the source narrative.
- Every non-null `redacted` must contain the literal token `[redacted]`.
- `ground_truths.<task>.truth` must be a member of the controlled
  vocabulary in `docs/controlled_vocabulary.md`; otherwise the value is
  set to null.
- `linked_figures` references are normalised to the form `Figure <N><letter>`
  (e.g. `Figure 1A`) and resolved against the subfigure manifest produced
  by the subfigure pipeline.

## Tier assignment

Each profile is assigned to exactly one quality tier based on coverage of
input modalities and diversity of available task labels. The criteria
below are applied in order and the first match wins.

| Tier | Criteria |
|---|---|
| **Gold**   | ≥4 of 6 task labels populated, full demographics, full clinical text (semiology + MRI report + EEG report), and ≥1 linked MRI or EEG subfigure |
| **Silver** | ≥3 of 6 task labels populated, any narrative text, and either ≥1 linked subfigure or full demographics |
| **Bronze** | ≥2 of 6 task labels populated and any narrative text |
| (dropped)  | otherwise — record is filtered out of the released benchmark |

The released dataset on the HuggingFace mirror contains
**834 Gold + 13,006 Silver + 11,897 Bronze = 25,737 patients** in total.
