"""Assign Tier A/B/C/D per patient based on label + input completeness.

Tier rules (top -> bottom, first match wins):
  A: >=4 non-null labels AND (semiology_text OR (mri_report_text AND eeg_report_text))
     AND age non-null AND sex non-null AND >=1 linked figure.
  B: >=3 non-null labels AND any narrative text AND (>=1 figure OR (age + sex)).
  C: >=2 non-null labels AND any narrative text.
  D: everything else.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_COLS = [
    "epilepsy_type_label", "seizure_type_label", "ez_localization_label",
    "aed_response_label", "surgery_outcome_label", "status_epilepticus_label",
]
NARRATIVE_COLS = ["semiology_text", "mri_report_text", "eeg_report_text"]


def _is_null(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    if isinstance(v, str) and not v.strip():
        return True
    return False


def _present(v) -> bool:
    return not _is_null(v)


def assign_tier(row: pd.Series) -> str:
    n_labels = sum(1 for c in LABEL_COLS if _present(row.get(c)))
    has_semiology = _present(row.get("semiology_text"))
    has_mri = _present(row.get("mri_report_text"))
    has_eeg = _present(row.get("eeg_report_text"))
    has_any_narr = has_semiology or has_mri or has_eeg
    has_age = _present(row.get("age"))
    has_sex = _present(row.get("sex"))
    n_figs = int(row.get("num_linked_figures") or 0)

    # Tier A
    if (n_labels >= 4
        and (has_semiology or (has_mri and has_eeg))
        and has_age and has_sex
        and n_figs >= 1):
        return "A"
    # Tier B
    if (n_labels >= 3
        and has_any_narr
        and (n_figs >= 1 or (has_age and has_sex))):
        return "B"
    # Tier C
    if n_labels >= 2 and has_any_narr:
        return "C"
    return "D"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-xlsx", required=True)
    ap.add_argument("--out-xlsx", required=True)
    args = ap.parse_args()

    df = pd.read_excel(args.in_xlsx, sheet_name="Patients")
    print(f"Input rows: {len(df)}")

    df["tier"] = df.apply(assign_tier, axis=1)
    dist = df["tier"].value_counts().to_dict()
    print("Tier distribution:")
    for t in "ABCD":
        n = dist.get(t, 0)
        pct = 100 * n / max(1, len(df))
        print(f"  {t}: {n:>8d}  ({pct:5.1f}%)")

    out_path = Path(args.out_xlsx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Patients", index=False)
        pd.DataFrame([{"tier": t, "count": dist.get(t, 0)} for t in "ABCD"]).to_excel(
            w, sheet_name="tier_distribution", index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
