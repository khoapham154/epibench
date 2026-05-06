"""Prepare train/test splits for v4 baselines.

Reads dataset/v4_current/01_combined_full_dataset/comprehensive_patients_v4_full.xlsx
and writes per-task CSVs:
    splits/{task}/train.csv      — Tier B (SILVER), 13,006 rows max
    splits/{task}/test_gold.csv  — Tier A (GOLD), 834 rows max
    splits/{task}/test_bronze.csv — Tier C (BRONZE), 11,897 rows max

Each CSV has columns:
  patient_id, pmc_id, age, sex, semiology_text, mri_report_text, eeg_report_text,
  mri_image_paths (json list), eeg_image_paths (json list),
  num_mri_images, num_eeg_images, {task}_label, {task}_label_id

Only patients with non-null label for the task are included in the per-task CSVs.

Outputs label_maps_v4.json with the integer ID mapping for each task.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("EPIBENCH_ROOT", "."))
V4 = ROOT / "dataset/v4_current/01_combined_full_dataset/comprehensive_patients_v4_full.xlsx"

TASKS = [
    "epilepsy_type",
    "seizure_type",
    "ez_localization",
    "aed_response",
    "surgery_outcome",
    "status_epilepticus",
]

# Label IDs follow ILAE class lists from pipeline/ilae_label_maps.py
LABEL_MAPS = {
    "epilepsy_type": {"Focal": 0, "Generalised": 1, "Combined Focal and Generalised": 2, "Unknown": 3},
    "seizure_type": {"Focal": 0, "Generalised": 1, "Unknown": 2, "Unclassified": 3},
    "ez_localization": {"Temporal": 0, "Extratemporal": 1, "Multifocal": 2, "Hemispheric": 3, "Unknown": 4},
    "aed_response": {"drug-responsive": 0, "drug-resistant": 1, "unspecified": 2},
    "surgery_outcome": {"Seizure-free": 0, "Improved": 1, "No improvement": 2, "Not applicable": 3},
    "status_epilepticus": {"Convulsive SE": 0, "Non-convulsive SE": 1, "Refractory SE": 2, "None": 3, "Unknown": 4},
}

KEEP_COLS = [
    "pmc_id", "patient_ref", "age", "sex",
    "semiology_text", "mri_report_text", "eeg_report_text",
    "demographics_notes",
    "mri_image_paths", "eeg_image_paths",
    "num_mri_images", "num_eeg_images", "num_linked_figures",
    "tier", "source", "keyword_source",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "splits"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading v4 from {V4}")
    df = pd.read_excel(V4, sheet_name="Patients")
    df = df[df["source"] == "pubmed"].copy()  # benchmark = PubMed only
    df["patient_id"] = df["pmc_id"].astype(str) + "_p" + df["patient_num"].astype(str)
    print(f"Total PubMed patients: {len(df)}")

    # Save label maps
    with (out_dir.parent / "label_maps_v4.json").open("w") as f:
        json.dump(LABEL_MAPS, f, indent=2)

    summary = []
    for task in TASKS:
        col = f"{task}_label"
        if col not in df.columns:
            print(f"[skip] no column {col}")
            continue
        labelled = df[df[col].notna()].copy()
        labelled[f"{task}_label_id"] = labelled[col].map(LABEL_MAPS[task])
        labelled = labelled[labelled[f"{task}_label_id"].notna()].copy()
        labelled[f"{task}_label_id"] = labelled[f"{task}_label_id"].astype(int)

        keep = ["patient_id"] + KEEP_COLS + [col, f"{task}_label_id"]
        sub = labelled[keep].copy()

        task_dir = out_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        train = sub[sub["tier"] == "B"]
        test_gold = sub[sub["tier"] == "A"]
        test_bronze = sub[sub["tier"] == "C"]
        train.to_csv(task_dir / "train.csv", index=False)
        test_gold.to_csv(task_dir / "test_gold.csv", index=False)
        test_bronze.to_csv(task_dir / "test_bronze.csv", index=False)

        n_classes = sub[f"{task}_label_id"].nunique()
        summary.append({
            "task": task,
            "train_silver": len(train),
            "test_gold": len(test_gold),
            "test_bronze": len(test_bronze),
            "n_classes_observed": int(n_classes),
        })
        print(f"  {task}: train={len(train)} test_gold={len(test_gold)} test_bronze={len(test_bronze)} classes={n_classes}")

    pd.DataFrame(summary).to_csv(out_dir / "split_summary.csv", index=False)
    print(f"\nWrote splits to {out_dir}")
    print(f"Wrote label maps to {out_dir.parent / 'label_maps_v4.json'}")


if __name__ == "__main__":
    main()
