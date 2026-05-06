"""Modality ablation grid on GOLD test set.

For two information-dense tasks (epilepsy_type, ez_localization) train
small heads under six regimes and report macro-F1 on GOLD:

  text_only        — TF-IDF + LogReg on patient text
  mri_only         — MedSigLIP-MRI mean-pooled subfigure features
  eeg_only         — MedSigLIP-EEG mean-pooled subfigure features
  text_mri         — concat(TF-IDF, MRI features)
  text_eeg         — concat(TF-IDF, EEG features)
  text_mri_eeg     — concat(TF-IDF, MRI, EEG)

Patients without a given modality contribute a zero-vector for that block,
plus a 0/1 availability flag (so the head can learn to down-weight zeros).
This makes the comparison fair across regimes (same N) and matches how a
real deployment handles missing modalities.

Output:
  results/modality_ablation.json
  results/modality_ablation.csv
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

THIS = Path(__file__).resolve().parent
SPLITS = THIS / "splits"
RESULTS = THIS / "results"
MED_DIR = THIS / "features" / "medsiglip"
LABEL_MAPS = json.loads((THIS / "label_maps_v4.json").read_text())

TASKS = ["epilepsy_type", "ez_localization"]
REGIMES = ["text_only", "mri_only", "eeg_only", "text_mri", "text_eeg", "text_mri_eeg"]


def _hash(p): return hashlib.sha1(p.encode()).hexdigest()[:16]


def _parse(s):
    if pd.isna(s) or s in ("", "[]"): return []
    try:
        v = ast.literal_eval(str(s))
        return [str(p) for p in v] if isinstance(v, list) else []
    except Exception:
        return []


def _patient_feat(paths) -> tuple[np.ndarray | None, int]:
    feats = []
    for p in paths:
        f = MED_DIR / f"{_hash(p)}.npy"
        if f.exists():
            feats.append(np.load(f))
    if not feats:
        return None, 0
    a = np.stack(feats).mean(0)
    a = a / (np.linalg.norm(a) + 1e-9)
    return a.astype(np.float32), len(feats)


def _build_text(df: pd.DataFrame) -> list[str]:
    out = []
    for _, r in df.iterrows():
        chunks = []
        for fld, lbl in [("semiology_text", "SEMIOLOGY"), ("mri_report_text", "MRI"),
                         ("eeg_report_text", "EEG"), ("demographics_notes", "DEMOGRAPHICS")]:
            v = r.get(fld)
            if pd.notna(v): chunks.append(f"{lbl}: {v}")
        for fld, lbl in [("age", "Age"), ("sex", "Sex")]:
            v = r.get(fld)
            if pd.notna(v): chunks.append(f"{lbl}: {v}")
        out.append(" | ".join(chunks) if chunks else "[no text]")
    return out


def _build_modality(df: pd.DataFrame, col: str, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (feat[N, dim], avail[N]). Patients without modality get zeros + avail=0."""
    feats = np.zeros((len(df), dim), dtype=np.float32)
    avail = np.zeros(len(df), dtype=np.float32)
    for i, (_, r) in enumerate(df.iterrows()):
        f, k = _patient_feat(_parse(r.get(col)))
        if f is not None:
            feats[i] = f
            avail[i] = 1.0
    return feats, avail


def _stack_X(text_v, mri_X, mri_a, eeg_X, eeg_a, regime: str):
    """Stack into a single feature matrix per regime. Text is sparse; image is dense."""
    cols = []
    if "text" in regime:
        cols.append(text_v)
    if "mri" in regime:
        cols.append(csr_matrix(np.hstack([mri_X, mri_a[:, None]])))
    if "eeg" in regime:
        cols.append(csr_matrix(np.hstack([eeg_X, eeg_a[:, None]])))
    return hstack(cols).tocsr() if len(cols) > 1 else cols[0]


def run_task(task: str) -> list[dict]:
    train = pd.read_csv(SPLITS / task / "train.csv")
    test = pd.read_csv(SPLITS / task / "test_gold.csv")
    label_col = f"{task}_label_id"
    ytr = train[label_col].astype(int).values
    yte = test[label_col].astype(int).values

    # Image features
    DIM = 1152  # MedSigLIP-448 pooled dim
    mri_tr, mri_a_tr = _build_modality(train, "mri_image_paths", DIM)
    eeg_tr, eeg_a_tr = _build_modality(train, "eeg_image_paths", DIM)
    mri_te, mri_a_te = _build_modality(test, "mri_image_paths", DIM)
    eeg_te, eeg_a_te = _build_modality(test, "eeg_image_paths", DIM)

    # Text features
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
    Xtr_text = vec.fit_transform(_build_text(train))
    Xte_text = vec.transform(_build_text(test))

    rows = []
    for regime in REGIMES:
        Xtr = _stack_X(Xtr_text, mri_tr, mri_a_tr, eeg_tr, eeg_a_tr, regime)
        Xte = _stack_X(Xte_text, mri_te, mri_a_te, eeg_te, eeg_a_te, regime)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        f1 = f1_score(yte, pred, average="macro", zero_division=0)
        # Coverage stats
        if regime == "mri_only":
            cov = float(mri_a_te.mean())
        elif regime == "eeg_only":
            cov = float(eeg_a_te.mean())
        elif regime == "text_mri":
            cov = float(mri_a_te.mean())
        elif regime == "text_eeg":
            cov = float(eeg_a_te.mean())
        elif regime == "text_mri_eeg":
            cov = float(((mri_a_te + eeg_a_te) > 0).mean())
        else:
            cov = 1.0
        rows.append({"task": task, "regime": regime, "macro_f1_gold": float(f1),
                     "n_test": int(len(test)), "test_coverage": cov,
                     "n_train": int(len(train))})
        print(f"  {task:20s} {regime:14s}  macro-F1={f1:.3f}  cov={cov:.1%}")
    return rows


def main() -> None:
    all_rows: list[dict] = []
    for task in TASKS:
        print(f"\n=== {task} ===")
        all_rows.extend(run_task(task))
    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS / "modality_ablation.csv", index=False)
    (RESULTS / "modality_ablation.json").write_text(json.dumps(all_rows, indent=2))
    print("\nWrote results/modality_ablation.{csv,json}")
    print(df.pivot(index="regime", columns="task", values="macro_f1_gold").to_markdown())


if __name__ == "__main__":
    main()
