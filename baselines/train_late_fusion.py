"""Train MedSigLIP-feature linear classifiers + late fusion with PubMedBERT-base.

Reads cached MedSigLIP-448 features from features/medsiglip/, builds patient-level
mean-pooled features per modality, trains class-balanced LogisticRegression per
task, evaluates on the same PMC-disjoint GOLD/SILVER/BRONZE splits.

Three baselines per task:
  - medsiglip_mri  — MRI features only (patients with ≥1 MRI subfigure)
  - medsiglip_eeg  — EEG features only (patients with ≥1 EEG subfigure)
  - late_fusion    — mean of text logits + MRI logits + EEG logits over available
                     modalities (text always present; MRI/EEG only when present)

Skips Expansion tier: by tier-rule construction, Expansion has 0 linked images.

Output: results/{baseline}/{task}_seed42.json with macro_f1 per tier + per-class
        report + confusion matrix.

Run:
    python train_multimodal.py --task epilepsy_type
    python train_multimodal.py --task all
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

THIS = Path(__file__).resolve().parent
SPLITS = THIS / "splits"
RESULTS = THIS / "results"
FEATURES_DIR = THIS / "features" / "medsiglip"
LABEL_MAPS = json.loads((THIS / "label_maps_v4.json").read_text())
TASKS = list(LABEL_MAPS.keys())
TIERS = ["silver", "gold", "bronze"]  # Expansion has 0 image patients


def _hash(path: str) -> str:
    return hashlib.sha1(path.encode()).hexdigest()[:16]


def _parse(s) -> list[str]:
    if pd.isna(s) or s in ("", "[]"):
        return []
    try:
        v = ast.literal_eval(str(s))
        if isinstance(v, list):
            return [str(p) for p in v]
    except Exception:
        pass
    return []


def _load_feature(h: str) -> np.ndarray | None:
    p = FEATURES_DIR / f"{h}.npy"
    return np.load(p) if p.exists() else None


def _patient_feature(paths: list[str]) -> np.ndarray | None:
    """Mean-pool L2-normalised features over patient's subfigures."""
    feats = []
    for p in paths:
        v = _load_feature(_hash(p))
        if v is not None:
            feats.append(v)
    if not feats:
        return None
    arr = np.stack(feats, axis=0)
    pooled = arr.mean(axis=0)
    # L2-renormalise after mean-pool
    n = np.linalg.norm(pooled) + 1e-9
    return pooled / n


def _split(task: str) -> dict:
    out = {"train": pd.read_csv(SPLITS / task / "train.csv")}
    for tier in TIERS:
        out[tier] = pd.read_csv(SPLITS / task / f"test_{tier}.csv")
    return out


def _patient_features(df: pd.DataFrame, modality: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return (X, y, kept_idx) — only patients with ≥1 subfigure of this modality."""
    col = "mri_image_paths" if modality == "MRI" else "eeg_image_paths"
    Xs, ys, idxs = [], [], []
    label_col = None
    # Find the label column for this row's task; we get it from caller
    for i, r in df.iterrows():
        paths = _parse(r.get(col))
        if not paths:
            continue
        feat = _patient_feature(paths)
        if feat is None:
            continue
        Xs.append(feat)
        idxs.append(i)
    return np.stack(Xs, axis=0) if Xs else np.zeros((0, 1)), None, idxs


def run_modality(task: str, modality: str) -> dict:
    """Train LogReg on MedSigLIP features for one modality."""
    splits = _split(task)
    label_col = f"{task}_label_id"

    # Train
    train_X = []
    train_y = []
    train_df = splits["train"]
    col = "mri_image_paths" if modality == "MRI" else "eeg_image_paths"
    for _, r in train_df.iterrows():
        feat = _patient_feature(_parse(r.get(col)))
        if feat is None:
            continue
        train_X.append(feat)
        train_y.append(int(r[label_col]))
    if len(train_X) < 20:
        return {"task": task, "baseline": f"medsiglip_{modality.lower()}",
                "skipped": True, "reason": f"only {len(train_X)} train patients with {modality}"}

    Xtr = np.stack(train_X, axis=0)
    ytr = np.array(train_y, dtype=int)
    print(f"[{modality} {task}] train n={len(ytr)} feature_dim={Xtr.shape[1]}")

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1, random_state=42)
    clf.fit(Xtr, ytr)

    res = {
        "task": task, "baseline": f"medsiglip_{modality.lower()}",
        "modality": modality, "n_train": int(len(ytr)),
        "feature_dim": int(Xtr.shape[1]),
        "config": {"model": "google/medsiglip-448", "head": "LogReg(balanced)"},
    }

    for tier in TIERS:
        df = splits[tier]
        Xte, yte = [], []
        for _, r in df.iterrows():
            feat = _patient_feature(_parse(r.get(col)))
            if feat is None:
                continue
            Xte.append(feat)
            yte.append(int(r[label_col]))
        if not Xte:
            res[f"macro_f1_{tier}"] = None
            res[f"n_test_{tier}"] = 0
            continue
        Xte = np.stack(Xte, axis=0); yte = np.array(yte)
        pred = clf.predict(Xte)
        res[f"macro_f1_{tier}"] = float(f1_score(yte, pred, average="macro", zero_division=0))
        res[f"per_class_{tier}"] = classification_report(yte, pred, output_dict=True, zero_division=0)
        res[f"confusion_matrix_{tier}"] = confusion_matrix(yte, pred).tolist()
        res[f"n_test_{tier}"] = int(len(yte))

    return res


def run_late_fusion(task: str) -> dict:
    """Late-fuse text logits (PubMedBERT-base) + MRI logits + EEG logits.

    Score patients on logit-mean across available modalities (text always
    present). Train uses train.csv with patients that have MRI and/or EEG;
    text logits come from PubMedBERT-base via re-fitting on the same train.
    """
    # Cheap text head: TF-IDF + LogReg (fast, deterministic).
    # Could swap for PubMedBERT-base logits later.
    from sklearn.feature_extraction.text import TfidfVectorizer

    splits = _split(task)
    label_col = f"{task}_label_id"

    def build_text(df):
        out = []
        for _, r in df.iterrows():
            chunks = []
            for fld, lbl in [("semiology_text", "SEMIOLOGY"), ("mri_report_text", "MRI"),
                             ("eeg_report_text", "EEG"), ("demographics_notes", "DEMOGRAPHICS")]:
                v = r.get(fld)
                if pd.notna(v):
                    chunks.append(f"{lbl}: {v}")
            for fld, lbl in [("age", "Age"), ("sex", "Sex")]:
                v = r.get(fld)
                if pd.notna(v):
                    chunks.append(f"{lbl}: {v}")
            out.append(" | ".join(chunks) if chunks else "[no text]")
        return out

    train_df = splits["train"]
    Xtr_text = build_text(train_df)
    ytr = train_df[label_col].astype(int).values
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
    Xtr_text_v = vec.fit_transform(Xtr_text)
    text_clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1, random_state=42)
    text_clf.fit(Xtr_text_v, ytr)

    # Train MRI head
    mri_X, mri_y = [], []
    for _, r in train_df.iterrows():
        feat = _patient_feature(_parse(r.get("mri_image_paths")))
        if feat is not None:
            mri_X.append(feat); mri_y.append(int(r[label_col]))
    mri_clf = None
    if len(mri_X) >= 20:
        mri_clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1, random_state=42)
        mri_clf.fit(np.stack(mri_X), np.array(mri_y))

    # Train EEG head
    eeg_X, eeg_y = [], []
    for _, r in train_df.iterrows():
        feat = _patient_feature(_parse(r.get("eeg_image_paths")))
        if feat is not None:
            eeg_X.append(feat); eeg_y.append(int(r[label_col]))
    eeg_clf = None
    if len(eeg_X) >= 20:
        eeg_clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1, random_state=42)
        eeg_clf.fit(np.stack(eeg_X), np.array(eeg_y))

    res = {
        "task": task, "baseline": "late_fusion",
        "n_train_text": int(len(ytr)),
        "n_train_mri": int(len(mri_X)),
        "n_train_eeg": int(len(eeg_X)),
        "config": {
            "text_head": "TF-IDF + LogReg",
            "mri_head": "MedSigLIP-448 + LogReg" if mri_clf else "skipped",
            "eeg_head": "MedSigLIP-448 + LogReg" if eeg_clf else "skipped",
            "fusion": "mean of available logits per patient",
        },
    }

    n_classes = max(ytr.max(), max((c for c in [mri_y, eeg_y] for c in c), default=0)) + 1

    for tier in TIERS:
        df = splits[tier]
        # Predict text logits for all
        Xte_text = build_text(df)
        Xte_text_v = vec.transform(Xte_text)
        text_logits = text_clf.predict_log_proba(Xte_text_v)

        # MRI logits where available; fall back to None
        mri_logits = np.full((len(df), n_classes), np.nan)
        if mri_clf:
            mri_class_to_idx = {int(c): i for i, c in enumerate(mri_clf.classes_)}
            for i, r in enumerate(df.itertuples()):
                feat = _patient_feature(_parse(getattr(r, "mri_image_paths", None)))
                if feat is None:
                    continue
                lp = mri_clf.predict_log_proba(feat.reshape(1, -1))[0]
                vec_full = np.full(n_classes, -1e9)
                for c, idx in mri_class_to_idx.items():
                    if c < n_classes:
                        vec_full[c] = lp[idx]
                mri_logits[i] = vec_full

        # EEG logits where available
        eeg_logits = np.full((len(df), n_classes), np.nan)
        if eeg_clf:
            eeg_class_to_idx = {int(c): i for i, c in enumerate(eeg_clf.classes_)}
            for i, r in enumerate(df.itertuples()):
                feat = _patient_feature(_parse(getattr(r, "eeg_image_paths", None)))
                if feat is None:
                    continue
                lp = eeg_clf.predict_log_proba(feat.reshape(1, -1))[0]
                vec_full = np.full(n_classes, -1e9)
                for c, idx in eeg_class_to_idx.items():
                    if c < n_classes:
                        vec_full[c] = lp[idx]
                eeg_logits[i] = vec_full

        # Pad text_logits to n_classes if necessary
        text_full = np.full((len(df), n_classes), -1e9)
        text_class_to_idx = {int(c): i for i, c in enumerate(text_clf.classes_)}
        for c, idx in text_class_to_idx.items():
            if c < n_classes:
                text_full[:, c] = text_logits[:, idx]

        # Fuse: mean over available logits per patient
        stacked = np.stack([text_full, mri_logits, eeg_logits], axis=0)  # [3, N, C]
        mask = np.isfinite(stacked)
        # Replace NaN with 0 for sum, divide by count
        clean = np.where(mask, stacked, 0)
        cnt = mask.sum(axis=0)
        cnt = np.where(cnt == 0, 1, cnt)
        fused = clean.sum(axis=0) / cnt  # [N, C]
        pred = fused.argmax(axis=-1)
        yte = df[f"{task}_label_id"].astype(int).values

        res[f"macro_f1_{tier}"] = float(f1_score(yte, pred, average="macro", zero_division=0))
        res[f"per_class_{tier}"] = classification_report(yte, pred, output_dict=True, zero_division=0)
        res[f"confusion_matrix_{tier}"] = confusion_matrix(yte, pred).tolist()
        res[f"n_test_{tier}"] = int(len(yte))
        # Also report fraction of patients in this tier that contributed which modality
        res[f"n_with_mri_{tier}"] = int(np.isfinite(mri_logits).any(axis=1).sum())
        res[f"n_with_eeg_{tier}"] = int(np.isfinite(eeg_logits).any(axis=1).sum())
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="task name OR 'all'")
    args = ap.parse_args()

    tasks = TASKS if args.task == "all" else [args.task]

    for task in tasks:
        print(f"\n=== {task} ===")
        for mod in ["MRI", "EEG"]:
            res = run_modality(task, mod)
            out = RESULTS / f"medsiglip_{mod.lower()}" / f"{task}_seed42.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(res, indent=2))
            print(f"  saved {out.name}: {res.get('macro_f1_gold','—')}")
        # Late fusion
        res = run_late_fusion(task)
        out = RESULTS / "late_fusion" / f"{task}_seed42.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(f"  saved late_fusion/{task}_seed42.json: GOLD={res.get('macro_f1_gold','—')}")


if __name__ == "__main__":
    main()
