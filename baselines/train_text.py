"""Text baselines for EpiBench v4: TF-IDF + LogReg, PubMedBERT fine-tune.

For each of 6 tasks:
  1. Load train (90% Tier-B SILVER) + 4 test partitions:
       test_silver.csv (10% Tier-B), test_gold.csv (Tier-A),
       test_bronze.csv (Tier-C), test_expansion.csv (Tier-D).
  2. Concatenate (semiology + mri_report + eeg_report + demographics_notes) as input text.
  3a. TF-IDF + LogReg (CPU only, ~1 min/task).
  3b. PubMedBERT fine-tune via HF transformers (1 GPU, ~30 min/task).
  4. Evaluate macro-F1, per-class P/R/F1, confusion matrix on each test partition.
  5. Write JSON: results/{baseline}/{task}_seed{N}.json with keys
       macro_f1_silver, macro_f1_gold, macro_f1_bronze, macro_f1_expansion,
       plus per-tier per_class and confusion_matrix.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

THIS = Path(__file__).resolve().parent
SPLITS = THIS / "splits"
RESULTS = THIS / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

TIERS = ["silver", "gold", "bronze", "expansion"]


def build_input_text(df: pd.DataFrame) -> list[str]:
    parts = []
    for _, r in df.iterrows():
        chunks = []
        if pd.notna(r.get("semiology_text")):
            chunks.append(f"SEMIOLOGY: {r['semiology_text']}")
        if pd.notna(r.get("mri_report_text")):
            chunks.append(f"MRI: {r['mri_report_text']}")
        if pd.notna(r.get("eeg_report_text")):
            chunks.append(f"EEG: {r['eeg_report_text']}")
        if pd.notna(r.get("demographics_notes")):
            chunks.append(f"DEMOGRAPHICS: {r['demographics_notes']}")
        if pd.notna(r.get("age")):
            chunks.append(f"Age: {r['age']}")
        if pd.notna(r.get("sex")):
            chunks.append(f"Sex: {r['sex']}")
        parts.append(" | ".join(chunks) if chunks else "[no text]")
    return parts


def _eval(yte: np.ndarray, pred: np.ndarray) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    return {
        "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
        "per_class": classification_report(yte, pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(yte, pred).tolist(),
        "n": int(len(yte)),
    }


def _load_tests(task: str) -> dict[str, pd.DataFrame]:
    tests: dict[str, pd.DataFrame] = {}
    for tier in TIERS:
        path = SPLITS / task / f"test_{tier}.csv"
        if path.exists():
            df = pd.read_csv(path)
        else:
            df = pd.DataFrame()
        tests[tier] = df
    return tests


def run_tfidf_with_regime(task: str, regime: str) -> dict:
    """TF-IDF baseline trained under cross-tier regime ('B' | 'B_C' | 'B_C_D')."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    train = pd.read_csv(SPLITS / task / "train.csv")
    if regime in ("B_C", "B_C_D"):
        train = pd.concat([train, pd.read_csv(SPLITS / task / "test_bronze.csv")], ignore_index=True)
    if regime == "B_C_D":
        train = pd.concat([train, pd.read_csv(SPLITS / task / "test_expansion.csv")], ignore_index=True)
    Xtr = build_input_text(train)
    ytr = train[f"{task}_label_id"].astype(int).values
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
    Xtr_v = vec.fit_transform(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
    clf.fit(Xtr_v, ytr)
    res: dict = {"task": task, "baseline": "tfidf_logreg",
                 "n_train": int(len(train)),
                 "config": {"train_regime": regime}}
    tests = _load_tests(task)
    # Only score GOLD here; cross-tier scaling test-set is GOLD by design.
    for tier in TIERS:
        df = tests[tier]
        if len(df) == 0:
            res[f"macro_f1_{tier}"] = None
            res[f"n_test_{tier}"] = 0
            continue
        # Skip BRONZE/Expansion if they were in training
        if regime in ("B_C", "B_C_D") and tier == "bronze":
            res[f"macro_f1_{tier}"] = None; res[f"n_test_{tier}"] = 0; continue
        if regime == "B_C_D" and tier == "expansion":
            res[f"macro_f1_{tier}"] = None; res[f"n_test_{tier}"] = 0; continue
        Xte = build_input_text(df)
        Xte_v = vec.transform(Xte)
        yte = df[f"{task}_label_id"].astype(int).values
        pred = clf.predict(Xte_v)
        ev = _eval(yte, pred)
        res[f"macro_f1_{tier}"] = ev["macro_f1"]
        res[f"per_class_{tier}"] = ev["per_class"]
        res[f"confusion_matrix_{tier}"] = ev["confusion_matrix"]
        res[f"n_test_{tier}"] = ev["n"]
    return res


def run_tfidf(task: str) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    train = pd.read_csv(SPLITS / task / "train.csv")
    Xtr = build_input_text(train)
    ytr = train[f"{task}_label_id"].astype(int).values

    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
    Xtr_v = vec.fit_transform(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
    clf.fit(Xtr_v, ytr)

    res: dict = {
        "task": task, "baseline": "tfidf_logreg",
        "n_train": int(len(train)),
    }
    tests = _load_tests(task)
    for tier in TIERS:
        df = tests[tier]
        if len(df) == 0:
            res[f"macro_f1_{tier}"] = None
            res[f"n_test_{tier}"] = 0
            continue
        Xte = build_input_text(df)
        Xte_v = vec.transform(Xte)
        yte = df[f"{task}_label_id"].astype(int).values
        pred = clf.predict(Xte_v)
        ev = _eval(yte, pred)
        res[f"macro_f1_{tier}"] = ev["macro_f1"]
        res[f"per_class_{tier}"] = ev["per_class"]
        res[f"confusion_matrix_{tier}"] = ev["confusion_matrix"]
        res[f"n_test_{tier}"] = ev["n"]
    return res


def run_pubmedbert(task: str, epochs: int = 3, batch: int = 16, lr: float = 2e-5,
                   seed: int = 42, model_id: str | None = None,
                   baseline_name: str = "pubmedbert",
                   max_length: int = 512, warmup_ratio: float = 0.0,
                   class_weights: bool = False, save_preds: bool = False,
                   train_regime: str = "B") -> dict:
    """train_regime: one of 'B' (default, train.csv only),
                     'B_C' (train.csv + test_bronze.csv),
                     'B_C_D' (train.csv + test_bronze.csv + test_expansion.csv).
    The Tier-C/D rows are pulled from their test_*.csv files (those CSVs are
    *patient* sets, not held-out splits; for the scaling study we move them
    into training and only test on GOLD).
    """
    import shutil
    import torch
    import torch.nn as nn
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer,
        DataCollatorWithPadding, Trainer, TrainingArguments,
    )

    torch.manual_seed(seed); np.random.seed(seed)

    train_df = pd.read_csv(SPLITS / task / "train.csv")
    if train_regime in ("B_C", "B_C_D"):
        bronze = pd.read_csv(SPLITS / task / "test_bronze.csv")
        train_df = pd.concat([train_df, bronze], ignore_index=True)
    if train_regime == "B_C_D":
        exp = pd.read_csv(SPLITS / task / "test_expansion.csv")
        train_df = pd.concat([train_df, exp], ignore_index=True)
    train_df["text"] = build_input_text(train_df)
    label_col = f"{task}_label_id"

    # Determine n_classes: include any class observed across train + all 4 tests
    tests = _load_tests(task)
    all_labels = set(train_df[label_col].astype(int).tolist())
    for tier in TIERS:
        if len(tests[tier]):
            all_labels.update(tests[tier][label_col].astype(int).tolist())
    n_classes = max(all_labels) + 1 if all_labels else 1

    if model_id is None:
        model_id = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    tok = AutoTokenizer.from_pretrained(model_id)

    def tokenize(b):
        return tok(b["text"], truncation=True, max_length=max_length)

    ds_train = Dataset.from_pandas(
        train_df[["text", label_col]].rename(columns={label_col: "labels"})
    ).map(tokenize, batched=True)

    out = THIS / "tmp_runs" / f"{baseline_name}_{task}_{train_regime}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=n_classes)
    # DeBERTa-v3 needs lower lr + warmup, and bf16 (fp16 disentangled-attention is unstable).
    is_deberta = "deberta" in model_id.lower()
    use_bf16 = is_deberta
    if is_deberta:
        lr = 1e-5
        warmup_ratio = max(warmup_ratio, 0.1)
    max_grad_norm = 1.0

    cw_tensor = None
    if class_weights:
        counts = np.bincount(train_df[label_col].astype(int).values, minlength=n_classes).astype(np.float32)
        cw = counts.sum() / (n_classes * np.maximum(counts, 1))
        cw_tensor = torch.from_numpy(cw.astype(np.float32))

    args = TrainingArguments(
        output_dir=str(out), num_train_epochs=epochs,
        per_device_train_batch_size=batch, per_device_eval_batch_size=batch * 2,
        learning_rate=lr, weight_decay=0.01, save_strategy="no",
        eval_strategy="no", logging_steps=500,
        fp16=not use_bf16, bf16=use_bf16,
        warmup_ratio=warmup_ratio, max_grad_norm=max_grad_norm,
        seed=seed,
        report_to=[], dataloader_num_workers=4,
    )

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss(
                weight=cw_tensor.to(logits.device) if cw_tensor is not None else None
            )
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    TrainerCls = WeightedTrainer if class_weights else Trainer
    trainer = TrainerCls(model=model, args=args, train_dataset=ds_train,
                         data_collator=DataCollatorWithPadding(tokenizer=tok))
    trainer.train()

    res: dict = {
        "task": task, "baseline": baseline_name,
        "n_train": int(len(train_df)),
        "config": {"epochs": epochs, "batch": batch, "lr": lr, "seed": seed,
                   "model": model_id, "max_length": max_length,
                   "warmup_ratio": warmup_ratio, "class_weights": class_weights,
                   "train_regime": train_regime},
    }
    preds_dir = None
    if save_preds:
        preds_dir = RESULTS / baseline_name / "predictions"
        preds_dir.mkdir(parents=True, exist_ok=True)
    for tier in TIERS:
        df = tests[tier]
        if len(df) == 0:
            res[f"macro_f1_{tier}"] = None
            res[f"n_test_{tier}"] = 0
            continue
        df = df.copy()
        df["text"] = build_input_text(df)
        ds = Dataset.from_pandas(
            df[["text", label_col]].rename(columns={label_col: "labels"})
        ).map(tokenize, batched=True)
        preds = trainer.predict(ds)
        pred = np.argmax(preds.predictions, axis=-1)
        yte = np.array(ds["labels"])
        ev = _eval(yte, pred)
        res[f"macro_f1_{tier}"] = ev["macro_f1"]
        res[f"per_class_{tier}"] = ev["per_class"]
        res[f"confusion_matrix_{tier}"] = ev["confusion_matrix"]
        res[f"n_test_{tier}"] = ev["n"]
        if preds_dir is not None:
            np.savez_compressed(preds_dir / f"{task}_{tier}_seed{seed}.npz",
                                logits=preds.predictions.astype(np.float32),
                                pred=pred, y=yte)

    shutil.rmtree(out, ignore_errors=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True,
                    help="tfidf | pubmedbert | biomedbert_large | deberta_v3_large | bio_clinicalbert | <custom>")
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_id", default=None,
                    help="Override HF model id (otherwise picked from --baseline)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--warmup_ratio", type=float, default=0.0)
    ap.add_argument("--class_weights", action="store_true")
    ap.add_argument("--save_preds", action="store_true")
    ap.add_argument("--train_regime", choices=["B", "B_C", "B_C_D"], default="B")
    ap.add_argument("--out_suffix", default=None,
                    help="If set, suffix for the output filename: {task}_seed{seed}{suffix}.json")
    args = ap.parse_args()

    BASELINE_MODELS = {
        "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "biomedbert_large": "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
        "deberta_v3_large": "microsoft/deberta-v3-large",
        "bio_clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    }
    if args.baseline == "tfidf":
        res = run_tfidf_with_regime(args.task, args.train_regime) \
            if args.train_regime != "B" else run_tfidf(args.task)
    else:
        model_id = args.model_id or BASELINE_MODELS.get(args.baseline)
        if model_id is None:
            raise SystemExit(f"Unknown baseline {args.baseline!r} and --model_id not provided")
        res = run_pubmedbert(args.task, seed=args.seed,
                             model_id=model_id, baseline_name=args.baseline,
                             epochs=args.epochs, max_length=args.max_length,
                             warmup_ratio=args.warmup_ratio,
                             class_weights=args.class_weights,
                             save_preds=args.save_preds,
                             train_regime=args.train_regime)
    suffix = args.out_suffix or ""
    out = RESULTS / args.baseline / f"{args.task}_seed{args.seed}{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(res, f, indent=2)

    print(f"\n=== {args.baseline} / {args.task} / seed={args.seed} ===")
    for tier in TIERS:
        f1 = res.get(f"macro_f1_{tier}")
        n = res.get(f"n_test_{tier}", 0)
        s = f"{f1:.4f}" if isinstance(f1, float) else "—"
        print(f"  macro-F1 ({tier:>9s}, n={n:>5}): {s}")
    print(f"  saved: {out}")


if __name__ == "__main__":
    main()
