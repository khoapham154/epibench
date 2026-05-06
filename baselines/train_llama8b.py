"""Llama-3.1-8B-Instruct as a fine-tunable encoder for the 6 EpiBench tasks.

Setup:
- meta-llama/Llama-3.1-8B-Instruct (already cached if HF_HOME has it; if not,
  this will download ~16 GB weights).
- LoRA via PEFT (r=16, alpha=32, dropout=0.05, target_modules q/k/v/o + gate/up/down)
- max_length 1024, bf16, class-weighted CE.
- 4 epochs (LoRA needs more than full-FT, but encoder is bigger).
- Pooled hidden state of the LAST token -> classification head.

Output: results/llama31_8b/{task}_seed42.json (same shape as the BERT baselines).

Run (one task at a time, ~10 min each on 1xA100):
    CUDA_VISIBLE_DEVICES=0 python train_llama8b.py --task epilepsy_type --seed 42
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    DataCollatorWithPadding, Trainer, TrainingArguments,
)

THIS = Path(__file__).resolve().parent
SPLITS = THIS / "splits"
RESULTS = THIS / "results"
TIERS = ["silver", "gold", "bronze", "expansion"]
MODEL_ID = os.environ.get("LLAMA_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
BASELINE = "llama31_8b"


def build_input_text(df: pd.DataFrame) -> list[str]:
    parts = []
    for _, r in df.iterrows():
        chunks = []
        for fld, lbl in [("semiology_text", "SEMIOLOGY"), ("mri_report_text", "MRI"),
                         ("eeg_report_text", "EEG"), ("demographics_notes", "DEMOGRAPHICS")]:
            v = r.get(fld)
            if pd.notna(v): chunks.append(f"{lbl}: {v}")
        for fld, lbl in [("age", "Age"), ("sex", "Sex")]:
            v = r.get(fld)
            if pd.notna(v): chunks.append(f"{lbl}: {v}")
        parts.append(" | ".join(chunks) if chunks else "[no text]")
    return parts


def _eval(yte, pred):
    return {
        "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
        "per_class": classification_report(yte, pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(yte, pred).tolist(),
        "n": int(len(yte)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_length", type=int, default=1024)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    train_df = pd.read_csv(SPLITS / args.task / "train.csv")
    train_df["text"] = build_input_text(train_df)
    label_col = f"{args.task}_label_id"

    tests = {}
    for tier in TIERS:
        p = SPLITS / args.task / f"test_{tier}.csv"
        if p.exists(): tests[tier] = pd.read_csv(p)
    all_labels = set(train_df[label_col].astype(int).tolist())
    for tier, df in tests.items():
        all_labels.update(df[label_col].astype(int).tolist())
    n_classes = max(all_labels) + 1

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize(b):
        return tok(b["text"], truncation=True, max_length=args.max_length)

    ds_train = Dataset.from_pandas(
        train_df[["text", label_col]].rename(columns={label_col: "labels"})
    ).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=n_classes, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.pad_token_id = tok.pad_token_id

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="SEQ_CLS",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    counts = np.bincount(train_df[label_col].astype(int).values, minlength=n_classes).astype(np.float32)
    cw = torch.from_numpy((counts.sum() / (n_classes * np.maximum(counts, 1))).astype(np.float32))

    out_dir = THIS / "tmp_runs" / f"{BASELINE}_{args.task}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(out_dir), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=args.batch * 2,
        learning_rate=args.lr, weight_decay=0.01, save_strategy="no",
        eval_strategy="no", logging_steps=200,
        bf16=True, warmup_ratio=0.05, max_grad_norm=1.0,
        seed=args.seed, report_to=[], dataloader_num_workers=4,
    )

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss(weight=cw.to(logits.device))
            return (loss_fct(logits, labels), outputs) if return_outputs else loss_fct(logits, labels)

    trainer = WeightedTrainer(model=model, args=targs, train_dataset=ds_train,
                              data_collator=DataCollatorWithPadding(tokenizer=tok))
    trainer.train()

    lora_cfg_d = lora_cfg.to_dict()
    if isinstance(lora_cfg_d.get("target_modules"), set):
        lora_cfg_d["target_modules"] = sorted(lora_cfg_d["target_modules"])
    res = {"task": args.task, "baseline": BASELINE, "n_train": int(len(train_df)),
           "config": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                      "seed": args.seed, "model": MODEL_ID,
                      "max_length": args.max_length, "lora": lora_cfg_d,
                      "class_weights": True}}
    preds_dir = RESULTS / BASELINE / "predictions"; preds_dir.mkdir(parents=True, exist_ok=True)
    for tier in TIERS:
        if tier not in tests:
            res[f"macro_f1_{tier}"] = None; res[f"n_test_{tier}"] = 0; continue
        df = tests[tier]
        if len(df) == 0:
            res[f"macro_f1_{tier}"] = None; res[f"n_test_{tier}"] = 0; continue
        df = df.copy(); df["text"] = build_input_text(df)
        ds = Dataset.from_pandas(
            df[["text", label_col]].rename(columns={label_col: "labels"})
        ).map(tokenize, batched=True)
        preds = trainer.predict(ds)
        pred = np.argmax(preds.predictions, axis=-1); yte = np.array(ds["labels"])
        ev = _eval(yte, pred)
        res[f"macro_f1_{tier}"] = ev["macro_f1"]
        res[f"per_class_{tier}"] = ev["per_class"]
        res[f"confusion_matrix_{tier}"] = ev["confusion_matrix"]
        res[f"n_test_{tier}"] = ev["n"]
        np.savez_compressed(preds_dir / f"{args.task}_{tier}_seed{args.seed}.npz",
                            logits=preds.predictions.astype(np.float32),
                            pred=pred, y=yte)

    out = RESULTS / BASELINE / f"{args.task}_seed{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))

    import shutil; shutil.rmtree(out_dir, ignore_errors=True)
    print(f"\n=== {BASELINE} / {args.task} ===")
    for t in TIERS:
        f1 = res.get(f"macro_f1_{t}"); s = f"{f1:.4f}" if isinstance(f1, float) else "—"
        print(f"  {t:>9s}: {s}")
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
