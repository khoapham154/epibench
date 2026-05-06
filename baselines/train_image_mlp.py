"""Train MLP heads (vs LogReg) on cached MedSigLIP / BiomedCLIP features per task per modality.

This is the cheap alternative to a full LoRA fine-tune of the vision encoder.
A 2-layer MLP usually buys 2-5 macro-F1 over the LogReg baseline by capturing
non-linear class boundaries in the image embedding space.

Run:
    python train_image_mlp_heads.py --variant medsiglip_mri_mlp
    python train_image_mlp_heads.py --variant all
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score

THIS = Path(__file__).resolve().parent
SPLITS = THIS / "splits"
RESULTS = THIS / "results"
MED_DIR = THIS / "features" / "medsiglip"
BMC_DIR = THIS / "features" / "biomedclip"
LABEL_MAPS = json.loads((THIS / "label_maps_v4.json").read_text())
TASKS = list(LABEL_MAPS.keys())
TIERS = ["silver", "gold", "bronze"]


def _hash(p): return hashlib.sha1(p.encode()).hexdigest()[:16]


def _parse(s):
    if pd.isna(s) or s in ("", "[]"): return []
    try:
        v = ast.literal_eval(str(s))
        return [str(p) for p in v] if isinstance(v, list) else []
    except Exception:
        return []


def _patient_feat(img_dir: Path, paths) -> np.ndarray | None:
    feats = []
    for p in paths:
        f = img_dir / f"{_hash(p)}.npy"
        if f.exists():
            feats.append(np.load(f))
    if not feats:
        return None
    a = np.stack(feats).mean(0)
    return a / (np.linalg.norm(a) + 1e-9)


class MLPHead(nn.Module):
    def __init__(self, dim_in, n_classes, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def _split(task):
    return {tier: pd.read_csv(SPLITS / task / f"test_{tier}.csv") for tier in TIERS} | \
           {"train": pd.read_csv(SPLITS / task / "train.csv")}


def run(task: str, modality: str, img_dir: Path, baseline_name: str,
        epochs: int = 60, lr: float = 1e-3, hidden: int = 256) -> dict:
    splits = _split(task)
    label_col = f"{task}_label_id"
    col = "mri_image_paths" if modality == "MRI" else "eeg_image_paths"

    Xtr, ytr = [], []
    for _, r in splits["train"].iterrows():
        f = _patient_feat(img_dir, _parse(r.get(col)))
        if f is not None:
            Xtr.append(f); ytr.append(int(r[label_col]))
    if len(Xtr) < 20:
        return {"task": task, "baseline": baseline_name, "skipped": True,
                "reason": f"only {len(Xtr)} train patients with {modality}"}
    Xtr = np.stack(Xtr); ytr = np.array(ytr)
    n_classes = int(ytr.max()) + 1
    for tier in TIERS:
        for _, r in splits[tier].iterrows():
            try:
                v = int(r[label_col])
                n_classes = max(n_classes, v + 1)
            except Exception:
                pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPHead(dim_in=Xtr.shape[1], n_classes=n_classes, hidden=hidden).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    counts = np.bincount(ytr, minlength=n_classes).astype(np.float32)
    weights = (counts.sum() / (n_classes * np.maximum(counts, 1))).astype(np.float32)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))

    Xtr_t = torch.from_numpy(Xtr).float().to(device)
    ytr_t = torch.from_numpy(ytr).long().to(device)
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(Xtr_t), generator=g)
    Xtr_t, ytr_t = Xtr_t[perm], ytr_t[perm]

    batch = 64
    model.train()
    for _ in range(epochs):
        for i in range(0, len(Xtr_t), batch):
            xb = Xtr_t[i: i + batch]; yb = ytr_t[i: i + batch]
            optim.zero_grad()
            criterion(model(xb), yb).backward()
            optim.step()

    model.eval()
    res = {"task": task, "baseline": baseline_name, "modality": modality,
           "n_train": int(len(ytr)), "feature_dim": int(Xtr.shape[1]),
           "config": {"head": f"MLP[{Xtr.shape[1]}->{hidden}->{hidden//2}->{n_classes}]",
                      "epochs": epochs, "lr": lr, "hidden": hidden}}

    for tier in TIERS:
        df = splits[tier]
        Xte, yte = [], []
        for _, r in df.iterrows():
            f = _patient_feat(img_dir, _parse(r.get(col)))
            if f is not None:
                Xte.append(f); yte.append(int(r[label_col]))
        if not Xte:
            res[f"macro_f1_{tier}"] = None; res[f"n_test_{tier}"] = 0
            continue
        with torch.no_grad():
            logits = model(torch.from_numpy(np.stack(Xte)).float().to(device))
            pred = logits.argmax(-1).cpu().numpy()
        yte = np.array(yte)
        res[f"macro_f1_{tier}"] = float(f1_score(yte, pred, average="macro", zero_division=0))
        res[f"per_class_{tier}"] = classification_report(yte, pred, output_dict=True, zero_division=0)
        res[f"confusion_matrix_{tier}"] = confusion_matrix(yte, pred).tolist()
        res[f"n_test_{tier}"] = int(len(yte))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["medsiglip_mri_mlp", "medsiglip_eeg_mlp",
                             "biomedclip_mri_mlp", "biomedclip_eeg_mlp", "all"])
    args = ap.parse_args()

    plan = {
        "medsiglip_mri_mlp": ("MRI", MED_DIR),
        "medsiglip_eeg_mlp": ("EEG", MED_DIR),
        "biomedclip_mri_mlp": ("MRI", BMC_DIR),
        "biomedclip_eeg_mlp": ("EEG", BMC_DIR),
    }
    variants = list(plan.keys()) if args.variant == "all" else [args.variant]

    for v in variants:
        modality, img_dir = plan[v]
        for task in TASKS:
            print(f"\n[{v}/{task}]")
            res = run(task, modality, img_dir, v)
            out = RESULTS / v / f"{task}_seed42.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(res, indent=2))
            if not res.get("skipped"):
                print(f"  GOLD={res.get('macro_f1_gold','—')}  saved {out.name}")
            else:
                print(f"  SKIP: {res.get('reason')}")


if __name__ == "__main__":
    main()
