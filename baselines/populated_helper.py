"""Shared helpers for the populated-only restriction.

Three things we want consistent across all new baselines:
  1. The label set for inference / scoring (drop classes the v4 release left empty).
  2. Logit-masking helper to push prediction probability of empty classes to 0.
  3. Macro-F1 scoring restricted to populated class IDs.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

THIS = Path(__file__).resolve().parent
LABEL_MAPS_FULL = json.loads((THIS / "label_maps_v4.json").read_text())
LABEL_MAPS_POPULATED = json.loads((THIS / "label_maps_v4_populated.json").read_text())
# Drop the _meta key from the populated map for downstream code
LABEL_MAPS_POPULATED.pop("_meta", None)


def populated_ids(task: str) -> list[int]:
    """Return the list of class IDs that have populated training examples for `task`.

    For tasks with no removed classes, this is just range(n_classes).
    """
    pop = LABEL_MAPS_POPULATED[task]
    return sorted(pop.values())


def populated_class_names(task: str) -> list[str]:
    """Return populated class names in the order matching populated_ids()."""
    pop = LABEL_MAPS_POPULATED[task]
    inv = {v: k for k, v in pop.items()}
    return [inv[i] for i in populated_ids(task)]


def n_classes_full(task: str) -> int:
    """Total number of declared classes (including empty ones), used for output-head sizing."""
    return max(LABEL_MAPS_FULL[task].values()) + 1


def mask_unpopulated_logits(logits: torch.Tensor, task: str) -> torch.Tensor:
    """Set logits for unpopulated classes to -inf so they're never argmax'd."""
    pop_ids = set(populated_ids(task))
    full_ids = list(range(logits.shape[-1]))
    mask_ids = [i for i in full_ids if i not in pop_ids]
    if not mask_ids:
        return logits
    out = logits.clone()
    out[..., mask_ids] = float("-inf")
    return out


def macro_f1_populated(y_true, y_pred, task: str) -> float:
    """Macro-F1 over populated classes only (so empty classes don't drag the average to 0)."""
    pop = populated_ids(task)
    return float(f1_score(y_true, y_pred, labels=pop, average="macro", zero_division=0))


def eval_populated(y_true, y_pred, task: str) -> dict:
    """Full evaluation dict matching the existing baseline output shape."""
    pop = populated_ids(task)
    return {
        "macro_f1": macro_f1_populated(y_true, y_pred, task),
        "per_class": classification_report(
            y_true, y_pred, labels=pop, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=pop).tolist(),
        "n": int(len(y_true)),
        "populated_ids": pop,
        "populated_classes": populated_class_names(task),
    }
