#!/usr/bin/env python3
"""
MedSigLIP-448 medical image classifier.

Uses google/medsiglip-448 with sigmoid loss (independent per-class probabilities).
Solves the "softmax forced winner" problem of CLIP — non-brain MRI no longer
gets classified as brain MRI just because it has higher MRI score than other categories.

22 categories: 8 target medical (brain MRI/CT/EEG variants) + 6 non-brain medical
(orbital, abdominal, spine, cardiac, SPECT, PET) + 8 noise (tables, charts, histology, etc.).

Decision rule:
    - target_max = max prob across target medical categories
    - distractor_max = max prob across non-brain medical AND noise categories
    - ACCEPT  if target_max > 0.5 AND target_max > distractor_max + 0.10
    - REJECT  if distractor_max > target_max + 0.10
    - BORDERLINE otherwise (route to VLM verifier)
"""

import re

# Filename-based fast-path rejection (PMC convention: tNNN = table, gNNN = graph in some journals)
# Conservative: only catches obvious table filenames where graphics-extraction tools assigned `t` prefix.
_TABLE_FILENAME_RE = re.compile(
    r"\.t\d{2,3}\.|"          # .t01., .t002., etc. (PLOS ONE tables)
    r"_t\d{2,3}_|"            # _t01_, etc.
    r"-t\d{2,3}-|"            # -t01-
    r"[._-]table[._-]?\d*\.|" # _table_1.jpg
    r"tab\d+\.",              # tab1.jpg
    re.IGNORECASE,
)


def is_likely_table_by_filename(path: str) -> bool:
    """Quick filename heuristic for table images (catches PMC `*.tNNN.*` pattern)."""
    if not path:
        return False
    basename = path.rsplit("/", 1)[-1]
    return bool(_TABLE_FILENAME_RE.search(basename))

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)


# Category groups
TARGET_BRAIN_MRI = [
    "axial brain MRI",
    "coronal brain MRI",
    "sagittal brain MRI",
    "T2 FLAIR brain MRI",
]

TARGET_BRAIN_OTHER = [
    "brain CT scan",
    "scalp EEG recording with multiple channels",
    "intracranial EEG with depth electrodes",
    "ictal EEG showing seizure activity",
]

NON_BRAIN_MEDICAL = [
    "orbital MRI of the eye",
    "abdominal MRI",
    "spine MRI",
    "cardiac MRI",
    "SPECT brain perfusion scan",
    "PET brain scan",
]

NOISE_CATEGORIES = [
    "data table with rows and columns",
    "boxplot of statistical data",
    "bar chart or graph",
    "scatter plot or line graph",
    "histology slide with H&E staining",
    "anatomical illustration drawing",
    "flowchart or schematic diagram",
    "clinical photograph of a patient",
]

ALL_PROMPTS = TARGET_BRAIN_MRI + TARGET_BRAIN_OTHER + NON_BRAIN_MEDICAL + NOISE_CATEGORIES

# Indices
N_BRAIN_MRI = len(TARGET_BRAIN_MRI)
N_BRAIN_OTHER = len(TARGET_BRAIN_OTHER)
N_NON_BRAIN = len(NON_BRAIN_MEDICAL)
N_NOISE = len(NOISE_CATEGORIES)

IDX_BRAIN_MRI = list(range(0, N_BRAIN_MRI))
IDX_BRAIN_OTHER = list(range(N_BRAIN_MRI, N_BRAIN_MRI + N_BRAIN_OTHER))
IDX_NON_BRAIN = list(range(
    N_BRAIN_MRI + N_BRAIN_OTHER,
    N_BRAIN_MRI + N_BRAIN_OTHER + N_NON_BRAIN,
))
IDX_NOISE = list(range(
    N_BRAIN_MRI + N_BRAIN_OTHER + N_NON_BRAIN,
    len(ALL_PROMPTS),
))

# Decision thresholds
# MedSigLIP sigmoid probs are typically in [0.0005, 0.05] range due to logit_bias=-10.
# We use RELATIVE ratios rather than absolute thresholds.
DEFAULT_ACCEPT_RATIO = 1.5          # target_max / distractor_max ratio for ACCEPT
DEFAULT_REJECT_RATIO = 1.5          # distractor_max / target_max ratio for REJECT
DEFAULT_MIN_TARGET_SCORE = 0.0005   # absolute floor: target must score above this


@dataclass
class MedSigLipResult:
    image_path: str
    category_scores: Dict[str, float]
    top_category: str
    top_score: float
    brain_mri_max: float
    brain_other_max: float  # CT, EEG variants
    non_brain_medical_max: float
    noise_max: float
    target_max: float       # = max(brain_mri_max, brain_other_max)
    distractor_max: float   # = max(non_brain_medical_max, noise_max)
    decision: str           # ACCEPT, REJECT, BORDERLINE
    modality: str           # "MRI", "EEG", "CT", "OTHER"


class MedSigLipClassifier:
    """22-category medical image classifier using MedSigLIP-448."""

    def __init__(
        self,
        model_id: str = "google/medsiglip-448",
        device: str = "cuda:0",
        accept_ratio: float = DEFAULT_ACCEPT_RATIO,
        reject_ratio: float = DEFAULT_REJECT_RATIO,
        min_target_score: float = DEFAULT_MIN_TARGET_SCORE,
    ):
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.accept_ratio = accept_ratio
        self.reject_ratio = reject_ratio
        self.min_target_score = min_target_score

        log.info(f"Loading {model_id} on {device}")
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

        self.logit_scale = self.model.logit_scale.exp().item()
        self.logit_bias = self.model.logit_bias.item() if hasattr(self.model, "logit_bias") else 0.0
        log.info(f"logit_scale={self.logit_scale:.3f}, logit_bias={self.logit_bias:.3f}")

        self._encode_text()

    def _encode_text(self):
        """Pre-encode all 22 prompts."""
        # Transformers 5.x quirk: full forward pass with dummy image needed
        dummy = Image.new("RGB", (448, 448), (128, 128, 128))
        inputs = self.processor(
            text=ALL_PROMPTS,
            images=[dummy] * len(ALL_PROMPTS),
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            self.text_features = outputs.text_embeds
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        log.info(f"Encoded {len(ALL_PROMPTS)} category prompts: {self.text_features.shape}")

    def _classify_pil(self, images: List[Image.Image]) -> np.ndarray:
        """Classify a batch of PIL images. Returns sigmoid probs of shape (N, 22)."""
        inputs = self.processor(
            images=images,
            text=[""] * len(images),
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            img_features = outputs.image_embeds
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            logits = (img_features @ self.text_features.T) * self.logit_scale + self.logit_bias
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def classify_batch(
        self,
        image_paths: List[str],
        batch_size: int = 64,
        progress_every: int = 20,
        num_workers: int = 10,
    ) -> List[MedSigLipResult]:
        """Classify a list of image paths with multi-worker preprocessing.

        Workers do the full image loading + preprocessing (resize, normalize, tensorize)
        so the main thread only does GPU inference. Set num_workers=0 to disable multiprocessing.
        """
        from torch.utils.data import Dataset, DataLoader
        import torch

        # Capture processor reference for use in workers
        processor_ref = self.processor

        class _PreprocessDataset(Dataset):
            """Each worker loads AND preprocesses an image into a tensor."""
            def __init__(self, paths):
                self.paths = paths

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, idx):
                path = self.paths[idx]
                try:
                    img = Image.open(path).convert("RGB")
                    # Full preprocessing in worker
                    pixel = processor_ref.image_processor(
                        images=[img], return_tensors="pt"
                    ).pixel_values[0]  # shape (C, H, W)
                    return idx, path, pixel, True, ""
                except Exception as e:
                    # Return a dummy tensor so collation doesn't fail
                    dummy = torch.zeros(3, 448, 448)
                    return idx, path, dummy, False, str(e)

        def _collate(batch):
            idxs = [b[0] for b in batch]
            paths = [b[1] for b in batch]
            pixels = torch.stack([b[2] for b in batch])
            valid = [b[3] for b in batch]
            errors = [b[4] for b in batch]
            return idxs, paths, pixels, valid, errors

        dataset = _PreprocessDataset(image_paths)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=_collate,
            pin_memory=True,
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )

        total = len(image_paths)
        results_dict = {}
        n_processed = 0

        for batch_idxs, batch_paths, batch_pixels, batch_valid, batch_errors in loader:
            batch_pixels = batch_pixels.to(self.device, non_blocking=True)

            with torch.no_grad():
                # Compute image features directly from pixel values
                img_features = self.model.vision_model(pixel_values=batch_pixels).pooler_output
                # Project if the model has a visual projection layer
                if hasattr(self.model, "visual_projection"):
                    img_features = self.model.visual_projection(img_features)
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                logits = (img_features @ self.text_features.T) * self.logit_scale + self.logit_bias
                probs = torch.sigmoid(logits).cpu().numpy()

            for k in range(len(batch_idxs)):
                idx = batch_idxs[k]
                path = batch_paths[k]
                if batch_valid[k]:
                    results_dict[idx] = self._make_result(path, probs[k])
                else:
                    results_dict[idx] = self._error_result(path, batch_errors[k])

            n_processed += len(batch_idxs)
            if (n_processed // batch_size) % progress_every == 0 or n_processed >= total:
                log.info(f"MedSigLIP classified {n_processed:,}/{total:,}")

        # Return in original order
        results = [results_dict[i] for i in range(total)]
        return results

    def _make_result(self, path: str, probs: np.ndarray) -> MedSigLipResult:
        """Compute decision from sigmoid probabilities."""
        scores = {ALL_PROMPTS[i]: float(probs[i]) for i in range(len(ALL_PROMPTS))}

        brain_mri_max = float(probs[IDX_BRAIN_MRI].max())
        brain_other_max = float(probs[IDX_BRAIN_OTHER].max())
        non_brain_med_max = float(probs[IDX_NON_BRAIN].max())
        noise_max = float(probs[IDX_NOISE].max())

        target_max = max(brain_mri_max, brain_other_max)
        distractor_max = max(non_brain_med_max, noise_max)

        # Decision based on ratio + top-1 category
        top_idx = int(probs.argmax())
        top_cat = ALL_PROMPTS[top_idx]
        top_in_target = top_idx in IDX_BRAIN_MRI or top_idx in IDX_BRAIN_OTHER

        # Avoid division by zero
        eps = 1e-8
        target_to_dist_ratio = target_max / max(distractor_max, eps)
        dist_to_target_ratio = distractor_max / max(target_max, eps)

        if (top_in_target
                and target_max >= self.min_target_score
                and target_to_dist_ratio >= self.accept_ratio):
            decision = "ACCEPT"
        elif (not top_in_target
              and dist_to_target_ratio >= self.reject_ratio):
            decision = "REJECT"
        else:
            decision = "BORDERLINE"

        # Modality from top-1
        if top_idx in IDX_BRAIN_MRI:
            modality = "MRI"
        elif "EEG" in top_cat:
            modality = "EEG"
        elif "CT" in top_cat:
            modality = "CT"
        else:
            modality = "OTHER"

        return MedSigLipResult(
            image_path=path,
            category_scores=scores,
            top_category=top_cat,
            top_score=float(probs[top_idx]),
            brain_mri_max=brain_mri_max,
            brain_other_max=brain_other_max,
            non_brain_medical_max=non_brain_med_max,
            noise_max=noise_max,
            target_max=target_max,
            distractor_max=distractor_max,
            decision=decision,
            modality=modality,
        )

    def _error_result(self, path: str, error: str) -> MedSigLipResult:
        return MedSigLipResult(
            image_path=path,
            category_scores={},
            top_category="ERROR",
            top_score=0.0,
            brain_mri_max=0.0,
            brain_other_max=0.0,
            non_brain_medical_max=0.0,
            noise_max=0.0,
            target_max=0.0,
            distractor_max=0.0,
            decision="REJECT",
            modality="OTHER",
        )


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="Image paths")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    clf = MedSigLipClassifier(device=args.device)
    results = clf.classify_batch(args.images, batch_size=args.batch_size)

    for r in results:
        print(f"\n{Path(r.image_path).name}")
        print(f"  Top: {r.top_category} ({r.top_score:.3f})")
        print(f"  brain_mri={r.brain_mri_max:.3f} brain_other={r.brain_other_max:.3f}")
        print(f"  non_brain_med={r.non_brain_medical_max:.3f} noise={r.noise_max:.3f}")
        print(f"  Decision: {r.decision} | Modality: {r.modality}")
