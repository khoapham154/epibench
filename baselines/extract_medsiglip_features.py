"""Extract MedSigLIP-448 image features for every linked subfigure in v4.

Reads all unique subfigure paths from `mri_image_paths` + `eeg_image_paths`
columns of `comprehensive_patients_v4_full.xlsx`, runs MedSigLIP-448 vision
encoder once per image, caches L2-normalised pooled features to disk.

Output: features/medsiglip/{path_hash}.npy  -> [feature_dim] float32
        features/medsiglip/manifest.parquet -> (path, hash, modality, dim)

Reuse: train_multimodal.py loads the manifest + cached features; mean-pools per
patient; trains a linear head per task; evaluates per tier.

Run:
    CUDA_VISIBLE_DEVICES=0 python extract_medsiglip_features.py
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

THIS = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("EPIBENCH_ROOT", "."))
V4 = ROOT / "dataset/v4_current/01_combined_full_dataset/comprehensive_patients_v4_full.xlsx"
FEATURES_DIR = THIS / "features" / "medsiglip"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = FEATURES_DIR / "manifest.parquet"


def _parse_paths(s) -> list[str]:
    if pd.isna(s) or s in ("", "[]"):
        return []
    try:
        v = ast.literal_eval(str(s))
        if isinstance(v, list):
            return [str(p) for p in v]
    except Exception:
        pass
    return []


def _hash(path: str) -> str:
    return hashlib.sha1(path.encode()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after N images (debug)")
    args = ap.parse_args()

    print(f"Loading {V4} ...")
    df = pd.read_excel(V4, sheet_name="Patients")
    print(f"  {len(df):,} patients")

    # Collect unique image paths with modality tags
    rows = []
    for _, r in df.iterrows():
        for p in _parse_paths(r.get("mri_image_paths")):
            rows.append({"path": p, "modality": "MRI", "patient_id": f"{r['pmc_id']}_p{r['patient_num']}"})
        for p in _parse_paths(r.get("eeg_image_paths")):
            rows.append({"path": p, "modality": "EEG", "patient_id": f"{r['pmc_id']}_p{r['patient_num']}"})
    pmap = pd.DataFrame(rows)
    pmap["hash"] = pmap["path"].apply(_hash)
    print(f"  {len(pmap):,} (path × patient) edges")
    unique = pmap.drop_duplicates(subset=["path"]).reset_index(drop=True)
    print(f"  {len(unique):,} unique image paths")

    # Skip already-cached
    have = {p.stem for p in FEATURES_DIR.glob("*.npy")}
    todo = unique[~unique["hash"].isin(have)].reset_index(drop=True)
    if args.limit:
        todo = todo.head(args.limit)
    print(f"  {len(todo):,} to extract (already-cached: {len(unique) - len(todo):,})")

    if len(todo) == 0:
        print("Nothing to do.")
    else:
        # Load model
        print("Loading google/medsiglip-448 ...")
        from transformers import AutoModel, AutoProcessor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModel.from_pretrained("google/medsiglip-448", dtype=dtype).to(device).eval()
        proc = AutoProcessor.from_pretrained("google/medsiglip-448")
        print(f"  device={device} dtype={dtype}")

        # Batch through images
        n_failed = 0
        for batch_start in tqdm(range(0, len(todo), args.batch_size), desc="batches"):
            chunk = todo.iloc[batch_start: batch_start + args.batch_size]
            images, hashes = [], []
            for _, r in chunk.iterrows():
                try:
                    img = Image.open(r["path"]).convert("RGB")
                    images.append(img)
                    hashes.append(r["hash"])
                except Exception:
                    n_failed += 1
                    continue
            if not images:
                continue
            try:
                inputs = proc(images=images, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model.get_image_features(**inputs)
                # MedSigLIP-448 returns BaseModelOutputWithPooling; pooler_output is [B, 1152]
                feats = out.pooler_output if hasattr(out, "pooler_output") else out
                feats = torch.nn.functional.normalize(feats, dim=-1).to(torch.float32).cpu().numpy()
                for h, v in zip(hashes, feats):
                    np.save(FEATURES_DIR / f"{h}.npy", v.astype(np.float32))
            except Exception as e:
                print(f"  batch {batch_start} failed: {e}")
                n_failed += len(images)
        print(f"  done. {n_failed} images failed to load.")

    # Build manifest of (path, hash, modality_first_seen, dim)
    have = sorted(FEATURES_DIR.glob("*.npy"))
    if have:
        sample = np.load(have[0])
        dim = int(sample.shape[-1])
    else:
        dim = 0
    manifest_rows = []
    seen = set()
    for _, r in unique.iterrows():
        if r["hash"] in seen:
            continue
        seen.add(r["hash"])
        manifest_rows.append({
            "path": r["path"], "hash": r["hash"],
            "modality": r["modality"], "dim": dim,
            "cached": (FEATURES_DIR / f"{r['hash']}.npy").exists(),
        })
    pd.DataFrame(manifest_rows).to_parquet(MANIFEST, index=False)
    print(f"Wrote {MANIFEST}  ({len(manifest_rows):,} entries, dim={dim})")


if __name__ == "__main__":
    main()
