"""Tier validity audit: re-label a sample of GOLD vs BRONZE patients via a
stricter LLM reviewer prompt and compute Cohen's kappa per (tier x task).

Tests the central dataset claim: "GOLD is higher label-quality than BRONZE."
If kappa(GOLD, reviewer) >> kappa(BRONZE, reviewer), the tier hierarchy is
empirically validated. Otherwise, the tiering is just a count-of-fields
heuristic, not a quality gradient.

Sample: per task, randomly sample N_GOLD=50 GOLD + N_BRONZE=50 BRONZE patients
(or all if fewer). Use Mistral-Small-3.2 (already serving on 8016/8017) as the
reviewer. Sends the full clinical text + age/sex + a strict ILAE prompt.

Output: results/tier_audit.json with kappa per (task, tier).

Run:
    VLLM_PORTS=8016:8017 python tier_audit.py
"""
from __future__ import annotations
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

THIS = Path(__file__).resolve().parent
SPLITS = THIS / "splits"
RESULTS = THIS / "results"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline" / "extraction"))
from llm_client import chat_completion, parallel_map, parse_json  # noqa: E402

LABEL_MAPS = json.loads((THIS / "label_maps_v4.json").read_text())
random.seed(20260428)

# Same ILAE definitions as zero_shot_llm.py — but stricter, one-shot reviewer.
ILAE_DEFS = {
    "epilepsy_type": """ILAE 2017 epilepsy classification:
  - Focal: seizures originating in networks limited to one hemisphere
  - Generalised: seizures originating in bilaterally distributed networks
  - Combined Focal and Generalised: same patient has both
  - Unknown: insufficient information""",
    "seizure_type": """ILAE 2017 seizure classification (drop "onset"):
  - Focal: clear focal onset
  - Generalised: bilateral simultaneous onset
  - Unknown: not observed
  - Unclassified: cannot classify even broadly""",
    "ez_localization": """Epileptogenic zone:
  - Temporal: temporal lobe single-side
  - Extratemporal: single non-temporal lobe
  - Multifocal: bilateral or non-contiguous
  - Hemispheric: full hemisphere
  - Unknown: non-localising""",
    "aed_response": """ILAE 2010 ASM response:
  - drug-resistant: failed >=2 ASMs
  - drug-responsive: seizure-free >=12 mo on ASMs
  - unspecified: response not reported""",
    "surgery_outcome": """Engel:
  - Seizure-free: Engel I
  - Improved: Engel II/III
  - No improvement: Engel IV
  - Not applicable: no surgery""",
    "status_epilepticus": """ILAE 2015 SE:
  - Convulsive SE
  - Non-convulsive SE
  - Refractory SE
  - None: explicitly stated no SE
  - Unknown: not discussed""",
}

N_PER_TIER = 50


def build_text(r) -> str:
    chunks = []
    for fld, lbl in [("semiology_text", "SEMIOLOGY"), ("mri_report_text", "MRI"),
                     ("eeg_report_text", "EEG"), ("demographics_notes", "DEMOGRAPHICS")]:
        v = r.get(fld)
        if pd.notna(v): chunks.append(f"{lbl}: {v}")
    for fld, lbl in [("age", "Age"), ("sex", "Sex")]:
        v = r.get(fld)
        if pd.notna(v): chunks.append(f"{lbl}: {v}")
    return "\n".join(chunks) if chunks else "[no clinical text]"


def review_one(task: str, text: str) -> str | None:
    classes = list(LABEL_MAPS[task].keys())
    sys_p = f"""You are a senior epileptologist re-reviewing a case for ILAE classification. Read the clinical text below and output ONE label.

{ILAE_DEFS[task]}

Output JSON only: {{"label": "<one of: {' | '.join(classes)}>"}}.
If insufficient info, output the most-uncertain class above (e.g. "Unknown")."""
    user = f"### Clinical text\n{text[:4000]}\n\n### Output JSON only."
    try:
        raw = chat_completion(
            [{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
            temperature=0.0, max_tokens=120, response_format_json=True,
        )
        d = parse_json(raw)
        return d.get("label")
    except Exception:
        return None


def audit_task(task: str) -> dict:
    label_col = f"{task}_label_id"
    cls_to_id = LABEL_MAPS[task]

    out: dict = {"task": task}
    for tier_name, csv_name in [("gold", "test_gold.csv"), ("bronze", "test_bronze.csv")]:
        df = pd.read_csv(SPLITS / task / csv_name)
        df = df[df[label_col].notna()].copy()
        n_avail = len(df)
        if n_avail == 0:
            out[tier_name] = {"n_sampled": 0, "kappa": None, "agreement": None}
            continue
        sample = df.sample(n=min(N_PER_TIER, n_avail), random_state=20260428)
        texts = [build_text(r) for _, r in sample.iterrows()]
        ids = sample[label_col].astype(int).values

        print(f"  [{task}/{tier_name}] reviewing {len(texts)}…")
        labels = parallel_map(lambda t: review_one(task, t), texts,
                              max_workers=16, desc=f"{task}/{tier_name}", print_every=20)

        reviewer_ids = []
        for lab in labels:
            if lab is None or lab not in cls_to_id:
                reviewer_ids.append(-1)
            else:
                reviewer_ids.append(cls_to_id[lab])
        ids_arr = np.array(ids); rev_arr = np.array(reviewer_ids)
        valid = rev_arr >= 0
        kappa = cohen_kappa_score(ids_arr[valid], rev_arr[valid]) if valid.any() else None
        agree = float((ids_arr[valid] == rev_arr[valid]).mean()) if valid.any() else None
        out[tier_name] = {
            "n_sampled": int(valid.sum()), "n_invalid": int((~valid).sum()),
            "kappa": float(kappa) if kappa is not None else None,
            "agreement": agree,
        }
        print(f"    {tier_name}: kappa={kappa:.3f} agree={agree:.1%} (n={int(valid.sum())}, invalid={int((~valid).sum())})")
    return out


def main() -> None:
    print(f"VLLM_PORTS={os.environ.get('VLLM_PORTS','default')}")
    rows = []
    for task in LABEL_MAPS:
        print(f"\n=== {task} ===")
        rows.append(audit_task(task))
    out_path = RESULTS / "tier_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))

    # Pretty-print
    summary = []
    for r in rows:
        for tier in ("gold", "bronze"):
            d = r.get(tier, {})
            summary.append({"task": r["task"], "tier": tier,
                            "kappa": d.get("kappa"),
                            "agreement": d.get("agreement"),
                            "n": d.get("n_sampled")})
    print("\n=== Summary ===")
    print(pd.DataFrame(summary).to_markdown(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
