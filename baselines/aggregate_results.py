"""Aggregate v4 baseline results into per-tier comparison table.

Reads baselines/results/{baseline}/*.json and produces:
  results/per_task_per_tier_table.md   — 3 baselines × 6 tasks × 4 tiers (72 cells)
  results/per_task_per_tier_table.csv  — same in CSV
  results/summary.json                 — best-baseline-per-tier headlines
  ../content/fig06_data.csv            — heatmap source for Figure 6
  ../content/06_section06_results.md   — overwritten with new per-tier prose
"""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

THIS = Path(__file__).resolve().parent
RESULTS = THIS / "results"
CONTENT = THIS.parent / "content"
CONTENT.mkdir(parents=True, exist_ok=True)

TASKS = ["epilepsy_type", "seizure_type", "ez_localization",
         "aed_response", "surgery_outcome", "status_epilepticus"]
# EpiBench v4 final benchmark uses 3 tiers (Tier-D / Expansion was dropped
# in round 4 due to label-quality concerns; see PAPER_CONTENT.md §3).
TIERS = ["silver", "gold", "bronze"]
TIER_LABEL = {"silver": "SILVER (held-out, PMC-disjoint)", "gold": "GOLD",
              "bronze": "BRONZE"}

BASELINE_ORDER = [
    ("tfidf", "TF-IDF + LogReg"),
    ("pubmedbert", "PubMedBERT-base"),
    ("pubmedbert_strong", "PubMedBERT-base (strong)"),
    ("bio_clinicalbert", "Bio_ClinicalBERT"),
    ("bio_clinicalbert_strong", "Bio_ClinicalBERT (strong)"),
    ("biomedbert_large", "BiomedBERT-large"),
    ("biomedbert_large_strong", "BiomedBERT-large (strong)"),
    ("text_ensemble", "Text-encoder ensemble (3-model strong)"),
    ("llama31_8b", "Llama-3.1-8B + LoRA"),
    ("deberta_v3_large", "DeBERTa-v3-large"),
    ("medsiglip_mri", "MedSigLIP-MRI (LR)"),
    ("medsiglip_eeg", "MedSigLIP-EEG (LR)"),
    ("medsiglip_mri_mlp", "MedSigLIP-MRI (MLP)"),
    ("medsiglip_eeg_mlp", "MedSigLIP-EEG (MLP)"),
    ("biomedclip_mri", "BiomedCLIP-MRI (LR)"),
    ("biomedclip_eeg", "BiomedCLIP-EEG (LR)"),
    ("biomedclip_mri_mlp", "BiomedCLIP-MRI (MLP)"),
    ("biomedclip_eeg_mlp", "BiomedCLIP-EEG (MLP)"),
    ("late_fusion", "Late fusion (text+MRI+EEG)"),
    ("pmb_late_fusion", "PMB-text + MedSigLIP late fusion"),
    ("joint_mlp_pmb_medsiglip", "Joint MLP (PMB+MedSigLIP)"),
    ("joint_mlp_pmb_biomedclip", "Joint MLP (PMB+BiomedCLIP)"),
    ("mistral_zero_shot", "Mistral-3.2-24B 0-shot+RAG"),
    ("mistral_fewshot_k3", "Mistral-3.2-24B few-shot k=3"),
    ("llama4_scout_zero_shot", "Llama-4-Scout-17B-16E 0-shot+RAG"),
    ("gpt_oss_120b_zero_shot", "GPT-OSS-120B 0-shot+RAG (v1)"),
    ("gpt_oss_120b_v2", "GPT-OSS-120B 0-shot+RAG (v2: strict+tolerant)"),
    # New round-5 LLM additions
    ("llama31_8b_zero_shot", "Llama-3.1-8B 0-shot+RAG"),
    ("qwen25_7b_zero_shot", "Qwen2.5-7B 0-shot+RAG"),
    ("medgemma_27b_zero_shot", "MedGemma-27B 0-shot+RAG"),
    ("gpt_oss_20b_zero_shot", "GPT-OSS-20B 0-shot+RAG"),
    ("qwen25_7b_lora", "Qwen2.5-7B + LoRA"),
    ("medgemma_4b_lora", "MedGemma-4B + LoRA"),
    ("gpt_oss_20b_lora", "GPT-OSS-20B + LoRA"),
    # Round-6 additions: populated-only zero-shot + attention-pool MedSigLIP + learned fusion
    ("medsiglip_mri_attn", "MedSigLIP-MRI (attn-pool)"),
    ("medsiglip_eeg_attn", "MedSigLIP-EEG (attn-pool)"),
    ("learned_fusion_v1", "Learned fusion (Llama-8B + MedSigLIP)"),
    ("mistral_zero_shot_populated", "Mistral-3.2-24B 0-shot+RAG (populated)"),
    ("qwen25_7b_zero_shot_populated", "Qwen2.5-7B 0-shot+RAG (populated)"),
    ("medgemma_27b_zero_shot_populated", "MedGemma-27B 0-shot+RAG (populated)"),
    ("llama4_scout_zs_populated", "Llama-4-Scout-17B-16E 0-shot+RAG (populated)"),
    ("gpt_oss_120b_zs_populated", "GPT-OSS-120B 0-shot+RAG (populated)"),
    # Round-7: vision-encoder LoRA + multimodal-LLM LoRA
    ("medsiglip_mri_vision_lora", "MedSigLIP-MRI (vision-LoRA)"),
    ("medsiglip_eeg_vision_lora", "MedSigLIP-EEG (vision-LoRA)"),
    ("medgemma_4b_multi_lora", "MedGemma-4B-Multi + LoRA"),
    ("qwen25_vl_7b_lora", "Qwen2.5-VL-7B + LoRA"),
]

# Baselines whose results are saved as one file per (task, tier) instead of one per (task, seed).
PER_TIER_BASELINES = {"mistral_zero_shot", "mistral_fewshot_k3",
                      "llama4_scout_zero_shot", "gpt_oss_120b_zero_shot",
                      "gpt_oss_120b_v2",
                      "llama31_8b_zero_shot", "qwen25_7b_zero_shot",
                      "medgemma_27b_zero_shot", "gpt_oss_20b_zero_shot",
                      # Populated-only zero-shot variants (round 6)
                      "mistral_zero_shot_populated",
                      "qwen25_7b_zero_shot_populated",
                      "medgemma_27b_zero_shot_populated",
                      "llama4_scout_zs_populated",
                      "gpt_oss_120b_zs_populated"}

# Strong recipe (8ep + class weights + 1024 ctx) reuses the underlying model dir
# but its files end in `_strong.json`. Map the synthetic baseline id back to the
# real dir, and tell the loader to filter for `*_strong.json` only.
STRONG_BASELINES = {
    "pubmedbert_strong": "pubmedbert",
    "bio_clinicalbert_strong": "bio_clinicalbert",
    "biomedbert_large_strong": "biomedbert_large",
}


def _load_baseline(baseline: str) -> dict[str, dict]:
    """Return {task: merged_dict_with_per_tier_keys}.

    For fine-tune baselines that have multiple seeds, return the MEAN macro-F1
    across all seeds (so seed-variance is averaged out and a single failing seed
    doesn't poison the headline number).
    """
    out: dict[str, dict] = {}
    real_dir = STRONG_BASELINES.get(baseline, baseline)
    base_dir = RESULTS / real_dir
    if not base_dir.exists():
        return {}

    is_strong = baseline in STRONG_BASELINES

    if baseline in PER_TIER_BASELINES:
        for f in sorted(base_dir.glob("*.json")):
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            task = d.get("task", f.stem.split("_")[0])
            tier = d.get("tier", "gold")
            entry = out.setdefault(task, {"task": task, "baseline": baseline})
            entry[f"macro_f1_{tier}"] = d.get(f"macro_f1_{tier}")
            entry[f"n_test_{tier}"] = d.get(f"n_test_{tier}")
        return out

    # Fine-tune baselines: aggregate across seeds (mean per tier).
    # The strong/non-strong filter only applies when the baseline is one of the
    # paired ones (pubmedbert vs pubmedbert_strong etc). Other baselines that
    # store *_strong files (e.g. text_ensemble) are accepted as-is.
    paired_baselines = set(STRONG_BASELINES.values()) | set(STRONG_BASELINES.keys())
    apply_strong_filter = baseline in paired_baselines
    by_task: dict[str, list[dict]] = {}
    for f in sorted(base_dir.glob("*.json")):
        is_strong_file = f.stem.endswith("_strong")
        is_regime_file = "_regime" in f.stem
        if is_regime_file:
            continue  # cross-tier scaling files — not part of headline tables
        if apply_strong_filter:
            if is_strong and not is_strong_file:
                continue
            if (not is_strong) and is_strong_file:
                continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        task = d.get("task", f.stem.split("_seed")[0])
        by_task.setdefault(task, []).append(d)

    for task, runs in by_task.items():
        merged: dict = {"task": task, "baseline": baseline,
                        "n_seeds": len(runs),
                        "seeds": [r.get("config", {}).get("seed") for r in runs]}
        for tier in ["silver", "gold", "bronze", "expansion"]:
            vals = [r.get(f"macro_f1_{tier}") for r in runs
                    if r.get(f"macro_f1_{tier}") is not None]
            n_tests = [r.get(f"n_test_{tier}") for r in runs
                       if r.get(f"n_test_{tier}") is not None]
            merged[f"macro_f1_{tier}"] = (sum(vals) / len(vals)) if vals else None
            merged[f"macro_f1_{tier}_per_seed"] = vals
            merged[f"n_test_{tier}"] = n_tests[0] if n_tests else 0
        out[task] = merged
    return out


def main() -> None:
    # Build a long-form DataFrame: (baseline, task, tier) → macro-F1
    rows: list[dict] = []
    for baseline_id, display in BASELINE_ORDER:
        per_task = _load_baseline(baseline_id)
        for task in TASKS:
            d = per_task.get(task) or {}
            for tier in TIERS:
                rows.append({
                    "baseline": display,
                    "task": task,
                    "tier": tier,
                    "macro_f1": d.get(f"macro_f1_{tier}"),
                    "n_test": d.get(f"n_test_{tier}", 0),
                })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "per_task_per_tier_table.csv", index=False)

    # Wide table: rows = baseline, cols = (task × tier)
    wide = df.pivot_table(index="baseline", columns=["task", "tier"], values="macro_f1")
    # Reorder
    wide = wide[[(t, tier) for t in TASKS for tier in TIERS]]
    wide.to_csv(RESULTS / "per_task_per_tier_table_wide.csv")

    # Markdown table — split into per-tier sub-tables for readability
    lines = ["# Per-task macro-F1 across 4 tiers (v4 only)", "",
             "Rows = baseline, columns = task. One sub-table per tier.", ""]
    for tier in TIERS:
        lines.append(f"## {TIER_LABEL[tier]}")
        lines.append("")
        sub = df[df["tier"] == tier].pivot(index="baseline", columns="task", values="macro_f1")
        sub = sub[TASKS]
        # Add EpiBench Score column (mean across tasks)
        sub["EpiBench Score"] = sub.mean(axis=1)
        lines.append(sub.map(lambda v: f"{v:.3f}" if pd.notna(v) else "—").to_markdown())
        lines.append("")
    md = "\n".join(lines)
    (RESULTS / "per_task_per_tier_table.md").write_text(md + "\n")

    # Per-tier EpiBench Scores (mean across tasks)
    summary: dict = {"per_tier_epibench": {}, "per_baseline_per_tier": {}}
    for tier in TIERS:
        sub = df[df["tier"] == tier]
        per_b = sub.groupby("baseline")["macro_f1"].mean()
        summary["per_baseline_per_tier"][tier] = per_b.to_dict()
        # Best baseline this tier
        best_b = per_b.idxmax() if not per_b.empty else None
        summary["per_tier_epibench"][tier] = {
            "best_baseline": best_b,
            "best_score": float(per_b.max()) if best_b else None,
        }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))

    # Heatmap CSV for Figure 6
    heatmap = df.pivot_table(index="baseline", columns=["task", "tier"], values="macro_f1")
    heatmap = heatmap[[(t, tier) for t in TASKS for tier in TIERS]]
    heatmap.to_csv(CONTENT / "fig06_data.csv")

    print(md)
    print("\nWrote:")
    print(f"  {RESULTS / 'per_task_per_tier_table.md'}")
    print(f"  {RESULTS / 'per_task_per_tier_table.csv'}")
    print(f"  {RESULTS / 'summary.json'}")
    print(f"  {CONTENT / 'fig06_data.csv'}")


if __name__ == "__main__":
    main()
