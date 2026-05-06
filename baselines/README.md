# Baselines

The 15 reported baselines from paper §4, plus the splits generator,
results aggregator, and the two appendix analyses (tier audit, modality
ablation).

## Splits

`prepare_splits.py` builds the four PMC-disjoint folds described in
Appendix D:

- **Train**: 90% of Silver PMCs
- **test_silver**: held-out 10% Silver PMCs (in-distribution)
- **test_gold**: Gold tier with leak-PMCs dropped (primary benchmark)
- **test_bronze**: Bronze tier with leak-PMCs dropped (cross-tier)

The folds are reproduced deterministically via
`GroupShuffleSplit(groups=pmc_id, test_size=0.10, random_state=42)`.

For convenience, `splits/<task>.csv` carries a compact
`(patient_id, pmc_id, fold)` manifest per task. Join against the released
HuggingFace dataset to recover the full clinical text.

## Per-baseline scripts

| Baseline | Script | Notes |
|---|---|---|
| TF-IDF + LogReg | `train_text.py --baseline tfidf` | CPU only, ≈12 min total |
| BiomedBERT-large | `train_text.py --baseline biomedbert_large` | 3-seed |
| PubMedBERT-base | `train_text.py --baseline pubmedbert` | 3-seed |
| Llama-3.1-8B + LoRA | `train_llama8b.py` | 3-seed loop, 8 GPUs |
| MedGemma-4B + LoRA | `train_text_lora.py --model google/medgemma-4b-it` | 3-seed |
| GPT-OSS-20B + LoRA | `train_text_lora.py --model openai-community/gpt-oss-20b` | 3-seed |
| Qwen2.5-7B + LoRA | `train_text_lora.py --model Qwen/Qwen2.5-7B-Instruct` | 3-seed |
| Late fusion (text+MRI+EEG) | `train_late_fusion.py` | Uses MedSigLIP features |
| MedSigLIP-MRI + MLP | `train_image_mlp.py --modality mri` | Frozen encoder + MLP head |
| MedSigLIP-EEG + MLP | `train_image_mlp.py --modality eeg` | Frozen encoder + MLP head |
| Qwen2.5-VL-32B + LoRA | `train_vlm.py --model Qwen/Qwen2.5-VL-32B-Instruct` | Joint vision + language LoRA |
| Qwen2.5-VL-7B + LoRA | `train_vlm.py --model Qwen/Qwen2.5-VL-7B-Instruct` | Joint vision + language LoRA |
| MedGemma-4B-Multi + LoRA | `train_vlm.py --model google/medgemma-4b-it --vision` | Multimodal mode |
| GPT-OSS-120B (zero-shot) | `zero_shot_llm.py --model openai-community/gpt-oss-120b --populated` | Controlled-vocab populated decoding |
| Mistral-3.2-24B (zero-shot) | `zero_shot_llm.py --model mistralai/Mistral-Small-3.2-24B-Instruct-2506 --populated` | |

Shared LoRA recipe: `r=16`, `α=32`, 2 epochs, batch 1 with gradient
accumulation 8, learning rate `1e-4`, bf16. VLMs additionally apply joint
LoRA on the vision tower (`q_proj, k_proj, v_proj, o_proj, qkv, proj`)
with up to 6 sub-figure panels per patient.

## Aggregation and analyses

```bash
python aggregate_results.py
# writes ../results/per_task_per_tier_table.{csv,md}

python tier_audit.py     # Cohen's kappa tier validity audit (Appendix F.1)
python modality_ablation.py   # Six-regime modality ablation (Appendix F.2)
```
