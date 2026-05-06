# EpiBench: Code release for anonymous NeurIPS 2026 submission

This repository contains the data-construction pipeline, baseline training
code, and evaluation scripts for **EpiBench** — a multimodal, multi-task
benchmark for clinical epilepsy management spanning six ILAE-aligned tasks
across three modalities (clinical text, MRI, EEG).

The released dataset is hosted on the anonymous HuggingFace account
[`NeurIPS-1899-ED-2026/EpiBench-NeurIPS2026`](https://huggingface.co/datasets/NeurIPS-1899-ED-2026/EpiBench-NeurIPS2026)
under CC-BY-NC-SA-4.0. This code repository accompanies the submission
and is licensed under MIT.

> **Anonymous notice.** This release is for double-blind peer review.
> Author identities, institutional affiliations, and server-specific paths
> have been removed. All hard-coded paths default to environment
> variables (`EPIBENCH_ROOT`, `EPIBENCH_MONET`).

## Repository layout

```
epibench/
├── pipeline/                # 5-stage neurologist–LLM extraction pipeline
│   ├── extraction/          # Patient extraction from PMC NXML / PDFs
│   │   ├── extract_patients.py     # Entry point
│   │   ├── paper_type_classifier.py
│   │   ├── llm_client.py            # vLLM wrapper for Mistral-Small-3.2-24B
│   │   ├── nxml_loader.py           # JATS XML parsing
│   │   ├── tier_assignment.py       # Gold / Silver / Bronze gating
│   │   ├── ilae_label_maps.py       # Controlled-vocab consolidators
│   │   └── tests/                   # Regression tests for ez_loc consolidator
│   ├── subfigure/           # Multi-panel figure decomposition
│   │   ├── detr_detection.py        # DAB-DETR (pmc-18m-dab-detr)
│   │   ├── medsiglip_classifier.py  # MedSigLIP-448 modality classifier
│   │   ├── llm_subcaption.py        # LLM subcaption extraction
│   │   ├── quality_check.py         # Perceptual hashing + dedup
│   │   ├── generate_dataset.py      # Final subfigure manifest
│   │   ├── config.py
│   │   └── utils.py
│   ├── retrieval/           # PubMed combinatorial retrieval (term lists in docs/taxonomy.md)
│   └── clustering/          # PubMedBERT + FAISS + UMAP + k-means for expert review
├── baselines/               # 15 reported baselines (paper §4)
│   ├── prepare_splits.py            # PMC-disjoint splits
│   ├── train_text.py                # TF-IDF + LogReg, BiomedBERT, PubMedBERT
│   ├── train_text_lora.py           # LoRA on text LLMs (Llama-3.1-8B, MedGemma-4B, GPT-OSS-20B, Qwen2.5-7B)
│   ├── train_llama8b.py             # Llama-3.1-8B-specific entry point with 3-seed loop
│   ├── train_vlm.py                 # Joint vision–language LoRA (Qwen2.5-VL-32B/7B, MedGemma-4B-Multi)
│   ├── train_late_fusion.py         # Multimodal late-fusion hybrid
│   ├── train_image_mlp.py           # MedSigLIP-MRI / EEG MLP heads
│   ├── extract_medsiglip_features.py
│   ├── zero_shot_llm.py             # GPT-OSS-120B / Mistral-3.2-24B zero-shot
│   ├── populated_helper.py          # Controlled-vocab populated decoding
│   ├── aggregate_results.py         # Build per-task per-tier results table
│   ├── tier_audit.py                # Cohen's kappa tier audit (Appendix F)
│   ├── modality_ablation.py         # Six-regime modality ablation (Appendix F)
│   └── splits/                      # Per-task (patient_id, pmc_id, fold) manifests
├── figures/
│   └── render_fig04_benchmark.py    # Reproduces the Figure 4 GOLD heatmap
├── results/                 # Latest results referenced by the paper
│   ├── per_task_per_tier_table.csv  # 46-baseline grid; 15-baseline subset is in §E
│   ├── per_task_per_tier_table.md   # Same, markdown view
│   ├── modality_ablation.csv        # Appendix F modality ablation
│   ├── tier_audit.json              # Appendix F tier validity audit
│   └── figure_4_gold_table.md       # Authoritative Figure 4 numbers
├── docs/
│   ├── taxonomy.md                  # 284-term curated ILAE taxonomy (Appendix A)
│   ├── prompts.md                   # 5 LLM prompts (Appendix B)
│   ├── schema.md                    # Output JSON schema (Appendix C)
│   └── controlled_vocabulary.md     # Per-task controlled vocab (Appendix C)
├── images/                  # Figures 1–4 from the paper, for offline rendering
├── requirements.txt
└── LICENSE
```

## Quick start

### Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=<hf-token>          # For gated models (Llama, MedGemma)
export EPIBENCH_ROOT=$(pwd)         # Paths default to ./
```

### Reproduce the benchmark on the released dataset

```bash
# 1. Pull the dataset
huggingface-cli download NeurIPS-1899-ED-2026/EpiBench-NeurIPS2026 \
  --repo-type dataset --local-dir ./dataset/v4_current

# 2. Build the splits (PMC-disjoint train/silver/gold/bronze per task)
python baselines/prepare_splits.py

# 3. Run a baseline (example: TF-IDF + LogReg on epilepsy_type)
python baselines/train_text.py \
  --task epilepsy_type \
  --baseline tfidf \
  --seed 42

# 4. Aggregate per-task per-tier results
python baselines/aggregate_results.py
```

### Reproduce the extraction pipeline (compute-heavy)

The full pipeline takes ≈88 wall-clock hours on 8× A100-80GB to extract
**42,259 patient candidates** from **169,121 unique PMC papers**, of which
**25,737** survive the tier-assignment gate.

```bash
# Serve Mistral-Small-3.2-24B on 8 vLLM instances (one per GPU)
bash pipeline/extraction/serve_vllm.sh

# Run extraction
python pipeline/extraction/extract_patients.py \
  --pmc_list ./pmc_lists/all_pmcs.csv \
  --out_dir  ./profiles
```

### Reproduce Figure 4

```bash
python figures/render_fig04_benchmark.py
# writes figures/output/figure_4_benchmark_heatmap.{png,pdf}
```

## The 15 reported baselines

Per paper §4, the following 15 baselines feed Figure 4. **Bold** = the
EpiBench Score winner per family.

| Family | Baseline | Params | Script |
|---|---|--:|---|
| VLM + LoRA | **Qwen2.5-VL-32B + LoRA** | 32B | `baselines/train_vlm.py` |
| VLM + LoRA | Qwen2.5-VL-7B + LoRA | 7B | `baselines/train_vlm.py` |
| VLM + LoRA | MedGemma-4B-Multi + LoRA | 4B | `baselines/train_vlm.py` |
| Text LLM + LoRA | **Llama-3.1-8B + LoRA** | 8B | `baselines/train_llama8b.py` |
| Text LLM + LoRA | MedGemma-4B + LoRA | 4B | `baselines/train_text_lora.py` |
| Text LLM + LoRA | GPT-OSS-20B + LoRA | 20B | `baselines/train_text_lora.py` |
| Text LLM + LoRA | Qwen2.5-7B + LoRA | 7B | `baselines/train_text_lora.py` |
| Late fusion | Late fusion (text+MRI+EEG) | — | `baselines/train_late_fusion.py` |
| Text encoder | BiomedBERT-large | 335M | `baselines/train_text.py --baseline biomedbert_large` |
| Text encoder | PubMedBERT-base | 110M | `baselines/train_text.py --baseline pubmedbert` |
| Zero-shot LLM | GPT-OSS-120B (zero-shot) | 120B | `baselines/zero_shot_llm.py --model gpt-oss-120b` |
| Zero-shot LLM | Mistral-3.2-24B (zero-shot) | 24B | `baselines/zero_shot_llm.py --model mistral-3.2-24b` |
| Image (frozen) | MedSigLIP-MRI + MLP | 438M | `baselines/train_image_mlp.py --modality mri` |
| Image (frozen) | MedSigLIP-EEG + MLP | 438M | `baselines/train_image_mlp.py --modality eeg` |
| Sparse text | TF-IDF + LogReg | — | `baselines/train_text.py --baseline tfidf` |

The headline result on the Gold tier (3-seed mean macro-F1, $n=121$–$425$
per task): **Qwen2.5-VL-32B + LoRA = 0.663 EpiBench Score**, beating the
strongest text-only baseline (Llama-3.1-8B + LoRA, 0.617) by **+4.6 pp**
and winning 4 of 6 ILAE tasks. Llama-3.1-8B + LoRA retains the two
narrative-driven tasks (`seizure_type`, `aed_response`).

See `results/figure_4_gold_table.md` for the full per-task table.

## License

- **Code**: MIT (this repository)
- **Dataset**: CC-BY-NC-SA-4.0 (released separately on HuggingFace)
- **Models**: each baseline obeys its upstream license; LLM checkpoints
  (Llama, Qwen, Mistral, MedGemma, GPT-OSS) are downloaded from
  HuggingFace under their respective terms.

## Citation

Withheld for double-blind review. Once the paper is accepted, this README
will carry a BibTeX entry for the EpiBench paper.
