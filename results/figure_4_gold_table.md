# Table 4. EpiBench benchmark — GOLD macro-F1 per task

We report GOLD macro-F1 across six International League Against Epilepsy (ILAE)
tasks for fifteen baselines, grouped by family.  Trained methods use the same
recipe per family (LoRA $r{=}16$, $\alpha{=}32$, 2 epochs, batch 1 with gradient
accumulation 8, lr $10^{-4}$); vision–language models additionally apply joint
LoRA on the vision tower and cap input panels at six per patient.  **Bold** marks
the per-column winner; the right-most column reports the **EpiBench Score**
(mean of six task macro-F1).

| Family    | Baseline                          | Params | Epilepsy<br>Type | Seizure<br>Type | EZ<br>Loc | AED<br>Resp | Surgery<br>Outcome | Status<br>Epilept. | **EpiBench<br>Score** |
|-----------|-----------------------------------|-------:|-----------------:|-----------------:|----------:|------------:|--------------------:|--------------------:|----------------------:|
| **VLM**   | **Qwen2.5-VL-32B + LoRA**         | 32B    | **0.745**        | 0.640            | **0.770** | 0.470       | **0.690**           | **0.660**           | **0.663**             |
| VLM       | Qwen2.5-VL-7B + LoRA              | 7B     | 0.720            | 0.620            | 0.745     | 0.460       | 0.660               | 0.625               | 0.638                 |
| VLM       | MedGemma-4B-Multi + LoRA          | 4B     | 0.665            | 0.605            | 0.715     | 0.450       | 0.625               | 0.620               | 0.613                 |
| Text-LLM  | Llama-3.1-8B + LoRA               | 8B     | 0.626            | **0.736**        | 0.732     | **0.491**   | 0.605               | 0.510               | 0.617                 |
| Text-LLM  | MedGemma-4B + LoRA                | 4B     | 0.595            | 0.715            | 0.715     | 0.485       | 0.580               | 0.545               | 0.606                 |
| Text-LLM  | GPT-OSS-20B + LoRA                | 20B    | 0.580            | 0.690            | 0.665     | 0.485       | 0.510               | 0.575               | 0.584                 |
| Text-LLM  | Qwen2.5-7B + LoRA                 | 7B     | 0.560            | 0.660            | 0.640     | 0.475       | 0.520               | 0.485               | 0.557                 |
| Fusion    | Late fusion (text + MRI + EEG)    | —      | 0.560            | 0.580            | 0.580     | 0.435       | 0.530               | 0.405               | 0.515                 |
| Text-BERT | BiomedBERT-large                  | 335M   | 0.530            | 0.565            | 0.605     | 0.430       | 0.510               | 0.395               | 0.506                 |
| Text-BERT | PubMedBERT                        | 110M   | 0.520            | 0.555            | 0.595     | 0.420       | 0.500               | 0.385               | 0.496                 |
| Zero-shot | GPT-OSS-120B, 0-shot              | 120B   | 0.475            | 0.410            | 0.510     | 0.380       | 0.405               | 0.335               | 0.419                 |
| Zero-shot | Mistral-3.2-24B, 0-shot           | 24B    | 0.465            | 0.405            | 0.500     | 0.385       | 0.410               | 0.330               | 0.416                 |
| Image-only | MedSigLIP-MRI (MLP)              | 438M   | 0.395            | 0.395            | 0.395     | 0.305       | 0.420               | 0.335               | 0.374                 |
| Image-only | MedSigLIP-EEG (MLP)              | 438M   | 0.380            | 0.355            | 0.360     | 0.295       | 0.330               | 0.295               | 0.336                 |
| Trivial   | TF-IDF + LogReg                   | —      | 0.585            | 0.625            | 0.605     | 0.425       | 0.475               | 0.380               | 0.516                 |

## Headline finding

**Qwen2.5-VL-32B + LoRA** attains an **EpiBench Score of 0.663** on Gold
(n=121–425 per task), a **$\Delta$ +4.6 pp** improvement over the highest-scoring
text-only baseline (Llama-3.1-8B + LoRA, 0.617). The win is concentrated on the
four imaging-grounded tasks where the clinical label depends on visual evidence
the dataset preserves:

- **epilepsy_type** ($\Delta$ +11.9 pp): Magnetic Resonance Imaging (MRI) lesion patterns and EEG
  ictal traces disambiguate focal versus generalised aetiology.
- **ez_localization** ($\Delta$ +3.8 pp): MRI is the diagnostic standard for the seizure focus, and
  EpiBench preserves the linked panels at sub-figure granularity.
- **surgery_outcome** ($\Delta$ +8.5 pp): pre-operative MRI features (mesial temporal sclerosis,
  cortical dysplasia) predict the Engel I outcome rate.
- **status_epilepticus** ($\Delta$ +15.0 pp): non-convulsive status is an electroencephalography
  (EEG) diagnosis; the model leverages the linked EEG panels directly.

Conversely, the **strong text-only Llama-3.1-8B + LoRA** retains the two narrative-driven tasks:
**seizure_type** (semiology described as natural language; $\Delta$ -9.6 pp for the strongest VLM)
and **aed_response** (multi-month anti-seizure-medication trial history; $\Delta$ -2.1 pp).
This pattern validates the **multimodal premise of EpiBench**: the **18,501** linked sub-figures we
extract from the **5,207** PMC case-report papers contribute measurable downstream signal precisely
where ILAE classification depends on imaging, and not where it depends on narrative.

## Reproducibility

Hyperparameters: `max_subfigs=6`, joint vision–language LoRA on
`q_proj, k_proj, v_proj, o_proj, qkv, proj`, gradient checkpointing,
2 epochs, batch 1 with gradient accumulation 8, lr $10^{-4}$, bf16,
on $8\times$ A100-80GB.  Per-task macro-F1 with seed-variance bands and
per-class confusion matrices are released in
`paper/baselines/results/per_task_per_tier_table.csv`.
