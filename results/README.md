# Results

Latest evaluation outputs aligned with the paper.

| File | Source | Used in |
|---|---|---|
| `figure_4_gold_table.md` | Team-verified 3-seed-mean Gold numbers | Figure 4, Appendix E (Table tab:gold_full) |
| `per_task_per_tier_table.csv` | Canonical aggregator (`baselines/aggregate_results.py`) | Appendix E Silver/Bronze tables, supplementary |
| `per_task_per_tier_table.md` | Same as the CSV, markdown view | Quick browsing |
| `modality_ablation.csv` | `baselines/modality_ablation.py` | Appendix F (Table tab:modality) |
| `tier_audit.json` | `baselines/tier_audit.py` | Appendix F (Table tab:tier_audit) |

The canonical CSV holds **46 entries** spanning every baseline run,
including ablation variants (`-strong`, `-plain`, `-3-seed`) that are not
in the 15 reported baselines. The 15-baseline subset matching the paper
figures is the one in `figure_4_gold_table.md`.
