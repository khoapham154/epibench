# Figure rendering

`render_fig04_benchmark.py` reproduces the GOLD benchmark heatmap shown
as Figure 4 of the paper. It renders the 15-baseline `(task, baseline)`
matrix using the team-verified 3-seed-mean values committed in
`../results/figure_4_gold_table.md`.

```bash
python render_fig04_benchmark.py
# outputs:
#   output/figure_4_benchmark_heatmap.png
#   output/figure_4_benchmark_heatmap.pdf
```

Figures 1, 2, and 3 in the paper (Overview, Pipeline, Case-study) are
included as static PNGs in `../images/` and are not regenerated from
code, since they involve manually composed panels (e.g. the highlighted
PMC5320722 case-study reading).
