# Pipeline

Five-stage neurologist–LLM pipeline for constructing EpiBench from open
PubMed Central case reports and clinical textbooks. Mirrors paper §3.

| Stage | Directory | What it does |
|---|---|---|
| 1. Term curation | `../docs/taxonomy.md` | 284 ILAE-aligned search terms across 6 categories, curated by expert neurologists |
| 2. Source retrieval | `retrieval/` | Combinatorial query expansion against PMC Open Access |
| 3. Patient extraction | `extraction/` | Multi-stage LLM extraction (paper-type gate, individual-patient identification, table parsing, patient linking, ground-truth structuring) |
| 3b. Subfigure decomposition | `subfigure/` | DAB-DETR multi-panel splitting + MedSigLIP-448 modality classification + BiomedCLIP caption co-validation |
| 4. Clustering for review | `clustering/` | PubMedBERT embeddings + FAISS + UMAP + k-means (KneeLocator) for stratified expert sampling |
| 5. Iterative refinement | (manual) | Five rounds of neurologist review feeding prompt revisions in `../docs/prompts.md` |

## Running the extraction

The full pipeline expects a vLLM endpoint serving Mistral-Small-3.2-24B
on each of 8 GPUs. Hardware footprint matches paper §G: ≈88 wall-clock
hours on 8× A100-80GB.

```bash
# 1. Spin up vLLM (one instance per GPU, ports 8010–8017)
bash extraction/serve_vllm.sh

# 2. Build the PMC list (or fetch the released list)
python retrieval/pubmed_retrieval.py --terms ../docs/taxonomy.md \
    --out ../pmc_lists/all_pmcs.csv

# 3. Run the per-paper extractor
python extraction/extract_patients.py \
    --pmc_list  ../pmc_lists/all_pmcs.csv \
    --vllm_base http://localhost:8010 \
    --out_dir   ../profiles
```

Each paper produces a `final_profiles.json` matching the schema in
`../docs/schema.md`. After all papers are processed, the tier assignment
in `extraction/tier_assignment.py` produces the Gold/Silver/Bronze gates
described in paper §3.5.
