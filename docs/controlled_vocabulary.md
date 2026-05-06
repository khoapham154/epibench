# ILAE-compliant controlled vocabulary per task

Mirrors Appendix C of the paper. The released vocabulary is the subset of
classes that remain populated after post-extraction consolidation. During
extraction, the LLM is given a broader vocabulary; consolidation rules in
`pipeline/extraction/ilae_label_maps.py` then map free-text labels onto
the canonical classes below.

| Task | Standard | Classes |
|---|---|---|
| `epilepsy_type` | ILAE 2017 | Focal \| Generalised \| Combined Focal and Generalised \| Unknown |
| `seizure_type` | ILAE 2017 | Focal \| Generalised \| Unknown \| Unclassified |
| `ez_localization` | — | Temporal \| Extratemporal \| Multifocal \| Hemispheric \| Unknown |
| `aed_response` | ILAE 2010 | drug-responsive (≥12 mo seizure-free on ASMs) \| drug-resistant (failed ≥2 adequate ASM trials) \| unspecified |
| `surgery_outcome` | Engel mapping | Seizure-free (Engel I) \| Improved (Engel II/III) |
| `status_epilepticus` | ILAE 2015 | Refractory SE \| Non-convulsive SE \| Unknown |

## Notes on consolidation

- For `surgery_outcome`, the original LLM vocabulary includes
  *No improvement (Engel IV)* and *Not applicable*, but these two classes
  were merged or removed when consolidation rules consistently failed to
  align with the LLM's output phrasings.
- For `status_epilepticus`, the original LLM vocabulary includes
  *Convulsive SE* and *None*, similarly removed.
- For `ez_localization`, sublobar Opercular / Frontobasal / Cingulate are
  consolidated into *Extratemporal*; bare *Left* / *Right* without lobe
  resolve to *Unknown*; bilateral or two-lobe compounds resolve to
  *Multifocal*. See `tests/test_ilae_consolidators.py` for the 13
  regression cases.
