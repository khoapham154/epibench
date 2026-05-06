"""Zero-shot LLM baseline: Mistral-Small-3.2-24B on the 6 EpiBench tasks.

For each test patient (GOLD tier), build a prompt with:
  - clinical narrative (semiology + MRI + EEG + demographics)
  - ILAE controlled vocabulary for the task (retrieved from PROMPT_DEFINITIONS)
  - "Choose one of: ..." instruction
  - JSON output format

Hits 4 vLLM servers running Mistral-Small-3.2 on ports 8014-8017 (GPUs 4-7).
NOTE: this assumes the LLM extraction servers are NOT running on those ports;
if they are (during full-corpus extraction), we use ports 8010-8013 instead.

Use:
    VLLM_PORTS=8014:8017 python zero_shot_mistral.py --task epilepsy_type
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import classification_report, confusion_matrix, f1_score

THIS = Path(__file__).resolve().parent
RESULTS = THIS / "results"

# Reuse the production llm_client (HTTP-only)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline" / "extraction"))
from llm_client import chat_completion, parse_json  # noqa: E402

LABEL_MAPS = json.loads((THIS / "label_maps_v4.json").read_text())
LABEL_MAPS_POPULATED = json.loads((THIS / "label_maps_v4_populated.json").read_text())
LABEL_MAPS_POPULATED.pop("_meta", None)


# ---------- Tolerant label parser ----------
# Maps common spelling variants / synonyms produced by LLMs back to the
# canonical label vocabulary. Applied after exact match fails.
SYNONYMS = {
    "epilepsy_type": {
        "generalized": "Generalised",
        "generalised": "Generalised",
        "focal": "Focal",
        "combined focal and generalized": "Combined Focal and Generalised",
        "combined focal and generalised": "Combined Focal and Generalised",
        "combined": "Combined Focal and Generalised",
        "unknown": "Unknown",
    },
    "seizure_type": {
        "focal": "Focal",
        "focal onset": "Focal",
        "generalized": "Generalised",
        "generalised": "Generalised",
        "generalized onset": "Generalised",
        "unknown": "Unknown",
        "unknown onset": "Unknown",
        "unclassified": "Unclassified",
    },
    "ez_localization": {
        "temporal": "Temporal",
        "extratemporal": "Extratemporal",
        "extra-temporal": "Extratemporal",
        "extra temporal": "Extratemporal",
        "frontal": "Extratemporal",
        "parietal": "Extratemporal",
        "occipital": "Extratemporal",
        "insular": "Extratemporal",
        "multifocal": "Multifocal",
        "multi-focal": "Multifocal",
        "bilateral": "Multifocal",
        "hemispheric": "Hemispheric",
        "unknown": "Unknown",
        "non-localising": "Unknown",
        "non-localizing": "Unknown",
    },
    "aed_response": {
        "drug-resistant": "drug-resistant",
        "drug resistant": "drug-resistant",
        "drug-refractory": "drug-resistant",
        "refractory": "drug-resistant",
        "drug-responsive": "drug-responsive",
        "drug responsive": "drug-responsive",
        "responsive": "drug-responsive",
        "seizure-free": "drug-responsive",
        "seizure free": "drug-responsive",
        "unspecified": "unspecified",
        "unknown": "unspecified",
    },
    "surgery_outcome": {
        "seizure-free": "Seizure-free",
        "seizure free": "Seizure-free",
        "engel i": "Seizure-free",
        "engel 1": "Seizure-free",
        "improved": "Improved",
        "engel ii": "Improved",
        "engel iii": "Improved",
        "no improvement": "No improvement",
        "engel iv": "No improvement",
        "not applicable": "Not applicable",
        "no surgery": "Not applicable",
    },
    "status_epilepticus": {
        "convulsive se": "Convulsive SE",
        "gcse": "Convulsive SE",
        "non-convulsive se": "Non-convulsive SE",
        "ncse": "Non-convulsive SE",
        "refractory se": "Refractory SE",
        "rse": "Refractory SE",
        "none": "None",
        "no se": "None",
        "unknown": "Unknown",
    },
}


def _normalize(s: str) -> str:
    return "".join(c.lower() for c in s.strip() if c.isalnum() or c in (" ", "-"))


def parse_label(task: str, raw: str | None) -> str | None:
    """Tolerant label match. Returns canonical class name or None if not parseable."""
    if not raw:
        return None
    cls_to_id = LABEL_MAPS[task]
    classes = list(cls_to_id.keys())
    # Exact match first
    if raw in cls_to_id:
        return raw
    norm = _normalize(raw)
    # Synonym lookup (lowercase, alnum-stripped)
    for k, v in SYNONYMS.get(task, {}).items():
        if _normalize(k) == norm:
            return v if v in cls_to_id else None
    # Substring contains a class name (e.g. "Generalised tonic-clonic")
    for cls in classes:
        if _normalize(cls) in norm:
            return cls
    # Synonym contained
    for k, v in SYNONYMS.get(task, {}).items():
        if _normalize(k) in norm and v in cls_to_id:
            return v
    return None

# ILAE definitions for each task — used as RAG context
ILAE_DEFS = {
    "epilepsy_type": """ILAE 2017 epilepsy classification:
  - "Focal": seizures originating in networks limited to one hemisphere
  - "Generalised": seizures originating in bilaterally distributed networks
  - "Combined Focal and Generalised": same patient has both types
  - "Unknown": insufficient information to determine""",
    "seizure_type": """ILAE 2017 seizure classification (drop the "onset" suffix):
  - "Focal": clear focal onset (motor / non-motor; aware / impaired awareness)
  - "Generalised": bilateral simultaneous onset (tonic-clonic, absence, myoclonic, atonic)
  - "Unknown": onset not observed/witnessed but evidence indicates one of the above
  - "Unclassified": insufficient information to classify even broadly""",
    "ez_localization": """Epileptogenic zone localisation:
  - "Temporal": temporal lobe (mesial or lateral), single lobe only
  - "Extratemporal": single non-temporal lobe (frontal/parietal/occipital/insular) or sublobar
  - "Multifocal": bilateral OR two or more non-contiguous lobes
  - "Hemispheric": entire hemisphere or large hemispheric region
  - "Unknown": stated to be uncertain or non-localising""",
    "aed_response": """ILAE 2010 drug-resistance definition:
  - "drug-resistant": failed adequate trials of >=2 tolerated, appropriately chosen ASMs
  - "drug-responsive": seizure-free for >=12 months on current ASMs
  - "unspecified": on treatment but response not reported / cannot be classified""",
    "surgery_outcome": """Engel surgical outcome:
  - "Seizure-free": Engel class I (free of disabling seizures)
  - "Improved": Engel class II or III (rare disabling seizures / worthwhile improvement)
  - "No improvement": Engel class IV (no worthwhile improvement)
  - "Not applicable": no surgery performed""",
    "status_epilepticus": """ILAE 2015 SE classification:
  - "Convulsive SE": generalized tonic-clonic SE, GCSE
  - "Non-convulsive SE": NCSE with/without impaired awareness, absence SE
  - "Refractory SE": SE continuing despite first-line benzodiazepine + one ASM
  - "None": paper explicitly states no history/episode of SE
  - "Unknown": SE status not discussed""",
}


def build_input_text(r) -> str:
    chunks = []
    if pd.notna(r.get("semiology_text")):
        chunks.append(f"SEMIOLOGY: {r['semiology_text']}")
    if pd.notna(r.get("mri_report_text")):
        chunks.append(f"MRI: {r['mri_report_text']}")
    if pd.notna(r.get("eeg_report_text")):
        chunks.append(f"EEG: {r['eeg_report_text']}")
    if pd.notna(r.get("demographics_notes")):
        chunks.append(f"DEMOGRAPHICS: {r['demographics_notes']}")
    if pd.notna(r.get("age")):
        chunks.append(f"Age: {r['age']}")
    if pd.notna(r.get("sex")):
        chunks.append(f"Sex: {r['sex']}")
    return "\n".join(chunks) if chunks else "[no clinical text available]"


def _ilae_def_filtered(task: str, classes: list[str]) -> str:
    """Return the ILAE definition for `task`, but only listing lines that mention
    one of the allowed classes (so the populated-only prompt doesn't tell the model
    about empty classes like 'No improvement' or 'Convulsive SE')."""
    full = ILAE_DEFS[task].split("\n")
    head, body = full[0], full[1:]
    keep = []
    for line in body:
        # Each definition line starts with `  - "ClassName"`; keep iff the class is allowed
        kept = any(f'"{c}"' in line for c in classes)
        if kept:
            keep.append(line)
    return "\n".join([head, *keep])


def predict_one(task: str, patient_text: str, *,
                max_tokens: int = 200, strict_prompt: bool = False,
                populated_only: bool = False) -> str | None:
    cls_map = LABEL_MAPS_POPULATED[task] if populated_only else LABEL_MAPS[task]
    classes = list(cls_map.keys())
    ilae = _ilae_def_filtered(task, classes) if populated_only else ILAE_DEFS[task]
    extra = ""
    if strict_prompt:
        extra = (f"\nThe value of \"label\" MUST be exactly one of these strings (copy verbatim): "
                 f"{' | '.join(repr(c) for c in classes)}.")
    sys_prompt = f"""You are a clinical epilepsy classifier. Given a patient's clinical text, output ONE label from the controlled vocabulary.

{ilae}

Output JSON only: {{"label": "<one of: {' | '.join(classes)}>", "rationale": "<one sentence>"}}.
Do NOT output any other text.{extra}"""
    user = f"PATIENT CLINICAL TEXT:\n{patient_text[:3500]}\n\nClassify the patient for task '{task}'. Output JSON."
    msgs = [{"role": "system", "content": sys_prompt},
            {"role": "user", "content": user}]
    try:
        raw = chat_completion(msgs, temperature=0.0, max_tokens=max_tokens, response_format_json=True)
        d = parse_json(raw)
        return d.get("label")
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--tier", default="gold", choices=["silver", "gold", "bronze", "expansion"])
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--baseline_name", required=True,
                    help="Subfolder under results/, e.g. llama4_scout_zero_shot")
    ap.add_argument("--model_label", required=True,
                    help="Model name string saved to JSON (display only)")
    ap.add_argument("--max_tokens", type=int, default=200)
    ap.add_argument("--strict_prompt", action="store_true",
                    help="Reinforce 'output exactly one of: ...' in the system prompt.")
    ap.add_argument("--populated_only", action="store_true",
                    help="Restrict prompt vocabulary + scoring to populated classes only "
                         "(drop empty classes like 'No improvement', 'Not applicable', "
                         "'Convulsive SE', 'None'). Macro-F1 is then computed over "
                         "populated class IDs only.")
    args = ap.parse_args()

    test = pd.read_csv(THIS / "splits" / args.task / f"test_{args.tier}.csv")
    if args.limit:
        test = test.head(args.limit)
    if len(test) == 0:
        print(f"[skip] no test data for {args.task} / tier={args.tier}")
        return

    cls_map_active = LABEL_MAPS_POPULATED[args.task] if args.populated_only else LABEL_MAPS[args.task]
    classes = list(cls_map_active.keys())
    cls_to_id = cls_map_active

    # Build all input texts
    texts = [build_input_text(r) for _, r in test.iterrows()]
    yte = test[f"{args.task}_label_id"].astype(int).values

    print(f"=== {args.baseline_name} / {args.task} / {args.tier} / n={len(test)} ===")
    t0 = time.time()

    def _one(i_text):
        i, txt = i_text
        return i, predict_one(args.task, txt,
                              max_tokens=args.max_tokens,
                              strict_prompt=args.strict_prompt,
                              populated_only=args.populated_only)

    preds_label: list[str | None] = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_one, (i, txt)): i for i, txt in enumerate(texts)}
        done = 0
        for fut in as_completed(futures):
            i, label = fut.result()
            preds_label[i] = label
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(texts)}")

    # Map predictions to ids using TOLERANT parser; count true unparseable.
    # Under --populated_only: an unparseable prediction is mapped to the most-frequent
    # populated class so we don't artificially default to class id 0 (which may have
    # been excluded under populated-only).
    pop_label_set = set(cls_to_id.values())
    fallback_id = sorted(pop_label_set)[0] if pop_label_set else 0
    pred_ids = []
    n_invalid = 0
    for raw_label in preds_label:
        canon = parse_label(args.task, raw_label)
        # Under populated-only, only accept predictions whose canonical id is populated
        if canon is None or (args.populated_only and LABEL_MAPS[args.task].get(canon) not in pop_label_set):
            n_invalid += 1
            pred_ids.append(fallback_id)
        else:
            # parse_label() returns the full-vocab name; remap through populated map if needed
            full_id = LABEL_MAPS[args.task][canon]
            pred_ids.append(full_id)

    if args.populated_only:
        pop_ids_list = sorted(pop_label_set)
        macro_f1 = float(f1_score(yte, pred_ids, labels=pop_ids_list, average="macro", zero_division=0))
        report = classification_report(yte, pred_ids, labels=pop_ids_list, output_dict=True, zero_division=0)
        cm = confusion_matrix(yte, pred_ids, labels=pop_ids_list).tolist()
    else:
        macro_f1 = float(f1_score(yte, pred_ids, average="macro", zero_division=0))
        report = classification_report(yte, pred_ids, output_dict=True, zero_division=0)
        cm = confusion_matrix(yte, pred_ids).tolist()
    elapsed = time.time() - t0

    res = {
        "task": args.task, "baseline": args.baseline_name,
        "tier": args.tier,
        "n_train": 0, f"n_test_{args.tier}": int(len(test)),
        f"macro_f1_{args.tier}": macro_f1,
        "n_invalid_predictions": int(n_invalid),
        "elapsed_seconds": elapsed,
        f"per_class_{args.tier}": report,
        f"confusion_matrix_{args.tier}": cm,
        "config": {"model": args.model_label,
                   "ports": os.environ.get("VLLM_PORTS", "unset"),
                   "workers": args.workers, "temperature": 0.0},
    }

    out = RESULTS / args.baseline_name / f"{args.task}_{args.tier}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(res, f, indent=2)

    # Save raw + canonicalised label predictions for later rescoring / calibration.
    preds_dir = RESULTS / args.baseline_name / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    pred_records = [{"i": i, "raw": raw, "canon_id": int(pred_ids[i])}
                    for i, raw in enumerate(preds_label)]
    (preds_dir / f"{args.task}_{args.tier}.json").write_text(json.dumps(pred_records, indent=2))

    print(f"  macro-F1 (GOLD): {macro_f1:.4f}")
    print(f"  invalid predictions: {n_invalid}/{len(test)} ({100*n_invalid/len(test):.1f}%)")
    print(f"  elapsed: {elapsed:.0f}s")
    print(f"  saved: {out}")


if __name__ == "__main__":
    main()
