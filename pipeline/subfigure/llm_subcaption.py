#!/usr/bin/env python3
"""
Step 4: LLM Subcaption Extraction — FAST VERSION

Key speedups over the original:
  1. REGEX FIRST: ~70% of subcaptions are parseable with regex (zero GPU cost)
  2. GROUP BY FIGURE: one LLM call per figure, not per subfigure
     (if a figure has 6 panels A-F, that's 1 call instead of 6)
  3. SMALLER MODEL: Qwen2.5-7B-Instruct is more than enough for text parsing
     (~10x faster than 72B, fits on 1 GPU)
  4. CUDA GRAPHS ENABLED: enforce_eager=False now that custom_all_reduce is
     disabled via env var (2x throughput)
  5. LARGER BATCHES: vLLM can batch many short prompts efficiently

Expected: ~35k subfigures in 1-2 hours instead of 95 hours.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# ==============================================================================
# ENV FIXES — must be before any torch/vLLM imports
# ==============================================================================
os.environ["VLLM_DISABLE_CUSTOM_ALL_REDUCE"] = "1"
os.environ.setdefault("NCCL_IB_DISABLE", "1")

if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
    val = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", val)

try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    print("⚠️ vLLM not available — using regex-only extraction")


# ==============================================================================
# REGEX-BASED SUBCAPTION PARSER (handles ~70% of cases, zero GPU cost)
# ==============================================================================

# Common caption patterns:
#   "(A) T1-weighted MRI showing lesion. (B) FLAIR sequence..."
#   "A: Axial MRI. B: Coronal view."
#   "A, preoperative MRI; B, postoperative MRI"
#   "(a-c) MRI sequences. (d) EEG recording."

SUBCAPTION_PATTERNS = [
    # (A) description. or (A) description;
    re.compile(
        r'\(([A-Za-z](?:\s*[-\u2013\u2014,&]\s*[A-Za-z])*)\)\s*[:\s]*(.+?)(?=\([A-Za-z](?:\s*[-\u2013\u2014,&]\s*[A-Za-z])*\)|$)',
        re.IGNORECASE | re.DOTALL
    ),
    # A: description. B: description. or A) description
    re.compile(
        r'(?:^|(?<=\.\s)|(?<=;\s)|(?<=\n))([A-Za-z](?:\s*[-\u2013\u2014,&]\s*[A-Za-z])*)[):]\s*(.+?)(?=(?:^|(?<=\.\s)|(?<=;\s)|(?<=\n))[A-Za-z](?:\s*[-\u2013\u2014,&]\s*[A-Za-z])*[):]|$)',
        re.IGNORECASE | re.DOTALL
    ),
    # A, description; B, description
    re.compile(
        r'(?:^|(?<=;\s)|(?<=\.\s))([A-Za-z]),\s*(.+?)(?=(?:;\s*)[A-Za-z],|$)',
        re.IGNORECASE | re.DOTALL
    ),
]


def _expand_label_range(label_str: str) -> List[str]:
    """Expand label ranges like 'A-C' or 'A, B' into individual labels."""
    label_str = label_str.strip()
    if len(label_str) == 1:
        return [label_str.upper()]
    range_match = re.match(r'^([A-Za-z])\s*[-\u2013\u2014]\s*([A-Za-z])$', label_str)
    if range_match:
        start, end = range_match.group(1).upper(), range_match.group(2).upper()
        return [chr(c) for c in range(ord(start), ord(end) + 1)]
    parts = re.split(r'[,&]\s*', label_str)
    return [p.strip().upper() for p in parts if len(p.strip()) == 1]


def regex_extract_subcaptions(caption: str) -> Dict[str, str]:
    """
    Try to extract per-label subcaptions using regex.
    Returns dict: {"A": "subcaption text", "B": "...", ...}
    Returns empty dict if parsing fails.
    """
    if not caption or not isinstance(caption, str):
        return {}

    caption = caption.strip()
    results = {}

    for pattern in SUBCAPTION_PATTERNS:
        matches = list(pattern.finditer(caption))
        if len(matches) >= 2:
            for m in matches:
                labels = _expand_label_range(m.group(1))
                text = m.group(2).strip().rstrip('.;,')
                for label in labels:
                    if label not in results:
                        results[label] = text
            break

    return results


def classify_modality_from_text(text: str) -> str:
    """Fast keyword-based modality classification with word boundaries."""
    import re
    t = text.lower()
    if re.search(r'\bflair\b', t):
        return 'MRI_FLAIR'
    if re.search(r'\bt1\b', t) and re.search(r'\bweight|[-\s]w\b|[-\s]wi\b', t):
        return 'MRI_T1_WEIGHTED'
    if re.search(r'\bt2\b', t) and re.search(r'\bweight|[-\s]w\b|[-\s]wi\b', t):
        return 'MRI_T2_WEIGHTED'
    if re.search(r'\bdwi\b|\bdiffusion\b', t):
        return 'MRI_DWI_ADC'
    if re.search(r'\badc\b', t) and re.search(r'\bmap\b|\bimag', t):
        return 'MRI_DWI_ADC'
    if re.search(r'\bgadolinium\b|contrast-enhanced|post-contrast|\bgd-', t):
        return 'MRI_CONTRAST'
    if re.search(r'\bswi\b|\bsusceptibility\b', t):
        return 'MRI_OTHER'
    if re.search(r'\bmri\b|\bmr\s|\bmagnetic\s+resonance\b', t):
        return 'MRI_OTHER'
    if re.search(r'\bictal\b', t) and not re.search(r'\binterictal\b', t):
        return 'EEG_ICTAL'
    if re.search(r'\binterictal\b', t):
        return 'EEG_INTERICTAL'
    if re.search(r'\bintracranial\b|\bdepth\s+electrode\b|\bseeg\b|\bieeg\b|\becog\b|\bsubdural\b', t):
        return 'EEG_INTRACRANIAL'
    if re.search(r'\beeg\b|\belectroencephalog', t):
        return 'EEG_SCALP'
    return 'UNKNOWN'


# ==============================================================================
# LLM-BASED EXTRACTION (for the ~30% regex can't handle)
# ==============================================================================

BATCH_SUBCAPTION_PROMPT = """Extract subcaptions from this figure caption. The figure has subfigures labeled: {labels}

CAPTION:
{caption}

For each subfigure label, extract its specific description and classify the imaging modality.

OUTPUT FORMAT (JSON only, no other text):
{{
  {label_template}
}}"""

LABEL_ENTRY_TEMPLATE = '"{label}": {{"subcaption": "description for {label}", "modality": "MRI_T1_WEIGHTED|MRI_T2_WEIGHTED|MRI_FLAIR|MRI_DWI_ADC|MRI_CONTRAST|MRI_OTHER|EEG_SCALP|EEG_ICTAL|EEG_INTERICTAL|EEG_INTRACRANIAL|EEG_OTHER|UNKNOWN"}}'


def _build_llm_prompt(caption: str, labels: List[str]) -> str:
    label_entries = ",\n  ".join(
        LABEL_ENTRY_TEMPLATE.format(label=l) for l in labels
    )
    return BATCH_SUBCAPTION_PROMPT.format(
        labels=", ".join(labels),
        caption=caption,
        label_template=label_entries,
    )


def _parse_llm_response(response: str, labels: List[str]) -> Dict[str, Dict]:
    results = {}
    try:
        brace_start = response.find('{')
        brace_end = response.rfind('}')
        if brace_start >= 0 and brace_end > brace_start:
            data = json.loads(response[brace_start:brace_end + 1])
            for label in labels:
                if label in data and isinstance(data[label], dict):
                    results[label] = data[label]
                elif label.lower() in data and isinstance(data[label.lower()], dict):
                    results[label] = data[label.lower()]
    except json.JSONDecodeError:
        for label in labels:
            pattern = rf'"{label}"\s*:\s*\{{([^}}]+)\}}'
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    results[label] = json.loads('{' + match.group(1) + '}')
                except json.JSONDecodeError:
                    pass
    return results


class FastLLMExtractor:
    """Lightweight vLLM wrapper — smaller model, CUDA graphs enabled."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 16384,
    ):
        self.llm = None
        self.sampling_params = None

        if not HAS_VLLM:
            return

        tp_attempts = _build_tp_attempts(tensor_parallel_size)

        for tp in tp_attempts:
            try:
                print(f"🔧 Loading {model_id} with tp={tp}...")
                self.llm = LLM(
                    model=model_id,
                    tensor_parallel_size=tp,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_model_len=max_model_len,
                    trust_remote_code=True,
                    enforce_eager=False,              # CUDA graphs ON (2x faster)
                    disable_custom_all_reduce=True,   # disable buggy kernel
                )
                print(f"✅ Loaded {model_id} (tp={tp})")
                break
            except Exception as e:
                print(f"❌ tp={tp} failed: {e}")
                self.llm = None
                _cleanup_gpu()

        if self.llm is None:
            print("⚠️ LLM init failed — regex-only mode")

        self.sampling_params = SamplingParams(
            temperature=0.05,
            top_p=0.9,
            max_tokens=2048,
            stop=["```"],
        ) if HAS_VLLM else None

    def batch_extract(
        self,
        figure_groups: List[Tuple[str, List[str]]],
    ) -> Dict[str, Dict[str, Dict]]:
        """
        Process multiple figures in one vLLM batch call.

        Args:
            figure_groups: List of (caption, [labels]) tuples
        Returns:
            Dict mapping caption -> {label: {subcaption, modality}}
        """
        if self.llm is None:
            return {}

        prompts = []
        captions = []
        label_lists = []

        for caption, labels in figure_groups:
            prompts.append(_build_llm_prompt(caption, labels))
            captions.append(caption)
            label_lists.append(labels)

        try:
            outputs = self.llm.generate(prompts, self.sampling_params)
        except Exception as e:
            print(f"⚠️ vLLM generate failed: {e}")
            return {}

        results = {}
        for caption, labels, output in zip(captions, label_lists, outputs):
            response = output.outputs[0].text.strip()
            parsed = _parse_llm_response(response, labels)
            results[caption] = parsed

        return results


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

@dataclass
class SubcaptionResult:
    subfigure_path: str
    source_image: str
    subfigure_label: str
    subcaption: str
    modality: str
    findings: List[str] = field(default_factory=list)
    anatomical_region: Optional[str] = None
    intext_references: List[Dict] = field(default_factory=list)
    confidence: float = 0.5

    def to_dict(self) -> Dict:
        return {
            'subfigure_path': self.subfigure_path,
            'source_image': self.source_image,
            'subfigure_label': self.subfigure_label,
            'subcaption': self.subcaption,
            'modality': self.modality,
            'findings': self.findings,
            'anatomical_region': self.anatomical_region,
            'intext_references': self.intext_references,
            'confidence': self.confidence,
        }


def _index_to_label(index: int) -> str:
    if index < 26:
        return chr(ord('A') + index)
    return f"Panel{index + 1}"


def run_subcaption_extraction(
    input_csv: str,
    output_dir: str,
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    batch_size: int = 128,
    tensor_parallel_size: int = 1,
) -> pd.DataFrame:
    """
    Two-pass extraction:
      Pass 1 (CPU): Regex parses ~70% of subcaptions instantly
      Pass 2 (GPU): LLM handles remaining ~30%, grouped by figure
    """
    print(f"\nReading input: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Total subfigures: {len(df):,}")

    # ------------------------------------------------------------------
    # PASS 1: Regex extraction (instant, no GPU)
    # ------------------------------------------------------------------
    print("\n[Pass 1] Regex subcaption extraction...")

    figure_groups = defaultdict(list)  # caption -> [(row_idx, label)]
    for idx, row in df.iterrows():
        caption = row.get('caption', '') or ''
        label = _index_to_label(row.get('subfigure_index', 0))
        figure_groups[caption].append((idx, label))

    print(f"  Unique figures: {len(figure_groups):,} "
          f"(avg {len(df)/max(len(figure_groups),1):.1f} subfigures/figure)")

    results = {}  # row_idx -> SubcaptionResult
    regex_hits = 0
    regex_misses_by_figure = []  # (caption, [(row_idx, label)])

    for caption, subfigs in tqdm(figure_groups.items(), desc="Regex parsing"):
        parsed = regex_extract_subcaptions(caption)
        labels_for_llm = []

        for row_idx, label in subfigs:
            row = df.loc[row_idx]
            if label in parsed:
                subcap = parsed[label]
                modality = classify_modality_from_text(subcap)
                if modality == 'UNKNOWN':
                    modality = classify_modality_from_text(caption)

                results[row_idx] = SubcaptionResult(
                    subfigure_path=row.get('subfigure_path', ''),
                    source_image=row.get('source_image', row.get('figure_path', '')),
                    subfigure_label=label,
                    subcaption=subcap,
                    modality=modality,
                    confidence=0.8,
                )
                regex_hits += 1
            else:
                labels_for_llm.append((row_idx, label))

        if labels_for_llm:
            regex_misses_by_figure.append((caption, labels_for_llm))

    regex_pct = regex_hits / len(df) * 100 if len(df) > 0 else 0
    llm_needed = len(df) - regex_hits
    print(f"  ✅ Regex extracted: {regex_hits:,} / {len(df):,} ({regex_pct:.1f}%)")
    print(f"  Remaining for LLM: {llm_needed:,} subfigures "
          f"across {len(regex_misses_by_figure):,} figures")

    # ------------------------------------------------------------------
    # PASS 2: LLM extraction for regex failures (grouped by figure)
    # ------------------------------------------------------------------
    if llm_needed > 0 and HAS_VLLM:
        print(f"\n[Pass 2] LLM extraction ({model_id})...")

        extractor = FastLLMExtractor(
            model_id=model_id,
            tensor_parallel_size=tensor_parallel_size,
        )

        if extractor.llm is not None:
            figure_batch = []
            batch_meta = []

            for caption, subfig_list in tqdm(
                regex_misses_by_figure, desc="Preparing LLM batches"
            ):
                labels = [label for _, label in subfig_list]
                figure_batch.append((caption, labels))
                batch_meta.append((caption, subfig_list))

                if len(figure_batch) >= batch_size:
                    _process_llm_batch(
                        extractor, figure_batch, batch_meta, df, results
                    )
                    figure_batch.clear()
                    batch_meta.clear()

            if figure_batch:
                _process_llm_batch(
                    extractor, figure_batch, batch_meta, df, results
                )

            del extractor
            _cleanup_gpu()

    # ------------------------------------------------------------------
    # Fill remaining gaps with fallback
    # ------------------------------------------------------------------
    for idx, row in df.iterrows():
        if idx not in results:
            caption = row.get('caption', '') or ''
            label = _index_to_label(row.get('subfigure_index', 0))
            modality = classify_modality_from_text(caption)
            results[idx] = SubcaptionResult(
                subfigure_path=row.get('subfigure_path', ''),
                source_image=row.get('source_image', row.get('figure_path', '')),
                subfigure_label=label,
                subcaption=caption,
                modality=modality,
                confidence=0.3,
            )

    # ------------------------------------------------------------------
    # Build output
    # ------------------------------------------------------------------
    all_results = [results[idx] for idx in range(len(df))]
    results_df = pd.DataFrame([r.to_dict() for r in all_results])

    merged_df = df.copy()
    for col in ['subcaption', 'modality', 'findings', 'anatomical_region', 'confidence']:
        if col in results_df.columns:
            merged_df[col] = results_df[col].values

    print(f"\n{'='*60}")
    print("SUBCAPTION EXTRACTION RESULTS")
    print(f"{'='*60}")
    print(f"Total subfigures: {len(all_results):,}")
    print(f"  Regex-extracted:  {sum(1 for r in all_results if r.confidence >= 0.7):,}")
    print(f"  LLM-extracted:    {sum(1 for r in all_results if 0.5 <= r.confidence < 0.7):,}")
    print(f"  Fallback:         {sum(1 for r in all_results if r.confidence < 0.5):,}")
    print(f"\nModality distribution:")
    print(results_df['modality'].value_counts().to_string())

    output_csv = os.path.join(output_dir, 'subfigure_subcaptions.csv')
    merged_df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved: {output_csv}")

    results_json = os.path.join(output_dir, 'subfigure_subcaptions.json')
    with open(results_json, 'w') as f:
        json.dump([r.to_dict() for r in all_results], f, indent=2)
    print(f"✅ Saved: {results_json}")

    return merged_df


def _process_llm_batch(
    extractor: FastLLMExtractor,
    figure_batch: List[Tuple[str, List[str]]],
    batch_meta: List[Tuple[str, List[Tuple[int, str]]]],
    df: pd.DataFrame,
    results: Dict[int, SubcaptionResult],
):
    """Send a batch of figures to the LLM and store results."""
    llm_results = extractor.batch_extract(figure_batch)

    valid_modalities = {
        'MRI_T1_WEIGHTED', 'MRI_T2_WEIGHTED', 'MRI_FLAIR',
        'MRI_DWI_ADC', 'MRI_CONTRAST', 'MRI_OTHER',
        'EEG_SCALP', 'EEG_ICTAL', 'EEG_INTERICTAL',
        'EEG_INTRACRANIAL', 'EEG_OTHER', 'UNKNOWN',
    }

    for caption, subfig_list in batch_meta:
        parsed = llm_results.get(caption, {})
        for row_idx, label in subfig_list:
            row = df.loc[row_idx]
            if label in parsed:
                entry = parsed[label]
                subcap = entry.get('subcaption', caption)
                modality = entry.get('modality', 'UNKNOWN')
                if modality not in valid_modalities:
                    modality = classify_modality_from_text(subcap)

                results[row_idx] = SubcaptionResult(
                    subfigure_path=row.get('subfigure_path', ''),
                    source_image=row.get('source_image', row.get('figure_path', '')),
                    subfigure_label=label,
                    subcaption=subcap,
                    modality=modality,
                    confidence=0.6,
                )


# ==============================================================================
# HELPERS
# ==============================================================================

def _build_tp_attempts(requested_tp: int) -> List[int]:
    attempts = []
    tp = requested_tp
    while tp >= 1:
        attempts.append(tp)
        tp //= 2
    if attempts[-1] != 1:
        attempts.append(1)
    return attempts


def _cleanup_gpu():
    try:
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ==============================================================================
# CLI  (same interface as original — drop-in replacement)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fast LLM subcaption extraction")
    parser.add_argument('--keyword', required=True)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct',
                        help='LLM model (default: 7B — fast enough for parsing)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Figures per LLM batch (not subfigures)')
    parser.add_argument('--tensor-parallel', type=int, default=1,
                        help='GPUs for tensor parallel (1 is fine for 7B)')
    parser.add_argument('--force', action='store_true')

    args = parser.parse_args()

    output_csv = os.path.join(args.input_dir, 'subfigure_subcaptions.csv')
    if os.path.exists(output_csv) and not args.force:
        print(f"Output exists: {output_csv} — use --force to reprocess")
        return

    input_csv = os.path.join(args.input_dir, 'subfigure_detections.csv')
    if not os.path.exists(input_csv):
        print(f"❌ Not found: {input_csv}")
        sys.exit(1)

    tensor_parallel = args.tensor_parallel
    if 'NUM_GPUS' in os.environ:
        env_gpus = int(os.environ['NUM_GPUS'])
        if '7B' in args.model or '7b' in args.model:
            tensor_parallel = min(env_gpus, 1)
            print(f"ℹ️  7B model: using tp=1 (ignoring NUM_GPUS={env_gpus})")
        else:
            tensor_parallel = env_gpus

    run_subcaption_extraction(
        input_csv=input_csv,
        output_dir=args.input_dir,
        model_id=args.model,
        batch_size=args.batch_size,
        tensor_parallel_size=tensor_parallel,
    )


if __name__ == '__main__':
    main()