#!/usr/bin/env python3
"""
Step 2: Enhanced Image Quality Check (INTEGRATED WITH EXISTING PIPELINE)

This version integrates:
1. Your existing clean_duplicate_renditions.py logic
2. Additional quality checks (size, corruption, format)
3. Perceptual hashing for near-duplicate detection
4. Quality metrics reporting

Features:
- Size validation (min/max dimensions, file size)
- Format validation (supported image formats)
- Corruption detection (try to load images)
- Duplicate rendition removal (same stem, different extensions)
- Perceptual hash-based duplicate detection
- Quality reporting
"""

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import pandas as pd
from tqdm import tqdm

try:
    from PIL import Image
    import imagehash
    import warnings
    # Suppress PIL warnings for corrupt TIFF metadata and large images
    warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)
    warnings.filterwarnings('ignore', message='.*TIFF.*')
    HAS_IMAGING = True
except ImportError:
    HAS_IMAGING = False
    print("⚠️ PIL/imagehash not available, some checks disabled")


# ==============================================================================
# CONFIGURATION
# ==============================================================================

PRIORITY_EXTENSIONS = {
    ".jpg": 0, ".jpeg": 0,
    ".png": 1,
    ".tif": 2, ".tiff": 2,
    ".bmp": 3,
    ".gif": 4,
}

SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.tiff', '.tif', '.bmp'}


@dataclass
class QualityMetrics:
    """Quality metrics for a single image."""
    image_path: str
    file_exists: bool = False
    file_size_mb: float = 0.0
    width: int = 0
    height: int = 0
    format: str = ""
    mode: str = ""
    is_corrupt: bool = False
    is_too_small: bool = False
    is_too_large: bool = False
    is_wrong_format: bool = False
    is_duplicate_rendition: bool = False
    perceptual_hash: str = ""
    passes_quality: bool = False
    failure_reason: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


# ==============================================================================
# DUPLICATE RENDITION DETECTION (From your existing code)
# ==============================================================================

def find_duplicate_renditions(
    image_paths: List[Path],
    root: Path
) -> Dict[str, List[Path]]:
    """
    Find duplicate renditions (same stem, different extensions).

    Returns dict mapping stem -> list of paths
    """
    by_stem = defaultdict(list)

    for path in image_paths:
        if path.suffix.lower() in SUPPORTED_FORMATS:
            # Get stem relative to a common parent
            try:
                rel_path = path.relative_to(root)
                parent = rel_path.parent
                stem = path.stem
                key = str(parent / stem)
            except ValueError:
                key = path.stem

            by_stem[key].append(path)

    # Filter to only groups with multiple extensions
    duplicates = {}
    for stem, paths in by_stem.items():
        exts = {p.suffix.lower() for p in paths}
        if len(exts) > 1:
            duplicates[stem] = paths

    return duplicates


def get_best_rendition(paths: List[Path]) -> Path:
    """Get best rendition based on extension priority."""
    def ext_priority(p: Path) -> int:
        return PRIORITY_EXTENSIONS.get(p.suffix.lower(), 99)

    return sorted(paths, key=ext_priority)[0]


# ==============================================================================
# PERCEPTUAL HASH DUPLICATE DETECTION
# ==============================================================================

def compute_perceptual_hash(image_path: Path) -> Optional[str]:
    """Compute perceptual hash for image."""
    if not HAS_IMAGING:
        return None

    try:
        img = Image.open(image_path)
        phash = imagehash.phash(img, hash_size=8)
        return str(phash)
    except Exception:
        return None


def find_perceptual_duplicates(
    image_metrics: List[QualityMetrics],
    threshold: int = 5
) -> Dict[str, List[str]]:
    """
    Find perceptually identical images using exact hash matching.

    Uses O(n) exact hash grouping instead of O(n^2) pairwise comparison.
    Returns dict mapping representative -> list of duplicates
    """
    if not HAS_IMAGING:
        return {}

    # Group by exact hash - O(n)
    by_hash = defaultdict(list)
    for metric in image_metrics:
        if metric.perceptual_hash:
            by_hash[metric.perceptual_hash].append(metric.image_path)

    # Only keep groups with duplicates (exact match)
    duplicates = {}
    for hash_val, paths in by_hash.items():
        if len(paths) > 1:
            duplicates[paths[0]] = paths[1:]

    return duplicates


# ==============================================================================
# QUALITY CHECKS
# ==============================================================================

def check_single_image(
    args: Tuple[str, int, int, float, float]
) -> QualityMetrics:
    """
    Check quality of a single image.

    Args:
        (image_path, min_size, max_size, min_file_size_mb, max_file_size_mb)
    """
    image_path, min_size, max_size, min_file_mb, max_file_mb = args

    metrics = QualityMetrics(image_path=image_path)
    path = Path(image_path)

    # Check file exists
    if not path.exists():
        metrics.failure_reason = "File not found"
        return metrics

    metrics.file_exists = True

    # Check file size
    try:
        file_size_bytes = path.stat().st_size
        metrics.file_size_mb = file_size_bytes / (1024 * 1024)

        if metrics.file_size_mb < min_file_mb:
            metrics.is_too_small = True
            metrics.failure_reason = f"File too small: {metrics.file_size_mb:.2f}MB < {min_file_mb}MB"
            return metrics

        if metrics.file_size_mb > max_file_mb:
            metrics.is_too_large = True
            metrics.failure_reason = f"File too large: {metrics.file_size_mb:.2f}MB > {max_file_mb}MB"
            return metrics
    except Exception as e:
        metrics.failure_reason = f"File size check failed: {e}"
        return metrics

    # Check format
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        metrics.is_wrong_format = True
        metrics.failure_reason = f"Unsupported format: {path.suffix}"
        return metrics

    # Try to load image
    if HAS_IMAGING:
        try:
            # Fast dimension check without loading full image
            with Image.open(path) as img:
                metrics.width, metrics.height = img.size
                metrics.format = img.format or ""
                metrics.mode = img.mode or ""

                # Check dimensions early
                if metrics.width < min_size or metrics.height < min_size:
                    metrics.is_too_small = True
                    metrics.failure_reason = f"Image too small: {metrics.width}x{metrics.height} < {min_size}"
                    return metrics

                if metrics.width > max_size or metrics.height > max_size:
                    metrics.is_too_large = True
                    metrics.failure_reason = f"Image too large: {metrics.width}x{metrics.height} > {max_size}"
                    return metrics

                # Reject decompression bombs (>89M pixels)
                pixel_count = metrics.width * metrics.height
                if pixel_count > 89_478_485:  # PIL's safety limit
                    metrics.is_too_large = True
                    metrics.failure_reason = f"Image too large: {pixel_count:,} pixels (potential decompression bomb)"
                    return metrics

                # Only compute perceptual hash for valid images
                metrics.perceptual_hash = compute_perceptual_hash(path) or ""

        except Exception as e:
            metrics.is_corrupt = True
            metrics.failure_reason = f"Image corrupt or unreadable: {e}"
            return metrics

    # Passed all checks
    metrics.passes_quality = True
    return metrics


# ==============================================================================
# MAIN QUALITY CHECK PIPELINE
# ==============================================================================

def run_quality_check(
    input_csv: Path,
    output_dir: Path,
    min_size: int = 100,
    max_size: int = 4096,
    min_file_mb: float = 0.001,
    max_file_mb: float = 50.0,
    workers: int = 32,
    remove_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Run comprehensive quality checks on images.

    Returns filtered DataFrame with quality images only.
    """
    print(f"\n{'='*60}")
    print("STEP 2: IMAGE QUALITY CHECK")
    print(f"{'='*60}\n")

    # Read input
    print(f"Reading input CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Total images: {len(df):,}")

    image_paths = df['figure_path'].tolist() if 'figure_path' in df.columns else df['image_path'].tolist()

    # Run quality checks in parallel
    print(f"\nRunning quality checks with {workers} workers...")
    check_args = [
        (str(path), min_size, max_size, min_file_mb, max_file_mb)
        for path in image_paths
    ]

    all_metrics = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(check_single_image, args): i
                  for i, args in enumerate(check_args)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Quality check"):
            try:
                metrics = future.result()
                all_metrics.append(metrics)
            except Exception as e:
                print(f"Error: {e}")
                # Create failed metric
                idx = futures[future]
                all_metrics.append(QualityMetrics(
                    image_path=check_args[idx][0],
                    failure_reason=str(e)
                ))

    # Create metrics DataFrame
    metrics_df = pd.DataFrame([m.to_dict() for m in all_metrics])

    # Find duplicate renditions
    print("\nDetecting duplicate renditions...")
    root = Path(image_paths[0]).parent.parent if image_paths else Path(".")
    valid_paths = [Path(m.image_path) for m in all_metrics if m.passes_quality]

    duplicate_groups = find_duplicate_renditions(valid_paths, root)
    print(f"Found {len(duplicate_groups)} duplicate rendition groups")

    # Mark duplicates
    duplicates_to_remove = set()
    if remove_duplicates and duplicate_groups:
        for stem, paths in duplicate_groups.items():
            best = get_best_rendition(paths)
            losers = [p for p in paths if p != best]
            duplicates_to_remove.update(str(p) for p in losers)

        # Update metrics
        for m in all_metrics:
            if m.image_path in duplicates_to_remove:
                m.is_duplicate_rendition = True
                m.passes_quality = False
                m.failure_reason = "Duplicate rendition (lower priority extension)"

    # Find perceptual duplicates
    if HAS_IMAGING:
        print("\nDetecting perceptually similar images...")
        passing_metrics = [m for m in all_metrics if m.passes_quality and m.perceptual_hash]
        perceptual_dupes = find_perceptual_duplicates(passing_metrics, threshold=5)

        if perceptual_dupes:
            print(f"Found {len(perceptual_dupes)} perceptual duplicate groups")

            # Mark perceptual duplicates
            all_perceptual_dupes = set()
            for rep, dupes in perceptual_dupes.items():
                all_perceptual_dupes.update(dupes)

            for m in all_metrics:
                if m.image_path in all_perceptual_dupes:
                    m.is_duplicate_rendition = True
                    m.passes_quality = False
                    if not m.failure_reason:
                        m.failure_reason = "Perceptual duplicate"

    # Filter to passing images
    passing_df = df[metrics_df['passes_quality']].copy()

    # Delete images that don't meet requirements
    failed_metrics = [m for m in all_metrics if not m.passes_quality and m.file_exists]
    if failed_metrics:
        print(f"\nDeleting {len(failed_metrics)} images that failed quality checks...")
        deleted_count = 0
        for m in failed_metrics:
            try:
                path = Path(m.image_path)
                if path.exists():
                    path.unlink()
                    deleted_count += 1
            except Exception as e:
                print(f"Failed to delete {m.image_path}: {e}")
        print(f"Deleted {deleted_count} files")

    # Print statistics
    print(f"\n{'='*60}")
    print("QUALITY CHECK RESULTS")
    print(f"{'='*60}")
    print(f"Total input:          {len(df):,}")
    print(f"Passed quality:       {len(passing_df):,} ({100*len(passing_df)/len(df):.1f}%)")
    print(f"\nFailure breakdown:")
    print(metrics_df['failure_reason'].value_counts().head(10).to_string())

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    passing_csv = output_dir / "quality_filtered.csv"
    passing_df.to_csv(passing_csv, index=False)
    print(f"\n✅ Passing images saved: {passing_csv}")

    metrics_csv = output_dir / "quality_report.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"✅ Quality report saved: {metrics_csv}")

    # Save duplicate report
    if duplicate_groups:
        duplicate_report = output_dir / "duplicate_report.json"
        with open(duplicate_report, 'w') as f:
            json.dump({
                'duplicate_renditions': {k: [str(p) for p in v] for k, v in duplicate_groups.items()},
                'perceptual_duplicates': {k: v for k, v in perceptual_dupes.items()} if HAS_IMAGING else {},
            }, f, indent=2)
        print(f"✅ Duplicate report saved: {duplicate_report}")

    return passing_df


# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Enhanced image quality check with duplicate detection"
    )
    parser.add_argument("--keyword", type=str, required=True,
                       help="Keyword (mri, eeg, semiology, etc.)")
    parser.add_argument("--input-dir", type=str, required=True,
                       help="Input directory (from step 1)")
    parser.add_argument("--min-size", type=int, default=100,
                       help="Minimum image dimension (default: 100)")
    parser.add_argument("--max-size", type=int, default=4096,
                       help="Maximum image dimension (default: 4096)")
    parser.add_argument("--min-file-mb", type=float, default=0.001,
                       help="Minimum file size in MB (default: 0.001)")
    parser.add_argument("--max-file-mb", type=float, default=50.0,
                       help="Maximum file size in MB (default: 50)")
    parser.add_argument("--workers", type=int, default=32,
                       help="Number of parallel workers")
    parser.add_argument("--no-remove-duplicates", action="store_true",
                       help="Don't remove duplicate renditions")
    parser.add_argument("--force", action="store_true",
                       help="Force reprocess even if output exists")

    args = parser.parse_args()

    # Setup paths
    input_dir = Path(args.input_dir)
    input_csv = input_dir / "biomedical_pairs.csv"
    output_csv = input_dir / "quality_filtered.csv"

    if not input_csv.exists():
        print(f"❌ Input CSV not found: {input_csv}")
        print("   Run step1_filter_biomedical_images.py first")
        return

    if output_csv.exists() and not args.force:
        print(f"✅ Output already exists: {output_csv}")
        print("   Use --force to reprocess")
        return

    # Run quality check
    run_quality_check(
        input_csv=input_csv,
        output_dir=input_dir,
        min_size=args.min_size,
        max_size=args.max_size,
        min_file_mb=args.min_file_mb,
        max_file_mb=args.max_file_mb,
        workers=args.workers,
        remove_duplicates=not args.no_remove_duplicates,
    )


if __name__ == "__main__":
    main()
