#!/usr/bin/env python3
"""
Step 5: Generate Final Subfigure Dataset

This script compiles all processed subfigures into a final dataset:
1. Merge all extraction results
2. Link subfigures to patient data (from existing Step 4 extraction)
3. Generate quality metrics and statistics
4. Create final output files (CSV, JSON, JSONL)

Output format compatible with Open-PMC dataset structure.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm


def load_patient_data(extract_dir: str, pmc_id: str) -> Optional[Dict]:
    """Load patient profiles from existing extraction."""
    profiles_path = os.path.join(extract_dir, pmc_id, 'final_profiles.json')
    
    if not os.path.exists(profiles_path):
        return None
    
    try:
        with open(profiles_path, 'r') as f:
            return json.load(f)
    except:
        return None


def link_subfigure_to_patients(
    subfigure: Dict,
    patient_data: Optional[Dict],
) -> List[Dict]:
    """
    Link a subfigure to relevant patients.
    
    Uses figure references in patient profiles to establish links.
    """
    if patient_data is None:
        return []
    
    linked_patients = []
    subfig_path = subfigure.get('subfigure_path', '')
    source_image = subfigure.get('source_image', '')
    
    # Extract figure ID from path
    fig_id = None
    for pattern in ['fig', 'figure', 'Fig', 'Figure']:
        if pattern in source_image:
            # Try to extract figure number
            import re
            match = re.search(rf'{pattern}[-_]?(\d+)', source_image, re.IGNORECASE)
            if match:
                fig_id = f"fig{match.group(1)}"
                break
    
    # Check each patient's linked figures
    for patient in patient_data:
        linked_figs = patient.get('linked_figures', [])
        
        # Check if this subfigure's source matches any linked figure
        for fig_ref in linked_figs:
            if fig_id and fig_id.lower() in fig_ref.lower():
                linked_patients.append({
                    'patient_ref': patient.get('patient_ref', ''),
                    'figure_ref': fig_ref,
                    'match_type': 'figure_id',
                })
                break
    
    return linked_patients


def generate_final_dataset(
    input_dir: str,
    extract_base_dir: str,
    output_dir: str,
    workers: int = 32,
) -> pd.DataFrame:
    """
    Generate the final subfigure dataset.
    
    Args:
        input_dir: Directory with subfigure extraction results
        extract_base_dir: Base directory with paper extractions
        output_dir: Output directory
        workers: Number of parallel workers
    
    Returns:
        Final dataset DataFrame
    """
    # Load subcaption results
    subcaptions_csv = os.path.join(input_dir, 'subfigure_subcaptions.csv')
    if not os.path.exists(subcaptions_csv):
        print(f"❌ Subcaptions CSV not found: {subcaptions_csv}")
        sys.exit(1)
    
    print(f"\nReading subcaption results: {subcaptions_csv}")
    df = pd.read_csv(subcaptions_csv)
    print(f"Total subfigures: {len(df):,}")
    
    # Load detailed JSON for additional fields
    subcaptions_json = os.path.join(input_dir, 'subfigure_subcaptions.json')
    subcaption_details = {}
    if os.path.exists(subcaptions_json):
        with open(subcaptions_json, 'r') as f:
            details_list = json.load(f)
            for detail in details_list:
                key = detail.get('subfigure_path', '')
                subcaption_details[key] = detail
    
    # Process each subfigure
    final_records = []
    print(f"\nGenerating final dataset...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        record = row.to_dict()
        
        # Get PMC ID
        pmc_id = record.get('pmc_id', '')
        if not pmc_id:
            # Try to extract from path
            source = record.get('source_image', '') or record.get('figure_path', '')
            import re
            match = re.search(r'PMC\d+', source)
            if match:
                pmc_id = match.group()
                record['pmc_id'] = pmc_id
        
        # Add detailed subcaption info
        subfig_path = record.get('subfigure_path', '')
        if subfig_path in subcaption_details:
            detail = subcaption_details[subfig_path]
            record['findings'] = detail.get('findings', [])
            record['anatomical_region'] = detail.get('anatomical_region', '')
            record['intext_references'] = detail.get('intext_references', [])
        
        # Link to patient data (if available)
        if pmc_id and extract_base_dir:
            patient_data = load_patient_data(extract_base_dir, pmc_id)
            linked = link_subfigure_to_patients(record, patient_data)
            record['linked_patients'] = linked
            record['num_linked_patients'] = len(linked)
        else:
            record['linked_patients'] = []
            record['num_linked_patients'] = 0
        
        # Add quality flags
        record['has_subcaption'] = bool(record.get('subcaption', ''))
        record['has_modality'] = record.get('modality', 'UNKNOWN') != 'UNKNOWN'
        record['has_findings'] = len(record.get('findings', [])) > 0
        
        final_records.append(record)
    
    # Create final DataFrame
    final_df = pd.DataFrame(final_records)
    
    # Compute statistics
    print(f"\n{'='*60}")
    print("FINAL DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"Total subfigures: {len(final_df):,}")
    print(f"Unique source images: {final_df['source_image'].nunique():,}")
    print(f"Unique PMC IDs: {final_df['pmc_id'].nunique():,}")
    
    print(f"\nModality distribution:")
    print(final_df['modality'].value_counts().head(10).to_string())
    
    has_subcaption = final_df['has_subcaption'].sum()
    has_modality = final_df['has_modality'].sum()
    has_findings = final_df['has_findings'].sum()
    has_patients = (final_df['num_linked_patients'] > 0).sum()
    
    print(f"\nQuality metrics:")
    print(f"  With subcaption: {has_subcaption:,} ({100*has_subcaption/len(final_df):.1f}%)")
    print(f"  With modality: {has_modality:,} ({100*has_modality/len(final_df):.1f}%)")
    print(f"  With findings: {has_findings:,} ({100*has_findings/len(final_df):.1f}%)")
    print(f"  Linked to patients: {has_patients:,} ({100*has_patients/len(final_df):.1f}%)")
    
    # Filter to MRI and EEG only for final output (exclude UNKNOWN)
    mri_eeg_df = final_df[
        final_df['modality'].str.contains('MRI|EEG', case=False, na=False)
    ]
    unknown_df = final_df[final_df['modality'] == 'UNKNOWN']
    print(f"\nMRI/EEG subfigures: {len(mri_eeg_df):,}")
    print(f"UNKNOWN (excluded): {len(unknown_df):,}")
    # Save UNKNOWN separately for optional review
    if len(unknown_df) > 0:
        unknown_path = os.path.join(output_dir, 'unknown_for_review.csv')
        unknown_df.to_csv(unknown_path, index=False)
        print(f"  Saved to {unknown_path}")
    
    # Create output directory structure
    final_subfig_dir = os.path.join(output_dir, 'final_subfigures')
    os.makedirs(final_subfig_dir, exist_ok=True)
    
    # Save final dataset (CSV)
    final_csv = os.path.join(output_dir, 'final_dataset.csv')
    final_df.to_csv(final_csv, index=False)
    print(f"\n✅ Saved final dataset CSV: {final_csv}")
    
    # Save MRI/EEG only dataset
    mri_eeg_csv = os.path.join(output_dir, 'mri_eeg_subfigures.csv')
    mri_eeg_df.to_csv(mri_eeg_csv, index=False)
    print(f"✅ Saved MRI/EEG dataset: {mri_eeg_csv}")
    
    # Save as JSON
    final_json = os.path.join(output_dir, 'final_dataset.json')
    with open(final_json, 'w') as f:
        # Convert lists to proper JSON
        records_for_json = []
        for record in final_records:
            r = record.copy()
            # Ensure lists are actual lists
            for key in ['findings', 'intext_references', 'linked_patients']:
                if key in r and isinstance(r[key], str):
                    try:
                        r[key] = json.loads(r[key])
                    except:
                        r[key] = []
            records_for_json.append(r)
        json.dump(records_for_json, f, indent=2)
    print(f"✅ Saved final dataset JSON: {final_json}")
    
    # Save as JSONL (for training)
    final_jsonl = os.path.join(output_dir, 'final_dataset.jsonl')
    with open(final_jsonl, 'w') as f:
        for record in records_for_json:
            f.write(json.dumps(record) + '\n')
    print(f"✅ Saved final dataset JSONL: {final_jsonl}")
    
    # Save statistics
    stats = {
        'total_subfigures': len(final_df),
        'unique_source_images': int(final_df['source_image'].nunique()),
        'unique_pmc_ids': int(final_df['pmc_id'].nunique()),
        'with_subcaption': int(has_subcaption),
        'with_modality': int(has_modality),
        'with_findings': int(has_findings),
        'linked_to_patients': int(has_patients),
        'mri_eeg_count': len(mri_eeg_df),
        'modality_distribution': final_df['modality'].value_counts().to_dict(),
    }
    stats_json = os.path.join(output_dir, 'dataset_statistics.json')
    with open(stats_json, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Saved statistics: {stats_json}")
    
    return final_df


def main():
    parser = argparse.ArgumentParser(description="Generate final subfigure dataset")
    parser.add_argument('--keyword', required=True, help='Keyword for dataset')
    parser.add_argument('--input-dir', required=True, help='Input directory')
    parser.add_argument('--extract-dir', default=None, help='Patient extraction base directory')
    parser.add_argument('--workers', type=int, default=32, help='Number of workers')
    
    args = parser.parse_args()
    
    # Set default extract directory
    extract_dir = args.extract_dir
    if extract_dir is None:
        extract_dir = f"/data/pubmed_epilepsy/downloads/{args.keyword}_extracted"
    
    # Run generation
    generate_final_dataset(
        input_dir=args.input_dir,
        extract_base_dir=extract_dir,
        output_dir=args.input_dir,
        workers=args.workers,
    )


if __name__ == '__main__':
    main()
