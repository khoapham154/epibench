#!/usr/bin/env python3
"""
Configuration for Subfigure Extraction Pipeline (CORRECTED VERSION)

Fixed issues:
1. Correct DAB-DETR model (vector-institute/pmc-18m-dab-detr)
2. Integration with existing pipeline paths
3. Added patient linking configuration
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class HardwareConfig:
    """Hardware configuration."""
    num_cpus: int = int(os.environ.get("NUM_CORES", 96))
    num_gpus: int = int(os.environ.get("NUM_GPUS", 8))
    gpu_memory_gb: int = 80  # Per GPU
    total_ram_gb: int = 1700
    cuda_visible_devices: str = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

    @property
    def worker_count(self) -> int:
        """Number of CPU workers (90% of cores)."""
        return int(self.num_cpus * 0.9)


@dataclass
class ModelConfig:
    """Model configuration for LLM and DETR."""
    # LLM settings
    llm_model_id: str = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-72B-Instruct")
    llm_tensor_parallel_size: int = int(os.environ.get("NUM_GPUS", 8))
    llm_gpu_memory_utilization: float = 0.92
    llm_max_model_len: int = 32768
    llm_batch_size: int = 32

    # DETR settings - FIXED: Use correct medical subfigure model
    detr_model_id: str = "vector-institute/pmc-18m-dab-detr"  # ✅ CORRECT MODEL
    detr_confidence_threshold: float = 0.5
    detr_nms_threshold: float = 0.5
    detr_batch_size: int = 64


@dataclass
class ImageConfig:
    """Image processing configuration."""
    # Size limits
    min_image_size: int = 100
    max_image_size: int = 4096
    min_subfigure_size: int = 50
    max_file_size_mb: int = 50  # Added: Reject huge files

    # Quality thresholds
    min_aspect_ratio: float = 0.1
    max_aspect_ratio: float = 10.0
    min_file_size_kb: int = 1

    # Duplicate detection settings
    perceptual_hash_threshold: int = 10  # Added for duplicate detection

    # Supported formats
    supported_formats: List[str] = field(default_factory=lambda: [
        '.jpg', '.jpeg', '.png', '.gif', '.tiff', '.tif', '.bmp'
    ])


@dataclass
class PathConfig:
    """Path configuration - matches existing pipeline."""
    # Base directories (match existing pipeline)
    data_base: Path = Path("/data/pubmed_epilepsy/downloads")
    output_base: Path = Path(os.environ.get("EPIBENCH_ROOT", "."))
    monet_base: Path = Path(os.environ.get("EPIBENCH_MONET", "./monet_pubmed"))

    def get_extract_dir(self, keyword: str) -> Path:
        """Get extraction directory for keyword."""
        return self.data_base / f"{keyword}_extracted"

    def get_pairs_csv(self, keyword: str) -> Path:
        """Get final pairs CSV path."""
        return self.monet_base / "out" / f"{keyword}_final_pairs.csv"

    def get_subfigure_output_dir(self, keyword: str) -> Path:
        """Get subfigure output directory."""
        return self.output_base / "subfigures" / keyword

    def get_patient_output_dir(self, keyword: str) -> Path:
        """Get patient extraction output directory (from existing pipeline)."""
        return self.output_base / keyword

    def get_log_dir(self) -> Path:
        """Get log directory."""
        return self.output_base / "logs" / "subfigure"

    def get_duplicates_dir(self, keyword: str) -> Path:
        """Get duplicates directory."""
        return self.monet_base / "duplicated_images" / keyword


@dataclass
class ModalityConfig:
    """Modality classification configuration."""
    # Target modalities
    target_modalities: List[str] = field(default_factory=lambda: ['MRI', 'EEG'])

    # Modality confidence threshold
    min_confidence: float = 0.2

    # Detailed modality types
    mri_types: List[str] = field(default_factory=lambda: [
        'MRI_T1_WEIGHTED', 'MRI_T2_WEIGHTED', 'MRI_FLAIR',
        'MRI_DWI_ADC', 'MRI_CONTRAST', 'MRI_OTHER'
    ])

    eeg_types: List[str] = field(default_factory=lambda: [
        'EEG_SCALP', 'EEG_ICTAL', 'EEG_INTERICTAL',
        'EEG_INTRACRANIAL', 'EEG_OTHER'
    ])


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    modality: ModalityConfig = field(default_factory=ModalityConfig)

    # Pipeline settings
    force_reprocess: bool = False
    skip_quality_check: bool = False
    skip_detr: bool = False
    remove_duplicates: bool = True
    link_to_patients: bool = True  # Added: Enable patient linking

    # Keywords to process
    keywords: List[str] = field(default_factory=lambda: [
        "mri", "eeg", "semiology", "asm", "syndromes"
    ])

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'hardware': {
                'num_cpus': self.hardware.num_cpus,
                'num_gpus': self.hardware.num_gpus,
                'gpu_memory_gb': self.hardware.gpu_memory_gb,
                'worker_count': self.hardware.worker_count,
            },
            'model': {
                'llm_model_id': self.model.llm_model_id,
                'llm_tensor_parallel_size': self.model.llm_tensor_parallel_size,
                'detr_model_id': self.model.detr_model_id,
            },
            'image': {
                'min_image_size': self.image.min_image_size,
                'max_image_size': self.image.max_image_size,
                'min_subfigure_size': self.image.min_subfigure_size,
                'max_file_size_mb': self.image.max_file_size_mb,
            },
            'modality': {
                'target_modalities': self.modality.target_modalities,
                'min_confidence': self.modality.min_confidence,
            },
            'pipeline': {
                'link_to_patients': self.link_to_patients,
                'remove_duplicates': self.remove_duplicates,
            }
        }


# Default configuration instance
DEFAULT_CONFIG = PipelineConfig()


def get_config(
    keyword: str,
    force: bool = False,
    skip_quality: bool = False,
    skip_detr: bool = False,
    link_patients: bool = True,
) -> PipelineConfig:
    """Get configuration for a specific keyword."""
    config = PipelineConfig(
        force_reprocess=force,
        skip_quality_check=skip_quality,
        skip_detr=skip_detr,
        link_to_patients=link_patients,
    )
    return config


if __name__ == '__main__':
    # Print default configuration
    import json
    config = DEFAULT_CONFIG
    print(json.dumps(config.to_dict(), indent=2))
