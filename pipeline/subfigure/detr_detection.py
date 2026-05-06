#!/usr/bin/env python3
"""
Step 3: DAB-DETR Subfigure Detection

This script uses DAB-DETR (Detection Transformer with Dynamic Anchor Boxes)
to detect subfigures in multi-panel medical images.

Following the Open-PMC approach:
- Detect individual subfigure panels in composite figures
- Extract bounding boxes for each subfigure
- Crop and save individual subfigures
- Handle both regular grids and irregular layouts

Model: We use a DETR-based model fine-tuned on biomedical subfigure detection
or fallback to general object detection with post-processing.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# Try imports
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from torchvision.ops import nms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Create dummy torch object for decorator
    class torch:
        @staticmethod
        def no_grad():
            def decorator(func):
                return func
            return decorator

try:
    from PIL import Image
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class BoundingBox:
    """Represents a detected subfigure bounding box."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    label: str = "subfigure"
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    def to_dict(self) -> Dict:
        return {
            'x1': int(self.x1), 'y1': int(self.y1),
            'x2': int(self.x2), 'y2': int(self.y2),
            'width': int(self.width), 'height': int(self.height),
            'confidence': float(self.confidence),
            'label': self.label,
        }


class SubfigureDetector:
    """
    DAB-DETR based subfigure detector.
    
    Uses DETR (DEtection TRansformer) architecture for detecting
    subfigures in multi-panel medical images.
    """
    
    def __init__(
        self,
        model_name: str = "vector-institute/pmc-18m-dab-detr",
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5,
    ):
        self.device = device if HAS_TORCH and torch.cuda.is_available() else "cpu"
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        if not HAS_TORCH or not HAS_TRANSFORMERS:
            print("⚠️ PyTorch or Transformers not available")
            self.model = None
            self.processor = None
            return
        
        print(f"Loading model: {model_name}")
        print(f"Device: {self.device}")
        
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForObjectDetection.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}")
            print("Falling back to heuristic detection")
            self.model = None
            self.processor = None
    
    @torch.no_grad()
    def detect(self, image: Image.Image) -> List[BoundingBox]:
        """
        Detect subfigures in an image.
        
        Args:
            image: PIL Image
        
        Returns:
            List of detected BoundingBox objects
        """
        if self.model is None:
            # Fallback to heuristic detection
            return self._heuristic_detect(image)
        
        # Prepare input
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        outputs = self.model(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([[image.height, image.width]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes,
            threshold=self.confidence_threshold
        )[0]
        
        boxes = []
        for score, label, box in zip(
            results["scores"].cpu().numpy(),
            results["labels"].cpu().numpy(),
            results["boxes"].cpu().numpy()
        ):
            x1, y1, x2, y2 = map(int, box)
            boxes.append(BoundingBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=float(score),
                label=self.model.config.id2label.get(label, "subfigure")
            ))
        
        # Apply NMS
        if len(boxes) > 0:
            boxes = self._apply_nms(boxes)
        
        # If no detections, try heuristic
        if len(boxes) == 0:
            boxes = self._heuristic_detect(image)
        
        return boxes
    
    def _apply_nms(self, boxes: List[BoundingBox]) -> List[BoundingBox]:
        """Apply non-maximum suppression to remove overlapping boxes."""
        if not HAS_TORCH or len(boxes) == 0:
            return boxes
        
        # Convert to tensors
        box_tensor = torch.tensor([
            [b.x1, b.y1, b.x2, b.y2] for b in boxes
        ], dtype=torch.float32)
        score_tensor = torch.tensor([b.confidence for b in boxes])
        
        # Apply NMS
        keep_indices = nms(box_tensor, score_tensor, self.nms_threshold)
        
        return [boxes[i] for i in keep_indices.tolist()]
    
    def _heuristic_detect(self, image: Image.Image) -> List[BoundingBox]:
        """
        Heuristic subfigure detection based on image analysis.
        
        Uses edge detection and line finding to identify grid patterns
        commonly found in multi-panel medical figures.
        """
        if not HAS_PIL:
            return []
        
        width, height = image.size
        
        # Convert to numpy for analysis
        try:
            import cv2
            img_array = np.array(image.convert('L'))  # Grayscale
            
            # Detect edges
            edges = cv2.Canny(img_array, 50, 150)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(
                edges, 1, np.pi/180, 
                threshold=100,
                minLineLength=min(width, height) * 0.3,
                maxLineGap=10
            )
            
            if lines is None or len(lines) < 2:
                # Single panel or no clear divisions
                return [BoundingBox(
                    x1=0, y1=0, x2=width, y2=height,
                    confidence=1.0, label="single_panel"
                )]
            
            # Find vertical and horizontal dividing lines
            v_lines = []  # Vertical lines (x positions)
            h_lines = []  # Horizontal lines (y positions)
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x1 - x2) < 10:  # Vertical line
                    v_lines.append((x1 + x2) // 2)
                elif abs(y1 - y2) < 10:  # Horizontal line
                    h_lines.append((y1 + y2) // 2)
            
            # Cluster line positions
            v_lines = self._cluster_lines(v_lines, width)
            h_lines = self._cluster_lines(h_lines, height)
            
            # Generate bounding boxes from grid
            boxes = self._grid_to_boxes(v_lines, h_lines, width, height)
            
            return boxes
            
        except ImportError:
            # No CV2, use simple grid heuristic
            return self._simple_grid_detect(image)
    
    def _cluster_lines(self, lines: List[int], max_val: int, threshold: int = 20) -> List[int]:
        """Cluster nearby lines and return cluster centers."""
        if not lines:
            return []
        
        lines = sorted(lines)
        clusters = [[lines[0]]]
        
        for line in lines[1:]:
            if line - clusters[-1][-1] < threshold:
                clusters[-1].append(line)
            else:
                clusters.append([line])
        
        # Return cluster centers, filtering edge clusters
        centers = []
        for cluster in clusters:
            center = sum(cluster) // len(cluster)
            # Filter lines too close to edges
            if center > max_val * 0.1 and center < max_val * 0.9:
                centers.append(center)
        
        return centers
    
    def _grid_to_boxes(
        self, 
        v_lines: List[int], 
        h_lines: List[int],
        width: int, 
        height: int
    ) -> List[BoundingBox]:
        """Convert grid lines to bounding boxes."""
        # Add image boundaries
        x_positions = [0] + sorted(v_lines) + [width]
        y_positions = [0] + sorted(h_lines) + [height]
        
        boxes = []
        for i in range(len(x_positions) - 1):
            for j in range(len(y_positions) - 1):
                x1, x2 = x_positions[i], x_positions[i + 1]
                y1, y2 = y_positions[j], y_positions[j + 1]
                
                # Filter very small boxes
                if (x2 - x1) > width * 0.1 and (y2 - y1) > height * 0.1:
                    boxes.append(BoundingBox(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        confidence=0.8,
                        label="subfigure"
                    ))
        
        return boxes
    
    def _simple_grid_detect(self, image: Image.Image) -> List[BoundingBox]:
        """Simple grid detection for 2x2, 1x2, 2x1 layouts."""
        width, height = image.size
        
        # Analyze aspect ratio
        aspect = width / height
        
        if 0.8 <= aspect <= 1.2:
            # Square-ish: try 2x2 grid
            boxes = [
                BoundingBox(0, 0, width//2, height//2, 0.7, "subfigure"),
                BoundingBox(width//2, 0, width, height//2, 0.7, "subfigure"),
                BoundingBox(0, height//2, width//2, height, 0.7, "subfigure"),
                BoundingBox(width//2, height//2, width, height, 0.7, "subfigure"),
            ]
        elif aspect > 1.5:
            # Wide: try 1x2 horizontal
            boxes = [
                BoundingBox(0, 0, width//2, height, 0.7, "subfigure"),
                BoundingBox(width//2, 0, width, height, 0.7, "subfigure"),
            ]
        elif aspect < 0.67:
            # Tall: try 2x1 vertical
            boxes = [
                BoundingBox(0, 0, width, height//2, 0.7, "subfigure"),
                BoundingBox(0, height//2, width, height, 0.7, "subfigure"),
            ]
        else:
            # Single panel
            boxes = [BoundingBox(0, 0, width, height, 1.0, "single_panel")]
        
        return boxes


def crop_subfigures(
    image_path: str,
    boxes: List[BoundingBox],
    output_dir: str,
    min_size: int = 50,
) -> List[Dict]:
    """
    Crop and save subfigures from an image.
    
    Args:
        image_path: Path to source image
        boxes: List of detected bounding boxes
        output_dir: Directory to save cropped subfigures
        min_size: Minimum subfigure dimension
    
    Returns:
        List of dicts with subfigure info
    """
    if not HAS_PIL:
        return []
    
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Failed to open image: {e}")
        return []
    
    subfigures = []
    base_name = Path(image_path).stem
    
    for i, box in enumerate(boxes):
        # Skip small boxes
        if box.width < min_size or box.height < min_size:
            continue
        
        # Crop
        cropped = img.crop((box.x1, box.y1, box.x2, box.y2))
        
        # Generate output filename
        subfig_name = f"{base_name}_subfig_{i:02d}.png"
        subfig_path = os.path.join(output_dir, subfig_name)
        
        # Save
        cropped.save(subfig_path, "PNG")
        
        subfigures.append({
            'source_image': image_path,
            'subfigure_path': subfig_path,
            'subfigure_index': i,
            'bbox': box.to_dict(),
        })
    
    return subfigures


def process_single_image(
    args: Tuple[str, str, 'SubfigureDetector', int]
) -> Dict:
    """Process a single image for subfigure detection."""
    image_path, output_dir, detector, min_size = args
    
    result = {
        'image_path': image_path,
        'success': False,
        'num_subfigures': 0,
        'is_multipanel': False,
        'subfigures': [],
        'error': None,
    }
    
    try:
        # Load image
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Detect subfigures
        boxes = detector.detect(img)
        
        # Determine if multi-panel
        result['is_multipanel'] = len(boxes) > 1 or (
            len(boxes) == 1 and boxes[0].label != "single_panel"
        )
        
        # Crop and save subfigures
        if result['is_multipanel'] and len(boxes) > 1:
            result['subfigures'] = crop_subfigures(
                image_path, boxes, output_dir, min_size
            )
        else:
            # Single panel - copy as-is
            result['subfigures'] = [{
                'source_image': image_path,
                'subfigure_path': image_path,  # Use original
                'subfigure_index': 0,
                'bbox': boxes[0].to_dict() if boxes else None,
            }]
        
        result['num_subfigures'] = len(result['subfigures'])
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def run_detection(
    input_csv: str,
    output_dir: str,
    batch_size: int = 64,
    min_subfigure_size: int = 50,
    model_name: str = "vector-institute/pmc-18m-dab-detr",
    confidence_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Run subfigure detection on all images.
    
    Args:
        input_csv: Path to quality_filtered.csv
        output_dir: Output directory
        batch_size: Batch size for GPU processing
        min_subfigure_size: Minimum subfigure dimension
        model_name: DETR model name
        confidence_threshold: Detection confidence threshold
    
    Returns:
        DataFrame with detection results
    """
    print(f"\nReading input CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Total images: {len(df):,}")
    
    # Create output directories
    subfig_dir = os.path.join(output_dir, 'cropped_subfigures')
    os.makedirs(subfig_dir, exist_ok=True)
    
    # Initialize detector
    detector = SubfigureDetector(
        model_name=model_name,
        confidence_threshold=confidence_threshold,
    )
    
    # Get unique image paths
    image_paths = df['figure_path'].unique().tolist()
    print(f"Unique images to process: {len(image_paths):,}")
    
    # Process images
    all_results = []
    print(f"\nDetecting subfigures...")
    
    for image_path in tqdm(image_paths, desc="Detection"):
        result = process_single_image((
            image_path, subfig_dir, detector, min_subfigure_size
        ))
        all_results.append(result)
    
    # Aggregate statistics
    successful = sum(1 for r in all_results if r['success'])
    multipanel = sum(1 for r in all_results if r['is_multipanel'])
    total_subfigs = sum(r['num_subfigures'] for r in all_results)
    
    print(f"\n{'='*60}")
    print("DETECTION RESULTS")
    print(f"{'='*60}")
    print(f"Images processed: {len(all_results):,}")
    print(f"Successful: {successful:,}")
    print(f"Multi-panel images: {multipanel:,}")
    print(f"Total subfigures extracted: {total_subfigs:,}")
    
    # Create expanded DataFrame with subfigures
    subfig_rows = []
    for result in all_results:
        if not result['success']:
            continue
        
        # Get original row data
        orig_rows = df[df['figure_path'] == result['image_path']].to_dict('records')
        if not orig_rows:
            continue
        orig_row = orig_rows[0]
        
        for subfig in result['subfigures']:
            row = orig_row.copy()
            row['source_image'] = result['image_path']
            row['subfigure_path'] = subfig['subfigure_path']
            row['subfigure_index'] = subfig['subfigure_index']
            row['is_multipanel'] = result['is_multipanel']
            row['bbox'] = json.dumps(subfig['bbox']) if subfig['bbox'] else None
            subfig_rows.append(row)
    
    subfig_df = pd.DataFrame(subfig_rows)
    
    # Save results
    output_csv = os.path.join(output_dir, 'subfigure_detections.csv')
    subfig_df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved detection results to: {output_csv}")
    
    # Save detailed results as JSON
    results_json = os.path.join(output_dir, 'subfigure_detections.json')
    with open(results_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✅ Saved detailed results to: {results_json}")
    
    return subfig_df


def main():
    parser = argparse.ArgumentParser(description="DAB-DETR subfigure detection")
    parser.add_argument('--keyword', required=True, help='Keyword for dataset')
    parser.add_argument('--input-dir', required=True, help='Input directory')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--min-subfigure-size', type=int, default=50, help='Min subfig size')
    parser.add_argument('--model', default='vector-institute/pmc-18m-dab-detr', help='DETR model')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--force', action='store_true', help='Force reprocess')
    
    args = parser.parse_args()
    
    # Check if output exists
    output_csv = os.path.join(args.input_dir, 'subfigure_detections.csv')
    if os.path.exists(output_csv) and not args.force:
        print(f"Output already exists: {output_csv}")
        print("Use --force to reprocess")
        return
    
    # Check input exists
    input_csv = os.path.join(args.input_dir, 'quality_filtered.csv')
    if not os.path.exists(input_csv):
        print(f"❌ Input CSV not found: {input_csv}")
        sys.exit(1)
    
    # Run detection
    run_detection(
        input_csv=input_csv,
        output_dir=args.input_dir,
        batch_size=args.batch_size,
        min_subfigure_size=args.min_subfigure_size,
        model_name=args.model,
        confidence_threshold=args.confidence,
    )


if __name__ == '__main__':
    main()
