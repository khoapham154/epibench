#!/usr/bin/env python3
"""
Utility Functions for Subfigure Extraction Pipeline

Common utilities used across all pipeline steps.
"""

import hashlib
import json
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from tqdm import tqdm


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def file_hash(filepath: str, algorithm: str = 'md5') -> Optional[str]:
    """Compute hash of a file."""
    try:
        hash_func = hashlib.new(algorithm)
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except:
        return None


def find_files(
    root_dir: str,
    extensions: List[str] = None,
    pattern: str = None,
    recursive: bool = True,
) -> List[str]:
    """
    Find files matching criteria.
    
    Args:
        root_dir: Root directory to search
        extensions: List of file extensions (e.g., ['.jpg', '.png'])
        pattern: Regex pattern to match filenames
        recursive: Whether to search recursively
    
    Returns:
        List of matching file paths
    """
    matches = []
    
    if extensions:
        extensions = [ext.lower() for ext in extensions]
    
    if pattern:
        pattern = re.compile(pattern)
    
    if recursive:
        for root, dirs, files in os.walk(root_dir):
            for filename in files:
                filepath = os.path.join(root, filename)
                
                if extensions:
                    if not any(filename.lower().endswith(ext) for ext in extensions):
                        continue
                
                if pattern:
                    if not pattern.search(filename):
                        continue
                
                matches.append(filepath)
    else:
        for filename in os.listdir(root_dir):
            filepath = os.path.join(root_dir, filename)
            if not os.path.isfile(filepath):
                continue
            
            if extensions:
                if not any(filename.lower().endswith(ext) for ext in extensions):
                    continue
            
            if pattern:
                if not pattern.search(filename):
                    continue
            
            matches.append(filepath)
    
    return matches


def parallel_process(
    items: List[Any],
    func: Callable,
    workers: int = 32,
    use_threads: bool = False,
    desc: str = "Processing",
    chunk_size: int = 1,
) -> List[Any]:
    """
    Process items in parallel.
    
    Args:
        items: List of items to process
        func: Function to apply to each item
        workers: Number of parallel workers
        use_threads: Use ThreadPoolExecutor instead of ProcessPoolExecutor
        desc: Description for progress bar
        chunk_size: Number of items per worker task
    
    Returns:
        List of results
    """
    if not items:
        return []
    
    results = []
    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    
    with executor_class(max_workers=workers) as executor:
        if chunk_size > 1:
            # Batch processing
            chunks = [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]
            futures = {executor.submit(func, chunk): i for i, chunk in enumerate(chunks)}
        else:
            futures = {executor.submit(func, item): i for i, item in enumerate(items)}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            try:
                result = future.result()
                if chunk_size > 1 and isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)
            except Exception as e:
                print(f"Error in parallel processing: {e}")
    
    return results


def load_json(filepath: str, default: Any = None) -> Any:
    """Load JSON file with error handling."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return default


def save_json(data: Any, filepath: str, indent: int = 2) -> bool:
    """Save data to JSON file."""
    try:
        ensure_dir(os.path.dirname(filepath))
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving JSON: {e}")
        return False


def extract_pmc_id(path: str) -> Optional[str]:
    """Extract PMC ID from a file path."""
    match = re.search(r'PMC\d+', str(path))
    return match.group() if match else None


def extract_figure_id(path: str) -> Optional[str]:
    """Extract figure ID from a file path."""
    patterns = [
        r'[Ff]ig(?:ure)?[-_]?(\d+)',
        r'[Ff](\d+)',
        r'figure(\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, str(path))
        if match:
            return f"fig{match.group(1)}"
    
    return None


def safe_copy(src: str, dst: str, overwrite: bool = False) -> bool:
    """Safely copy a file."""
    try:
        if os.path.exists(dst) and not overwrite:
            return True
        ensure_dir(os.path.dirname(dst))
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        print(f"Error copying {src} to {dst}: {e}")
        return False


def batch_iterator(items: List[Any], batch_size: int):
    """Iterate over items in batches."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    return text.strip()


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate text to maximum length."""
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


class ProgressTracker:
    """Track progress across pipeline steps."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.progress_file = os.path.join(output_dir, 'progress.json')
        self.progress = self._load()
    
    def _load(self) -> Dict:
        return load_json(self.progress_file, default={
            'completed_steps': [],
            'step_stats': {},
            'errors': [],
        })
    
    def save(self):
        save_json(self.progress, self.progress_file)
    
    def mark_step_complete(self, step: str, stats: Dict = None):
        if step not in self.progress['completed_steps']:
            self.progress['completed_steps'].append(step)
        if stats:
            self.progress['step_stats'][step] = stats
        self.save()
    
    def is_step_complete(self, step: str) -> bool:
        return step in self.progress['completed_steps']
    
    def add_error(self, step: str, error: str):
        self.progress['errors'].append({
            'step': step,
            'error': error,
        })
        self.save()


def get_image_dimensions(filepath: str) -> Optional[Tuple[int, int]]:
    """Get image dimensions without loading full image."""
    try:
        from PIL import Image
        with Image.open(filepath) as img:
            return img.size  # (width, height)
    except:
        return None


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


if __name__ == '__main__':
    # Test utilities
    print("Testing utilities...")
    
    # Test find_files
    files = find_files('/tmp', extensions=['.txt'], recursive=False)
    print(f"Found {len(files)} .txt files in /tmp")
    
    # Test extract_pmc_id
    pmc_id = extract_pmc_id('/data/PMC12345/figure1.png')
    print(f"Extracted PMC ID: {pmc_id}")
    
    # Test format_size
    print(f"1024 bytes = {format_size(1024)}")
    print(f"1048576 bytes = {format_size(1048576)}")
    
    print("✅ Utilities test complete")
