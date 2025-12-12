"""
Hard Negative Dataset - Weighted Sampling for Fine-tuning

Dataset class có trọng số cao cho hard negatives.
Author: Lockdown
Date: Nov 26, 2025
"""

import json
import random
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter
from torch.utils.data import Dataset, WeightedRandomSampler

# Import base dataset
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.fact_checking.train import ViFactCheckDataset

class HardNegativeViFactCheckDataset(Dataset):
    """Dataset với weighted sampling cho hard negatives."""
    
    def __init__(
        self, 
        jsonl_path: str, 
        tokenizer, 
        max_len: int = 512,
        hard_negatives_info: Optional[Dict] = None,
        hard_negative_weight: float = 2.0,
        hard_negative_ratio: float = 0.5
    ):
        # Load base dataset
        self.base_dataset = ViFactCheckDataset(jsonl_path, tokenizer, max_len)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Hard negative configuration
        self.hard_negatives_info = hard_negatives_info or {}
        self.hard_negative_weight = hard_negative_weight
        self.hard_negative_ratio = hard_negative_ratio
        
        # Determine which samples are hard negatives
        self._setup_hard_negatives()
        
        # Create sample weights for weighted sampling
        self._create_sample_weights()
        
    def _setup_hard_negatives(self):
        """Xác định indices của hard negatives."""
        self.is_hard_negative = [False] * len(self.base_dataset)
        self.hard_negative_indices = set()
        
        if not self.hard_negatives_info:
            return
        
        # Get hard negatives for this split (assuming train split)
        hard_negatives = self.hard_negatives_info.get('hard_negatives', [])
        for hn in hard_negatives:
            if hn.get('split') == 'train':  # Only use train hard negatives for training
                idx = hn.get('index')
                if idx is not None and 0 <= idx < len(self.base_dataset):
                    self.is_hard_negative[idx] = True
                    self.hard_negative_indices.add(idx)
        
        print(f"🎯 Loaded {len(self.hard_negative_indices)} hard negatives")
        
    def _create_sample_weights(self):
        """Tạo trọng số cho từng sample (hard negatives có trọng số cao hơn)."""
        self.sample_weights = []
        
        for i in range(len(self.base_dataset)):
            if self.is_hard_negative[i]:
                weight = self.hard_negative_weight
            else:
                weight = 1.0
            self.sample_weights.append(weight)
        
        # Normalize weights
        total_weight = sum(self.sample_weights)
        self.sample_weights = [w / total_weight for w in self.sample_weights]
        
        print(f"📊 Sample weights: HN={self.hard_negative_weight}, Normal=1.0")
        
    def create_weighted_sampler(self) -> WeightedRandomSampler:
        """Tạo WeightedRandomSampler cho DataLoader."""
        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.sample_weights),
            replacement=True
        )
    
    def get_hard_negative_stats(self) -> Dict[str, Any]:
        """Thống kê về hard negatives."""
        # Count labels in hard negatives
        hn_label_counts = Counter()
        normal_label_counts = Counter()
        
        for i in range(len(self.base_dataset)):
            sample = self.base_dataset[i]
            label = sample['labels'].item()
            
            if self.is_hard_negative[i]:
                hn_label_counts[label] += 1
            else:
                normal_label_counts[label] += 1
        
        return {
            'total_samples': len(self.base_dataset),
            'hard_negatives_count': len(self.hard_negative_indices),
            'hard_negative_ratio': len(self.hard_negative_indices) / len(self.base_dataset),
            'hard_negative_label_dist': dict(hn_label_counts),
            'normal_label_dist': dict(normal_label_counts),
            'average_weight': np.mean(self.sample_weights)
        }
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        return self.base_dataset[idx]


class BalancedHardNegativeDataset(Dataset):
    """Alternative: Dataset cân bằng tỷ lệ hard negatives."""
    
    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        max_len: int = 512,
        hard_negatives_info: Optional[Dict] = None,
        target_hn_ratio: float = 0.3,
        seed: int = 42
    ):
        self.base_dataset = ViFactCheckDataset(jsonl_path, tokenizer, max_len)
        self.target_hn_ratio = target_hn_ratio
        
        # Set seed for reproducible sampling
        random.seed(seed)
        np.random.seed(seed)
        
        # Create balanced dataset
        self._create_balanced_dataset(hard_negatives_info)
        
    def _create_balanced_dataset(self, hard_negatives_info: Dict):
        """Tạo tập cân bằng với tỷ lệ hard negatives mong muốn."""
        # Identify hard negatives
        hard_negative_indices = set()
        if hard_negatives_info:
            for hn in hard_negatives_info.get('hard_negatives', []):
                if hn.get('split') == 'train':
                    idx = hn.get('index')
                    if idx is not None and 0 <= idx < len(self.base_dataset):
                        hard_negative_indices.add(idx)
        
        # Separate indices
        hn_indices = list(hard_negative_indices)
        normal_indices = [i for i in range(len(self.base_dataset)) if i not in hard_negative_indices]
        
        # Calculate target counts
        total_samples = len(self.base_dataset)
        target_hn_count = int(total_samples * self.target_hn_ratio)
        target_normal_count = total_samples - target_hn_count
        
        # Sample to reach target ratio
        if len(hn_indices) > target_hn_count:
            # Too many hard negatives, sample down
            hn_indices = random.sample(hn_indices, target_hn_count)
        else:
            # Too few hard negatives, repeat some
            while len(hn_indices) < target_hn_count:
                hn_indices.extend(random.sample(hard_negative_indices, 
                                               min(len(hard_negative_indices), target_hn_count - len(hn_indices))))
        
        if len(normal_indices) > target_normal_count:
            normal_indices = random.sample(normal_indices, target_normal_count)
        
        # Combine and shuffle
        self.selected_indices = hn_indices + normal_indices
        random.shuffle(self.selected_indices)
        
        print(f"🎯 Balanced dataset: {len(hn_indices)} HN + {len(normal_indices)} normal = {len(self.selected_indices)} total")
        print(f"📊 Target HN ratio: {self.target_hn_ratio:.1%}, Actual: {len(hn_indices)/len(self.selected_indices):.1%}")
        
    def __len__(self):
        return len(self.selected_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.selected_indices[idx]
        return self.base_dataset[actual_idx]


def load_hard_negatives_info(cache_path: str) -> Optional[Dict]:
    """Load hard negatives từ cache file."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        print(f"⚠️ Hard negatives cache not found: {cache_path}")
        return None
    
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded hard negatives info from: {cache_path}")
        return data
    except Exception as e:
        print(f"❌ Error loading hard negatives: {e}")
        return None


def create_hard_negative_dataset(
    jsonl_path: str,
    tokenizer,
    config: Dict[str, Any],
    use_balanced: bool = False
) -> Dataset:
    """Factory function để tạo hard negative dataset."""
    
    # Load hard negatives info
    cache_path = config['data'].get('hard_negatives_cache')
    if cache_path and not Path(cache_path).is_absolute():
        cache_path = project_root / cache_path
        
    hard_negatives_info = load_hard_negatives_info(str(cache_path)) if cache_path else None
    
    # Get parameters from config
    max_len = config['model'].get('max_length', 512)
    hard_negative_weight = config['training'].get('hard_negative_weight', 2.0)
    hard_negative_ratio = config['training'].get('hard_negative_ratio', 0.5)
    
    if use_balanced:
        return BalancedHardNegativeDataset(
            jsonl_path=jsonl_path,
            tokenizer=tokenizer,
            max_len=max_len,
            hard_negatives_info=hard_negatives_info,
            target_hn_ratio=hard_negative_ratio,
            seed=config['training'].get('seed', 42)
        )
    else:
        return HardNegativeViFactCheckDataset(
            jsonl_path=jsonl_path,
            tokenizer=tokenizer,
            max_len=max_len,
            hard_negatives_info=hard_negatives_info,
            hard_negative_weight=hard_negative_weight,
            hard_negative_ratio=hard_negative_ratio
        )
