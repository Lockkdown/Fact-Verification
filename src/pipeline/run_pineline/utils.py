"""
Utils cho ViFactCheck Pipeline - Clean Version

Author: Lockdown
Date: Nov 27, 2025
"""

import json
import time
import torch
from pathlib import Path
from typing import Dict, Any
from transformers import AutoTokenizer

from .config import PipelineConfig


def load_factcheck_model_3label(config: PipelineConfig):
    """
    Load ViFactCheck 3-label model từ checkpoint.
    
    Returns:
        (model, tokenizer, device)
    """
    from src.pipeline.fact_checking.model import PhoBERTFactCheck
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.phobert_model_name)
    
    # Load model với 3 labels (Support / Refute / NEI)
    model = PhoBERTFactCheck(
        pretrained_name=config.phobert_model_name,
        num_classes=3
    )
    
    # Load checkpoint
    checkpoint_path = Path(config.factcheck_3label_checkpoint)
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"⚠️ Warning: Checkpoint {checkpoint_path} not found!")
    
    model.to(device)
    model.eval()
    
    return model, tokenizer, device


def save_json(data: Dict[str, Any], filepath: str):
    """Save dict to JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


class Timer:
    """Context manager để đo timing."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.name:
            print(f"[Timer] {self.name}: {self.elapsed:.2f}s")
