import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict


class NLIDataset(Dataset):
    """Dataset cho NLI task từ JSONL.
    
    Format JSONL (RAW TEXT):
        {"premise": "...", "hypothesis": "...", "label": 0, "pair_id": "..."}
    
    Labels:
        0: Entailment
        1: Neutral
        2: Contradiction
        3: OTHER
    """
    
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 128):
        """Load dataset từ file JSONL và PRE-TOKENIZE.
        
        Args:
            jsonl_path: Đường dẫn file JSONL
            tokenizer: PhoBERT tokenizer
            max_length: Max sequence length (default: 128)
        """
        print(f"📁 Loading dataset from {jsonl_path}...")
        
        # Load raw data
        raw_samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                raw_samples.append(json.loads(line.strip()))
        
        print(f"✓ Loaded {len(raw_samples)} raw samples")
        print(f"⚡ Pre-tokenizing with max_length={max_length}...")
        
        # ⚡ PRE-TOKENIZE 1 lần duy nhất
        # Suppress warning về overflowing tokens
        import warnings
        from transformers import logging as transformers_logging
        
        # Save original level
        original_level = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()  # Only show errors
        
        self.samples = []
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*overflowing tokens.*')
            
            for sample in raw_samples:
                encoding = tokenizer(
                    sample['premise'],
                    sample['hypothesis'],
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'  # Return PyTorch tensors
                )
                self.samples.append({
                    'input_ids': encoding['input_ids'].squeeze(0),  # Remove batch dim
                    'attention_mask': encoding['attention_mask'].squeeze(0),
                    'label': torch.tensor(sample['label'], dtype=torch.long)
                })
        
        # Restore original level
        transformers_logging.set_verbosity(original_level)
        
        print(f"✓ Pre-tokenized {len(self.samples)} samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Lấy 1 sample.
        
        Returns:
            Dict chứa input_ids, attention_mask, label (tensor)
        """
        # ⚡ Chỉ trả về data đã tokenized sẵn
        return self.samples[idx]
    
    def get_labels(self) -> List[int]:
        """Lấy tất cả labels (để tính class weights nếu cần)."""
        return [item['label'].item() for item in self.samples]
