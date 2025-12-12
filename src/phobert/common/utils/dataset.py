"""Dataset loader cho PhoBERT Classification từ JSONL files."""

import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict


class FakeNewsDataset(Dataset):
    """Dataset cho fake news classification từ JSONL.
    
    Format JSONL:
        {"text": "...", "label": 0/1, "input_ids": [...], "attention_mask": [...]}
    """
    
    def __init__(self, jsonl_path: str):
        """Load dataset từ file JSONL.
        
        Args:
            jsonl_path: Đường dẫn file JSONL
        """
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
        
        print(f"✓ Loaded {len(self.data)} samples từ {jsonl_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Lấy 1 sample.
        
        Returns:
            Dict chứa input_ids, attention_mask, label (tensor)
        """
        item = self.data[idx]
        
        # Truncate to PhoBERT's max position embeddings (258)
        max_len = 256  # An toàn hơn là 258
        input_ids = item['input_ids'][:max_len]
        attention_mask = item['attention_mask'][:max_len]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label': torch.tensor(item['label'], dtype=torch.long)
        }
    
    def get_labels(self) -> List[int]:
        """Lấy tất cả labels (để tính class weights)."""
        return [item['label'] for item in self.data]
