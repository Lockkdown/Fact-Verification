"""
Hard Negative Mining Module - Error Analysis & Sample Selection

Phân tích các mẫu model hiện tại phân loại sai để tạo hard negatives cho fine-tuning.
Author: Lockdown
Date: Nov 26, 2025
"""

import os
import json
import torch
import numpy as np
import logging
# Suppress specific PyTorch warnings
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import model và dataset từ fact_checking
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.fact_checking.model import PhoBERTFactCheck
from src.pipeline.fact_checking.train import ViFactCheckDataset, LABEL_MAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HardNegativeMiner:
    """Phân tích lỗi và tạo hard negatives từ model hiện tại."""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.model_path = model_path
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        logger.info(f"🔍 Loading model from {model_path}")
        self.model, self.tokenizer = self._load_model()
        
        # Mapping từ int -> string labels
        self.id_to_label = {0: 'SUPPORT', 1: 'REFUTE', 2: 'NEI'}
        self.label_to_id = {'SUPPORT': 0, 'REFUTE': 1, 'NEI': 2}
        
        # Storage for results
        self.error_analysis = {}
        self.hard_negatives = []
        
    def _load_model(self) -> Tuple[PhoBERTFactCheck, Any]:
        """Load pre-trained model and tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model = PhoBERTFactCheck(num_classes=3)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        logger.info("✅ Model loaded successfully")
        return model, tokenizer
    
    def analyze_errors_on_split(self, split: str) -> Dict[str, Any]:
        """Phân tích lỗi trên một split (dev/train)."""
        logger.info(f"🔍 Analyzing errors on {split} split...")
        
        # Load dataset
        data_path = self.config['data'][f'{split}_path']
        if not Path(data_path).is_absolute():
            data_path = project_root / data_path
            
        dataset = ViFactCheckDataset(data_path, self.tokenizer, max_len=512)
        
        # Collect predictions
        all_preds = []
        all_labels = []
        all_probs = []
        all_samples = []
        
        with torch.no_grad():
            for i, sample in enumerate(dataset):
                # Process single sample
                input_ids = sample['input_ids'].unsqueeze(0).to(self.device)
                attention_mask = sample['attention_mask'].unsqueeze(0).to(self.device)
                true_label = sample['labels'].item()
                
                # Model prediction
                outputs = self.model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)
                pred_label = torch.argmax(outputs, dim=1).item()
                confidence = probs.max().item()
                
                all_preds.append(pred_label)
                all_labels.append(true_label)
                all_probs.append(probs.cpu().numpy()[0])
                
                # Store sample info for hard negative selection
                all_samples.append({
                    'index': i,
                    'pred_label': pred_label,
                    'true_label': true_label,
                    'confidence': confidence,
                    'probs': probs.cpu().numpy()[0].tolist(), # Convert ndarray to list
                    'is_correct': pred_label == true_label
                })
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=list(LABEL_MAP.values()), 
                                     digits=4, output_dict=True)
        cm = confusion_matrix(all_labels, all_preds)
        
        # Error analysis by class
        errors_by_class = defaultdict(list)
        confidence_by_correctness = {'correct': [], 'incorrect': []}
        
        for sample in all_samples:
            if sample['is_correct']:
                confidence_by_correctness['correct'].append(sample['confidence'])
            else:
                confidence_by_correctness['incorrect'].append(sample['confidence'])
                true_class = self.id_to_label[sample['true_label']]
                pred_class = self.id_to_label[sample['pred_label']]
                errors_by_class[f"{true_class}→{pred_class}"].append(sample)
        
        logger.info(f"📊 {split} Accuracy: {accuracy:.4f}")
        logger.info(f"📊 Error patterns: {dict([(k, len(v)) for k, v in errors_by_class.items()])}")
        
        return {
            'split': split,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'samples': all_samples,
            'errors_by_class': dict(errors_by_class),
            'confidence_stats': {
                'correct_mean': float(np.mean(confidence_by_correctness['correct'])),
                'incorrect_mean': float(np.mean(confidence_by_correctness['incorrect']))
            }
        }
    
    def select_hard_negatives(self, analysis_results: List[Dict]) -> List[Dict]:
        """Chọn hard negatives theo chiến thuật Top-K (Target Pool Size)."""
        logger.info("🎯 Selecting hard negatives (Top-K Strategy)...")
        
        config = self.config.get('hard_negative_config', {})
        # Target pool size: Ưu tiên số lượng mẫu (ví dụ: 1000)
        target_count = config.get('target_hard_negative_count', 1000)
        min_conf_threshold = self.config['training'].get('min_confidence_threshold', 0.9)
        logger.info(f"🎯 Selection Criteria: Incorrect predictions OR (Correct and Confidence < {min_conf_threshold})")
        
        all_errors = []
        
        for result in analysis_results:
            split = result['split']
            # Thu thập TẤT CẢ các mẫu sai HOẶC đúng nhưng kém tự tin
            for sample in result['samples']:
                is_hard = False
                error_type = ""
                
                if not sample['is_correct']:
                    is_hard = True
                    error_type = f"{self.id_to_label[sample['true_label']]}→{self.id_to_label[sample['pred_label']]}"
                elif sample['is_correct'] and sample['confidence'] < min_conf_threshold:
                    is_hard = True
                    error_type = f"LOW_CONF ({sample['confidence']:.2f})"
                    
                if is_hard:
                    sample_data = {
                        'split': split,
                        'index': sample['index'],
                        'true_label': sample['true_label'],
                        'pred_label': sample['pred_label'],
                        'confidence': sample['confidence'],
                        'error_type': error_type,
                        'weight': self.config['training'].get('hard_negative_weight', 2.0)
                    }
                    all_errors.append(sample_data)
        
        # Sắp xếp theo confidence tăng dần (Ưu tiên mẫu có confidence thấp nhất - tức là model bối rối nhất)
        # Lưu ý: Logic cũ là sort confidence giảm dần cho mẫu sai (sai mà tự tin cao là nguy hiểm).
        # Nhưng logic mới trộn cả 2, nên sort confidence thấp dần (bối rối nhất) có vẻ hợp lý hơn cho cả 2 loại.
        all_errors.sort(key=lambda x: x['confidence'], reverse=False)
        
        # Lấy Top-K
        hard_negatives = all_errors[:target_count]
        
        # Thống kê theo loại lỗi
        hard_negatives_by_error_type = defaultdict(list)
        for hn in hard_negatives:
            hard_negatives_by_error_type[hn['error_type']].append(hn)

        logger.info(f"✅ Selected {len(hard_negatives)} hard negatives (Target: {target_count})")
        logger.info(f"📊 By error type: {dict([(k, len(v)) for k, v in hard_negatives_by_error_type.items()])}")
        
        return hard_negatives
    
    def run_analysis(self, splits: List[str]) -> Tuple[List[Dict], List[Dict]]:
        """Chạy toàn bộ quy trình phân tích."""
        logger.info("🚀 Starting hard negative mining analysis...")
        
        # Analyze each split
        analysis_results = []
        for split in splits:
            result = self.analyze_errors_on_split(split)
            analysis_results.append(result)
        
        # Select hard negatives
        hard_negatives = self.select_hard_negatives(analysis_results)
        
        # Save results
        self.error_analysis = {
            'analysis_results': analysis_results,
            'hard_negatives': hard_negatives,
            'config': self.config
        }
        
        return analysis_results, hard_negatives
    
    def save_results(self, output_path: str):
        """Lưu kết quả phân tích."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.error_analysis, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Analysis results saved to: {output_path}")


def main():
    """Demo chạy hard negative mining."""
    # Load config
    config_path = project_root / "config/fact_checking/train_config_hard_negatives.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Model path
    model_path = project_root / config['model']['pretrained_model_path']
    
    # Initialize miner
    miner = HardNegativeMiner(str(model_path), config)
    
    # Run analysis
    splits = config['hard_negative_config']['error_analysis_splits']
    analysis_results, hard_negatives = miner.run_analysis(splits)
    
    # Save results
    output_path = project_root / config['data']['hard_negatives_cache']
    miner.save_results(str(output_path))
    
    print(f"🎉 Hard negative mining completed!")
    print(f"📊 Found {len(hard_negatives)} hard negatives")


if __name__ == "__main__":
    main()
