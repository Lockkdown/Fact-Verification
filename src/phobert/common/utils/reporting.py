"""Report generation cho PhoBERT training results."""

import json
from pathlib import Path
from typing import Dict
import numpy as np


class ReportGenerator:
    """Generate training reports in multiple formats."""
    
    @staticmethod
    def generate_metrics_report(metrics: Dict, save_path: str):
        """Tạo báo cáo metrics dạng text file.
        
        Args:
            metrics: Dict chứa các metrics
            save_path: Đường dẫn lưu file (.txt)
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("PHOBERT FAKE NEWS CLASSIFICATION - METRICS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Core metrics
            f.write("1. CORE METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"  • Accuracy:          {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
            f.write(f"  • Macro Precision:   {metrics['macro_precision']:.4f}\n")
            f.write(f"  • Macro Recall:      {metrics['macro_recall']:.4f}\n")
            f.write(f"  • Macro F1-Score:    {metrics['macro_f1']:.4f}\n")
            
            # AUC scores
            if 'roc_auc' in metrics:
                f.write(f"\n2. AUC SCORES\n")
                f.write("-" * 80 + "\n")
                f.write(f"  • ROC-AUC:           {metrics['roc_auc']:.4f}\n")
                f.write(f"  • PR-AUC:            {metrics['pr_auc']:.4f}\n")
            
            # Per-class metrics
            f.write(f"\n3. PER-CLASS METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Class 0 (Real News):\n")
            f.write(f"    - Precision:       {metrics['precision_class_0']:.4f}\n")
            f.write(f"    - Recall:          {metrics['recall_class_0']:.4f}\n")
            f.write(f"    - F1-Score:        {metrics['f1_class_0']:.4f}\n")
            f.write(f"\n")
            f.write(f"  Class 1 (Fake News):\n")
            f.write(f"    - Precision:       {metrics['precision_class_1']:.4f}\n")
            f.write(f"    - Recall:          {metrics['recall_class_1']:.4f}\n")
            f.write(f"    - F1-Score:        {metrics['f1_class_1']:.4f}\n")
            
            # Confusion matrix
            if 'confusion_matrix' in metrics:
                cm = metrics['confusion_matrix']
                f.write(f"\n4. CONFUSION MATRIX\n")
                f.write("-" * 80 + "\n")
                f.write(f"                    Predicted\n")
                f.write(f"                    Real        Fake\n")
                f.write(f"  Actual Real      {cm[0,0]:5d}       {cm[0,1]:5d}\n")
                f.write(f"  Actual Fake      {cm[1,0]:5d}       {cm[1,1]:5d}\n")
                f.write(f"\n")
                f.write(f"  True Negatives (TN):  {metrics['tn']:5d}\n")
                f.write(f"  False Positives (FP): {metrics['fp']:5d}\n")
                f.write(f"  False Negatives (FN): {metrics['fn']:5d}\n")
                f.write(f"  True Positives (TP):  {metrics['tp']:5d}\n")
            
            # Calibration metrics (if available)
            if 'ece' in metrics:
                f.write(f"\n5. CALIBRATION METRICS\n")
                f.write("-" * 80 + "\n")
                f.write(f"  • Expected Calibration Error (ECE): {metrics['ece']:.4f}\n")
                f.write(f"  • Maximum Calibration Error (MCE):  {metrics['mce']:.4f}\n")
                f.write(f"  • Brier Score:                      {metrics['brier_score']:.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"✓ Metrics report saved: {save_path}")
    
    @staticmethod
    def save_confusion_matrix_data(cm: np.ndarray, save_path: str):
        """Lưu confusion matrix dạng JSON.
        
        Args:
            cm: Confusion matrix (2x2 numpy array)
            save_path: Đường dẫn lưu file (.json)
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract values
        tn, fp, fn, tp = cm.ravel()
        
        data = {
            "confusion_matrix": cm.tolist(),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "total_samples": int(cm.sum()),
            "labels": ["Real", "Fake"]
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Confusion matrix saved: {save_path}")
    
    @staticmethod
    def generate_training_summary(
        best_val_f1: float,
        final_metrics: Dict,
        train_config: Dict,
        save_path: str
    ):
        """Tạo tổng kết training session.
        
        Args:
            best_val_f1: Best validation F1 score
            final_metrics: Final evaluation metrics
            train_config: Training configuration
            save_path: Đường dẫn lưu file (.txt)
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("PHOBERT FAKE NEWS CLASSIFICATION - TRAINING SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # Training config
            f.write("TRAINING CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            for key, value in train_config.items():
                f.write(f"  {key:25s}: {value}\n")
            
            # Best results
            f.write(f"\n\nBEST VALIDATION RESULTS\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Best Validation F1:      {best_val_f1:.4f}\n")
            f.write(f"  Final Accuracy:          {final_metrics['accuracy']:.4f}\n")
            f.write(f"  Final ROC-AUC:           {final_metrics['roc_auc']:.4f}\n")
            f.write(f"  Final PR-AUC:            {final_metrics['pr_auc']:.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"✓ Training summary saved: {save_path}")
