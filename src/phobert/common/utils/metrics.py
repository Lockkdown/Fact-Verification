"""Metrics computation cho fake news classification."""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
from typing import Dict, Tuple


class MetricsCalculator:
    """Tính toán tất cả metrics cho classification."""
    
    @staticmethod
    def compute_core_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                            y_prob: np.ndarray = None) -> Dict:
        """Tính core metrics (bắt buộc).
        
        Args:
            y_true: Ground truth labels (N,)
            y_pred: Predicted labels (N,)
            y_prob: Predicted probabilities cho class 1 (N,) - optional
        
        Returns:
            Dict chứa accuracy, precision, recall, f1, confusion matrix
        """
        # Accuracy
        acc = accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1 (macro & per-class)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Macro averages
        macro_p = precision.mean()
        macro_r = recall.mean()
        macro_f1 = f1.mean()
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        metrics = {
            'accuracy': acc,
            'macro_precision': macro_p,
            'macro_recall': macro_r,
            'macro_f1': macro_f1,
            'precision_class_0': precision[0],
            'precision_class_1': precision[1],
            'recall_class_0': recall[0],
            'recall_class_1': recall[1],
            'f1_class_0': f1[0],
            'f1_class_1': f1[1],
            'confusion_matrix': cm,
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
        }
        
        # ROC-AUC & PR-AUC nếu có probabilities
        if y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                metrics['pr_auc'] = average_precision_score(y_true, y_prob)
            except:
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0
        
        return metrics
    
    @staticmethod
    def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, 
                               method: str = 'f1') -> Tuple[float, float]:
        """Tìm threshold tối ưu.
        
        Args:
            y_true: Ground truth (N,)
            y_prob: Probabilities cho class 1 (N,)
            method: 'f1' hoặc 'youden'
        
        Returns:
            (optimal_threshold, metric_value)
        """
        if method == 'f1':
            # Maximize F1 score
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            best_f1 = f1_scores[best_idx]
            return best_threshold, best_f1
        
        elif method == 'youden':
            # Maximize Youden's J statistic: J = sensitivity + specificity - 1
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            best_threshold = thresholds[best_idx]
            best_j = j_scores[best_idx]
            return best_threshold, best_j
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, 
                                    n_bins: int = 10) -> Dict:
        """Tính calibration metrics (ECE, MCE, Brier).
        
        Args:
            y_true: Ground truth (N,)
            y_prob: Predicted probabilities (N,)
            n_bins: Số bins cho ECE/MCE
        
        Returns:
            Dict chứa ece, mce, brier_score
        """
        # Brier score
        brier = np.mean((y_prob - y_true) ** 2)
        
        # ECE & MCE
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        mce = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Samples trong bin này
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                
                # Calibration error
                cal_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                
                ece += prop_in_bin * cal_error
                mce = max(mce, cal_error)
        
        return {
            'ece': ece,
            'mce': mce,
            'brier_score': brier
        }
    
    @staticmethod
    def print_metrics_report(metrics: Dict, prefix: str = ""):
        """In báo cáo metrics đẹp.
        
        Args:
            metrics: Dict metrics
            prefix: Prefix cho mỗi dòng (ví dụ: "Val")
        """
        print(f"\n{'='*60}")
        print(f"{prefix} METRICS REPORT")
        print(f"{'='*60}")
        
        # Core metrics
        print(f"\n📊 Core Metrics:")
        print(f"  • Accuracy:       {metrics['accuracy']:.4f}")
        print(f"  • Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"  • Macro Recall:    {metrics['macro_recall']:.4f}")
        print(f"  • Macro F1:        {metrics['macro_f1']:.4f}")
        
        # Per-class
        print(f"\n📈 Per-Class Metrics:")
        print(f"  Class 0 (Real):  P={metrics['precision_class_0']:.4f}, R={metrics['recall_class_0']:.4f}, F1={metrics['f1_class_0']:.4f}")
        print(f"  Class 1 (Fake):  P={metrics['precision_class_1']:.4f}, R={metrics['recall_class_1']:.4f}, F1={metrics['f1_class_1']:.4f}")
        
        # Confusion matrix
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            print(f"\n🔢 Confusion Matrix:")
            print(f"           Predicted")
            print(f"           Real  Fake")
            print(f"  Real    [{cm[0,0]:5d} {cm[0,1]:5d}]")
            print(f"  Fake    [{cm[1,0]:5d} {cm[1,1]:5d}]")
            print(f"  TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")
        
        # ROC-AUC & PR-AUC
        if 'roc_auc' in metrics:
            print(f"\n🎯 AUC Scores:")
            print(f"  • ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  • PR-AUC:  {metrics['pr_auc']:.4f}")
        
        # Calibration
        if 'ece' in metrics:
            print(f"\n📐 Calibration Metrics:")
            print(f"  • ECE (Expected Calibration Error): {metrics['ece']:.4f}")
            print(f"  • MCE (Maximum Calibration Error):  {metrics['mce']:.4f}")
            print(f"  • Brier Score: {metrics['brier_score']:.4f}")
        
        # Threshold tuning
        if 'optimal_threshold_f1' in metrics:
            print(f"\n⚙️  Optimal Thresholds:")
            print(f"  • F1-optimal:     {metrics['optimal_threshold_f1']:.4f} (F1={metrics['best_f1']:.4f})")
        if 'optimal_threshold_youden' in metrics:
            print(f"  • Youden-optimal: {metrics['optimal_threshold_youden']:.4f} (J={metrics['best_youden']:.4f})")
        
        print(f"\n{'='*60}\n")
