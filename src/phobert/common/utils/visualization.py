"""Visualization utilities cho training metrics."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict


sns.set_style('whitegrid')


class MetricsVisualizer:
    """Vẽ biểu đồ cho training metrics."""
    
    @staticmethod
    def plot_learning_curves(train_losses: List[float], val_losses: List[float],
                            train_f1s: List[float], val_f1s: List[float],
                            save_path: str):
        """Vẽ learning curves (loss & F1).
        
        Args:
            train_losses: Train loss theo epoch
            val_losses: Val loss theo epoch
            train_f1s: Train F1 theo epoch
            val_f1s: Val F1 theo epoch
            save_path: Đường dẫn lưu figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curve
        axes[0].plot(epochs, train_losses, 'o-', label='Train Loss', 
                    linewidth=3, markersize=8, color='#E63946', alpha=0.8)
        axes[0].plot(epochs, val_losses, 's-', label='Val Loss', 
                    linewidth=3, markersize=8, color='#457B9D', alpha=0.8)
        
        # Highlight best val loss
        best_val_idx = np.argmin(val_losses)
        axes[0].plot(epochs[best_val_idx], val_losses[best_val_idx], 
                    '*', markersize=20, color='gold', markeredgecolor='black', 
                    markeredgewidth=1.5, label=f'Best Val Loss (Epoch {best_val_idx+1})', zorder=5)
        
        axes[0].set_xlabel('Epoch', fontsize=14, fontweight='bold', labelpad=10)
        axes[0].set_ylabel('Loss', fontsize=14, fontweight='bold', labelpad=10)
        axes[0].set_title('Training & Validation Loss', fontsize=16, fontweight='bold', pad=15)
        axes[0].legend(fontsize=11, loc='best', framealpha=0.9)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].tick_params(axis='both', which='major', labelsize=11)
        
        # F1 curve
        axes[1].plot(epochs, train_f1s, 'o-', label='Train Macro-F1', 
                    linewidth=3, markersize=8, color='#2A9D8F', alpha=0.8)
        axes[1].plot(epochs, val_f1s, 's-', label='Val Macro-F1', 
                    linewidth=3, markersize=8, color='#F4A261', alpha=0.8)
        
        # Highlight best val F1
        best_f1_idx = np.argmax(val_f1s)
        axes[1].plot(epochs[best_f1_idx], val_f1s[best_f1_idx], 
                    '*', markersize=20, color='gold', markeredgecolor='black', 
                    markeredgewidth=1.5, label=f'Best Val F1 (Epoch {best_f1_idx+1})', zorder=5)
        
        axes[1].set_xlabel('Epoch', fontsize=14, fontweight='bold', labelpad=10)
        axes[1].set_ylabel('Macro-F1 Score', fontsize=14, fontweight='bold', labelpad=10)
        axes[1].set_title('Training & Validation Macro-F1', fontsize=16, fontweight='bold', pad=15)
        axes[1].legend(fontsize=11, loc='best', framealpha=0.9)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].tick_params(axis='both', which='major', labelsize=11)
        
        # Set y-axis limits cho F1 (0-1)
        axes[1].set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.2)
        plt.close()
        print(f"✓ Đã lưu learning curves: {save_path}")
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                             save_path: str):
        """Vẽ confusion matrix.
        
        Args:
            cm: Confusion matrix (2x2)
            class_names: Tên các classes
            save_path: Đường dẫn lưu figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Vẽ heatmap
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Sample Count'}, 
                   ax=ax, 
                   linewidths=2, linecolor='white',
                   square=True,
                   cbar=True)
        
        # Manually add annotations to ensure all cells have text
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                value = cm[i][j]
                # Use white text for dark cells, black for light cells
                text_color = 'white' if cm[i][j] > cm.max() / 2 else 'black'
                ax.text(j + 0.5, i + 0.5, f'{value}',
                       ha='center', va='center',
                       fontsize=20, fontweight='bold',
                       color=text_color)
        
        # Labels và title
        ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel('True Label', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=15)
        
        # Tăng font size cho tick labels
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Thêm thông tin chi tiết
        total = cm.sum()
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tn + tp) / total
        
        info_text = f'Total: {total:,} | Accuracy: {accuracy:.2%}\n'
        info_text += f'TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}'
        
        ax.text(0.5, -0.15, info_text, 
               transform=ax.transAxes,
               ha='center', va='top',
               fontsize=11, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.2)
        plt.close()
        print(f"✓ Đã lưu confusion matrix: {save_path}")
    
    @staticmethod
    def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_score: float,
                      save_path: str):
        """Vẽ ROC curve.
        
        Args:
            fpr: False positive rate
            tpr: True positive rate
            auc_score: AUC score
            save_path: Đường dẫn lưu figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # ROC curve
        ax.plot(fpr, tpr, linewidth=3, label=f'ROC Curve (AUC = {auc_score:.4f})', color='#2E86AB')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.50)', alpha=0.6)
        
        # Fill area under curve
        ax.fill_between(fpr, tpr, alpha=0.2, color='#2E86AB')
        
        # Labels và styling
        ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel('True Positive Rate (Recall)', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_title('ROC Curve (Receiver Operating Characteristic)', fontsize=16, fontweight='bold', pad=15)
        ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Set limits
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        
        # Tick params
        ax.tick_params(axis='both', which='major', labelsize=11)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.2)
        plt.close()
        print(f"✓ Đã lưu ROC curve: {save_path}")
    
    @staticmethod
    def plot_pr_curve(precision: np.ndarray, recall: np.ndarray, 
                     ap_score: float, save_path: str):
        """Vẽ Precision-Recall curve.
        
        Args:
            precision: Precision values
            recall: Recall values
            ap_score: Average Precision score
            save_path: Đường dẫn lưu figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # PR curve
        ax.plot(recall, precision, linewidth=3, 
               label=f'PR Curve (AP = {ap_score:.4f})', color='#A23B72')
        
        # Fill area under curve
        ax.fill_between(recall, precision, alpha=0.2, color='#A23B72')
        
        # Baseline (no-skill classifier)
        no_skill = len(precision[precision > 0]) / len(precision) if len(precision) > 0 else 0.5
        ax.plot([0, 1], [no_skill, no_skill], 'k--', linewidth=2, 
               label=f'No Skill (AP = {no_skill:.2f})', alpha=0.6)
        
        # Labels và styling
        ax.set_xlabel('Recall (Sensitivity)', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_title('Precision-Recall Curve', fontsize=16, fontweight='bold', pad=15)
        ax.legend(fontsize=12, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Set limits
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        
        # Tick params
        ax.tick_params(axis='both', which='major', labelsize=11)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.2)
        plt.close()
        print(f"✓ Đã lưu PR curve: {save_path}")
    
    @staticmethod
    def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray,
                              n_bins: int = 10, save_path: str = None):
        """Vẽ calibration curve (reliability diagram).
        
        Args:
            y_true: Ground truth
            y_prob: Predicted probabilities
            n_bins: Số bins
            save_path: Đường dẫn lưu
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Tính empirical probabilities
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        empirical_probs = []
        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i+1])
            if mask.sum() > 0:
                empirical_probs.append(y_true[mask].mean())
            else:
                empirical_probs.append(bin_centers[i])
        
        # Plot
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=1)
        ax.plot(bin_centers, empirical_probs, 'o-', linewidth=2, 
               label='Model Calibration', markersize=8)
        
        ax.set_xlabel('Predicted Probability', fontsize=12)
        ax.set_ylabel('Empirical Probability', fontsize=12)
        ax.set_title('Calibration Curve', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Đã lưu calibration curve: {save_path}")
        else:
            plt.show()
