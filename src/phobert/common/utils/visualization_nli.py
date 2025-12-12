"""Visualization utilities cho NLI multi-class (4 classes)."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize


sns.set_style('whitegrid')


class NLIVisualizer:
    """Visualizer cho NLI task (4 classes)."""
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                              class_names: List[str], title: str, save_path: str):
        """Vẽ confusion matrix cho multi-class.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            class_names: Tên các classes
            title: Title của plot
            save_path: Đường dẫn lưu
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Heatmap
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'},
                   ax=ax,
                   linewidths=2, linecolor='white',
                   square=True)
        
        # Manually add annotations to ensure all cells have text
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                value = cm[i][j]
                # Use white text for dark cells, black for light cells
                text_color = 'white' if cm[i][j] > cm.max() / 2 else 'black'
                ax.text(j + 0.5, i + 0.5, f'{value}',
                       ha='center', va='center',
                       fontsize=14, fontweight='bold',
                       color=text_color)
        
        # Labels
        ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel('True Label', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax.tick_params(axis='both', which='major', labelsize=11)
        
        # Accuracy info
        accuracy = np.trace(cm) / cm.sum()
        total = cm.sum()
        info_text = f'Total: {total:,} samples | Accuracy: {accuracy:.2%}'
        ax.text(0.5, -0.12, info_text,
               transform=ax.transAxes,
               ha='center', va='top',
               fontsize=11,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.2)
        plt.close()
    
    @staticmethod
    def plot_per_class_f1(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: List[str], title: str, save_path: str):
        """Vẽ per-class F1 scores.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            class_names: Tên các classes
            title: Title của plot
            save_path: Đường dẫn lưu
        """
        from sklearn.metrics import f1_score
        
        # Compute per-class F1
        f1_scores = f1_score(y_true, y_pred, average=None)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Bar chart
        colors = ['#E63946', '#457B9D', '#2A9D8F', '#F4A261']
        bars = ax.bar(range(len(class_names)), f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Value labels on bars
        for i, (bar, score) in enumerate(zip(bars, f1_scores)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}',
                   ha='center', va='bottom',
                   fontsize=12, fontweight='bold')
        
        # Styling
        ax.set_xlabel('Class', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel('F1 Score', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=11)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.tick_params(axis='both', which='major', labelsize=11)
        
        # Macro-F1 line
        macro_f1 = f1_scores.mean()
        ax.axhline(macro_f1, color='red', linestyle='--', linewidth=2, 
                  label=f'Macro-F1: {macro_f1:.3f}', alpha=0.7)
        ax.legend(fontsize=11, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.2)
        plt.close()
    
    @staticmethod
    def plot_roc_curves(y_true: np.ndarray, y_probs: np.ndarray,
                       class_names: List[str], title: str, save_path: str):
        """Vẽ ROC curves cho multi-class (one-vs-rest).
        
        Args:
            y_true: Ground truth labels (N,)
            y_probs: Predicted probabilities (N, num_classes)
            class_names: Tên các classes
            title: Title của plot
            save_path: Đường dẫn lưu
        """
        n_classes = len(class_names)
        
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#E63946', '#457B9D', '#2A9D8F', '#F4A261']
        
        # ROC curve for each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, linewidth=2.5, color=colors[i],
                   label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
        
        # Diagonal line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC = 0.500)', alpha=0.6)
        
        # Styling
        ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.tick_params(axis='both', which='major', labelsize=11)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.2)
        plt.close()
    
    @staticmethod
    def plot_pr_curves(y_true: np.ndarray, y_probs: np.ndarray,
                      class_names: List[str], title: str, save_path: str):
        """Vẽ Precision-Recall curves cho multi-class (one-vs-rest).
        
        Args:
            y_true: Ground truth labels (N,)
            y_probs: Predicted probabilities (N, num_classes)
            class_names: Tên các classes
            title: Title của plot
            save_path: Đường dẫn lưu
        """
        n_classes = len(class_names)
        
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#E63946', '#457B9D', '#2A9D8F', '#F4A261']
        
        # PR curve for each class
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
            ap_score = average_precision_score(y_true_bin[:, i], y_probs[:, i])
            
            ax.plot(recall, precision, linewidth=2.5, color=colors[i],
                   label=f'{class_names[i]} (AP = {ap_score:.3f})')
        
        # Styling
        ax.set_xlabel('Recall', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel('Precision', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.tick_params(axis='both', which='major', labelsize=11)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.2)
        plt.close()
    
    @staticmethod
    def plot_training_curves(train_losses: List[float], val_losses: List[float],
                            train_f1s: List[float] = None, val_f1s: List[float] = None,
                            save_path: str = None):
        """Vẽ training curves.
        
        Args:
            train_losses: Train losses theo epoch
            val_losses: Val losses theo epoch
            train_f1s: Train F1s theo epoch (optional)
            val_f1s: Val F1s theo epoch (optional)
            save_path: Đường dẫn lưu
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curve
        axes[0].plot(epochs, train_losses, 'o-', label='Train Loss',
                    linewidth=2.5, markersize=8, color='#E63946', alpha=0.8)
        axes[0].plot(epochs, val_losses, 's-', label='Val Loss',
                    linewidth=2.5, markersize=8, color='#457B9D', alpha=0.8)
        
        best_val_idx = np.argmin(val_losses)
        axes[0].plot(epochs[best_val_idx], val_losses[best_val_idx],
                    '*', markersize=20, color='gold', markeredgecolor='black',
                    markeredgewidth=1.5, label=f'Best (Epoch {best_val_idx+1})', zorder=5)
        
        axes[0].set_xlabel('Epoch', fontsize=13, fontweight='bold', labelpad=10)
        axes[0].set_ylabel('Loss', fontsize=13, fontweight='bold', labelpad=10)
        axes[0].set_title('Loss Curves', fontsize=15, fontweight='bold', pad=15)
        axes[0].legend(fontsize=11, loc='best', framealpha=0.9)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].tick_params(axis='both', which='major', labelsize=10)
        
        # F1 curve (chỉ vẽ nếu có F1 data)
        if val_f1s is not None:
            if train_f1s is not None:
                axes[1].plot(epochs, train_f1s, 'o-', label='Train Macro-F1',
                            linewidth=2.5, markersize=8, color='#2A9D8F', alpha=0.8)
            axes[1].plot(epochs, val_f1s, 's-', label='Val Macro-F1',
                        linewidth=2.5, markersize=8, color='#F4A261', alpha=0.8)
            
            best_f1_idx = np.argmax(val_f1s)
            axes[1].plot(epochs[best_f1_idx], val_f1s[best_f1_idx],
                        '*', markersize=20, color='gold', markeredgecolor='black',
                        markeredgewidth=1.5, label=f'Best (Epoch {best_f1_idx+1})', zorder=5)
        
        axes[1].set_xlabel('Epoch', fontsize=13, fontweight='bold', labelpad=10)
        axes[1].set_ylabel('Macro-F1', fontsize=13, fontweight='bold', labelpad=10)
        axes[1].set_title('Macro-F1 Curves', fontsize=15, fontweight='bold', pad=15)
        axes[1].legend(fontsize=11, loc='best', framealpha=0.9)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].set_ylim([0, 1.05])
        axes[1].tick_params(axis='both', which='major', labelsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.2)
        plt.close()
