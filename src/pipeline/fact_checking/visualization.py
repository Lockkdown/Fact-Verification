"""Visualization utilities cho ViFactCheck (3 classes)."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List
from sklearn.metrics import (
    confusion_matrix, f1_score, 
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

sns.set_style('whitegrid')


def plot_all_visualizations(
    train_losses, dev_losses,
    train_accs, dev_accs,
    train_f1s, dev_f1s,
    best_dev_labels, best_dev_preds, best_dev_probs,
    save_dir: Path
):
    """Generate all visualizations for ViFactCheck training.
    
    Args:
        train_losses: Train losses per epoch
        dev_losses: Dev losses per epoch
        train_accs: Train accuracies per epoch
        dev_accs: Dev accuracies per epoch
        train_f1s: Train F1s per epoch
        dev_f1s: Dev F1s per epoch
        best_dev_labels: Best dev true labels
        best_dev_preds: Best dev predictions
        best_dev_probs: Best dev probabilities
        save_dir: Directory to save plots
    """
    CLASS_NAMES = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
    
    print("\n📊 GENERATING VISUALIZATIONS...")
    print(f"{'='*60}")
    
    # 1. Training curves (Loss + F1)
    print("\n📈 Generating training curves...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curve
    axes[0].plot(epochs, train_losses, 'o-', label='Train Loss',
                linewidth=2.5, markersize=8, color='#E63946', alpha=0.8)
    axes[0].plot(epochs, dev_losses, 's-', label='Dev Loss',
                linewidth=2.5, markersize=8, color='#457B9D', alpha=0.8)
    
    best_loss_idx = np.argmin(dev_losses)
    axes[0].plot(epochs[best_loss_idx], dev_losses[best_loss_idx],
                '*', markersize=20, color='gold', markeredgecolor='black',
                markeredgewidth=1.5, label=f'Best (Epoch {best_loss_idx+1})', zorder=5)
    
    axes[0].set_xlabel('Epoch', fontsize=13, fontweight='bold', labelpad=10)
    axes[0].set_ylabel('Loss', fontsize=13, fontweight='bold', labelpad=10)
    axes[0].set_title('Loss Curves', fontsize=15, fontweight='bold', pad=15)
    axes[0].legend(fontsize=11, loc='best', framealpha=0.9)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # F1 curve
    axes[1].plot(epochs, train_f1s, 'o-', label='Train Macro-F1',
                linewidth=2.5, markersize=8, color='#2A9D8F', alpha=0.8)
    axes[1].plot(epochs, dev_f1s, 's-', label='Dev Macro-F1',
                linewidth=2.5, markersize=8, color='#F4A261', alpha=0.8)
    
    best_f1_idx = np.argmax(dev_f1s)
    axes[1].plot(epochs[best_f1_idx], dev_f1s[best_f1_idx],
                '*', markersize=20, color='gold', markeredgecolor='black',
                markeredgewidth=1.5, label=f'Best (Epoch {best_f1_idx+1})', zorder=5)
    
    axes[1].set_xlabel('Epoch', fontsize=13, fontweight='bold', labelpad=10)
    axes[1].set_ylabel('Macro-F1', fontsize=13, fontweight='bold', labelpad=10)
    axes[1].set_title('Macro-F1 Curves', fontsize=15, fontweight='bold', pad=15)
    axes[1].legend(fontsize=11, loc='best', framealpha=0.9)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("✅ Training curves saved")
    
    # 2. Confusion Matrix
    print("\n📊 Generating confusion matrix...")
    cm = confusion_matrix(best_dev_labels, best_dev_preds)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
               xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
               cbar_kws={'label': 'Count'}, ax=ax,
               linewidths=2, linecolor='white', square=True)
    
    # Add annotations manually
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            value = cm[i][j]
            text_color = 'white' if cm[i][j] > cm.max() / 2 else 'black'
            ax.text(j + 0.5, i + 0.5, f'{value}',
                   ha='center', va='center',
                   fontsize=14, fontweight='bold',
                   color=text_color)
    
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title('Dev Set Confusion Matrix', fontsize=16, fontweight='bold', pad=15)
    
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
    plt.savefig(save_dir / 'confusion_matrix.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("✅ Confusion matrix saved")
    
    # 3. Per-class F1 scores
    print("\n📊 Generating per-class F1...")
    f1_scores = f1_score(best_dev_labels, best_dev_preds, average=None)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#E63946', '#457B9D', '#2A9D8F']
    bars = ax.bar(range(len(CLASS_NAMES)), f1_scores, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=1.5)
    
    # Value labels on bars
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{score:.3f}',
               ha='center', va='bottom',
               fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Class', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('F1 Score', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title('Per-Class F1 Scores', fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Macro-F1 line
    macro_f1 = f1_scores.mean()
    ax.axhline(macro_f1, color='red', linestyle='--', linewidth=2, 
              label=f'Macro-F1: {macro_f1:.3f}', alpha=0.7)
    ax.legend(fontsize=11, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'per_class_f1.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("✅ Per-class F1 saved")
    
    # 4. ROC Curves (One-vs-Rest)
    print("\n📊 Generating ROC curves...")
    n_classes = 3
    y_true_bin = label_binarize(best_dev_labels, classes=range(n_classes))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#E63946', '#457B9D', '#2A9D8F']
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], best_dev_probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, linewidth=2.5, color=colors[i],
               label=f'{CLASS_NAMES[i]} (AUC = {roc_auc:.3f})')
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC = 0.500)', alpha=0.6)
    
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title('ROC Curves (One-vs-Rest)', fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    plt.tight_layout()
    plt.savefig(save_dir / 'roc_curves.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("✅ ROC curves saved")
    
    # 5. Precision-Recall Curves
    print("\n📊 Generating PR curves...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], best_dev_probs[:, i])
        ap_score = average_precision_score(y_true_bin[:, i], best_dev_probs[:, i])
        
        ax.plot(recall, precision, linewidth=2.5, color=colors[i],
               label=f'{CLASS_NAMES[i]} (AP = {ap_score:.3f})')
    
    ax.set_xlabel('Recall', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Precision', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title('Precision-Recall Curves', fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    plt.tight_layout()
    plt.savefig(save_dir / 'pr_curves.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("✅ PR curves saved")
    
    print(f"\n{'='*60}")
    print(f"✅ ALL VISUALIZATIONS SAVED TO: {save_dir}")
    print(f"{'='*60}\n")
