import torch
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

from model import PhoBERTFactCheck
from train import ViFactCheckDataset, LABEL_MAP

sns.set_style('whitegrid')

def evaluate_only(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")

    # Load Model
    print(f"📂 Loading model from {args.model_path}...")
    
    # Try to load config from model folder if possible to get max_len
    # But for now, default to 512 as per optimized training
    max_len = 512 
    
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    # Handle both full checkpoint dict and direct state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    model = PhoBERTFactCheck(num_classes=3)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Tokenizer & Data
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    project_root = Path(__file__).parent.parent.parent.parent
    data_dir = project_root / "dataset" / "processed" / "vifactcheck"
    
    # Select split
    if args.split == 'dev':
        data_path = data_dir / "vifactcheck_dev.jsonl"
    elif args.split == 'test':
        data_path = data_dir / "vifactcheck_test.jsonl"
    else:
        data_path = data_dir / "vifactcheck_train.jsonl"
        
    print(f"📉 Evaluating on: {data_path} (Max Len: {max_len})")
    ds = ViFactCheckDataset(data_path, tokenizer, max_len=max_len)
    
    # Update padding value dynamically
    global PAD_TOKEN_ID
    PAD_TOKEN_ID = tokenizer.pad_token_id
    
    def collate_wrapper(batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=PAD_TOKEN_ID)
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels_stacked = torch.stack(labels)
        
        return {
            'input_ids': input_ids_padded,
            'attention_mask': attention_mask_padded,
            'labels': labels_stacked
        }

    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_wrapper)

    # Inference
    preds, true_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, mask)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(outputs, dim=1)
            
            preds.extend(pred.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    preds = np.array(preds)
    true_labels = np.array(true_labels)
    all_probs = np.array(all_probs)

    # Metrics
    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average='macro')
    report = classification_report(true_labels, preds, target_names=list(LABEL_MAP.values()), digits=4)
    report_dict = classification_report(true_labels, preds, target_names=list(LABEL_MAP.values()), digits=4, output_dict=True)
    cm = confusion_matrix(true_labels, preds)
    
    print(f"\n📊 Evaluation on {args.split.upper()}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Automatic Save Logic
    # If output_dir is provided, use it. Otherwise, save relative to model path.
    if args.output_dir:
        eval_dir = Path(args.output_dir)
    else:
        # Default: Save to 'eval' folder next to 'checkpoints' folder of the model
        # e.g. .../U8_HighLR/checkpoints/model.pt -> .../U8_HighLR/eval
        model_path = Path(args.model_path)
        if model_path.parent.name == 'checkpoints':
             eval_dir = model_path.parent.parent / "eval"
        else:
             # Fallback to project root eval if structure is weird
             project_root = Path(__file__).parent.parent.parent.parent
             eval_dir = project_root / "results" / "fact_checking" / "eval"
             
    metrics_dir = eval_dir / "metrics"
    plots_dir = eval_dir / "plots"
    
    print(f"\n💾 Saving results to: {eval_dir}")
    
    # Create directories if not exist
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Output files
    txt_save_path = metrics_dir / f"eval_results_{args.split}.txt"
    json_save_path = metrics_dir / f"eval_metrics_{args.split}.json"
    
    # Save Text Report
    with open(txt_save_path, "w", encoding="utf-8") as f:
        f.write(f"Evaluation Split: {args.split.upper()}\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {f1:.4f}\n\n")
        f.write(f"=== CLASSIFICATION REPORT ===\n{report}\n\n")
        f.write(f"=== CONFUSION MATRIX ===\n{cm}\n")
    print(f"✅ Text report saved to {txt_save_path}")
    
    # Save JSON Metrics
    metrics_data = {
        "split": args.split,
        "accuracy": acc,
        "macro_f1": f1,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist()
    }
    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"✅ JSON metrics saved to {json_save_path}")
    
    # Generate Plots
    print("\n📊 GENERATING EVALUATION PLOTS...")
    CLASS_NAMES = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
    colors = ['#E63946', '#457B9D', '#2A9D8F']
    
    # 1. Confusion Matrix
    print("📊 Generating confusion matrix...")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
               xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
               cbar_kws={'label': 'Count'}, ax=ax,
               linewidths=2, linecolor='white', square=True)
    
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
    ax.set_title(f'{args.split.upper()} Set Confusion Matrix', fontsize=16, fontweight='bold', pad=15)
    
    info_text = f'Total: {len(true_labels):,} samples | Accuracy: {acc:.2%}'
    ax.text(0.5, -0.12, info_text, transform=ax.transAxes, ha='center', va='top',
           fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(plots_dir / f'confusion_matrix_{args.split}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("✅ Confusion matrix saved")
    
    # 2. Per-class F1 scores
    print("📊 Generating per-class F1...")
    f1_scores = f1_score(true_labels, preds, average=None)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(CLASS_NAMES)), f1_scores, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=1.5)
    
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{score:.3f}', ha='center', va='bottom',
               fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Class', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('F1 Score', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title(f'Per-Class F1 Scores ({args.split.upper()})', fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    macro_f1 = f1_scores.mean()
    ax.axhline(macro_f1, color='red', linestyle='--', linewidth=2, 
              label=f'Macro-F1: {macro_f1:.3f}', alpha=0.7)
    ax.legend(fontsize=11, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(plots_dir / f'per_class_f1_{args.split}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("✅ Per-class F1 saved")
    
    # 3. ROC Curves
    print("📊 Generating ROC curves...")
    n_classes = 3
    y_true_bin = label_binarize(true_labels, classes=range(n_classes))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2.5, color=colors[i],
               label=f'{CLASS_NAMES[i]} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC = 0.500)', alpha=0.6)
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title(f'ROC Curves ({args.split.upper()})', fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    plt.tight_layout()
    plt.savefig(plots_dir / f'roc_curves_{args.split}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("✅ ROC curves saved")
    
    # 4. Precision-Recall Curves
    print("📊 Generating PR curves...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], all_probs[:, i])
        ap_score = average_precision_score(y_true_bin[:, i], all_probs[:, i])
        ax.plot(recall, precision, linewidth=2.5, color=colors[i],
               label=f'{CLASS_NAMES[i]} (AP = {ap_score:.3f})')
    
    ax.set_xlabel('Recall', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Precision', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title(f'Precision-Recall Curves ({args.split.upper()})', fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    plt.tight_layout()
    plt.savefig(plots_dir / f'pr_curves_{args.split}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("✅ PR curves saved")
    
    print(f"\n{'='*60}")
    print(f"✅ ALL PLOTS SAVED TO: {plots_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"])
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation results")
    parser.add_argument("--save_result", type=str, default=None)
    
    args = parser.parse_args()
    evaluate_only(args)
