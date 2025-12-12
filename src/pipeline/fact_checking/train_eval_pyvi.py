"""
Train & Evaluate PhoBERT trên dữ liệu ViFactCheck được preprocess với pyvi
Kết quả lưu vào results/fact_checking/pyvi
"""
import os
import json
import argparse
import gc
import random
import warnings
import logging
from pathlib import Path
import sys

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from src.pipeline.fact_checking.model import PhoBERTFactCheck
from src.pipeline.fact_checking.visualization import plot_all_visualizations

# === CONFIGURATION ===
LABEL_MAP = {0: 'SUPPORT', 1: 'REFUTE', 2: 'NEI'}

# Paths
PYVI_DATA_DIR = project_root / "dataset" / "processed" / "vifactcheck_pyvi"
OUTPUT_DIR = project_root / "results" / "fact_checking" / "pyvi"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer_grouped_parameters_llrd(model, learning_rate, weight_decay, layer_decay=0.9):
    """
    Layer-wise Learning Rate Decay (LLRD) for PhoBERT/Roberta.
    Assigns lower learning rates to lower layers to prevent catastrophic forgetting.
    """
    opt_parameters = [] 
    named_parameters = list(model.named_parameters())
    
    # 1. Embeddings (Lowest LR)
    no_decay = ["bias", "LayerNorm.weight"]
    embed_params = [(n, p) for n, p in named_parameters if "embeddings" in n]
    embed_lr = learning_rate * (layer_decay ** 13)  # 12 layers + 1 embedding
    
    opt_parameters.append({
        "params": [p for n, p in embed_params if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
        "lr": embed_lr
    })
    opt_parameters.append({
        "params": [p for n, p in embed_params if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
        "lr": embed_lr
    })
    
    # 2. Encoder Layers (12 layers)
    for layer_i in range(12):
        layer_params = [(n, p) for n, p in named_parameters if f"encoder.layer.{layer_i}." in n]
        layer_lr = learning_rate * (layer_decay ** (12 - layer_i))
        
        opt_parameters.append({
            "params": [p for n, p in layer_params if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
            "lr": layer_lr
        })
        opt_parameters.append({
            "params": [p for n, p in layer_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": layer_lr
        })
        
    # 3. Classifier & Pooler (Head) - Highest LR (Base LR)
    head_params = [(n, p) for n, p in named_parameters if "classifier" in n or "pooler" in n]
    opt_parameters.append({
        "params": [p for n, p in head_params if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
        "lr": learning_rate
    })
    opt_parameters.append({
        "params": [p for n, p in head_params if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
        "lr": learning_rate
    })
    
    return opt_parameters


class ViFactCheckDataset(Dataset):
    """Dataset loader for ViFactCheck JSONL with DYNAMIC PADDING support."""
    def __init__(self, jsonl_path, tokenizer, max_len=256):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.path = jsonl_path
        
        if not Path(jsonl_path).exists():
            raise FileNotFoundError(f"❌ Data file not found: {jsonl_path}")

        print(f"📂 Loading data from {jsonl_path}...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.samples.append(item)
        print(f"✅ Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*overflowing tokens.*')
            encoding = self.tokenizer(
                item['statement'],
                item['evidence'],
                truncation=True,
                padding=False,
                max_length=self.max_len,
                return_tensors='pt'
            )
            
        label = item['label']
        if isinstance(label, str):
            label = int(label)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def train_epoch(model, dataloader, optimizer, criterion, device, scaler, scheduler, accumulation_steps, use_fp16=True):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(tqdm(dataloader, desc="Training", mininterval=1.0)):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        
        with autocast(device_type='cuda', enabled=use_fp16):
            outputs = model(input_ids, mask)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
            
        scaler.scale(loss).backward()
        
        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * accumulation_steps
        
        with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    all_probs = torch.cat(all_probs).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_labels, all_preds, all_probs


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    val_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            outputs = model(input_ids, mask)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    all_probs = torch.cat(all_probs).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return val_loss / len(loader), acc, f1, all_labels, all_preds, all_probs


def load_config(config_path: str) -> dict:
    """Load config from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Train & Eval PhoBERT with pyvi preprocessed data")
    parser.add_argument("--config", type=str, default="config/fact_checking/train_config_pyvi.json",
                       help="Path to config file")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate (skip training)")
    parser.add_argument("--model-path", type=str, default=None, help="Path to pretrained model (for eval-only)")
    parser.add_argument("--workers", type=int, default=1, help="Number of dataloader workers")
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    
    print(f"⚙️ Loading config from: {config_path}")
    cfg = load_config(config_path)
    
    # Extract config sections
    model_cfg = cfg['model']
    data_cfg = cfg['data']
    train_cfg = cfg['training']
    out_cfg = cfg['output']
    
    # Setup
    set_seed(train_cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directories
    output_dir = project_root / out_cfg['output_dir']
    checkpoint_dir = project_root / out_cfg['checkpoint_dir']
    metrics_dir = project_root / out_cfg['metrics_dir']
    plots_dir = project_root / out_cfg['plots_dir']
    
    for d in [checkpoint_dir, metrics_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("PHOBERT TRAINING WITH PYVI DATA")
    print(f"{'='*60}")
    print(f"🚀 Device: {device}")
    print(f"📂 Output dir: {output_dir}")
    print(f"⚡ Dynamic Padding: ENABLED")
    print(f"⚡ Workers: {args.workers}")
    
    # Data paths
    train_path = project_root / data_cfg['train_path']
    dev_path = project_root / data_cfg['dev_path']
    test_path = project_root / data_cfg['test_path']
    
    if not train_path.exists():
        print(f"\n❌ Training data not found: {train_path}")
        print("   Run preprocessing first:")
        print("   python src/pipeline/preprocessing/preprocessing_vifactcheck_pyvi.py")
        return
    
    # Tokenizer & Dataset
    print(f"\n📦 Loading tokenizer: {model_cfg['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg['base_model'])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    train_ds = ViFactCheckDataset(train_path, tokenizer, model_cfg['max_length'])
    dev_ds = ViFactCheckDataset(dev_path, tokenizer, model_cfg['max_length'])
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=train_cfg['batch_size'], 
        shuffle=True, 
        num_workers=args.workers, 
        pin_memory=True,
        collate_fn=data_collator
    )
    
    dev_loader = DataLoader(
        dev_ds, 
        batch_size=train_cfg['batch_size'] * 2, 
        num_workers=args.workers, 
        pin_memory=True,
        collate_fn=data_collator
    )
    
    # Model
    model = PhoBERTFactCheck(
        pretrained_name=model_cfg['base_model'],
        num_classes=model_cfg['num_classes'], 
        dropout_rate=model_cfg['dropout_rate'],
        unfreeze_last_n_layers=model_cfg['unfreeze_last_n_layers']
    )
    model.to(device)
    
    # Eval only mode
    if args.eval_only:
        if args.model_path is None:
            args.model_path = str(checkpoint_dir / "best_model.pt")
        
        print(f"\n📊 EVALUATION ONLY MODE")
        print(f"Loading model: {args.model_path}")
        
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Evaluate on test set
        test_ds = ViFactCheckDataset(test_path, tokenizer, model_cfg['max_length'])
        test_loader = DataLoader(
            test_ds, 
            batch_size=train_cfg['batch_size'] * 2, 
            num_workers=args.workers, 
            pin_memory=True,
            collate_fn=data_collator
        )
        
        test_loss, test_acc, test_f1, test_labels, test_preds, test_probs = evaluate(model, test_loader, device)
        
        print(f"\n{'='*60}")
        print("TEST SET RESULTS (pyvi)")
        print(f"{'='*60}")
        print(f"Accuracy: {test_acc:.4f}")
        print(f"Macro F1: {test_f1:.4f}")
        
        report = classification_report(test_labels, test_preds, target_names=list(LABEL_MAP.values()), digits=4)
        print(f"\n{report}")
        
        cm = confusion_matrix(test_labels, test_preds)
        print(f"Confusion Matrix:\n{cm}")
        
        # Save results
        metrics_data = {
            "split": "test",
            "accuracy": float(test_acc),
            "macro_f1": float(test_f1),
            "classification_report": classification_report(test_labels, test_preds, target_names=list(LABEL_MAP.values()), digits=4, output_dict=True),
            "confusion_matrix": cm.tolist()
        }
        
        with open(metrics_dir / 'eval_metrics_test.json', 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"\n✅ Results saved to {metrics_dir / 'eval_metrics_test.json'}")
        return
    
    # Training with LLRD
    layer_decay = train_cfg.get('layer_decay', 1.0)
    if layer_decay < 1.0:
        print(f"🛡️ Using Layer-wise Learning Rate Decay (LLRD) with factor: {layer_decay}")
        optimizer_params = get_optimizer_grouped_parameters_llrd(
            model, train_cfg['learning_rate'], train_cfg['weight_decay'], layer_decay
        )
        optimizer = torch.optim.AdamW(optimizer_params)
    else:
        print("⚡ Using standard AdamW optimizer (Uniform LR)")
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg['learning_rate'], weight_decay=train_cfg['weight_decay'])
    
    steps_per_epoch = len(train_loader) // train_cfg['accumulation_steps']
    total_steps = steps_per_epoch * train_cfg['epochs']
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(train_cfg['warmup_ratio'] * total_steps), 
        num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg['label_smoothing'])
    
    # FP16 Mixed Precision
    use_fp16 = train_cfg.get('fp16', True)
    scaler = GradScaler(enabled=use_fp16)
    print(f"⚡ FP16 Mixed Precision: {'ENABLED' if use_fp16 else 'DISABLED'}")
    
    # Training loop
    best_score = -float('inf')
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    train_losses, dev_losses = [], []
    train_accs, dev_accs = [], []
    train_f1s, dev_f1s = [], []
    
    print(f"\n🔥 Start Training for {train_cfg['epochs']} epochs...\n")
    
    for epoch in range(1, train_cfg['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"=== Epoch {epoch}/{train_cfg['epochs']} ===")
        print(f"{'='*60}")
        
        train_loss, train_labels, train_preds, train_probs = train_epoch(
            model, train_loader, optimizer, criterion, device, 
            scaler, scheduler, train_cfg['accumulation_steps'], use_fp16
        )
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        
        print(f"📉 Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        
        dev_loss, dev_acc, dev_f1, dev_labels, dev_preds, dev_probs = evaluate(model, dev_loader, device)
        print(f"📊 Dev   Loss: {dev_loss:.4f} | Acc: {dev_acc:.4f} | F1: {dev_f1:.4f}")
        
        # Smart save logic
        overfit_gap = train_acc - dev_acc
        if overfit_gap > 0.10:
            print(f"⚠️  Overfitting detected! Gap: {overfit_gap:.2%}")
            
        penalty = max(0, overfit_gap - 0.10)
        current_score = dev_acc - penalty
        
        print(f"⚖️  Smart Score: {current_score:.4f} (Gap Penalty: {penalty:.4f})")
        
        train_losses.append(train_loss)
        dev_losses.append(dev_loss)
        train_accs.append(train_acc)
        dev_accs.append(dev_acc)
        train_f1s.append(train_f1)
        dev_f1s.append(dev_f1)
        
        if current_score > best_score:
            best_score = current_score
            best_f1 = dev_f1
            best_epoch = epoch
            patience_counter = 0
            
            best_dev_labels = dev_labels.copy()
            best_dev_preds = dev_preds.copy()
            best_dev_probs = dev_probs.copy()
            
            print(f"🔥 New Best Model! (Score: {best_score:.4f}) Saving...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': best_score,
                'best_f1': dev_f1,
                'dev_acc': dev_acc,
                'train_acc': train_acc
            }, checkpoint_dir / "best_model.pt")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. Patience: {patience_counter}/{train_cfg['patience']}")
            if patience_counter >= train_cfg['patience']:
                print("\n⏹️ Early stopping triggered!")
                break
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # Save training history
    print(f"\n{'='*60}")
    print("💾 Saving training history...")
    print(f"{'='*60}")
    
    history_data = {
        'train_losses': [float(x) for x in train_losses],
        'dev_losses': [float(x) for x in dev_losses],
        'train_accs': [float(x) for x in train_accs],
        'dev_accs': [float(x) for x in dev_accs],
        'train_f1s': [float(x) for x in train_f1s],
        'dev_f1s': [float(x) for x in dev_f1s],
        'best_epoch': int(best_epoch),
        'best_score': float(best_score)
    }
    
    with open(metrics_dir / 'training_history.json', 'w') as f:
        json.dump(history_data, f, indent=2)
    
    # Final summary
    best_idx = int(best_epoch) - 1
    best_dev_acc = dev_accs[best_idx]
    best_dev_f1_val = dev_f1s[best_idx]
    best_train_acc = train_accs[best_idx]
    final_gap = best_train_acc - best_dev_acc
    
    cls_report_dict = classification_report(
        best_dev_labels, 
        best_dev_preds, 
        target_names=[LABEL_MAP[i] for i in range(len(LABEL_MAP))], 
        digits=4,
        output_dict=True
    )
    
    cm = confusion_matrix(best_dev_labels, best_dev_preds)
    
    detailed_metrics = {
        "best_epoch": int(best_epoch),
        "smart_score": float(best_score),
        "final_gap": float(final_gap),
        "classification_report": cls_report_dict,
        "confusion_matrix": cm.tolist()
    }
    
    with open(metrics_dir / 'best_model_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(detailed_metrics, f, indent=2)
    
    cls_report_str = classification_report(
        best_dev_labels, 
        best_dev_preds, 
        target_names=[LABEL_MAP[i] for i in range(len(LABEL_MAP))], 
        digits=4
    )
    
    summary_text = f"""============================================================
=== FINAL TRAINING SUMMARY (pyvi) ===
============================================================
Best Epoch:       {best_epoch}
Smart Score:      {best_score:.4f}
------------------------------------------------------------
Best Dev Acc:     {best_dev_acc:.4f}
Best Dev F1:      {best_dev_f1_val:.4f}
Train Acc:        {best_train_acc:.4f}
Overfitting Gap:  {final_gap:.2%}
------------------------------------------------------------

=== CLASSIFICATION REPORT ===
{cls_report_str}

=== CONFUSION MATRIX ===
{cm}
============================================================
"""
    
    with open(metrics_dir / 'summary_report.txt', 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    # Generate plots
    plot_all_visualizations(
        train_losses=train_losses,
        dev_losses=dev_losses,
        train_accs=train_accs,
        dev_accs=dev_accs,
        train_f1s=train_f1s,
        dev_f1s=dev_f1s,
        best_dev_labels=best_dev_labels,
        best_dev_preds=best_dev_preds,
        best_dev_probs=best_dev_probs,
        save_dir=plots_dir
    )
    
    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETED (pyvi)!")
    print(f"{'='*60}")
    print(f"Best model saved to: {checkpoint_dir / 'best_model.pt'}")
    print(f"Best dev Acc: {best_dev_acc:.4f}")
    print(f"Best dev F1: {best_dev_f1_val:.4f}")
    print(f"\n💡 To evaluate on test set:")
    print(f"   python src/pipeline/fact_checking/train_eval_pyvi.py --eval-only")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
