import os
import json
import argparse
import gc
import random
import warnings
import logging
from pathlib import Path
import sys

# Add project root to sys.path to fix module import errors
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Disable HuggingFace progress bars (faster loading)
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Suppress specific PyTorch warnings
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler  # PyTorch 2.0+
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Suppress tokenizer warnings
warnings.filterwarnings('ignore')
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# Import Model & Visualization (Absolute Imports)
from src.pipeline.fact_checking.model import PhoBERTFactCheck
from src.pipeline.fact_checking.visualization import plot_all_visualizations

# === CONFIGURATION ===
LABEL_MAP = {0: 'SUPPORT', 1: 'REFUTE', 2: 'NEI'}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        # Format: Statement (Claim) [SEP] Evidence
        
        # DYNAMIC PADDING: Don't pad here (padding=False), just truncate
        # Collator will handle padding to longest in batch
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*overflowing tokens.*')
            encoding = self.tokenizer(
                item['statement'],
                item['evidence'],
                truncation=True,
                padding=False,  # IMPORTANT: No padding here!
                max_length=self.max_len,
                return_tensors='pt'
            )
            
        # Handle both int and string labels (safety)
        label = item['label']
        if isinstance(label, str):
            label = int(label)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, dataloader, optimizer, criterion, device, scaler, scheduler, accumulation_steps):
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
        
        # Mixed Precision Training (PyTorch 2.0+ compatible)
        with autocast(device_type='cuda', enabled=True):
            outputs = model(input_ids, mask)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
            
        scaler.scale(loss).backward()
        
        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Scheduler step MUST be inside accumulation block
            scheduler.step()
            
            optimizer.zero_grad()
            
        total_loss += loss.item() * accumulation_steps
        
        # Collect predictions
        with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate all batches
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
    embed_lr = learning_rate * (layer_decay ** 13) # Assuming 12 layers + 1 embedding
    
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
    # PhoBERT has layers 0 to 11
    # Higher layers (closer to head) get higher LR
    for layer_i in range(12):
        layer_params = [(n, p) for n, p in named_parameters if f"encoder.layer.{layer_i}." in n]
        # Layer 11 -> decay^1, Layer 0 -> decay^12
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/fact_checking/train_config.json")
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers (increase for speed)")
    args = parser.parse_args()
    
    # 1. Load Config
    config_path = Path(args.config)
    project_root = Path(__file__).parent.parent.parent.parent
    
    if not config_path.is_absolute():
        config_path = project_root / config_path
        
    print(f"⚙️ Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        cfg = json.load(f)
        
    # Setup Paths
    data_cfg = cfg['data']
    train_cfg = cfg['training']
    model_cfg = cfg['model']
    out_cfg = cfg['output']
    
    output_dir = project_root / out_cfg['output_dir']
    checkpoint_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    
    for d in [checkpoint_dir, metrics_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # 2. Setup
    set_seed(train_cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")
    print(f"⚡ Dynamic Padding: ENABLED")
    print(f"⚡ DataLoader Workers: {args.workers}")
    
    # 3. Tokenizer & Dataset
    print(f"Tokenizer: {model_cfg['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg['base_model'])
    
    # Data Collator for Dynamic Padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    train_ds = ViFactCheckDataset(project_root / data_cfg['train_path'], tokenizer, model_cfg['max_length'])
    dev_ds = ViFactCheckDataset(project_root / data_cfg['dev_path'], tokenizer, model_cfg['max_length'])
    
    # Use num_workers for faster loading
    train_loader = DataLoader(
        train_ds, 
        batch_size=train_cfg['batch_size'], 
        shuffle=True, 
        num_workers=args.workers, 
        pin_memory=True,
        collate_fn=data_collator  # Use collator for dynamic padding
    )
    
    dev_loader = DataLoader(
        dev_ds, 
        batch_size=train_cfg['batch_size'] * 2, 
        num_workers=args.workers, 
        pin_memory=True,
        collate_fn=data_collator
    )
    
    # 4. Model
    model = PhoBERTFactCheck(
        pretrained_name=model_cfg['base_model'],
        num_classes=model_cfg['num_classes'], 
        dropout_rate=model_cfg['dropout_rate'],
        unfreeze_last_n_layers=model_cfg.get('unfreeze_last_n_layers', -1)
    )
    model.to(device)
    
    # 5. Optimizer & Scheduler
    # Use LLRD if specified in config, else standard AdamW
    layer_decay = train_cfg.get('layer_decay', 1.0)
    if layer_decay < 1.0:
        print(f"🛡️ Using Layer-wise Learning Rate Decay (LLRD) with factor: {layer_decay}")
        optimizer_params = get_optimizer_grouped_parameters_llrd(
            model, 
            train_cfg['learning_rate'], 
            train_cfg['weight_decay'], 
            layer_decay
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
    scaler = GradScaler()
    
    # 6. Training Loop
    best_score = -float('inf')
    best_f1 = 0.0  # Initialize best_f1 to avoid NameError
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
            scaler, scheduler, train_cfg['accumulation_steps']
        )
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        
        print(f"📉 Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        
        dev_loss, dev_acc, dev_f1, dev_labels, dev_preds, dev_probs = evaluate(model, dev_loader, device)
        print(f"📊 Dev   Loss: {dev_loss:.4f} | Acc: {dev_acc:.4f} | F1: {dev_f1:.4f}")
        
        # --- SMART SAVE LOGIC ---
        overfit_gap = train_acc - dev_acc
        if overfit_gap > 0.10:
            print(f"⚠️  Overfitting detected! Gap: {overfit_gap:.2%}")
            
        # Score Calculation: Penalize if gap > 10%
        # If gap <= 10%, score = dev_acc (Pure Acc)
        # If gap > 10%, score = dev_acc - (gap - 0.10) (Penalized)
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
    
    print(f"✅ Training history saved")
    
    # --- SUMMARY REPORT GENERATION ---
    # Calculate metrics for the BEST model (at best_epoch)
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
        output_dict=True  # Get as dictionary for JSON saving
    )
    
    cm = confusion_matrix(best_dev_labels, best_dev_preds) # Calculate CM here, BEFORE usage
    
    # Save detailed metrics to JSON
    detailed_metrics = {
        "best_epoch": int(best_epoch),
        "smart_score": float(best_score),
        "final_gap": float(final_gap),
        "classification_report": cls_report_dict,
        "confusion_matrix": cm.tolist() # Convert numpy array to list for JSON serialization
    }
    
    metrics_json_path = metrics_dir / 'best_model_metrics.json'
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_metrics, f, indent=2)
        
    print(f"📊 Detailed metrics saved: {metrics_json_path}")

    cls_report_str = classification_report(
        best_dev_labels, 
        best_dev_preds, 
        target_names=[LABEL_MAP[i] for i in range(len(LABEL_MAP))], 
        digits=4
    )
    
    summary_text = f"""============================================================
=== FINAL TRAINING SUMMARY ===
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
    
    summary_path = metrics_dir / 'summary_report.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
        
    print(f"📄 Summary report generated: {summary_path}")
    
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
    print("✅ TRAINING COMPLETED!")
    print(f"{'='*60}")
    print(f"Best model saved to: {checkpoint_dir / 'best_model.pt'}")
    print(f"Best dev F1: {best_f1:.4f}")
    print(f"Training history: {metrics_dir / 'training_history.json'}")
    print(f"Plots: {plots_dir}")
    print(f"\n💡 To evaluate on test set, run:")
    print(f"   python src/pipeline/fact_checking/evaluate.py \\")
    print(f"       --model_path {checkpoint_dir / 'best_model.pt'} \\")
    print(f"       --split test")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
