"""
Hard Negative Fine-tuning Script

Fine-tune model với focus vào hard negatives để cải thiện accuracy.
Author: Lockdown
Date: Nov 26, 2025
"""

import os
import json
import argparse
import gc
import random
import warnings
import logging
from pathlib import Path

# Disable HuggingFace progress bars
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import logging
# Suppress specific PyTorch warnings
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Import modules
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.fact_checking.model import PhoBERTFactCheck
from src.pipeline.fact_checking.train import ViFactCheckDataset, LABEL_MAP, set_seed, get_optimizer_grouped_parameters_llrd
from src.pipeline.fact_checking.visualization import plot_all_visualizations
from src.pipeline.fine_tune.hard_negative_dataset import create_hard_negative_dataset

warnings.filterwarnings('ignore')
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

def train_epoch(model, dataloader, optimizer, criterion, device, scaler, scheduler, accumulation_steps):
    """Train cho 1 epoch với hard negatives."""
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
    
    all_probs = torch.cat(all_probs).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_labels, all_preds, all_probs

def evaluate(model, loader, device):
    """Evaluation function."""
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

def load_pretrained_model(model_path: str, device: torch.device) -> PhoBERTFactCheck:
    """Load pre-trained model từ checkpoint."""
    print(f"📂 Loading pre-trained model from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        if 'dev_acc' in checkpoint:
            print(f"📊 Pre-trained model Dev Acc: {checkpoint['dev_acc']:.4f}")
    else:
        state_dict = checkpoint
        
    model = PhoBERTFactCheck(num_classes=3)
    model.load_state_dict(state_dict)
    print("✅ Pre-trained model loaded successfully")
    
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/fact_checking/train_config_hard_negatives.json")
    parser.add_argument("--workers", type=int, default=1, help="Number of dataloader workers")
    parser.add_argument("--use-balanced", action="store_true", help="Use balanced hard negative dataset")
    parser.add_argument("--run-mining", action="store_true", help="Run hard negative mining first")
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
        
    print(f"⚙️ Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        cfg = json.load(f)
        
    # Run hard negative mining if requested
    if args.run_mining:
        print("\n🔍 Running Hard Negative Mining first...")
        from src.pipeline.fine_tune.hard_negative_miner import HardNegativeMiner
        
        pretrained_model_path = project_root / cfg['model']['pretrained_model_path']
        miner = HardNegativeMiner(str(pretrained_model_path), cfg)
        
        splits = cfg['hard_negative_config']['error_analysis_splits']
        analysis_results, hard_negatives = miner.run_analysis(splits)
        
        # Save results
        cache_path = project_root / cfg['data']['hard_negatives_cache']
        miner.save_results(str(cache_path))
        print(f"✅ Mining completed. Found {len(hard_negatives)} hard negatives.\n")
    
    # Setup paths
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
        
    # Setup
    set_seed(train_cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")
    print(f"⚡ Hard Negative Fine-tuning Mode")
    print(f"⚡ DataLoader Workers: {args.workers}")
    print(f"⚡ Use Balanced Dataset: {args.use_balanced}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_cfg['base_model'])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Datasets với hard negative sampling
    print("\n📂 Loading datasets with hard negative weighting...")
    
    # Train dataset (với hard negatives)
    train_ds = create_hard_negative_dataset(
        jsonl_path=str(project_root / data_cfg['train_path']),
        tokenizer=tokenizer,
        config=cfg,
        use_balanced=args.use_balanced
    )
    
    # Regular dev dataset (không cần hard negatives cho evaluation)
    dev_ds = ViFactCheckDataset(project_root / data_cfg['dev_path'], tokenizer, model_cfg['max_length'])
    
    # Show hard negative stats
    if hasattr(train_ds, 'get_hard_negative_stats'):
        stats = train_ds.get_hard_negative_stats()
        print(f"📊 Hard Negative Stats:")
        print(f"   Total samples: {stats['total_samples']}")
        print(f"   Hard negatives: {stats['hard_negatives_count']} ({stats['hard_negative_ratio']:.1%})")
        print(f"   HN label dist: {stats['hard_negative_label_dist']}")
    
    # DataLoaders
    if hasattr(train_ds, 'create_weighted_sampler') and not args.use_balanced:
        sampler = train_ds.create_weighted_sampler()
        train_loader = DataLoader(
            train_ds,
            batch_size=train_cfg['batch_size'],
            sampler=sampler,  # Use weighted sampler
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=data_collator
        )
        print("🎯 Using WeightedRandomSampler for hard negatives")
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=train_cfg['batch_size'],
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=data_collator
        )
        print("🔄 Using regular shuffling")
    
    dev_loader = DataLoader(
        dev_ds, 
        batch_size=train_cfg['batch_size'] * 2, 
        num_workers=args.workers, 
        pin_memory=True,
        collate_fn=data_collator
    )
    
    # Model - Load pre-trained
    pretrained_model_path = project_root / model_cfg['pretrained_model_path']
    model = load_pretrained_model(str(pretrained_model_path), device)
    model.to(device)
    
    # Fine-tuning với learning rate thấp hơn
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
        print("⚡ Using standard AdamW optimizer")
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
    
    # Training Loop
    best_score = -float('inf')
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    train_losses, dev_losses = [], []
    train_accs, dev_accs = [], []
    train_f1s, dev_f1s = [], []
    
    print(f"\n🔥 Starting Hard Negative Fine-tuning for {train_cfg['epochs']} epochs...\n")
    
    for epoch in range(1, train_cfg['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"=== Epoch {epoch}/{train_cfg['epochs']} (Hard Negative Fine-tuning) ===")
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
        
        # Smart save logic (same as original)
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
                'train_acc': train_acc,
                'is_hard_negative_finetuned': True
            }, checkpoint_dir / "model_hard_negatives.pt")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. Patience: {patience_counter}/{train_cfg['patience']}")
            if patience_counter >= train_cfg['patience']:
                print("\n⏹️ Early stopping triggered!")
                break
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # Save results (similar to original train.py)
    print(f"\n{'='*60}")
    print("💾 Saving training history and generating reports...")
    print(f"{'='*60}")
    
    history_data = {
        'train_losses': [float(x) for x in train_losses],
        'dev_losses': [float(x) for x in dev_losses],
        'train_accs': [float(x) for x in train_accs],
        'dev_accs': [float(x) for x in dev_accs],
        'train_f1s': [float(x) for x in train_f1s],
        'dev_f1s': [float(x) for x in dev_f1s],
        'best_epoch': int(best_epoch),
        'best_score': float(best_score),
        'is_hard_negative_finetuned': True
    }
    
    with open(metrics_dir / 'training_history.json', 'w') as f:
        json.dump(history_data, f, indent=2)
    
    # Final metrics and report
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
        "training_type": "hard_negative_finetuning",
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
=== HARD NEGATIVE FINE-TUNING SUMMARY ===
============================================================
Training Type:    Hard Negative Fine-tuning
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
    print("✅ HARD NEGATIVE FINE-TUNING COMPLETED!")
    print(f"{'='*60}")
    print(f"Best model saved to: {checkpoint_dir / 'model_hard_negatives.pt'}")
    print(f"Best dev accuracy: {best_dev_acc:.4f}")
    print(f"Improvement over baseline: {best_dev_acc - 0.7524:.4f} (+{(best_dev_acc - 0.7524)*100:.2f}%)")
    print(f"Training history: {metrics_dir / 'training_history.json'}")
    print(f"Plots: {plots_dir}")
    
    # --- AUTO EVALUATE LOGIC ---
    print(f"\n{'='*60}")
    print("🚀 Evaluation on dev set")
    print(f"{'='*60}")
    import subprocess
    import sys
    
    eval_script = project_root / "src" / "pipeline" / "fact_checking" / "evaluate.py"
    best_model_path = checkpoint_dir / 'model_hard_negatives.pt'
    
    # Fix Unicode Error on Windows
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    # Run Eval Dev
    cmd_dev = [sys.executable, str(eval_script), "--model_path", str(best_model_path), "--split", "dev"]
    print(f"Command: {' '.join(cmd_dev)}")
    subprocess.run(cmd_dev, env=env)
    print(f"\n✅ Evaluation on dev set completed successfully")
    
    print(f"\n{'='*60}")
    print("🚀 Evaluation on test set")
    print(f"{'='*60}")
    
    # Run Eval Test
    cmd_test = [sys.executable, str(eval_script), "--model_path", str(best_model_path), "--split", "test"]
    subprocess.run(cmd_test, env=env)
    print(f"\n✅ Evaluation on test set completed successfully")
    # ---------------------------

    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
