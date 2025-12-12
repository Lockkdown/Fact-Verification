"""
Hyperparameter Search Script for ViFactCheck Training
Sequential grid search to find optimal config on single GPU.
"""

import os
import json
import csv
import gc
import shutil
from pathlib import Path
from datetime import datetime
from itertools import product
import argparse
import logging

# Suppress distributed warnings
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score
from torch.amp import autocast, GradScaler

# Import từ train.py
import sys
sys.path.append(str(Path(__file__).parent))
from train import ViFactCheckDataset, set_seed, train_epoch, evaluate
from model import PhoBERTFactCheck


def load_search_grid(grid_path):
    """Load grid search configuration."""
    with open(grid_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_configs(grid):
    """Generate all config combinations from grid."""
    search_params = grid['search_params']
    param_names = list(search_params.keys())
    param_values = [search_params[k] for k in param_names]
    
    configs = []
    for combo in product(*param_values):
        config = grid['fixed_params'].copy()
        config.update(dict(zip(param_names, combo)))
        configs.append(config)
    
    return configs


def train_single_config(config_id, config, data_paths, output_base, device):
    """Train a single config and return results."""
    print(f"\n{'='*80}")
    print(f"CONFIG {config_id + 1}")
    print(f"{'='*80}")
    print(f"unfreeze_layers: {config.get('unfreeze_last_n_layers', -1)}, dropout: {config['dropout_rate']}")
    print(f"lr: {config['learning_rate']}, wd: {config['weight_decay']}, warmup: {config['warmup_ratio']}, smoothing: {config['label_smoothing']}")
    print(f"{'='*80}\n")
    
    # Setup directories
    config_name = f"config_{config_id:02d}_unfz{config.get('unfreeze_last_n_layers', -1)}_drop{config['dropout_rate']}"
    config_dir = output_base / config_name
    checkpoint_dir = config_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seed
    set_seed(config['seed'])
    
    # Tokenizer & Dataset
    tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    project_root = Path(__file__).parent.parent.parent.parent
    train_ds = ViFactCheckDataset(project_root / data_paths['train_path'], tokenizer, config['max_length'])
    dev_ds = ViFactCheckDataset(project_root / data_paths['dev_path'], tokenizer, config['max_length'])
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=2,
        pin_memory=True,
        collate_fn=data_collator
    )
    
    dev_loader = DataLoader(
        dev_ds, 
        batch_size=config['batch_size'] * 2, 
        num_workers=2,
        pin_memory=True,
        collate_fn=data_collator
    )
    
    # Model
    model = PhoBERTFactCheck(
        pretrained_name=config['base_model'],
        num_classes=config['num_classes'],
        dropout_rate=config['dropout_rate'],
        unfreeze_last_n_layers=config.get('unfreeze_last_n_layers', -1)
    )
    model.to(device)
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    steps_per_epoch = len(train_loader) // config['accumulation_steps']
    total_steps = steps_per_epoch * config['epochs']
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config['warmup_ratio'] * total_steps),
        num_training_steps=total_steps
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    scaler = GradScaler()
    
    # Training loop
    best_f1 = 0
    best_epoch = 0
    best_dev_acc = 0
    best_train_acc = 0
    patience_counter = 0
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n--- Epoch {epoch}/{config['epochs']} ---")
        
        train_loss, train_labels, train_preds, train_probs = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler, scheduler, config['accumulation_steps']
        )
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        
        dev_loss, dev_acc, dev_f1, dev_labels, dev_preds, dev_probs = evaluate(model, dev_loader, device)
        print(f"Dev   Loss: {dev_loss:.4f} | Acc: {dev_acc:.4f} | F1: {dev_f1:.4f}")
        
        # Track best
        if dev_f1 > best_f1:
            prev_best = best_f1
            best_f1 = dev_f1
            best_epoch = epoch
            best_dev_acc = dev_acc
            best_train_acc = train_acc
            patience_counter = 0
            
            # Calculate gap for logging
            current_gap = train_acc - dev_acc
            gap_warning = f" (⚠️ Gap: {current_gap:.2%})" if current_gap > 0.10 else ""
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_f1': best_f1,
                'dev_acc': dev_acc,
                'train_acc': train_acc,
                'gap': current_gap
            }, checkpoint_dir / "best_model.pt")
            print(f"🔥 New Best Model! Saving...{gap_warning}")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Cleanup
        del train_labels, train_preds, train_probs
        del dev_labels, dev_preds, dev_probs
        gc.collect()
        torch.cuda.empty_cache()
    
    # Results
    overfit_gap = best_train_acc - best_dev_acc
    
    results = {
        'config_id': config_id,
        'config_name': config_name,
        'unfreeze_layers': config.get('unfreeze_last_n_layers', -1),
        'dropout': config['dropout_rate'],
        'learning_rate': config['learning_rate'],
        'weight_decay': config['weight_decay'],
        'warmup_ratio': config['warmup_ratio'],
        'label_smoothing': config['label_smoothing'],
        'best_epoch': best_epoch,
        'best_dev_acc': best_dev_acc,
        'best_dev_f1': best_f1,
        'best_train_acc': best_train_acc,
        'overfit_gap': overfit_gap,
        'checkpoint_path': str(checkpoint_dir / "best_model.pt")
    }
    
    # Cleanup
    del model, optimizer, scheduler, train_loader, dev_loader
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=str, default="config/fact_checking/search_grid.json")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Load grid
    project_root = Path(__file__).parent.parent.parent.parent
    grid_path = project_root / args.grid
    print(f"📋 Loading grid from: {grid_path}")
    grid = load_search_grid(grid_path)
    
    # Generate configs
    configs = generate_configs(grid)
    print(f"🔢 Total configs to search: {len(configs)}")
    
    # Setup output
    output_base = project_root / grid['output_dir']
    output_base.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = output_base / f"search_results_{timestamp}.csv"
    
    # Resume support
    completed_ids = set()
    if args.resume and results_csv.exists():
        with open(results_csv, 'r') as f:
            reader = csv.DictReader(f)
            completed_ids = {int(row['config_id']) for row in reader}
        print(f"📂 Resuming: {len(completed_ids)} configs already completed")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")
    
    # CSV header
    fieldnames = [
        'config_id', 'config_name', 'unfreeze_layers', 'dropout',
        'learning_rate', 'weight_decay', 'warmup_ratio', 'label_smoothing',
        'best_epoch', 'best_dev_acc', 'best_dev_f1', 
        'best_train_acc', 'overfit_gap', 'checkpoint_path'
    ]
    
    # Write header if new file
    if not results_csv.exists():
        with open(results_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    # Run search
    print(f"\n🔍 Starting hyperparameter search...")
    print(f"📊 Results will be saved to: {results_csv}\n")
    
    for i, config in enumerate(configs):
        if i in completed_ids:
            print(f"⏭️  Skipping config {i} (already completed)")
            continue
        
        try:
            results = train_single_config(i, config, grid['data_paths'], output_base, device)
            
            # Append to CSV
            with open(results_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(results)
            
            print(f"\n✅ Config {i} completed: Dev Acc={results['best_dev_acc']:.4f}, F1={results['best_dev_f1']:.4f}")
            
        except Exception as e:
            print(f"\n❌ Config {i} failed: {e}")
            continue
    
    print(f"\n{'='*80}")
    print("🎉 HYPERPARAMETER SEARCH COMPLETED!")
    print(f"{'='*80}")
    print(f"📊 Results saved to: {results_csv}")
    print(f"📁 Checkpoints saved to: {output_base}")
    print(f"\n💡 To view results:")
    print(f"   import pandas as pd")
    print(f"   df = pd.read_csv('{results_csv}')")
    print(f"   df.sort_values('best_dev_f1', ascending=False)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
