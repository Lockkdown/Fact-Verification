import torch
import torch.nn as nn
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import sys

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Suppress specific PyTorch warnings
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

from src.pipeline.fact_checking.model import PhoBERTFactCheck
from src.pipeline.fact_checking.train import ViFactCheckDataset, LABEL_MAP

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path, device):
    """Load a trained model."""
    logger.info(f"📂 Loading model from {model_path}")
    model = PhoBERTFactCheck(num_classes=3)
    # Set weights_only=False to support loading checkpoints with numpy scalars
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model

def ensemble_evaluate(model1_path, model2_path, test_path, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🚀 Device: {device}")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    # Load Datasets
    logger.info(f"📂 Loading test data from {test_path}")
    test_ds = ViFactCheckDataset(test_path, tokenizer, max_len=512)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=0, # Safe for Windows
        collate_fn=data_collator,
        shuffle=False
    )
    
    # Load Models
    model1 = load_model(model1_path, device)
    model2 = load_model(model2_path, device)
    
    logger.info("⚡ Starting Ensemble Evaluation (Soft Voting)...")
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Ensembling"):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Model 1 Inference
            outputs1 = model1(input_ids, mask)
            probs1 = torch.softmax(outputs1, dim=1)
            
            # Model 2 Inference
            outputs2 = model2(input_ids, mask)
            probs2 = torch.softmax(outputs2, dim=1)
            
            # Average Probabilities (Soft Voting)
            avg_probs = (probs1 + probs2) / 2
            preds = torch.argmax(avg_probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    print("\n" + "="*60)
    print("🔥 ENSEMBLE RESULTS 🔥")
    print("="*60)
    print(f"Model A: {Path(model1_path).name}")
    print(f"Model B: {Path(model2_path).name}")
    print("-" * 60)
    print(f"✅ Test Accuracy: {acc:.4f}")
    print(f"✅ Macro F1:      {macro_f1:.4f}")
    print("-" * 60)
    
    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=list(LABEL_MAP.values()), digits=4))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default="results/fact_checking/final_optimized/checkpoints/best_model.pt")
    parser.add_argument("--finetuned", type=str, default="results/fact_checking/hard_negatives/checkpoints/model_hard_negatives.pt")
    parser.add_argument("--test_path", type=str, default="dataset/processed/vifactcheck/vifactcheck_test.jsonl")
    
    args = parser.parse_args()
    
    project_path = Path(__file__).parent.parent.parent.parent
    baseline_full = project_path / args.baseline
    finetuned_full = project_path / args.finetuned
    test_full = project_path / args.test_path
    
    ensemble_evaluate(str(baseline_full), str(finetuned_full), str(test_full))
