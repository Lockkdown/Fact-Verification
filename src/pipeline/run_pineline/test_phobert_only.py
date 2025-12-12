"""
Quick test: PhoBERT accuracy on VnCoreNLP pre-processed data (NO DEBATE).

Purpose: Verify that Model Accuracy matches the original 75.33% baseline.
This confirms data consistency between training and inference.

Usage:
    python test_phobert_only.py --split test --max-samples 200
    python test_phobert_only.py --split test  # Full test set
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from collections import Counter

# Add project root to path
project_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer
from src.pipeline.fact_checking.model import PhoBERTFactCheck


def load_vifactcheck_jsonl(split: str, max_samples: int = None):
    """Load pre-processed JSONL data (VnCoreNLP segmented)."""
    jsonl_path = project_root / "dataset" / "processed" / "vifactcheck" / f"vifactcheck_{split}.jsonl"
    
    if not jsonl_path.exists():
        raise FileNotFoundError(f"File not found: {jsonl_path}")
    
    label_map = {0: "Support", 1: "Refute", 2: "NOT_ENOUGH_INFO"}
    samples = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line.strip())
            samples.append({
                "id": item.get("sample_id", i + 1),
                "statement": item["statement"],
                "evidence": item["evidence"],
                "gold_label": label_map[item["label"]],
                "gold_label_id": item["label"]
            })
    
    return samples


def main():
    parser = argparse.ArgumentParser(description='Test PhoBERT accuracy (no debate)')
    parser.add_argument('--split', type=str, default='test', choices=['dev', 'test'])
    parser.add_argument('--max-samples', type=int, default=None, help='Limit samples for quick test')
    args = parser.parse_args()
    
    print("=" * 60)
    print("PhoBERT ACCURACY TEST (VnCoreNLP Data, NO DEBATE)")
    print("=" * 60)
    
    # Load model
    checkpoint_path = project_root / "results/fact_checking/hard_negatives/checkpoints/model_hard_negatives.pt"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"\n📦 Loading model from: {checkpoint_path}")
    print(f"🖥️  Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    model = PhoBERTFactCheck(pretrained_name="vinai/phobert-base", num_classes=3)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Load data
    print(f"\n📂 Loading {args.split} set (VnCoreNLP pre-processed)...")
    samples = load_vifactcheck_jsonl(args.split, args.max_samples)
    print(f"✅ Loaded {len(samples)} samples")
    
    # Label distribution
    gold_dist = Counter(s["gold_label"] for s in samples)
    print(f"📊 Gold distribution: {dict(gold_dist)}")
    
    # Predict
    label_map = {0: "Support", 1: "Refute", 2: "NOT_ENOUGH_INFO"}
    correct = 0
    predictions = []
    
    print(f"\n🔄 Running predictions...")
    for sample in tqdm(samples, desc="Predicting"):
        inputs = tokenizer(
            sample["statement"],
            sample["evidence"],
            max_length=256,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            probs = torch.softmax(logits, dim=-1).squeeze()
            pred_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[pred_id].item()
        
        pred_label = label_map[pred_id]
        is_correct = (pred_label == sample["gold_label"])
        
        if is_correct:
            correct += 1
        
        predictions.append({
            "id": sample["id"],
            "gold": sample["gold_label"],
            "pred": pred_label,
            "confidence": confidence,
            "correct": is_correct
        })
    
    # Results
    accuracy = correct / len(samples)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"✅ Correct: {correct}/{len(samples)}")
    print(f"📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class accuracy
    print("\n📋 Per-class breakdown:")
    for label in ["Support", "Refute", "NOT_ENOUGH_INFO"]:
        label_samples = [p for p in predictions if p["gold"] == label]
        label_correct = sum(1 for p in label_samples if p["correct"])
        if label_samples:
            print(f"  {label}: {label_correct}/{len(label_samples)} = {label_correct/len(label_samples)*100:.2f}%")
    
    # Confusion matrix
    print("\n🔢 Confusion Matrix:")
    labels = ["Support", "Refute", "NOT_ENOUGH_INFO"]
    header = "True \\ Pred"
    print(f"{header:<20} {'Support':>10} {'Refute':>10} {'NEI':>10}")
    print("-" * 52)
    for true_label in labels:
        row = []
        for pred_label in labels:
            count = sum(1 for p in predictions if p["gold"] == true_label and p["pred"] == pred_label)
            row.append(count)
        print(f"{true_label:<20} {row[0]:>10} {row[1]:>10} {row[2]:>10}")
    
    # Compare with expected
    print("\n" + "=" * 60)
    print("COMPARISON WITH EXPECTED BASELINE")
    print("=" * 60)
    expected = 0.7533  # 75.33% from training
    diff = accuracy - expected
    print(f"Expected (from training): {expected*100:.2f}%")
    print(f"Actual (this test):       {accuracy*100:.2f}%")
    print(f"Difference:               {diff*100:+.2f}%")
    
    if abs(diff) < 0.01:
        print("\n✅ MATCH! Data is consistent with training.")
    elif abs(diff) < 0.03:
        print("\n⚠️  CLOSE. Small variance is acceptable.")
    else:
        print("\n❌ MISMATCH! Check data preprocessing.")


if __name__ == "__main__":
    main()
