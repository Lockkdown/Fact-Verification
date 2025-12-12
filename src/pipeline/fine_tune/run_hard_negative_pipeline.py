"""
Hard Negative Pipeline Runner - One-click solution

Chạy toàn bộ quy trình: Mining -> Fine-tuning -> Evaluation
Author: Lockdown  
Date: Nov 26, 2025
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent.parent.parent

def run_command(cmd, cwd=None, description=""):
    """Chạy lệnh và xử lý kết quả."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        cwd=cwd or project_root,
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Error in {description}")
        sys.exit(1)
    else:
        print(f"✅ {description} completed successfully")

def check_prerequisites():
    """Kiểm tra các file cần thiết."""
    print("🔍 Checking prerequisites...")
    
    # Check config file
    config_path = project_root / "config/fact_checking/train_config_hard_negatives.json"
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    # Check baseline model
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    baseline_model_path = project_root / config['model']['pretrained_model_path']
    if not baseline_model_path.exists():
        print(f"❌ Baseline model not found: {baseline_model_path}")
        print("💡 Please train the baseline model first using:")
        print("   python src/pipeline/fact_checking/train.py --config config/fact_checking/train_config_optimized.json")
        return False
    
    print("✅ All prerequisites satisfied")
    return True

def main():
    parser = argparse.ArgumentParser(description="Hard Negative Pipeline Runner")
    
    # Pipeline steps
    parser.add_argument("--skip-mining", action="store_true", help="Skip hard negative mining (use cached results)")
    parser.add_argument("--skip-training", action="store_true", help="Skip fine-tuning")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip final evaluation")
    
    # Training options
    parser.add_argument("--use-balanced", action="store_true", help="Use balanced hard negative dataset")
    parser.add_argument("--workers", type=int, default=1, help="Number of dataloader workers")
    
    # Evaluation options  
    parser.add_argument("--eval-splits", nargs="+", default=["dev", "test"], choices=["dev", "test"], 
                       help="Splits to evaluate on")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🎯 HARD NEGATIVE FINE-TUNING PIPELINE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Skip mining: {args.skip_mining}")
    print(f"Skip training: {args.skip_training}")
    print(f"Skip evaluation: {args.skip_evaluation}")
    print(f"Use balanced dataset: {args.use_balanced}")
    print("="*80)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Step 1: Hard Negative Mining
    if not args.skip_mining:
        run_command([
            sys.executable, 
            "src/pipeline/fine_tune/hard_negative_miner.py"
        ], description="Hard Negative Mining")
    else:
        print("\n⏭️ Skipping hard negative mining (using cached results)")
    
    # Step 2: Fine-tuning
    if not args.skip_training:
        cmd = [
            sys.executable,
            "src/pipeline/fine_tune/train_hard_negatives.py",
            "--workers", str(args.workers)
        ]
        
        if args.use_balanced:
            cmd.append("--use-balanced")
        
        run_command(cmd, description="Hard Negative Fine-tuning")
    else:
        print("\n⏭️ Skipping fine-tuning")
    
    # Step 3: Evaluation
    if not args.skip_evaluation:
        model_path = "results/fact_checking/hard_negatives/checkpoints/model_hard_negatives.pt"
        
        for split in args.eval_splits:
            run_command([
                sys.executable,
                "src/pipeline/fact_checking/evaluate.py", 
                "--model_path", model_path,
                "--split", split
            ], description=f"Evaluation on {split} set")
    else:
        print("\n⏭️ Skipping evaluation")
    
    # Final summary
    print("\n" + "="*80)
    print("🎉 HARD NEGATIVE PIPELINE COMPLETED!")
    print("="*80)
    
    if not args.skip_training:
        print(f"📂 Model saved: results/fact_checking/hard_negatives/checkpoints/model_hard_negatives.pt")
        print(f"📊 Metrics: results/fact_checking/hard_negatives/metrics/")
        print(f"📈 Plots: results/fact_checking/hard_negatives/plots/")
    
    if not args.skip_evaluation:
        print(f"📊 Evaluation results: results/fact_checking/eval/")
    
    print(f"\n⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    main()
