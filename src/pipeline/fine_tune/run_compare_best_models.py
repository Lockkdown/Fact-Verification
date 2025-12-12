import sys
import argparse
import json
from pathlib import Path
import subprocess
import pandas as pd

# Add project root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def run_evaluation(model_path, split="test"):
    """Chạy script evaluate.py cho một model cụ thể"""
    print(f"\n🔍 Evaluating model: {model_path.parent.parent.name}")
    print(f"   Split: {split.upper()}")
    
    eval_script = project_root / "src" / "pipeline" / "fact_checking" / "evaluate.py"
    
    cmd = [
        sys.executable,
        str(eval_script),
        "--model_path", str(model_path),
        "--split", split
    ]
    
    # Fix Unicode Error on Windows by forcing UTF-8 encoding for subprocess
    import os
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=env)
    
    if result.returncode != 0:
        print(f"❌ Error evaluating model: {result.stderr}")
        print(f"   Stdout dump: {result.stdout[:500]}...") # Print first 500 chars of stdout for clue
        return None
        
    # Parse output to get metrics (hacky but effective)
    output = result.stdout
    acc = 0.0
    f1 = 0.0
    
    for line in output.split('\n'):
        if "Accuracy:" in line:
            try:
                acc = float(line.split(":")[1].strip())
            except: pass
        if "Macro F1:" in line:
            try:
                f1 = float(line.split(":")[1].strip())
            except: pass
            
    print(f"   ✅ Result: Acc={acc:.4f}, F1={f1:.4f}")
    return {"name": model_path.parent.parent.name, "acc": acc, "f1": f1, "path": str(model_path)}

def main():
    print("⚔️  BATTLE OF CHAMPIONS: U8 vs U10 (Test Set) ⚔️")
    print("="*60)
    
    # Define Candidates
    base_dir = project_root / "results" / "fact_checking" / "grid_search_v7"
    
    candidates = [
        base_dir / "U8_HighLR_Aggressive" / "checkpoints" / "best_model.pt",
        base_dir / "U10_BestV6_G5_LR2e5_Decay085" / "checkpoints" / "best_model.pt"
    ]
    
    results = []
    
    for model_path in candidates:
        if not model_path.exists():
            print(f"⚠️ Model not found: {model_path}")
            continue
            
        res = run_evaluation(model_path, split="test")
        if res:
            results.append(res)
            
    if not results:
        print("\n❌ No results collected. Please check the errors above.")
        return

    # Print Comparison Table
    print("\n\n🏆 FINAL RESULTS (TEST SET)")
    print("="*60)
    print(f"{'Model Name':<40} | {'Test Acc':<10} | {'Test F1':<10}")
    print("-" * 66)
    
    # Sort by Acc descending
    results.sort(key=lambda x: x['acc'], reverse=True)
    
    for r in results:
        print(f"{r['name']:<40} | {r['acc']:.4f}     | {r['f1']:.4f}")
        
    print("="*60)
    
    best_model = results[0]
    print(f"\n👑 THE WINNER IS: {best_model['name']}")
    print(f"   Please use this model for the Debate System!")

if __name__ == "__main__":
    main()
