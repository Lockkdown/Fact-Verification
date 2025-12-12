import json
import os
import argparse
import sys
from pathlib import Path
import copy
import torch

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.fact_checking import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_config", type=str, default="config/fact_checking/search_grid_v6_full.json")
    args = parser.parse_args()
    
    grid_config_path = project_root / args.grid_config
    print(f"🚀 Starting Grid Search from: {grid_config_path}")
    
    with open(grid_config_path, 'r') as f:
        grid_cfg = json.load(f)
        
    base_config_path = project_root / grid_cfg['base_config']
    with open(base_config_path, 'r') as f:
        base_cfg = json.load(f)
        
    experiments = grid_cfg['experiments']
    output_base_dir = project_root / grid_cfg['output_dir']
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📋 Total experiments: {len(experiments)}")
    print(f"📂 Output directory: {output_base_dir}")
    
    results_summary = []
    
    for i, exp in enumerate(experiments):
        exp_name = exp['name']
        print(f"\n{'='*80}")
        print(f"▶️  Running Experiment {i+1}/{len(experiments)}: {exp_name}")
        print(f"{'='*80}")
        
        # 1. Prepare Config
        current_cfg = copy.deepcopy(base_cfg)
        
        # Apply fixed params
        for k, v in grid_cfg['fixed_params'].items():
            if k in current_cfg['model']: current_cfg['model'][k] = v
            elif k in current_cfg['training']: current_cfg['training'][k] = v
            
        # Apply data paths
        for k, v in grid_cfg['data_paths'].items():
            current_cfg['data'][k] = v
            
        # Apply experiment specific params
        for k, v in exp.items():
            if k == 'name': continue
            if k in current_cfg['training']:
                current_cfg['training'][k] = v
                print(f"   • {k}: {v}")
            elif k in current_cfg['model']:
                current_cfg['model'][k] = v
                print(f"   • {k}: {v}")
                
        # Set output dir specific to experiment
        exp_dir = output_base_dir / exp_name
        current_cfg['output']['output_dir'] = str(exp_dir.relative_to(project_root))
        
        # Save temp config
        temp_config_path = output_base_dir / f"temp_config_{exp_name}.json"
        with open(temp_config_path, 'w') as f:
            json.dump(current_cfg, f, indent=2)
            
        # 2. Run Training
        # Hack args for train.main
        # Force workers=1 for stability on Windows
        sys.argv = [sys.argv[0], "--config", str(temp_config_path), "--workers", "1"]
        
        try:
            # Clear CUDA cache before run
            torch.cuda.empty_cache()
            
            # Call train main
            train.main()
            
            # 3. Collect Result
            metrics_path = exp_dir / "metrics" / "best_model_metrics.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                    
                result = {
                    "name": exp_name,
                    "smart_score": metrics.get('smart_score', 0),
                    "best_epoch": metrics.get('best_epoch', 0),
                    "final_gap": metrics.get('final_gap', 0),
                    "f1_macro": metrics['classification_report']['macro avg']['f1-score'],
                    "acc": metrics['classification_report']['accuracy'],
                    "params": {k:v for k,v in exp.items() if k != 'name'}
                }
                results_summary.append(result)
                print(f"✅ Experiment {exp_name} finished. Score: {result['smart_score']:.4f}")
                
                # --- INCREMENTAL SAVE (Save immediately after each run) ---
                # This ensures we don't lose data if script crashes later
                import csv
                csv_path = output_base_dir / "grid_search_summary.csv"
                
                # Collect all possible param keys from all results so far
                all_param_keys = set()
                for r in results_summary:
                    all_param_keys.update(r['params'].keys())
                param_keys_list = sorted(list(all_param_keys))
                
                fieldnames = ["name", "smart_score", "acc", "f1_macro", "final_gap", "best_epoch"] + param_keys_list
                
                try:
                    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for r in results_summary:
                            row = {
                                "name": r['name'],
                                "smart_score": r['smart_score'],
                                "acc": r['acc'],
                                "f1_macro": r['f1_macro'],
                                "final_gap": r['final_gap'],
                                "best_epoch": r['best_epoch']
                            }
                            for k in param_keys_list:
                                row[k] = r['params'].get(k, '')
                            writer.writerow(row)
                    print(f"   💾 Updated CSV summary: {csv_path}")
                except Exception as csv_err:
                    print(f"   ⚠️ Failed to update CSV: {csv_err}")
                # -------------------------------------------------------
                
            else:
                print(f"⚠️  Metrics file not found for {exp_name}")
                
        except Exception as e:
            print(f"❌ Experiment {exp_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            
    # 4. Save Final Summary
    print(f"\n{'='*80}")
    print("🏆 GRID SEARCH COMPLETED! SUMMARY:")
    print(f"{'='*80}")
    
    # Sort by smart_score descending
    results_summary.sort(key=lambda x: x['smart_score'], reverse=True)
    
    summary_path = output_base_dir / "grid_search_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
        
    print(f"{'Name':<40} | {'Score':<8} | {'Acc':<8} | {'F1':<8} | {'Gap':<8}")
    print("-" * 85)
    for r in results_summary:
        print(f"{r['name']:<40} | {r['smart_score']:.4f}   | {r['acc']:.4f}   | {r['f1_macro']:.4f}   | {r['final_gap']:.2%}")
        
    print(f"\n💾 Full summary saved to: {summary_path}")
    print(f"📊 CSV summary saved to:  {csv_path}")

if __name__ == "__main__":
    main()
