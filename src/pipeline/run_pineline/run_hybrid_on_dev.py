"""
Run Hybrid Analysis on DEV set to find optimal threshold.

This script (v2 - Dec 24, 2025):
1. Supports per-policy threshold tuning (Option A)
2. Handles multiple EarlyStop max_K results (k3, k5, k7)
3. Saves optimal threshold per policy to config

Usage:
    # Analyze all available EarlyStop policies
    python run_hybrid_on_dev.py --analyze
    
    # Analyze specific max_K
    python run_hybrid_on_dev.py --analyze --max-k 3 5 7
    
    # Run debate first then analyze
    python run_hybrid_on_dev.py --run-debate --analyze

Author: Lockdown
Date: Dec 06, 2025
Updated: Dec 24, 2025 - Option A (per-policy threshold)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.run_pineline.hybrid_analysis import HybridAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Paths
DEV_RESULTS_BASE = PROJECT_ROOT / "results" / "vifactcheck" / "dev" / "full_debate"
DEV_HYBRID_OUTPUT = PROJECT_ROOT / "results" / "vifactcheck" / "dev" / "hybrid_analysis"
OPTIMAL_THRESHOLD_FILE = PROJECT_ROOT / "config" / "debate" / "optimal_threshold.json"

# Supported max_K values
SUPPORTED_MAX_K = [3, 5, 7]


def get_results_path(max_k: int) -> Path:
    """Get path to results file for a specific max_K."""
    return DEV_RESULTS_BASE / f"earlystop_k{max_k}" / "vifactcheck_dev_results.json"


def find_available_policies() -> List[int]:
    """Find which max_K policies have results available."""
    available = []
    for k in SUPPORTED_MAX_K:
        path = get_results_path(k)
        if path.exists():
            available.append(k)
    return available


def run_debate_on_dev(max_k_list: List[int] = None):
    """Run full debate pipeline on DEV set for specified max_K values."""
    if max_k_list is None:
        max_k_list = SUPPORTED_MAX_K
    
    logger.info("="*70)
    logger.info("STEP 1: Running Full Debate on DEV set")
    logger.info(f"        Configs: max_K = {max_k_list}")
    logger.info("="*70)
    
    import subprocess
    
    # Build max-rounds argument
    max_rounds_arg = [str(k) for k in max_k_list]
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "pipeline" / "run_pineline" / "eval_vifactcheck_pipeline.py"),
        "--splits", "dev",
        "--full-debate",
        "--async-debate",
        "--batch-size", "10",
        "--max-rounds", *max_rounds_arg
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
        logger.info("✅ Debate on DEV completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Debate failed with error: {e}")
        return False


def run_hybrid_analysis_for_policy(max_k: int, thresholds: List[float]) -> Optional[Dict]:
    """Run hybrid analysis for a specific max_K policy."""
    results_path = get_results_path(max_k)
    
    if not results_path.exists():
        logger.warning(f"⚠️  Results not found for max_K={max_k}: {results_path}")
        return None
    
    # Create output directory for this policy
    output_dir = DEV_HYBRID_OUTPUT / f"earlystop_k{max_k}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n📊 Analyzing EarlyStop max_K={max_k}")
    logger.info(f"   Input:  {results_path}")
    logger.info(f"   Output: {output_dir}")
    
    analyzer = HybridAnalyzer(str(results_path))
    
    report = analyzer.generate_full_report(
        output_dir=str(output_dir),
        thresholds=thresholds
    )
    
    # Extract optimal threshold
    optimal = report["optimal_threshold"]
    logger.info(f"   🎯 Optimal threshold: {optimal['threshold']}")
    logger.info(f"      Hybrid Accuracy: {optimal['hybrid_accuracy']:.2%}")
    logger.info(f"      Skip Ratio: {optimal['skip_ratio']:.1%}")
    
    return report


def run_hybrid_analysis(max_k_list: List[int] = None) -> Dict[int, Dict]:
    """Run hybrid analysis on DEV results for all specified max_K policies.
    
    Option A: Per-policy threshold tuning.
    Each max_K gets its own optimal threshold.
    """
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Hybrid Analysis (Option A - Per-Policy Threshold)")
    logger.info("="*70)
    
    # Find available policies if not specified
    if max_k_list is None:
        max_k_list = find_available_policies()
    
    if not max_k_list:
        logger.error("❌ No EarlyStop results found in DEV. Run --run-debate first.")
        return {}
    
    logger.info(f"📋 Analyzing policies: max_K = {max_k_list}")
    
    # Test thresholds
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
    
    # Analyze each policy
    all_reports = {}
    for max_k in max_k_list:
        report = run_hybrid_analysis_for_policy(max_k, thresholds)
        if report:
            all_reports[max_k] = report
    
    # Save all thresholds to config
    if all_reports:
        save_optimal_thresholds(all_reports)
    
    return all_reports


def save_optimal_thresholds(all_reports: Dict[int, Dict]):
    """Save optimal thresholds for all policies to config file.
    
    Also updates debate_config.json with per-policy thresholds.
    
    Structure:
    {
        "version": "2.0",
        "tuning_method": "per_policy",
        "tuned_on": "dev",
        "policies": {
            "earlystop_k3": {"threshold": 0.75, "hybrid_accuracy": 0.85, ...},
            "earlystop_k5": {"threshold": 0.70, "hybrid_accuracy": 0.86, ...},
            "earlystop_k7": {"threshold": 0.65, "hybrid_accuracy": 0.87, ...}
        },
        "default_policy": "earlystop_k5",
        "default_threshold": 0.70
    }
    """
    OPTIMAL_THRESHOLD_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    policies = {}
    for max_k, report in all_reports.items():
        policy_name = f"earlystop_k{max_k}"
        optimal = report["optimal_threshold"]
        policies[policy_name] = {
            "threshold": optimal["threshold"],
            "hybrid_accuracy": optimal["hybrid_accuracy"],
            "skip_ratio": optimal["skip_ratio"],
            "model_accuracy": report["baseline"]["model_accuracy"],
            "debate_accuracy": report["baseline"]["debate_accuracy"]
        }
    
    # Determine default (prefer k5 if available, else highest k)
    if "earlystop_k5" in policies:
        default_policy = "earlystop_k5"
    else:
        default_policy = f"earlystop_k{max(all_reports.keys())}"
    
    config = {
        "version": "2.0",
        "tuning_method": "per_policy",
        "tuned_on": "dev",
        "policies": policies,
        "default_policy": default_policy,
        "default_threshold": policies[default_policy]["threshold"],
        "note": "Option A: Each EarlyStop policy has its own optimal threshold tuned on DEV."
    }
    
    with open(OPTIMAL_THRESHOLD_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✅ Optimal thresholds saved to: {OPTIMAL_THRESHOLD_FILE}")
    logger.info(f"   Default policy: {default_policy}")
    logger.info(f"   Default threshold: {config['default_threshold']}")
    
    # Also update debate_config.json with per-policy thresholds
    _update_debate_config_thresholds(policies, config['default_threshold'])
    
    # Print summary table
    logger.info("\n" + "="*70)
    logger.info("📋 THRESHOLD SUMMARY (Per-Policy)")
    logger.info("="*70)
    logger.info(f"{'Policy':<20} {'Threshold':>10} {'Hybrid Acc':>12} {'Skip %':>10}")
    logger.info("-"*55)
    for policy, data in policies.items():
        logger.info(f"{policy:<20} {data['threshold']:>10.2f} {data['hybrid_accuracy']:>11.2%} {data['skip_ratio']:>9.1%}")
    logger.info("="*70)


def _update_debate_config_thresholds(policies: Dict, default_threshold: float):
    """Update debate_config.json with per-policy thresholds."""
    debate_config_path = PROJECT_ROOT / "config" / "debate" / "debate_config.json"
    
    try:
        with open(debate_config_path, 'r', encoding='utf-8') as f:
            debate_config = json.load(f)
        
        # Update hybrid_strategy section
        if 'hybrid_strategy' not in debate_config:
            debate_config['hybrid_strategy'] = {}
        
        debate_config['hybrid_strategy']['default_threshold'] = default_threshold
        debate_config['hybrid_strategy']['per_policy_thresholds'] = {
            policy: data['threshold'] for policy, data in policies.items()
        }
        
        with open(debate_config_path, 'w', encoding='utf-8') as f:
            json.dump(debate_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Also updated: {debate_config_path}")
    except Exception as e:
        logger.warning(f"⚠️  Could not update debate_config.json: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Hybrid Analysis on DEV set to find optimal threshold (Option A: per-policy)"
    )
    parser.add_argument(
        "--run-debate",
        action="store_true",
        help="Run full debate pipeline on DEV set"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run hybrid analysis on existing DEV results"
    )
    parser.add_argument(
        "--max-k",
        nargs="+",
        type=int,
        default=None,
        help="Specify max_K values to process (default: all available)"
    )
    
    args = parser.parse_args()
    
    if not args.run_debate and not args.analyze:
        parser.print_help()
        print("\n⚠️  Please specify at least one action: --run-debate or --analyze")
        return
    
    print("\n" + "="*70)
    print("HYBRID THRESHOLD TUNING ON DEV SET")
    print("Option A: Per-Policy Threshold (Dec 24, 2025)")
    print("="*70)
    print("Protocol:")
    print("  1. Tune separate threshold for EACH max_K policy on DEV")
    print("  2. Apply corresponding threshold on TEST per policy")
    print("  3. Report TEST results per policy in paper/thesis")
    print("="*70 + "\n")
    
    # Determine which max_K values to process
    max_k_list = args.max_k if args.max_k else None
    
    # Step 1: Run debate if requested
    if args.run_debate:
        success = run_debate_on_dev(max_k_list)
        if not success:
            logger.error("Failed to run debate. Exiting.")
            return
    
    # Step 2: Analyze if requested
    if args.analyze:
        reports = run_hybrid_analysis(max_k_list)
        if reports:
            print("\n✅ Analysis complete!")
            print(f"   📂 Output: {DEV_HYBRID_OUTPUT}")
            print(f"   � Config: {OPTIMAL_THRESHOLD_FILE}")
            print("\n💡 Next step: Run Hybrid on TEST with these thresholds")


if __name__ == "__main__":
    main()
