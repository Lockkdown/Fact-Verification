"""
Run Hybrid Analysis on DEV set to find optimal threshold.

This script:
1. Runs full debate on DEV set (if not already done)
2. Runs hybrid analysis to find optimal threshold
3. Saves the optimal threshold for use on TEST set

Usage:
    # Step 1: Run debate on dev (if needed)
    python run_hybrid_on_dev.py --run-debate
    
    # Step 2: Analyze and find optimal threshold (if debate results exist)
    python run_hybrid_on_dev.py --analyze
    
    # Both steps
    python run_hybrid_on_dev.py --run-debate --analyze

Author: Lockdown
Date: Dec 06, 2025
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

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
DEV_DEBATE_RESULTS = PROJECT_ROOT / "results" / "vifactcheck" / "dev" / "full_debate" / "vifactcheck_dev_results.json"
DEV_HYBRID_OUTPUT = PROJECT_ROOT / "results" / "vifactcheck" / "dev" / "hybrid_analysis"
OPTIMAL_THRESHOLD_FILE = PROJECT_ROOT / "config" / "debate" / "optimal_threshold.json"


def run_debate_on_dev():
    """Run full debate pipeline on DEV set."""
    logger.info("="*70)
    logger.info("STEP 1: Running Full Debate on DEV set")
    logger.info("="*70)
    
    # Check if results already exist
    if DEV_DEBATE_RESULTS.exists():
        logger.info(f"⚠️  DEV debate results already exist at: {DEV_DEBATE_RESULTS}")
        response = input("Do you want to re-run? (y/N): ").strip().lower()
        if response != 'y':
            logger.info("Skipping debate run. Using existing results.")
            return True
    
    # Run eval_vifactcheck_pipeline.py with dev split
    import subprocess
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "pipeline" / "run_pineline" / "eval_vifactcheck_pipeline.py"),
        "--splits", "dev",
        "--debate",
        "--async-debate",
        "--batch-size", "10"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
        logger.info("✅ Debate on DEV completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Debate failed with error: {e}")
        return False


def run_hybrid_analysis():
    """Run hybrid analysis on DEV results to find optimal threshold."""
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Running Hybrid Analysis on DEV set")
    logger.info("="*70)
    
    # Check if debate results exist
    if not DEV_DEBATE_RESULTS.exists():
        logger.error(f"❌ DEV debate results not found at: {DEV_DEBATE_RESULTS}")
        logger.error("Please run with --run-debate first.")
        return None
    
    # Create output directory
    DEV_HYBRID_OUTPUT.mkdir(parents=True, exist_ok=True)
    
    # Run analysis
    logger.info(f"📂 Input: {DEV_DEBATE_RESULTS}")
    logger.info(f"📂 Output: {DEV_HYBRID_OUTPUT}")
    
    analyzer = HybridAnalyzer(str(DEV_DEBATE_RESULTS))
    
    # Test thresholds
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
    
    report = analyzer.generate_full_report(
        output_dir=str(DEV_HYBRID_OUTPUT),
        thresholds=thresholds
    )
    
    # Extract optimal threshold
    optimal_threshold = report["optimal_threshold"]["threshold"]
    optimal_accuracy = report["optimal_threshold"]["hybrid_accuracy"]
    
    logger.info("\n" + "="*70)
    logger.info("🎯 OPTIMAL THRESHOLD FOUND (on DEV set)")
    logger.info("="*70)
    logger.info(f"   Threshold: {optimal_threshold}")
    logger.info(f"   Hybrid Accuracy: {optimal_accuracy:.2%}")
    logger.info(f"   Skip Ratio: {report['optimal_threshold']['skip_ratio']:.1%}")
    logger.info("="*70)
    
    # Save optimal threshold to config
    save_optimal_threshold(optimal_threshold, report)
    
    return report


def save_optimal_threshold(threshold: float, report: dict):
    """Save optimal threshold to config file for use on TEST set."""
    OPTIMAL_THRESHOLD_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    config = {
        "optimal_threshold": threshold,
        "tuned_on": "dev",
        "dev_results": {
            "model_accuracy": report["baseline"]["model_accuracy"],
            "debate_accuracy": report["baseline"]["debate_accuracy"],
            "hybrid_accuracy": report["optimal_threshold"]["hybrid_accuracy"],
            "skip_ratio": report["optimal_threshold"]["skip_ratio"]
        },
        "note": "This threshold was tuned on DEV set. Apply it on TEST set for final evaluation."
    }
    
    with open(OPTIMAL_THRESHOLD_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✅ Optimal threshold saved to: {OPTIMAL_THRESHOLD_FILE}")
    logger.info("   Use this threshold when evaluating on TEST set.")


def main():
    parser = argparse.ArgumentParser(
        description="Run Hybrid Analysis on DEV set to find optimal threshold"
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
    
    args = parser.parse_args()
    
    if not args.run_debate and not args.analyze:
        parser.print_help()
        print("\n⚠️  Please specify at least one action: --run-debate or --analyze")
        return
    
    print("\n" + "="*70)
    print("HYBRID THRESHOLD TUNING ON DEV SET")
    print("="*70)
    print("Following ML best practices:")
    print("  1. Tune threshold on DEV set")
    print("  2. Apply optimal threshold on TEST set")
    print("  3. Report TEST results in paper/thesis")
    print("="*70 + "\n")
    
    # Step 1: Run debate if requested
    if args.run_debate:
        success = run_debate_on_dev()
        if not success:
            logger.error("Failed to run debate. Exiting.")
            return
    
    # Step 2: Analyze if requested
    if args.analyze:
        report = run_hybrid_analysis()
        if report:
            print("\n✅ Analysis complete! Check the output directory for plots and reports.")
            print(f"   📂 {DEV_HYBRID_OUTPUT}")


if __name__ == "__main__":
    main()
