"""
Full ViFactCheck Pipeline Evaluation - Production Ready

Features:
- Progress bar with tqdm
- Checkpoint/Resume support
- Multi-split evaluation (dev + test)
- Incorrect samples export
- Memory efficient processing
- Integrated calibration
- Auto metrics & plots generation

Usage:
    # Dev only
    python eval_vifactcheck_pipeline.py --splits dev --debate --async-debate
    
    # Dev + Test with calibration
    python eval_vifactcheck_pipeline.py --splits dev test --debate --async-debate --calibration
    
    # Resume from checkpoint
    python eval_vifactcheck_pipeline.py --splits test --resume --debate --async-debate

Author: Lockdown
Date: Nov 16, 2025
"""

import os
import sys
import argparse
import json
import time
import gc
import logging
import asyncio
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipeline.run_pineline.config import PipelineConfig
from src.pipeline.run_pineline.article_pipeline import ViFactCheckPipeline
from src.pipeline.debate.debate_metrics import DebateMetricsTracker, DebateSampleMetrics, RoundMetrics
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# DEBATE CONFIG MANAGEMENT (Dec 2025 - Consensus-Based)
# ============================================================================

def update_debate_config(max_rounds: Optional[int], fixed_mode: bool = False) -> None:
    """Update debate_config.json with max_rounds and stop_on_consensus.
    
    Logic (Dec 24, 2025 - align with scope):
    - Fixed mode: stop_on_consensus=False (chạy đủ K rounds)
    - EarlyStop mode: stop_on_consensus=True (dừng khi unanimous+stable)
    
    Args:
        max_rounds: Max rounds limit, or None for unlimited (15)
        fixed_mode: If True, disable early stopping (run full K rounds)
    """
    config_path = project_root / 'config' / 'debate' / 'debate_config.json'
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    config['max_rounds'] = max_rounds
    config['stop_on_consensus'] = not fixed_mode  # False for fixed, True for earlystop
    
    if fixed_mode:
        mode_str = f"FIXED {max_rounds} rounds (no early stop)"
    elif max_rounds is None:
        mode_str = "EARLYSTOP (max safety=15, stop on unanimous+stable)"
    else:
        mode_str = f"EARLYSTOP max {max_rounds} rounds (stop on unanimous+stable)"
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📝 Updated debate_config.json: {mode_str}")


def parse_max_rounds_arg(max_rounds_str: str) -> Optional[int]:
    """Parse max_rounds argument to int or None."""
    if max_rounds_str.lower() in ['unlimited', 'none', 'inf']:
        return None
    return int(max_rounds_str)


def analyze_debate_metrics(results: List[Dict[str, Any]], max_rounds: Optional[int] = None) -> Dict[str, Any]:
    """Analyze debate metrics from results.
    
    Args:
        results: List of result dicts from pipeline
        max_rounds: Max rounds config (for calculating hit_cap_rate)
    
    Returns:
        Dict with:
        - avg_rounds_used: Average rounds used per sample
        - hit_cap_rate: % samples that hit max rounds (chạm trần)
        - early_stop_rate: % samples that stopped early (dừng sớm)
        - consensus_rate_per_round: {round: % samples reaching consensus at this round}
        - error_analysis: {correct_consensus, wrong_consensus, no_consensus}
    """
    consensus_rounds = []
    rounds_used_list = []
    hit_cap_count = 0
    early_stop_count = 0
    error_counts = {"correct_consensus": 0, "wrong_consensus": 0, "no_consensus": 0}
    total_debate_samples = 0
    
    for r in results:
        if not r.get("debate_metrics"):
            continue
            
        total_debate_samples += 1
        metrics = r["debate_metrics"]
        consensus_round = metrics.get("consensus_round")
        rounds_used = metrics.get("rounds_used", 0)
        stop_reason = metrics.get("stop_reason", "")
        early_stopped = metrics.get("early_stopped", False)
        
        # Track rounds used
        if rounds_used:
            rounds_used_list.append(rounds_used)
        
        # Track hit_cap vs early_stop
        if stop_reason == "max_rounds_reached":
            hit_cap_count += 1
        elif early_stopped or consensus_round is not None:
            early_stop_count += 1
        
        # Track consensus round
        if consensus_round is not None:
            consensus_rounds.append(consensus_round)
        
        # Error analysis: compare final_verdict with ground_truth
        final_verdict = r.get("final_verdict", "").upper()
        ground_truth = r.get("label", "").upper()
        
        # Normalize labels
        label_map = {"SUPPORT": "SUPPORTED", "REFUTE": "REFUTED", "SUPPORTS": "SUPPORTED", "REFUTES": "REFUTED"}
        final_verdict = label_map.get(final_verdict, final_verdict)
        ground_truth = label_map.get(ground_truth, ground_truth)
        
        is_correct = final_verdict == ground_truth
        has_consensus = consensus_round is not None
        
        if has_consensus and is_correct:
            error_counts["correct_consensus"] += 1
        elif has_consensus and not is_correct:
            error_counts["wrong_consensus"] += 1
        else:
            error_counts["no_consensus"] += 1
    
    # Calculate consensus rate per round
    consensus_rate_per_round = {}
    if consensus_rounds:
        from collections import Counter
        round_counts = Counter(consensus_rounds)
        for round_num, count in sorted(round_counts.items()):
            consensus_rate_per_round[round_num] = count / total_debate_samples if total_debate_samples > 0 else 0
    
    # Calculate key metrics
    avg_rounds_used = sum(rounds_used_list) / len(rounds_used_list) if rounds_used_list else 0
    hit_cap_rate = hit_cap_count / total_debate_samples if total_debate_samples > 0 else 0
    early_stop_rate = early_stop_count / total_debate_samples if total_debate_samples > 0 else 0
    avg_rounds_to_consensus = sum(consensus_rounds) / len(consensus_rounds) if consensus_rounds else None
    
    return {
        "total_debate_samples": total_debate_samples,
        # NEW: 3 key metrics for paper
        "avg_rounds_used": avg_rounds_used,
        "hit_cap_rate": hit_cap_rate,
        "early_stop_rate": early_stop_rate,
        # Existing metrics
        "consensus_rate_per_round": consensus_rate_per_round,
        "avg_rounds_to_consensus": avg_rounds_to_consensus,
        "samples_with_consensus": len(consensus_rounds),
        "consensus_rate_total": len(consensus_rounds) / total_debate_samples if total_debate_samples > 0 else 0,
        "error_analysis": error_counts
    }


def generate_comparison_report(
    all_experiment_results: Dict[str, Dict[str, Any]],
    output_dir: Path,
    debate_mode: str
) -> Path:
    """Generate a comparison report across all max_rounds configs.
    
    Args:
        all_experiment_results: {config_name: {split: results_data}}
        output_dir: Base output directory
        debate_mode: "full_debate" or "hybrid_debate"
    
    Returns:
        Path to saved report file
    """
    report_dir = output_dir / "comparison_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Build comparison table with debate metrics
    comparison_data = []
    for config_name, splits_data in all_experiment_results.items():
        for split, results in splits_data.items():
            # Analyze debate metrics if results available
            debate_analysis = {}
            if results.get("results"):
                debate_analysis = analyze_debate_metrics(results["results"])
            
            comparison_data.append({
                "max_rounds": config_name,
                "split": split,
                "debate_mode": debate_mode,
                "model_accuracy": results.get("model_accuracy", 0),
                "final_accuracy": results.get("final_accuracy", 0),
                "improvement": results.get("final_accuracy", 0) - results.get("model_accuracy", 0),
                "total_samples": results.get("total_samples", 0),
                "avg_time_per_sample": results.get("avg_per_sample", 0),
                # KEY METRICS for paper (Dec 2025)
                "avg_rounds_used": debate_analysis.get("avg_rounds_used", 0),
                "hit_cap_rate": debate_analysis.get("hit_cap_rate", 0),
                "early_stop_rate": debate_analysis.get("early_stop_rate", 0),
                # Consensus metrics
                "consensus_rate": debate_analysis.get("consensus_rate_total", 0),
                "avg_rounds_to_consensus": debate_analysis.get("avg_rounds_to_consensus"),
                "consensus_rate_per_round": debate_analysis.get("consensus_rate_per_round", {}),
                "error_analysis": debate_analysis.get("error_analysis", {}),
            })
    
    # Save as JSON
    report_file = report_dir / f"comparison_{debate_mode}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    
    # Print comparison table (paper format)
    logger.info("\n" + "="*140)
    logger.info(f"COMPARISON REPORT - {debate_mode.upper()}".center(140))
    logger.info("="*140)
    logger.info(f"{'Config':<12} {'Split':<6} {'Acc':<7} {'F1':<7} {'AvgRnds':<8} {'HitCap%':<9} {'Early%':<8} {'Consens%':<10} {'CorrectC':<9} {'WrongC':<8}")
    logger.info("-"*140)
    
    for row in comparison_data:
        error = row.get("error_analysis", {})
        logger.info(
            f"{row['max_rounds']:<12} "
            f"{row['split']:<6} "
            f"{row['final_accuracy']:.1%}  "
            f"{'--':<7} "  # F1 placeholder - computed separately
            f"{row['avg_rounds_used']:.2f}    "
            f"{row['hit_cap_rate']:.1%}     "
            f"{row['early_stop_rate']:.1%}   "
            f"{row['consensus_rate']:.1%}      "
            f"{error.get('correct_consensus', 0):<9} "
            f"{error.get('wrong_consensus', 0):<8}"
        )
    
    logger.info("="*140)
    
    # Print key metrics summary
    logger.info("\n📊 KEY DEBATE METRICS:")
    for row in comparison_data:
        logger.info(f"  {row['max_rounds']} ({row['split']}):")
        logger.info(f"    - avg_rounds_used: {row['avg_rounds_used']:.2f}")
        logger.info(f"    - hit_cap_rate: {row['hit_cap_rate']:.1%} (chạm trần)")
        logger.info(f"    - early_stop_rate: {row['early_stop_rate']:.1%} (dừng sớm)")
    
    # Print consensus rate per round details
    logger.info("\n📊 CONSENSUS RATE PER ROUND:")
    for row in comparison_data:
        if row.get("consensus_rate_per_round"):
            logger.info(f"  {row['max_rounds']} ({row['split']}):")
            for round_num, rate in row["consensus_rate_per_round"].items():
                logger.info(f"    Round {round_num}: {rate:.1%}")
    
    logger.info(f"\n📊 Report saved to: {report_file}")
    
    return report_file


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def save_checkpoint(
    results: List[Dict[str, Any]],
    split: str,
    checkpoint_dir: Path,
    metadata: Optional[Dict] = None
) -> Path:
    """Save checkpoint file."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_data = {
        "split": split,
        "timestamp": datetime.now().isoformat(),
        "total_processed": len(results),
        "results": results,
        "metadata": metadata or {}
    }
    
    checkpoint_file = checkpoint_dir / f"checkpoint_{split}_{len(results)}.json"
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"💾 Checkpoint saved: {checkpoint_file} ({len(results)} samples)")
    return checkpoint_file


def load_checkpoint(checkpoint_dir: Path, split: str) -> Optional[Dict[str, Any]]:
    """Load latest checkpoint for split."""
    if not checkpoint_dir.exists():
        return None
    
    # Find all checkpoints for this split
    checkpoints = list(checkpoint_dir.glob(f"checkpoint_{split}_*.json"))
    if not checkpoints:
        return None
    
    # Get latest (by sample count)
    latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
    
    logger.info(f"📂 Loading checkpoint: {latest}")
    with open(latest, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"✅ Resumed from {len(data['results'])} processed samples")
    return data


# ============================================================================
# DATA LOADING
# ============================================================================

def load_vifactcheck_split(split: str, max_samples: int = None) -> List[Dict]:
    """
    Load ViFactCheck dev/test split from PRE-PROCESSED JSONL files.
    
    ⚠️ Updated Dec 01, 2025: Now using PyVi pre-processed data to match
    the new PyVi-trained model (83.48% accuracy).
    
    Args:
        split: 'dev' or 'test'
        max_samples: Max samples to load (None = all)
    
    Returns:
        List of samples with: statement, evidence, gold_label, etc.
    """
    from pathlib import Path
    
    logger.info(f"📂 Loading ViFactCheck {split} set (PyVi pre-processed)...")
    
    # Path to pre-processed JSONL files
    # ⚡ Updated: Using PyVi data to match training (Dec 01, 2025)
    project_root = Path(__file__).parent.parent.parent.parent
    jsonl_path = project_root / "dataset" / "processed" / "vifactcheck_pyvi" / f"vifactcheck_{split}.jsonl"
    
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Pre-processed file not found: {jsonl_path}\n"
            f"Please run preprocessing_vifactcheck_pyvi.py first."
        )
    
    # Label mapping: 0=Support, 1=Refute, 2=NOT_ENOUGH_INFO
    label_map = {0: "Support", 1: "Refute", 2: "NOT_ENOUGH_INFO"}
    
    samples = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            
            item = json.loads(line.strip())
            
            samples.append({
                "id": item.get("sample_id", i + 1),
                "statement": item["statement"],  # Already PyVi segmented
                "evidence": item["evidence"],    # Already PyVi segmented
                "context": item.get("context", ""),
                "gold_label": label_map[item["label"]],
                "gold_label_id": item["label"]
            })
    
    # Print stats
    label_counts = {}
    for s in samples:
        label = s["gold_label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    total_in_file = sum(1 for _ in open(jsonl_path, 'r', encoding='utf-8'))
    
    if max_samples:
        logger.info(f"✅ Loaded {len(samples)} samples (limited from {total_in_file} total)")
    else:
        logger.info(f"✅ Loaded {len(samples)} samples")
    logger.info(f"📊 Distribution: {label_counts}")
    logger.info(f"🔧 Tokenization: PyVi (consistent with training)")
    
    return samples


# ============================================================================
# PIPELINE PROCESSING
# ============================================================================

def process_sample(
    pipeline: ViFactCheckPipeline,
    sample: Dict[str, Any],
    config: PipelineConfig
) -> Dict[str, Any]:
    """Process single sample using new ViFactCheckPipeline."""
    
    statement = sample['statement'].strip()
    evidence = sample['evidence'].strip()
    
    # Run pipeline (Gold Evidence mode - no context)
    result = pipeline.predict(
        statement=statement,
        evidence=evidence,
        use_debate=config.use_debate
    )
    
    return _build_result(sample, result)


async def process_sample_async(
    pipeline: ViFactCheckPipeline,
    sample: Dict[str, Any],
    config: PipelineConfig
) -> Dict[str, Any]:
    """Async version of process_sample for batch processing."""
    
    statement = sample['statement'].strip()
    evidence = sample['evidence'].strip()
    
    # Run async pipeline (Gold Evidence mode - no context)
    result = await pipeline.predict_async(
        statement=statement,
        evidence=evidence,
        use_debate=config.use_debate
    )
    
    return _build_result(sample, result)


def _build_result(sample: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """Build final result dict from pipeline output."""
    
    # Extract verdicts from result
    model_verdict = result.get("model_verdict", "NOT_ENOUGH_INFO")
    final_verdict = result.get("final_verdict", "NOT_ENOUGH_INFO")
    
    # Build debate info if available
    if result.get("debate_verdict"):
        debate_info = {
            "verdict": result.get("debate_verdict"),
            "confidence": result.get("debate_confidence", 0.0),
            "reasoning": result.get("debate_reasoning", "")
        }
        debate_result = {
            "verdict": result.get("debate_verdict"),
            "confidence": result.get("debate_confidence", 0.0),
            "reasoning": result.get("debate_reasoning", ""),
            "round_1_verdicts": result.get("debate_round_1_verdicts", {}),
            "all_rounds_verdicts": result.get("debate_all_rounds_verdicts", []),
            "metrics": result.get("debate_metrics", {}),
            "xai": result.get("debate_xai", {})
        }
    else:
        debate_info = None
        debate_result = None
    
    # Normalize verdicts
    verdict_map = {
        "Support": "Support", "SUPPORTS": "Support", "SUPPORTED": "Support",
        "Refute": "Refute", "REFUTES": "Refute", "REFUTED": "Refute",
        "NEI": "NOT_ENOUGH_INFO", "NOT_ENOUGH_INFO": "NOT_ENOUGH_INFO",
        "UNVERIFIED": "NOT_ENOUGH_INFO"
    }
    model_verdict = verdict_map.get(model_verdict, "NOT_ENOUGH_INFO")
    final_verdict = verdict_map.get(final_verdict, "NOT_ENOUGH_INFO")

    # Display-normalized labels for consistent reporting (SUPPORTED/REFUTED/NEI)
    def _to_debate_label(label: str) -> str:
        if not label:
            return "NEI"
        v = str(label).upper().strip()
        if v in ["SUPPORT", "SUPPORTS", "SUPPORTED", "0"]:
            return "SUPPORTED"
        if v in ["REFUTE", "REFUTES", "REFUTED", "1"]:
            return "REFUTED"
        return "NEI"

    gold_label_norm = _to_debate_label(sample.get("gold_label", ""))
    model_verdict_norm = _to_debate_label(model_verdict)
    final_verdict_norm = _to_debate_label(final_verdict)
    
    # Check correctness
    model_correct = (model_verdict == sample["gold_label"])
    final_correct = (final_verdict == sample["gold_label"])
    
    return {
        "id": sample["id"],
        "statement": sample["statement"],
        "evidence": sample["evidence"],  # Gold evidence for XAI comparison
        "gold_label": sample["gold_label"],
        "gold_label_norm": gold_label_norm,
        "model_verdict": model_verdict,
        "model_verdict_norm": model_verdict_norm,
        "final_verdict": final_verdict,
        "final_verdict_norm": final_verdict_norm,
        "model_correct": model_correct,
        "final_correct": final_correct,
        "debate_info": debate_info,
        "debate_result": debate_result,
        "verdict_3label_probs": result.get("model_probs")
    }


def run_pipeline_on_split(
    split: str,
    config: PipelineConfig,
    checkpoint_every: int = 50,
    resume: bool = False,
    output_dir: Path = None,
    max_samples: Optional[int] = None,
    batch_size: int = 1,
    quiet: bool = False
) -> Dict[str, Any]:
    """
    Run pipeline on a split with checkpoint support.
    
    Args:
        split: 'dev' or 'test'
        config: Pipeline configuration
        checkpoint_every: Save checkpoint every N samples
        resume: Resume from checkpoint if available
        output_dir: Output directory
        max_samples: Maximum number of samples to process (None = all)
        batch_size: Number of concurrent samples to process (Default: 1)
        quiet: Suppress detailed pipeline logs (show only progress bar)
    
    Returns:
        Dict with results, timing, metrics
    """
    
    logger.info("\n" + "="*80)
    logger.info(f"VIFACTCHECK EVALUATION - {split.upper()}".center(80))
    logger.info("="*80)
    logger.info(f"💬 Debate: {'✓ Enabled' if config.use_debate else '✗ Disabled'}")
    logger.info(f"⚡ Async: {'✓ Enabled' if config.use_async_debate else '✗ Disabled'}")
    if batch_size > 1:
        logger.info(f"� Batch Processing: {batch_size} samples concurrently")
    logger.info(f"� Checkpoint: Every {checkpoint_every} samples")
    logger.info("="*80 + "\n")
    
    # Setup directories
    checkpoint_dir = output_dir / split / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load samples
    samples = load_vifactcheck_split(split, max_samples=max_samples)
    
    # Check for existing checkpoint
    start_idx = 0
    results = []
    
    if resume:
        checkpoint_data = load_checkpoint(checkpoint_dir, split)
        if checkpoint_data:
            results = checkpoint_data["results"]
            start_idx = len(results)
            logger.info(f"▶️  Resuming from sample {start_idx + 1}/{len(samples)}\n")
    
    if start_idx == 0:
        logger.info(f"▶️  Starting fresh evaluation\n")
    
    # Initialize pipeline
    pipeline = ViFactCheckPipeline(config)
    start_time = time.time()
    
    # Suppress pipeline logs if quiet mode OR batch mode (too messy)
    if quiet or batch_size > 1:
        # Save original log level
        original_level = logging.getLogger().level
        # Set to WARNING to hide INFO logs from pipeline
        logging.getLogger().setLevel(logging.WARNING)
    
    # Process samples with progress bar
    samples_to_process = samples[start_idx:]
    
    with tqdm(
        total=len(samples),
        initial=start_idx,
        desc=f"Processing {split}",
        unit="sample",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    ) as pbar:
        
        if batch_size > 1:
            # --- ASYNC CONTINUOUS WORKER MODE (Semaphore) ---
            logger.info(f"🚀 Starting continuous async processing with {batch_size} concurrent workers...")
            
            async def run_with_semaphore(pipeline_ref):
                try:
                    # Limit concurrency
                    sem = asyncio.Semaphore(batch_size)
                    
                    async def bounded_process(sample):
                        async with sem:
                            # Small random stagger (0.0-0.5s) to prevent exact burst overlap
                            await asyncio.sleep(random.uniform(0.0, 0.5))
                            try:
                                return await process_sample_async(pipeline_ref, sample, config)
                            except Exception as e:
                                logger.error(f"Error processing sample {sample['id']}: {e}")
                                return {
                                    "id": sample["id"],
                                    "statement": sample.get("statement", ""),
                                    "gold_label": sample.get("gold_label", ""),
                                    "final_verdict": "ERROR",
                                    "final_correct": False,
                                    "model_correct": False,
                                    "error": str(e)
                                }

                    # Create all tasks
                    tasks = [bounded_process(s) for s in samples_to_process]
                    
                    # Process tasks as they complete (Continuous Feed)
                    for coro in asyncio.as_completed(tasks):
                        res = await coro
                        results.append(res)
                        
                        # Update progress
                        current_count = len(results)
                        recent_acc = sum(r.get('final_correct', False) for r in results) / current_count if results else 0
                        pbar.set_postfix_str(f"Acc: {recent_acc:.1%}")
                        pbar.update(1)
                        
                        # Checkpoint
                        if current_count % checkpoint_every == 0:
                            # Sort results by ID before saving to keep file organized
                            sorted_results = sorted(results, key=lambda x: x["id"])
                            save_checkpoint(sorted_results, split, checkpoint_dir)
                            gc.collect()
                            
                finally:
                    # Clean shutdown of network sessions
                    logger.info("🛑 Cleaning up async sessions...")
                    await pipeline_ref.shutdown()

            # Run the main async loop
            try:
                asyncio.run(run_with_semaphore(pipeline))
            except Exception as e:
                logger.error(f"Critical error in async loop: {e}")
                # Try to save whatever we have
                if results:
                    sorted_results = sorted(results, key=lambda x: x["id"])
                    save_checkpoint(sorted_results, split, checkpoint_dir)
        
        else:
            # --- SEQUENTIAL MODE ---
            for idx, sample in enumerate(samples_to_process, start=start_idx + 1):
                try:
                    result = process_sample(pipeline, sample, config)
                    results.append(result)
                    
                    # Update progress bar with status
                    status = "✓" if result["final_correct"] else "✗"
                    pbar.set_postfix_str(
                        f"{status} [{result['final_verdict'][:3]}] "
                        f"Acc: {sum(r['final_correct'] for r in results)/len(results):.1%}"
                    )
                    pbar.update(1)
                    
                    # Save checkpoint
                    if idx % checkpoint_every == 0:
                        save_checkpoint(results, split, checkpoint_dir)
                    
                    # Memory cleanup every 100 samples
                    if idx % 100 == 0:
                        gc.collect()
                
                except Exception as e:
                    logger.error(f"Error processing sample {sample['id']}: {e}")
                    pbar.update(1)
                    results.append({
                        "id": sample["id"],
                        "statement": sample["statement"],
                        "gold_label": sample["gold_label"],
                        "model_verdict": "ERROR",
                        "final_verdict": "ERROR",
                        "verdict_3label_probs": None,
                        "model_correct": False,
                        "final_correct": False,
                        "error": str(e)
                    })
    
    # Restore original log level if quiet mode
    if quiet:
        logging.getLogger().setLevel(original_level)
    
    # Final checkpoint
    save_checkpoint(results, split, checkpoint_dir)
    
    elapsed = time.time() - start_time
    
    # Compute accuracy
    model_correct = sum(1 for r in results if r.get("model_correct", False))
    final_correct = sum(1 for r in results if r.get("final_correct", False))
    
    logger.info(f"\n✅ {split.upper()} Completed!")
    logger.info(f"📊 Model Accuracy: {model_correct}/{len(results)} = {model_correct/len(results):.2%}")
    logger.info(f"📊 Final Accuracy: {final_correct}/{len(results)} = {final_correct/len(results):.2%}")
    logger.info(f"⏱️  Time: {elapsed:.2f}s ({elapsed/len(results):.2f}s per sample)")
    
    return {
        "split": split,
        "total_samples": len(results),
        "model_accuracy": model_correct / len(results),
        "final_accuracy": final_correct / len(results),
        "total_seconds": elapsed,
        "avg_per_sample": elapsed / len(results),
        "results": results
    }


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(
    data: Dict[str, Any],
    split: str,
    output_dir: Path,
    skip_split_subdir: bool = False
) -> Dict[str, Path]:
    """
    Save results to multiple files.
    
    Args:
        skip_split_subdir: If True, don't add split as subdirectory (for hybrid mode)
    
    Returns:
        Dict with paths to saved files
    """
    
    if skip_split_subdir:
        split_dir = output_dir
    else:
        split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    # 1. Full results
    full_results_file = split_dir / f"vifactcheck_{split}_results.json"
    with open(full_results_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    saved_files["full_results"] = full_results_file
    logger.info(f"💾 Full results: {full_results_file}")
    
    # 2. Incorrect samples only
    incorrect_samples = [
        r for r in data["results"]
        if not r.get("final_correct", False) and r.get("final_verdict") != "ERROR"
    ]
    
    incorrect_file = split_dir / f"vifactcheck_{split}_incorrect.json"
    incorrect_data = {
        "total_incorrect": len(incorrect_samples),
        "accuracy": data["final_accuracy"],
        "samples": incorrect_samples
    }
    with open(incorrect_file, 'w', encoding='utf-8') as f:
        json.dump(incorrect_data, f, ensure_ascii=False, indent=2)
    saved_files["incorrect"] = incorrect_file
    logger.info(f"💾 Incorrect samples: {incorrect_file} ({len(incorrect_samples)} samples)")
    
    # 3. Summary stats
    summary_file = split_dir / f"summary_{split}.json"
    summary = {
        "split": split,
        "total_samples": data["total_samples"],
        "model_accuracy": data["model_accuracy"],
        "final_accuracy": data["final_accuracy"],
        "improvement": data["final_accuracy"] - data["model_accuracy"],
        "total_seconds": data["total_seconds"],
        "avg_per_sample": data["avg_per_sample"],
        "timestamp": datetime.now().isoformat()
    }
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    saved_files["summary"] = summary_file
    logger.info(f"💾 Summary: {summary_file}")
    
    # 4. Debate Metrics & Visualizations (if debate was enabled)
    has_debate = any(r.get("debate_info") for r in data["results"])
    if has_debate:
        logger.info(f"\n📊 Generating Debate Metrics & Visualizations...")
        
        # Save metrics to metrics/ folder (Dec 24, 2025 cleanup)
        metrics_dir = split_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metrics tracker
        tracker = DebateMetricsTracker()
        
        for r in data["results"]:
            if r.get("final_verdict") == "ERROR":
                continue
            
            # Extract gold label for accuracy computation
            gold_label = r.get("gold_label", "")
            
            # Extract debate metrics
            debate_metrics = r.get("debate_result", {}).get("metrics", {}) if r.get("debate_result") else {}
            
            # ✅ FIX: Parse ALL rounds verdicts for metrics visualization
            round_metrics_list = []
            debate_result = r.get("debate_result", {}) if r.get("debate_result") else {}
            all_rounds_verdicts = debate_result.get("all_rounds_verdicts", [])
            
            # If all_rounds_verdicts available, use it (new format)
            if all_rounds_verdicts:
                from collections import Counter
                prev_majority = None
                for round_idx, round_data in enumerate(all_rounds_verdicts):
                    round_num = round_idx + 1
                    verdicts = {name: data.get("verdict", "NEI") for name, data in round_data.items()}
                    confidences = {name: data.get("confidence", 0.0) for name, data in round_data.items()}
                    reasonings = {name: data.get("reasoning", "") for name, data in round_data.items()}
                    roles = {name: data.get("role", "") for name, data in round_data.items()}
                    
                    # XAI Interaction fields (Round 2+)
                    agree_with = {name: data.get("agree_with", []) or [] for name, data in round_data.items()}
                    agree_reasons = {name: data.get("agree_reason", "") or "" for name, data in round_data.items()}
                    disagree_with = {name: data.get("disagree_with", []) or [] for name, data in round_data.items()}
                    disagree_reasons = {name: data.get("disagree_reason", "") or "" for name, data in round_data.items()}
                    changed = {name: data.get("changed", False) for name, data in round_data.items()}
                    change_reasons = {name: data.get("change_reason", "") or "" for name, data in round_data.items()}
                    
                    verdict_counts = Counter(verdicts.values())
                    majority_verdict = verdict_counts.most_common(1)[0][0] if verdict_counts else "NEI"
                    agreement_ratio = max(verdict_counts.values()) / len(verdicts) if verdicts else 0.0
                    verdict_changed = (prev_majority is not None and majority_verdict != prev_majority)
                    
                    # Compute if majority verdict is correct
                    majority_correct = False
                    if majority_verdict and gold_label:
                        # Normalize verdicts for comparison
                        m_norm = majority_verdict.upper()
                        if m_norm in ['SUPPORTED', 'SUPPORT']:
                            m_norm = 'Support'
                        elif m_norm in ['REFUTED', 'REFUTE']:
                            m_norm = 'Refute'
                        else:
                            m_norm = 'NOT_ENOUGH_INFO'
                        majority_correct = (m_norm == gold_label)
                    
                    round_metrics_list.append(RoundMetrics(
                        round_num=round_num,
                        verdicts=verdicts,
                        confidences=confidences,
                        reasonings=reasonings,
                        roles=roles,
                        agreement_ratio=agreement_ratio,
                        majority_verdict=majority_verdict,
                        verdict_changed_from_prev=verdict_changed,
                        correct=majority_correct,
                        # XAI fields
                        agree_with=agree_with,
                        agree_reasons=agree_reasons,
                        disagree_with=disagree_with,
                        disagree_reasons=disagree_reasons,
                        changed=changed,
                        change_reasons=change_reasons
                    ))
                    prev_majority = majority_verdict
            else:
                # Fallback: use round_1_verdicts only (old format)
                round_1_verdicts = debate_result.get("round_1_verdicts", {})
                if round_1_verdicts:
                    from collections import Counter
                    verdicts = {name: data.get("verdict", "NEI") for name, data in round_1_verdicts.items()}
                    confidences = {name: data.get("confidence", 0.0) for name, data in round_1_verdicts.items()}
                    reasonings = {name: data.get("reasoning", "") for name, data in round_1_verdicts.items()}
                    roles = {name: data.get("role", "") for name, data in round_1_verdicts.items()}
                    
                    verdict_counts = Counter(verdicts.values())
                    majority_verdict = verdict_counts.most_common(1)[0][0] if verdict_counts else "NEI"
                    agreement_ratio = max(verdict_counts.values()) / len(verdicts) if verdicts else 0.0
                    
                    # Compute if majority verdict is correct
                    majority_correct = False
                    if majority_verdict and gold_label:
                        # Normalize verdicts for comparison
                        m_norm = majority_verdict.upper()
                        if m_norm in ['SUPPORTED', 'SUPPORT']:
                            m_norm = 'Support'
                        elif m_norm in ['REFUTED', 'REFUTE']:
                            m_norm = 'Refute'
                        else:
                            m_norm = 'NOT_ENOUGH_INFO'
                        majority_correct = (m_norm == gold_label)
                    
                    round_metrics_list.append(RoundMetrics(
                        round_num=1,
                        verdicts=verdicts,
                        confidences=confidences,
                        reasonings=reasonings,
                        roles=roles,
                        agreement_ratio=agreement_ratio,
                        majority_verdict=majority_verdict,
                        verdict_changed_from_prev=False,
                        correct=majority_correct
                    ))
            
            sample = DebateSampleMetrics(
                sample_id=str(r.get("id", "")),
                statement=r.get("statement", "")[:200],
                gold_label=r.get("gold_label", ""),
                model_verdict=r.get("model_verdict", "NEI"),
                model_confidence=r.get("verdict_3label_probs", {}).get(r.get("model_verdict", "NEI"), 0.0) if r.get("verdict_3label_probs") else 0.0,
                final_verdict=r.get("final_verdict", "NEI"),
                final_confidence=r.get("debate_info", {}).get("confidence", 0.0) if r.get("debate_info") else 0.0,
                rounds_used=debate_metrics.get("rounds_used", 1),
                early_stopped=debate_metrics.get("early_stopped", False),
                stop_reason=debate_metrics.get("stop_reason", ""),
                mvp_agent=debate_metrics.get("mvp_agent", "Unknown"),
                round_metrics=round_metrics_list  # ✅ FIX: Now includes XAI data
            )
            tracker.add_sample(sample)
        
        # Save metrics JSON
        metrics_file = metrics_dir / f"debate_metrics_{split}.json"
        tracker.save_metrics(str(metrics_file))
        saved_files["debate_metrics"] = metrics_file
        
        
        # Print debate summary
        debate_summary = tracker.get_summary()
        logger.info(f"\n💬 Debate Summary ({split.upper()}):")
        logger.info(f"  Model Accuracy:  {debate_summary.get('model_accuracy', 0):.2%}")
        logger.info(f"  Final Accuracy:  {debate_summary.get('final_accuracy', 0):.2%}")
        logger.info(f"  Debate Gain:     {debate_summary.get('debate_gain', 0):+.2%}")
        logger.info(f"  Fixed:           {debate_summary.get('fixed_count', 0)} samples")
        logger.info(f"  Broke:           {debate_summary.get('broke_count', 0)} samples")
        logger.info(f"  Avg Rounds:      {debate_summary.get('avg_rounds', 0):.2f}")
    
    return saved_files


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full ViFactCheck Pipeline Evaluation - Production Ready"
    )
    
    # Splits to evaluate
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["dev"],
        choices=["dev", "test"],
        help="Splits to evaluate (default: dev)"
    )
    
    # Pipeline config
    parser.add_argument(
        "--debate",
        action="store_true",
        help="Enable debate system (legacy flag, use --full-debate or --hybrid-debate instead)"
    )
    parser.add_argument(
        "--async-debate",
        action="store_true",
        help="Enable async debate (faster)"
    )
    
    # Debate Mode Selection (DOWN Framework)
    debate_mode = parser.add_mutually_exclusive_group()
    debate_mode.add_argument(
        "--full-debate",
        action="store_true",
        help="Run debate on ALL samples (ignore Hybrid threshold)"
    )
    debate_mode.add_argument(
        "--hybrid-debate",
        action="store_true",
        help="Run debate ONLY when model confidence < threshold (DOWN Framework)"
    )
    
    # Consensus-Based Debate Config (Dec 2025)
    parser.add_argument(
        "--max-rounds",
        nargs="+",
        default=None,
        help="Max rounds configs to test (e.g., --max-rounds 3 5 7). If not set, uses config file value."
    )
    parser.add_argument(
        "--run-all-configs",
        action="store_true",
        help="Run all max_rounds configs (3, 5, 7) and generate comparison report"
    )
    parser.add_argument(
        "--run-both-modes",
        action="store_true",
        help="Run BOTH early-stop and fixed modes for the selected max_rounds configs (scope: 3, 5, 7)"
    )
    parser.add_argument(
        "--fixed",
        action="store_true",
        help="Fixed rounds mode: run exactly K rounds without early stopping (default: early stop enabled)"
    )
    
    # Checkpoint/Resume
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Save checkpoint every N samples (default: 50)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/vifactcheck for full-debate, results/vifactcheck/hybrid_debate for hybrid-debate)"
    )
    
    # Calibration
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Fit temperature on dev, apply on test (requires both splits)"
    )
    
    # Max samples (for testing)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per split (None = all, for testing)"
    )
    
    # Batch size (Concurrency)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of concurrent debates (Default: 1). Recommended: 5-8."
    )
    
    # Quiet mode
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed pipeline logs (show only progress bar)"
    )
    
    # Auto evaluation & plots
    parser.add_argument(
        "--full-report",
        action="store_true",
        default=True,
        help="Auto run evaluation + generate plots after pipeline (default: True)"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip evaluation & plots generation"
    )
    
    args = parser.parse_args()
    
    # Setup output directory based on mode
    # Structure: results/vifactcheck/{split}/ for full debate
    #            results/vifactcheck/{split}/hybrid_debate/ for hybrid mode
    if args.output_dir is None:
        base_dir = Path(project_root) / "results" / "vifactcheck"
        if args.hybrid_debate:
            # Hybrid mode: inside each split's folder (e.g., test/hybrid_debate/)
            # Note: actual path will be {base_dir}/{split}/hybrid_debate/ set during processing
            args.output_dir = base_dir
            args.hybrid_subdir = True  # Flag to append hybrid_debate to split dir
        else:
            # Full debate or default: standard directory
            args.output_dir = base_dir
            args.hybrid_subdir = False
    else:
        args.output_dir = Path(args.output_dir)
        args.hybrid_subdir = False
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup config
    config = PipelineConfig()
    config.save_intermediate = False
    
    # Debate mode handling
    if args.full_debate:
        config.use_debate = True
        config.hybrid_enabled = False  # Force disable hybrid
        debate_mode_str = "FULL (All Samples)"
    elif args.hybrid_debate:
        config.use_debate = True
        config.hybrid_enabled = True  # Enable hybrid
        debate_mode_str = "HYBRID (Confidence-Aware)"
    else:
        config.use_debate = args.debate
        config.hybrid_enabled = True  # Default: hybrid if debate enabled
        debate_mode_str = "HYBRID" if args.debate else "DISABLED"
    
    config.use_async_debate = args.async_debate
    
    # Determine max_rounds configs to run
    if args.run_all_configs or args.run_both_modes:
        # Scope-aligned (Dec 2025): Only 3-5-7
        max_rounds_configs = ["3", "5", "7"]
    elif args.max_rounds:
        max_rounds_configs = args.max_rounds
    else:
        max_rounds_configs = [None]  # Use config file value (single run)
    
    debate_mode_folder = "hybrid_debate" if args.hybrid_subdir else "full_debate"
    
    logger.info("\n" + "="*80)
    logger.info("VIFACTCHECK FULL EVALUATION".center(80))
    logger.info("="*80)
    logger.info(f"📋 Splits: {', '.join(args.splits)}")
    logger.info(f"💬 Debate: {'✓' if config.use_debate else '✗'}")
    logger.info(f"⚡ Async: {'✓' if args.async_debate else '✗'}")
    if config.use_debate:
        logger.info(f"🎯 Debate Mode: {debate_mode_str}")
        if args.run_all_configs or args.max_rounds:
            logger.info(f"🔄 Max Rounds Configs: {max_rounds_configs}")
        if args.hybrid_subdir:
            logger.info(f"📁 Output Mode: HYBRID → {{split}}/hybrid_debate/")
        else:
            logger.info(f"📁 Output Mode: FULL DEBATE → {{split}}/full_debate/")
        logger.info(f"📦 Evidence: Gold (from dataset)")
    logger.info(f"💾 Checkpoint: Every {args.checkpoint_every} samples")
    logger.info(f"🔄 Resume: {'✓' if args.resume else '✗'}")
    if args.batch_size > 1:
        logger.info(f"🚀 Batch Size: {args.batch_size} concurrent samples")
    if args.max_samples:
        logger.info(f"⚠️  Max Samples: {args.max_samples} per split (TESTING MODE)")
    if args.quiet:
        logger.info(f"🔇 Quiet Mode: ✓ (pipeline logs suppressed)")
    logger.info(f"📊 Base Output: {args.output_dir}")
    logger.info("="*80)

    # Store results for all configs (for comparison report)
    all_experiment_results = {}
    
    # Adaptive strategy: Start with round 3, progressively increase if hit_cap_rate >= threshold
    adaptive_threshold = 0.06  # 6% threshold
    configs_to_run = []

    # If running both modes in one command, do NOT use adaptive scheduling.
    # We want full coverage: (earlystop + fixed) x (3,5,7)
    if args.run_both_modes:
        configs_to_run = max_rounds_configs
    elif args.run_all_configs:
        # Adaptive mode: Start with 3, add more configs based on hit_cap_rate
        configs_to_run = ["3"]  # Always start with 3
        logger.info(f"\n🎯 ADAPTIVE STRATEGY: Starting with max_rounds=3")
        logger.info(f"   Will continue to higher rounds if hit_cap_rate >= {adaptive_threshold:.0%}")
    else:
        configs_to_run = max_rounds_configs

    # Determine which debate execution modes to run
    modes_to_run = [args.fixed]
    if args.run_both_modes:
        modes_to_run = [False, True]  # earlystop then fixed

    # Loop through modes x max_rounds configs
    for fixed_mode in modes_to_run:
        # Loop through max_rounds configs
        for idx, max_rounds_str in enumerate(configs_to_run):
            if max_rounds_str is not None:
                max_rounds = parse_max_rounds_arg(max_rounds_str)
                update_debate_config(max_rounds, fixed_mode=fixed_mode)
                config_name = max_rounds_str
            else:
                config_name = "default"
            
            # Determine mode prefix for folder naming (Dec 24, 2025 - align with scope)
            if fixed_mode:
                mode_prefix = "fixed"
                mode_display = f"FIXED K={config_name}"
            else:
                mode_prefix = "earlystop"
                mode_display = f"EARLYSTOP max={config_name}"
            
            logger.info(f"\n{'#'*80}")
            logger.info(f"CONFIG: {mode_display}".center(80))
            logger.info(f"{'#'*80}")
            
            # Unique key so comparison report can include both modes
            config_key = f"{mode_prefix}_{config_name}"
            all_experiment_results[config_key] = {}

            # Process each split for this config
            for split in args.splits:
                logger.info(f"\n{'='*80}")
                logger.info(f"Processing {split.upper()} split ({mode_display})".center(80))
                logger.info(f"{'='*80}\n")
                
                # Determine output directory for this split + config
                # Structure: results/vifactcheck/{split}/{debate_mode}/{mode_prefix}_k{rounds}/
                # Dec 24, 2025: folder naming aligned with scope (fixed_k3, earlystop_k7, etc.)
                if config_name in ['none', 'inf', 'default']:
                    config_folder = f"{mode_prefix}_k7"  # Safety default
                else:
                    config_folder = f"{mode_prefix}_k{config_name}"
                
                if args.hybrid_subdir:
                    split_output_dir = args.output_dir / split / "hybrid_debate" / config_folder
                else:
                    split_output_dir = args.output_dir / split / "full_debate" / config_folder
                
                # Run pipeline
                split_results = run_pipeline_on_split(
                    split=split,
                    config=config,
                    checkpoint_every=args.checkpoint_every,
                    resume=args.resume,
                    output_dir=split_output_dir,
                    max_samples=args.max_samples,
                    batch_size=args.batch_size,
                    quiet=args.quiet
                )
                
                # Save results
                saved_files = save_results(split_results, split, split_output_dir, skip_split_subdir=True)
                
                # Store for comparison report
                all_experiment_results[config_key][split] = split_results

                # Run evaluation & plots (default: True, skip if --no-report)
                if args.full_report and not args.no_report:
                    logger.info(f"\n📊 Generating metrics & plots for {split}...")
                    try:
                        # Step 1: Call evaluate_results.py to compute metrics
                        import subprocess
                        eval_script = Path(__file__).parent / "evaluate_results.py"
                        result_file = saved_files["full_results"]
                        
                        # Create metrics directory and set proper metrics file path
                        metrics_dir = split_output_dir / "metrics"
                        metrics_dir.mkdir(parents=True, exist_ok=True)
                        metrics_file = metrics_dir / f"metrics_{split}.json"
                        
                        subprocess.run([
                            sys.executable,
                            str(eval_script),
                            str(result_file),
                            "--save-metrics",
                            str(metrics_file)
                        ], check=True)
                        
                        logger.info(f"✅ Metrics saved to: {metrics_file}")
                        
                        # Print metrics summary to console
                        logger.info(f"\n📊 ===== METRICS SUMMARY ({split.upper()}) =====")
                        with open(metrics_file, 'r', encoding='utf-8') as f:
                            metrics = json.load(f)
                        
                        logger.info(f"\n📈 Overall:")
                        logger.info(f"  Model Accuracy:  {metrics['model_accuracy']:.2%}")
                        logger.info(f"  Final Accuracy:  {metrics['final_accuracy']:.2%}")
                        
                        logger.info(f"\n📊 Final Macro-Averaged:")
                        final_macro = metrics["final_macro"]
                        logger.info(f"  Precision: {final_macro['macro_precision']:.4f}")
                        logger.info(f"  Recall:    {final_macro['macro_recall']:.4f}")
                        logger.info(f"  F1:        {final_macro['macro_f1']:.4f}")
                        
                        logger.info(f"\n📋 Per-Class (Final):")
                        logger.info(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
                        logger.info("-" * 70)
                        for label, m in metrics["final_per_class"].items():
                            logger.info(f"{label:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10}")
                        
                        if metrics.get("debate_impact"):
                            debate = metrics["debate_impact"]
                            logger.info(f"\n💬 Debate Impact:")
                            logger.info(f"  Fixed:  {debate['fixed']} samples ({debate['fix_rate']:.2f}%)")
                            logger.info(f"  Broken: {debate['broken']} samples ({debate['break_rate']:.2f}%)")
                        
                        logger.info(f"\n{'='*80}\n")
                        
                        # Step 2: Generate plots using plot_metrics.py
                        logger.info(f"\n📈 Generating plots for {split}...")
                        import importlib.util
                        spec = importlib.util.spec_from_file_location(
                            "plot_metrics",
                            Path(__file__).parent / "plot_metrics.py"
                        )
                        plot_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(plot_module)
                        
                        with open(metrics_file, 'r', encoding='utf-8') as f:
                            metrics = json.load(f)
                        
                        charts_dir = split_output_dir / "charts"
                        plot_module.generate_all_plots(metrics, split_results["results"], split.upper(), charts_dir)
                        logger.info(f"✅ All plots saved to: {charts_dir}/")
                    except Exception as e:
                        logger.warning(f"⚠️  Could not generate metrics/plots: {e}")
                        import traceback
                        traceback.print_exc()

            # Adaptive strategy: Check if we need to run higher rounds
            # Only apply to single-mode run-all-configs (not run-both-modes)
            if (not args.run_both_modes) and args.run_all_configs and idx == len(configs_to_run) - 1:
                all_splits_results = []
                for split_name, split_data in all_experiment_results[config_key].items():
                    if split_data.get("results"):
                        all_splits_results.extend(split_data["results"])
                
                if all_splits_results:
                    metrics = analyze_debate_metrics(all_splits_results)
                    hit_cap_rate = metrics.get("hit_cap_rate", 0)
                    logger.info(f"\n📊 ADAPTIVE DECISION for max_rounds={config_name}:")
                    logger.info(f"   hit_cap_rate = {hit_cap_rate:.1%}")
                    if hit_cap_rate >= adaptive_threshold:
                        next_config_map = {"3": "5", "5": "7"}
                        next_config = next_config_map.get(config_name)
                        if next_config:
                            configs_to_run.append(next_config)
                            logger.info(f"   ✅ hit_cap_rate >= {adaptive_threshold:.0%} → Adding max_rounds={next_config}")
                        else:
                            logger.info(f"   ⏹️  Already at max_rounds=7, no higher config available")
                    else:
                        logger.info(f"   ⏹️  hit_cap_rate < {adaptive_threshold:.0%} → Stopping adaptive strategy")

    # Calibration workflow
    if args.calibration and "dev" in args.splits and "test" in args.splits:
        logger.info("\n" + "="*80)
        logger.info("CALIBRATION WORKFLOW".center(80))
        logger.info("="*80)
        logger.info("📊 Fitting temperature on dev set...")
        logger.info("📊 Applying calibrated T on test set...")
        logger.info("⚠️  TODO: Implement calibration workflow")
        # TODO: Call calibration_3label.py here
    
    # Generate comparison report if multiple configs
    if len(max_rounds_configs) > 1:
        debate_mode = "hybrid_debate" if args.hybrid_subdir else "full_debate"
        generate_comparison_report(all_experiment_results, args.output_dir, debate_mode)
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE".center(80))
    logger.info("="*80)
    
    for config_name, splits_data in all_experiment_results.items():
        logger.info(f"\n📊 Config: max_rounds = {config_name}")
        for split, data in splits_data.items():
            logger.info(f"  {split.upper()}:")
            logger.info(f"    Model Accuracy:  {data.get('model_accuracy', 0):.2%}")
            logger.info(f"    Final Accuracy:  {data.get('final_accuracy', 0):.2%}")
            logger.info(f"    Improvement:     {data.get('final_accuracy', 0) - data.get('model_accuracy', 0):+.2%}")
    
    logger.info("\n✅ All evaluations completed successfully!")


if __name__ == "__main__":
    main()
