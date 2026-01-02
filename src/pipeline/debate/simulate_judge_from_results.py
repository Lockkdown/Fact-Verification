"""
Judge Simulation Script - Replace Majority Vote with Judge Decisions

Tạo judge decisions từ existing debate results để thỏa mãn yêu cầu của giảng viên.
Script này sẽ:
1. Load existing results files (k3, k5, k7...)  
2. Simulate judge decisions từ debate arguments
3. Replace majority vote với judge reasoning
4. Generate comparison table để verify consistency

Author: Lockdown
Date: Dec 31, 2025
"""

import json
import logging
import os
import sys
import re
import time
import asyncio
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import pandas as pd

from dotenv import load_dotenv

# Add project root to path để import modules
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Use relative imports from current location
from judge_v2 import JudgeV2, RoundEvaluation, FinalVerdict
from debator import DebateArgument, Evidence

# For LLMClient
llm_client_path = project_root / "src" / "pipeline" / "debate"
sys.path.insert(0, str(llm_client_path))
from llm_client import LLMClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def _resolve_env_placeholders(obj: Any) -> Any:
    """Recursively resolve ${ENV_VAR} placeholders inside dict/list/string values."""
    if isinstance(obj, dict):
        return {k: _resolve_env_placeholders(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_placeholders(v) for v in obj]
    if isinstance(obj, str):
        def repl(match: re.Match) -> str:
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))

        return re.sub(r"\$\{([A-Z0-9_]+)\}", repl, obj)
    return obj


class JudgeSimulator:
    """
    Simulate judge decisions từ existing debate results.
    """
    
    def __init__(self, models_config_path: str):
        """
        Args:
            models_config_path: Path to models config with API keys
        """
        # Load env vars (.env) so ${VAR} placeholders can be resolved
        load_dotenv(dotenv_path=project_root / ".env", override=False)

        # Load models config
        with open(models_config_path, 'r', encoding='utf-8') as f:
            self.models_config = _resolve_env_placeholders(json.load(f))
        
        # Setup LLM client
        self.llm_client = LLMClient()
        
        # Get judge model config
        judge_model_config = self.models_config.get('judge', {})
        if not judge_model_config:
            logger.warning("No judge config found, using debator[0] as fallback")
            judge_model_config = self.models_config.get('debators', [{}])[0]
        
        logger.info(f"Using judge model: {judge_model_config.get('model', 'Unknown')}")
        logger.info(f"Using judge base_url: {judge_model_config.get('base_url', 'Unknown')}")
        
        # Initialize Judge V2
        self.judge = JudgeV2(
            model_config=judge_model_config,
            llm_client=self.llm_client,
            generate_xai=False  # Set to True if you want XAI
        )
    
    def convert_to_debate_argument(self, agent_data: Dict, agent_name: str, round_num: int) -> DebateArgument:
        """
        Convert existing result data to DebateArgument object.
        """
        # Map agent names to roles for consistency
        role_map = {
            "x-ai/grok-4-fast": "Truth Seeker A",
            "google/gemini-2.5-flash": "Truth Seeker B", 
            "openai/gpt-4o-mini": "Truth Seeker C"
        }
        
        return DebateArgument(
            debator_name=agent_name,
            role=role_map.get(agent_name, agent_name),
            round_num=round_num,
            verdict=agent_data.get('verdict', 'NEI'),
            confidence=float(agent_data.get('confidence', 0.5)),
            reasoning=agent_data.get('reasoning', ''),
            key_points=agent_data.get('key_points', []),
            evidence_citations=[],  # Not available in existing results
            # Round 2+ fields
            agree_with=agent_data.get('agree_with'),
            agree_reason=agent_data.get('agree_reason'),
            disagree_with=agent_data.get('disagree_with'),
            disagree_reason=agent_data.get('disagree_reason'),
            changed=agent_data.get('changed', False),
            change_reason=agent_data.get('change_reason'),
            # Structured fields
            parts=agent_data.get('parts'),
            decision_change=agent_data.get('decision_change'),
            changed_from=agent_data.get('changed_from'),
            change_trigger=agent_data.get('change_trigger'),
            rebuttals=agent_data.get('rebuttals'),
            key_parts_checked=agent_data.get('key_parts_checked')
        )
    
    def create_evidence_from_text(self, evidence_text: str) -> List[Evidence]:
        """
        Create Evidence objects from evidence text.
        """
        if not evidence_text:
            return []
        
        return [Evidence(
            text=evidence_text,
            source="dataset",
            rank=1,
            nli_score={},
            relevance_score=1.0,
            evidence_type="DIRECT"
        )]
    
    def simulate_judge_for_sample(self, sample: Dict) -> Dict:
        """
        Simulate judge decisions for one sample.
        
        Args:
            sample: One sample from results file
            
        Returns:
            Updated sample with judge decisions
        """
        try:
            # Reset per-sample judge state (avoid carrying round_evaluations across samples)
            self.judge.round_evaluations = []

            # Extract basic info
            claim = sample.get('statement', '')
            evidence_text = sample.get('evidence', '')
            evidence_list = self.create_evidence_from_text(evidence_text)
            
            # Get debate history from all_rounds_verdicts
            debate_result_in = sample.get('debate_result')
            if not isinstance(debate_result_in, dict):
                # Hybrid skip path (MODEL_HIGH_CONF) often has debate_result = null
                updated_sample = sample.copy()
                final_v = updated_sample.get('final_verdict_norm', updated_sample.get('final_verdict', ''))
                updated_sample['judge_vs_majority'] = {
                    'original_majority_verdict': final_v,
                    'judge_verdict': final_v,
                    'verdict_match': True,
                    'original_confidence': updated_sample.get('debate_result', {}).get('confidence', 0.5) if isinstance(updated_sample.get('debate_result'), dict) else 0.5,
                    'judge_confidence': 0.5
                }
                return updated_sample

            all_rounds_verdicts = debate_result_in.get('all_rounds_verdicts', [])
            if not all_rounds_verdicts:
                logger.warning(f"No debate history for sample {sample.get('id')}")
                return sample
            
            # Convert to DebateArgument objects
            debate_history = []
            for round_num, round_data in enumerate(all_rounds_verdicts, 1):
                round_arguments = []
                for agent_name, agent_data in round_data.items():
                    arg = self.convert_to_debate_argument(agent_data, agent_name, round_num)
                    round_arguments.append(arg)
                debate_history.append(round_arguments)
            
            # Simulate judge evaluation for each round
            judge_evaluations = []
            previous_majority_verdict = None
            
            for round_num, round_arguments in enumerate(debate_history, 1):
                evaluation = self.judge.evaluate_round(
                    claim=claim,
                    evidence_list=evidence_list,
                    round_num=round_num,
                    round_arguments=round_arguments,
                    previous_majority_verdict=previous_majority_verdict,
                    max_rounds=len(debate_history)
                )
                
                judge_evaluations.append(evaluation)
                previous_majority_verdict = evaluation.majority_verdict
                
                # Check if judge decided to stop
                if evaluation.early_stop_decision and round_num < len(debate_history):
                    logger.info(f"Judge would have stopped at round {round_num}, but using full history")
            
            # Make final decision
            original_stop_reason = sample.get('debate_result', {}).get('stop_reason', 'max_rounds_reached')
            
            final_verdict = self.judge.make_final_decision(
                claim=claim,
                evidence_list=evidence_list,
                debate_history=debate_history,
                stop_reason=original_stop_reason
            )

            # Derive judge stop info from judge evaluations for consistency
            judge_stop_round = None
            for ev in judge_evaluations:
                if ev.early_stop_decision:
                    judge_stop_round = ev.round_num
                    break
            if judge_stop_round is None:
                judge_stop_round = len(debate_history)

            judge_early_stopped = judge_stop_round < len(debate_history)
            judge_stop_reason = "unanimous_stable_consensus" if judge_early_stopped else "max_rounds_reached"
            
            # Update sample with judge decisions
            updated_sample = sample.copy()
            
            # Replace final verdict info
            updated_sample['final_verdict'] = final_verdict.verdict
            updated_sample['final_verdict_norm'] = final_verdict.verdict  
            
            # Update debate_result with judge info
            debate_result = updated_sample.get('debate_result', {}).copy()
            debate_result.update({
                'verdict': final_verdict.verdict,
                'confidence': 0.5,
                'reasoning': final_verdict.reasoning,
                'decision_path': final_verdict.decision_path,
                'judge_evaluations': [asdict(eval) for eval in judge_evaluations],
                'early_stopped': judge_early_stopped,
                'stop_reason': judge_stop_reason,
                'judge_stop_round': judge_stop_round,
                'original_stop_reason': original_stop_reason
            })
            
            updated_sample['debate_result'] = debate_result
            
            # Add judge vs majority comparison
            original_verdict = sample.get('final_verdict_norm', sample.get('final_verdict', ''))
            updated_sample['judge_vs_majority'] = {
                'original_majority_verdict': original_verdict,
                'judge_verdict': final_verdict.verdict,
                'verdict_match': (original_verdict == final_verdict.verdict),
                'original_confidence': sample.get('debate_result', {}).get('confidence', 0.5),
                'judge_confidence': 0.5
            }
            
            return updated_sample
            
        except Exception as e:
            logger.error(f"Error simulating judge for sample {sample.get('id')}: {e}")
            return sample

    async def simulate_judge_for_sample_async(self, sample: Dict) -> Dict:
        """Async version of simulate_judge_for_sample for batch processing."""
        try:
            # Reset per-sample judge state (avoid carrying round_evaluations across samples)
            self.judge.round_evaluations = []

            # Extract basic info
            claim = sample.get('statement', '')
            evidence_text = sample.get('evidence', '')
            evidence_list = self.create_evidence_from_text(evidence_text)
            
            # Get debate history from all_rounds_verdicts
            debate_result_in = sample.get('debate_result')
            if not isinstance(debate_result_in, dict):
                # Hybrid skip path (MODEL_HIGH_CONF) often has debate_result = null
                updated_sample = sample.copy()
                final_v = updated_sample.get('final_verdict_norm', updated_sample.get('final_verdict', ''))
                updated_sample['judge_vs_majority'] = {
                    'original_majority_verdict': final_v,
                    'judge_verdict': final_v,
                    'verdict_match': True,
                    'original_confidence': updated_sample.get('debate_result', {}).get('confidence', 0.5) if isinstance(updated_sample.get('debate_result'), dict) else 0.5,
                    'judge_confidence': 0.5
                }
                return updated_sample

            all_rounds_verdicts = debate_result_in.get('all_rounds_verdicts', [])
            if not all_rounds_verdicts:
                return sample
            
            # Convert to DebateArgument objects
            debate_history = []
            for round_num, round_data in enumerate(all_rounds_verdicts, 1):
                round_arguments = []
                for agent_name, agent_data in round_data.items():
                    arg = self.convert_to_debate_argument(agent_data, agent_name, round_num)
                    round_arguments.append(arg)
                debate_history.append(round_arguments)
            
            # Simulate judge evaluation for each round (using async versions)
            judge_evaluations = []
            previous_majority_verdict = None
            
            for round_num, round_arguments in enumerate(debate_history, 1):
                evaluation = await self.judge.evaluate_round_async(
                    claim=claim,
                    evidence_list=evidence_list,
                    round_num=round_num,
                    round_arguments=round_arguments,
                    previous_majority_verdict=previous_majority_verdict,
                    max_rounds=len(debate_history)
                )
                
                judge_evaluations.append(evaluation)
                previous_majority_verdict = evaluation.majority_verdict
                
                # Check if judge decided to stop
                if evaluation.early_stop_decision and round_num < len(debate_history):
                    logger.info(f"Judge would have stopped at round {round_num}, but using full history")
            
            # Make final decision (using async version)
            original_stop_reason = sample.get('debate_result', {}).get('stop_reason', 'max_rounds_reached')
            
            final_verdict = await self.judge.make_final_decision_async(
                claim=claim,
                evidence_list=evidence_list,
                debate_history=debate_history,
                stop_reason=original_stop_reason
            )

            # Derive judge stop info from judge evaluations for consistency
            judge_stop_round = None
            for ev in judge_evaluations:
                if ev.early_stop_decision:
                    judge_stop_round = ev.round_num
                    break
            if judge_stop_round is None:
                judge_stop_round = len(debate_history)

            judge_early_stopped = judge_stop_round < len(debate_history)
            judge_stop_reason = "unanimous_stable_consensus" if judge_early_stopped else "max_rounds_reached"
            
            # Update sample with judge decisions
            updated_sample = sample.copy()
            
            # Replace final verdict info
            updated_sample['final_verdict'] = final_verdict.verdict
            updated_sample['final_verdict_norm'] = final_verdict.verdict  
            
            # Update debate_result with judge info
            debate_result = updated_sample.get('debate_result', {}).copy()
            debate_result.update({
                'verdict': final_verdict.verdict,
                'confidence': 0.5,
                'reasoning': final_verdict.reasoning,
                'decision_path': final_verdict.decision_path,
                'judge_evaluations': [asdict(eval) for eval in judge_evaluations],
                'early_stopped': judge_early_stopped,
                'stop_reason': judge_stop_reason,
                'judge_stop_round': judge_stop_round,
                'original_stop_reason': original_stop_reason
            })
            
            updated_sample['debate_result'] = debate_result
            
            # Add judge vs majority comparison
            original_verdict = sample.get('final_verdict_norm', sample.get('final_verdict', ''))
            updated_sample['judge_vs_majority'] = {
                'original_majority_verdict': original_verdict,
                'judge_verdict': final_verdict.verdict,
                'verdict_match': (original_verdict == final_verdict.verdict),
                'original_confidence': sample.get('debate_result', {}).get('confidence', 0.5),
                'judge_confidence': 0.5
            }
            
            return updated_sample
            
        except Exception as e:
            logger.error(f"Error in async judge simulation for sample {sample.get('id')}: {e}")
            return sample
    
    def process_results_file(self, input_file: str, output_file: str, batch_size: int = 1) -> Dict[str, Any]:
        """
        Process one results file and generate judge decisions.
        
        Args:
            input_file: Path to input results file
            output_file: Path to output file with judge decisions
            batch_size: Number of concurrent judge calls (1 = sequential)
            
        Returns:
            Summary statistics
        """
        logger.info(f"Processing {input_file}...")
        logger.info(f"Will save judge results to {output_file}")
        
        # Load results
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process each sample
        updated_samples = []
        match_count = 0
        skipped_no_debate = 0
        total_samples = len(data.get('results', []))
        
        logger.info(f"Starting judge simulation for {total_samples} samples...")
        logger.info("⚠️  This will make API calls to DeepSeek V3.1 - make sure .env is configured!")

        # Reduce console noise from judge_v2 during batch runs
        logging.getLogger("judge_v2").setLevel(logging.WARNING)

        start_time = time.time()

        if tqdm is not None:
            pbar = tqdm(total=total_samples, desc="Judging", unit="sample")
        else:
            pbar = None
        
        samples_to_process = data.get('results', [])
        
        if batch_size > 1:
            # --- CONCURRENT BATCH MODE (ThreadPoolExecutor) ---
            logger.info(f"🚀 Starting concurrent batch processing with {batch_size} workers...")
            
            try:
                with ThreadPoolExecutor(max_workers=batch_size) as executor:
                    # Submit all tasks
                    future_to_sample = {
                        executor.submit(self.simulate_judge_for_sample, sample): sample 
                        for sample in samples_to_process
                    }
                    
                    # Process results as they complete
                    try:
                        for future in as_completed(future_to_sample):
                            sample = future_to_sample[future]
                            try:
                                result = future.result()
                                updated_samples.append(result)

                                # Count skipped samples (no judge injection possible)
                                if result.get('debate_result') is None:
                                    skipped_no_debate += 1
                                
                                # Count matches
                                if result.get('judge_vs_majority', {}).get('verdict_match', False):
                                    match_count += 1
                                
                                # Update progress
                                current_count = len(updated_samples)
                                match_rate_current = match_count / current_count if current_count > 0 else 0
                                
                                if pbar is not None:
                                    elapsed = time.time() - start_time
                                    rate = current_count / elapsed if elapsed > 0 else 0.0
                                    remaining = (total_samples - current_count) / rate if rate > 0 else 0.0
                                    pbar.set_postfix_str(f"Match: {match_rate_current:.1%} | ETA {_format_seconds(remaining)}")
                                    pbar.update(1)
                                else:
                                    if current_count % 10 == 0 or current_count == total_samples:
                                        elapsed = time.time() - start_time
                                        rate = current_count / elapsed if elapsed > 0 else 0.0
                                        remaining = (total_samples - current_count) / rate if rate > 0 else 0.0
                                        pct = (current_count / total_samples) * 100 if total_samples else 0
                                        print(
                                            f"\rProgress: {current_count}/{total_samples} ({pct:5.1f}%) | Match: {match_rate_current:.1%} | ETA {_format_seconds(remaining)}",
                                            end="",
                                            flush=True,
                                        )
                            
                            except Exception as e:
                                logger.error(f"Error processing sample {sample.get('id')}: {e}")
                                # Add original sample on error
                                updated_samples.append(sample)
                                current_count = len(updated_samples)
                                if pbar is not None:
                                    pbar.update(1)
                    
                    except KeyboardInterrupt:
                        logger.warning("🛑 KeyboardInterrupt detected! Canceling remaining tasks...")
                        # Cancel all pending futures
                        for future in future_to_sample:
                            future.cancel()
                        # Executor will shutdown gracefully when exiting context
                        raise
                
            except Exception as e:
                logger.error(f"Concurrent processing failed: {e}")
                logger.info("Falling back to sequential processing...")
                # Fall back to sequential if concurrent fails
                batch_size = 1
        
        if batch_size == 1:
            # --- SEQUENTIAL MODE ---
            for i, sample in enumerate(samples_to_process, 1):
                updated_sample = self.simulate_judge_for_sample(sample)
                updated_samples.append(updated_sample)

                if updated_sample.get('debate_result') is None:
                    skipped_no_debate += 1

                # Count matches
                if updated_sample.get('judge_vs_majority', {}).get('verdict_match', False):
                    match_count += 1

                # Progress update
                match_rate_current = match_count / i if i > 0 else 0
                
                if pbar is not None:
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0.0
                    remaining = (total_samples - i) / rate if rate > 0 else 0.0
                    pbar.set_postfix_str(f"Match: {match_rate_current:.1%} | ETA {_format_seconds(remaining)}")
                    pbar.update(1)
                else:
                    if i % 10 == 0 or i == total_samples:
                        elapsed = time.time() - start_time
                        rate = i / elapsed if elapsed > 0 else 0.0
                        remaining = (total_samples - i) / rate if rate > 0 else 0.0
                        pct = (i / total_samples) * 100 if total_samples else 0
                        print(
                            f"\rProgress: {i}/{total_samples} ({pct:5.1f}%) | Match: {match_rate_current:.1%} | ETA {_format_seconds(remaining)}",
                            end="",
                            flush=True,
                        )

        if pbar is not None:
            pbar.close()
        else:
            print("", flush=True)
        
        # Update data with processed samples
        updated_data = data.copy()
        updated_data['results'] = updated_samples
        
        # Add summary stats
        match_rate = match_count / total_samples if total_samples > 0 else 0
        updated_data['judge_simulation_stats'] = {
            'total_samples': total_samples,
            'verdict_matches': match_count,
            'match_rate': match_rate,
            'skipped_no_debate': skipped_no_debate,
            'processed_with_judge': True,
            'simulation_date': '2025-12-31'
        }
        
        # Save updated results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Saved judge results to {output_file}")
        logger.info(f"📊 Judge vs Majority match rate: {match_rate:.1%}")
        
        return {
            'input_file': input_file,
            'output_file': output_file,
            'total_samples': total_samples,
            'verdict_matches': match_count,
            'match_rate': match_rate
        }
    
    def create_comparison_report(self, stats_list: List[Dict], output_dir: str, suffix: str = ""):
        """
        Create comparison report across all processed files.
        """
        # Create comparison DataFrame
        comparison_data = []
        for stats in stats_list:
            comparison_data.append({
                'Configuration': os.path.basename(os.path.dirname(stats['input_file'])),
                'Input File': os.path.basename(stats['input_file']),
                'Output File': os.path.basename(stats['output_file']),
                'Total Samples': stats['total_samples'],
                'Verdict Matches': stats['verdict_matches'],
                'Match Rate': f"{stats['match_rate']:.1%}",
                'Disagreements': stats['total_samples'] - stats['verdict_matches']
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Save as CSV with suffix for individual k processing
        comparison_file = os.path.join(output_dir, f'judge_vs_majority_comparison{suffix}.csv')
        df.to_csv(comparison_file, index=False)
        
        # Create summary
        total_samples = sum(s['total_samples'] for s in stats_list)
        total_matches = sum(s['verdict_matches'] for s in stats_list)
        overall_match_rate = total_matches / total_samples if total_samples > 0 else 0
        
        summary = {
            'overall_stats': {
                'total_samples_processed': total_samples,
                'total_verdict_matches': total_matches,
                'overall_match_rate': overall_match_rate,
                'files_processed': len(stats_list),
                'simulation_date': '2025-12-31'
            },
            'per_file_stats': stats_list,
            'methodology_note': "Judge decisions were generated based on existing debate arguments to replace majority voting mechanism. This maintains methodological consistency while providing more sophisticated reasoning."
        }
        
        summary_file = os.path.join(output_dir, f'judge_simulation_summary{suffix}.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📊 Overall match rate: {overall_match_rate:.1%}")
        logger.info(f"📄 Comparison report saved to {comparison_file}")
        logger.info(f"📋 Summary saved to {summary_file}")


def main():
    """Main execution function."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Simulate judge decisions from existing debate results")
    parser.add_argument(
        '--k', 
        type=str, 
        choices=['3', '5', '7', 'all'], 
        default='all',
        help='Which k value to process: 3, 5, 7, or all (default: all)'
    )
    parser.add_argument(
        '--split', 
        type=str, 
        choices=['test', 'dev'], 
        default='test',
        help='Which data split to process: test or dev (default: test)'
    )

    parser.add_argument(
        '--input-mode',
        type=str,
        choices=['full_debate', 'hybrid_debate'],
        default='full_debate',
        help='Which results folder to read from: full_debate or hybrid_debate (default: full_debate)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Number of concurrent judge calls (default: 1 = sequential). Recommended: 3-5 for speed.'
    )
    
    args = parser.parse_args()
    
    # Configuration paths
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    models_config_path = project_root / "config" / "debate" / "models_config.json"
    
    # Results directories - save in SAME directories as original files
    results_base = project_root / "results" / "vifactcheck" / args.split / args.input_mode
    
    # Define all possible files
    all_files_map = {
        '3': ("earlystop_k3/vifactcheck_test_results.json", "earlystop_k3/vifactcheck_test_results_judge.json"),
        '5': ("earlystop_k5/vifactcheck_test_results.json", "earlystop_k5/vifactcheck_test_results_judge.json"),
        '7': ("earlystop_k7/vifactcheck_test_results.json", "earlystop_k7/vifactcheck_test_results_judge.json")
    }
    
    # Adjust filenames for dev split
    if args.split == 'dev':
        all_files_map = {
            '3': ("earlystop_k3/vifactcheck_dev_results.json", "earlystop_k3/vifactcheck_dev_results_judge.json"),
            '5': ("earlystop_k5/vifactcheck_dev_results.json", "earlystop_k5/vifactcheck_dev_results_judge.json"), 
            '7': ("earlystop_k7/vifactcheck_dev_results.json", "earlystop_k7/vifactcheck_dev_results_judge.json")
        }
    
    # Select files to process based on argument
    if args.k == 'all':
        files_to_process = list(all_files_map.values())
        logger.info(f"🚀 Processing ALL k values (3, 5, 7) for {args.split} split")
    else:
        files_to_process = [all_files_map[args.k]]
        logger.info(f"🎯 Processing k={args.k} only for {args.split} split")
    
    # Check models config
    if not models_config_path.exists():
        logger.error(f"Models config not found: {models_config_path}")
        return
    
    # Check results base directory
    if not results_base.exists():
        logger.error(f"Results directory not found: {results_base}")
        return
    
    # Initialize simulator
    simulator = JudgeSimulator(
        models_config_path=str(models_config_path)
    )
    
    # Process each file
    all_stats = []
    for input_file_rel, output_file_rel in files_to_process:
        input_file = results_base / input_file_rel
        output_file = results_base / output_file_rel  # Same directory as input
        
        if not input_file.exists():
            logger.warning(f"Input file not found: {input_file}")
            continue
        
        stats = simulator.process_results_file(
            input_file=str(input_file),
            output_file=str(output_file),
            batch_size=args.batch_size
        )
        all_stats.append(stats)
    
    # Create comparison report in results base directory
    if all_stats:
        output_suffix = f"_k{args.k}" if args.k != 'all' else ""
        simulator.create_comparison_report(all_stats, str(results_base), suffix=output_suffix)
    
    logger.info("🎉 Judge simulation completed!")
    logger.info(f"💡 Files saved for k={args.k} in {args.split} split directories")
    if args.batch_size > 1:
        logger.info(f"🚀 Used async batch processing with {args.batch_size} concurrent workers")


if __name__ == "__main__":
    main()
