"""
Fix Judge Mismatches Script - Re-run Judge for specific samples with verdict_match=false

This script:
1. Loads judge results file
2. Finds samples with verdict_match=false
3. Re-runs JudgeV2.make_final_decision for those samples
4. Updates the file with corrected results

Author: Lockdown
Date: Jan 01, 2026
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import asdict

from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Use relative imports from current location
from judge_v2 import JudgeV2
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


def _resolve_env_placeholders(obj: Any) -> Any:
    """Recursively resolve ${ENV_VAR} placeholders inside dict/list/string values."""
    import re
    if isinstance(obj, dict):
        return {k: _resolve_env_placeholders(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_placeholders(v) for v in obj]
    if isinstance(obj, str):
        def repl(match) -> str:
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))
        return re.sub(r"\$\{([A-Z0-9_]+)\}", repl, obj)
    return obj


def convert_to_debate_argument(agent_data: Dict, agent_name: str, round_num: int) -> DebateArgument:
    """Convert existing result data to DebateArgument object."""
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


def create_evidence_from_text(evidence_text: str) -> List[Evidence]:
    """Create Evidence objects from evidence text."""
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


def fix_sample_with_judge(sample: Dict, judge: JudgeV2) -> Dict:
    """Re-run judge for a single sample and return updated sample."""
    try:
        # Reset judge state
        judge.round_evaluations = []
        
        # Extract basic info
        claim = sample.get('statement', '')
        evidence_text = sample.get('evidence', '')
        evidence_list = create_evidence_from_text(evidence_text)
        
        # Get debate history from all_rounds_verdicts
        debate_result_in = sample.get('debate_result')
        if not isinstance(debate_result_in, dict):
            logger.warning(f"Sample {sample.get('id')} has no debate_result, skipping")
            return sample

        all_rounds_verdicts = debate_result_in.get('all_rounds_verdicts', [])
        if not all_rounds_verdicts:
            logger.warning(f"Sample {sample.get('id')} has no debate history, skipping")
            return sample
        
        # Convert to DebateArgument objects
        debate_history = []
        for round_num, round_data in enumerate(all_rounds_verdicts, 1):
            round_arguments = []
            for agent_name, agent_data in round_data.items():
                arg = convert_to_debate_argument(agent_data, agent_name, round_num)
                round_arguments.append(arg)
            debate_history.append(round_arguments)
        
        # Simulate judge evaluation for each round
        judge_evaluations = []
        previous_majority_verdict = None
        
        for round_num, round_arguments in enumerate(debate_history, 1):
            evaluation = judge.evaluate_round(
                claim=claim,
                evidence_list=evidence_list,
                round_num=round_num,
                round_arguments=round_arguments,
                previous_majority_verdict=previous_majority_verdict,
                max_rounds=len(debate_history)
            )
            
            judge_evaluations.append(evaluation)
            previous_majority_verdict = evaluation.majority_verdict
        
        # Make final decision
        original_stop_reason = sample.get('debate_result', {}).get('original_stop_reason', 'max_rounds_reached')
        
        final_verdict = judge.make_final_decision(
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
        
        # Update sample with NEW judge decisions
        updated_sample = sample.copy()
        
        # Replace final verdict info
        updated_sample['final_verdict'] = final_verdict.verdict
        updated_sample['final_verdict_norm'] = final_verdict.verdict  
        
        # Update debate_result with NEW judge info
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
        
        # Calculate TRUE majority verdict from debate votes (not from old judge)
        final_round_verdicts = []
        if debate_history:
            for arg in debate_history[-1]:  # Last round votes
                final_round_verdicts.append(arg.verdict)
        
        # Count votes to determine majority
        from collections import Counter
        vote_counts = Counter(final_round_verdicts)
        if vote_counts:
            true_majority_verdict = max(vote_counts.items(), key=lambda x: x[1])[0]
        else:
            true_majority_verdict = "NEI"  # Fallback
        
        # Update judge vs majority comparison using TRUE majority
        updated_sample['judge_vs_majority'] = {
            'original_majority_verdict': true_majority_verdict,
            'judge_verdict': final_verdict.verdict,
            'verdict_match': (true_majority_verdict == final_verdict.verdict),
            'original_confidence': sample.get('debate_result', {}).get('confidence', 0.5),
            'judge_confidence': 0.5
        }
        
        logger.info(f"✅ Fixed sample {sample.get('id')}: {true_majority_verdict} -> {final_verdict.verdict} (match: {true_majority_verdict == final_verdict.verdict})")
        
        return updated_sample
        
    except Exception as e:
        logger.error(f"Error fixing sample {sample.get('id')}: {e}")
        return sample


def main():
    """Main function to fix judge mismatches."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix judge mismatches by re-running Judge for specific samples")
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to judge results file with mismatches'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Output file path (default: overwrite input file)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show which samples would be fixed without actually fixing them'
    )
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    output_file = Path(args.output_file) if args.output_file else input_file
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1
    
    # Load env vars and models config
    load_dotenv(dotenv_path=project_root / ".env", override=False)
    models_config_path = project_root / "config" / "debate" / "models_config.json"
    
    if not models_config_path.exists():
        logger.error(f"Models config not found: {models_config_path}")
        return 1
    
    with open(models_config_path, 'r', encoding='utf-8') as f:
        models_config = _resolve_env_placeholders(json.load(f))
    
    # Setup LLM client and Judge
    llm_client = LLMClient()
    judge_model_config = models_config.get('judge', {})
    if not judge_model_config:
        logger.warning("No judge config found, using debator[0] as fallback")
        judge_model_config = models_config.get('debators', [{}])[0]
    
    judge = JudgeV2(
        model_config=judge_model_config,
        llm_client=llm_client,
        generate_xai=False
    )
    
    logger.info(f"Using judge model: {judge_model_config.get('model', 'Unknown')}")
    
    # Load results file
    logger.info(f"Loading results from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Find samples with verdict_match=false
    mismatch_samples = []
    for i, sample in enumerate(data.get('results', [])):
        judge_vs_majority = sample.get('judge_vs_majority', {})
        if not judge_vs_majority.get('verdict_match', True):
            mismatch_samples.append((i, sample))
    
    if not mismatch_samples:
        logger.info("✅ No mismatches found - all samples already match!")
        return 0
    
    logger.info(f"🔍 Found {len(mismatch_samples)} samples with verdict_match=false:")
    for i, sample in mismatch_samples:
        judge_vs_majority = sample.get('judge_vs_majority', {})
        original = judge_vs_majority.get('original_majority_verdict', 'Unknown')
        judge_verdict = judge_vs_majority.get('judge_verdict', 'Unknown')
        logger.info(f"  - Sample {sample.get('id')} (index {i}): {original} -> {judge_verdict}")
    
    if args.dry_run:
        logger.info("🏃 Dry run mode - no changes will be made")
        return 0
    
    # Re-run judge for mismatch samples
    logger.info(f"🔄 Re-running judge for {len(mismatch_samples)} samples...")
    
    fixed_count = 0
    for i, sample in mismatch_samples:
        logger.info(f"Fixing sample {sample.get('id')} (index {i})...")
        fixed_sample = fix_sample_with_judge(sample, judge)
        
        # Check if fix worked
        new_judge_vs_majority = fixed_sample.get('judge_vs_majority', {})
        if new_judge_vs_majority.get('verdict_match', False):
            data['results'][i] = fixed_sample
            fixed_count += 1
        else:
            logger.warning(f"❌ Failed to fix sample {sample.get('id')}")
    
    # Update judge_simulation_stats
    if 'judge_simulation_stats' in data:
        original_matches = data['judge_simulation_stats'].get('verdict_matches', 0)
        new_matches = original_matches + fixed_count
        total_samples = data['judge_simulation_stats'].get('total_samples', 0)
        new_match_rate = new_matches / total_samples if total_samples > 0 else 0
        
        data['judge_simulation_stats']['verdict_matches'] = new_matches
        data['judge_simulation_stats']['match_rate'] = new_match_rate
        
        logger.info(f"📊 Updated stats: {new_matches}/{total_samples} matches ({new_match_rate:.1%})")
    
    # Save fixed results
    logger.info(f"💾 Saving fixed results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"🎉 Fixed {fixed_count}/{len(mismatch_samples)} samples successfully!")
    
    if fixed_count == len(mismatch_samples):
        logger.info("✅ All mismatches have been resolved!")
    else:
        logger.warning(f"⚠️  {len(mismatch_samples) - fixed_count} samples still have issues")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
