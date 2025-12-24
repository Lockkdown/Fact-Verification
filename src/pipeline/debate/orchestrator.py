"""
Consensus-Based Multi-Round Debate Orchestrator.

Flow (v2.0 - Dec 22, 2025):
- Round 1: Independent reasoning (đọc & tự phán)
- Round 2...N: Cross-examination (tranh luận dựa trên câu trả lời của agents khác)
- Dừng khi: 3/3 agents đồng thuận HOẶC đạt max rounds
- Judge: LLM call để tổng hợp verdict

Note: XAI generation removed from pipeline (Dec 24, 2025)
      XAI is now only used in demo UI via xai_debate.py

Author: Lockdown
Date: Dec 22, 2025 (Refactored for true debate)
"""

import json
import logging
import os
import re
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from collections import Counter
import numpy as np
from dotenv import load_dotenv

from .debator import Debator, GenericDebator, Evidence, DebateArgument
from .judge import Judge, FinalVerdict, JudgeR2Anchor
from .llm_client import LLMClient
from .hybrid_strategy import HybridStrategy

logger = logging.getLogger(__name__)


class AdaptiveDebateOrchestrator:
    """
    Orchestrator cho adaptive multi-agent debate.
    Implements early stopping strategies để optimize cost.
    """
    
    def __init__(
        self,
        models_config_path: Optional[str] = None,
        debate_config_path: Optional[str] = None,
        env_file: Optional[str] = None,
        environment: Optional[str] = None
    ):
        """
        Args:
            models_config_path: Path to models_config.json (auto-detect if None)
            debate_config_path: Path to debate_config.json (auto-detect if None)
            env_file: Optional path to .env file (auto-detect if None)
            environment: 'production' or 'testing' (auto-detect from .env if None)
        """
        
        # Load .env file
        if env_file:
            load_dotenv(env_file)
        else:
            # Auto-detect .env from project root
            project_root = Path(__file__).parent.parent.parent.parent
            env_path = project_root / '.env'
            if env_path.exists():
                load_dotenv(env_path)
                logger.info(f"Loaded environment from: {env_path}")
            else:
                logger.warning("No .env file found, using system environment variables")
        
        # Determine environment
        if environment is None:
            environment = os.getenv('ENVIRONMENT', 'development').lower()
        
        # Auto-detect config paths if not provided
        if models_config_path is None or debate_config_path is None:
            config_dir = Path(__file__).parent.parent.parent.parent / 'config' / 'debate'
            
            if models_config_path is None:
                # Use single config file for all environments
                models_config_path = str(config_dir / 'models_config.json')
            
            if debate_config_path is None:
                debate_config_path = str(config_dir / 'debate_config.json')
        
        logger.info(f"Using environment: {environment}")
        logger.info(f"Models config: {models_config_path}")
        logger.info(f"Debate config: {debate_config_path}")
        
        # Load configs
        with open(models_config_path, 'r', encoding='utf-8') as f:
            models_config_str = f.read()
        
        with open(debate_config_path, 'r', encoding='utf-8') as f:
            debate_config_str = f.read()
        
        # Replace environment variables in configs
        models_config_str = self._replace_env_vars(models_config_str)
        debate_config_str = self._replace_env_vars(debate_config_str)
        
        # Parse JSON
        self.models_config = json.loads(models_config_str)
        self.debate_config = json.loads(debate_config_str)
        
        # Initialize LLM client
        self.llm_client = LLMClient(
            retry_config=self.debate_config.get('retry', {})
        )
        
        # Initialize debators
        self.debators = self._init_debators()
        
        # Initialize judge
        self.judge = self._init_judge()
        
        logger.info(f"Initialized debate system with {len(self.debators)} debators (Gold Evidence mode)")
        logger.info(f"Strategy: {self.debate_config['debate_strategy']}")
        logger.info(f"Rounds: {self.debate_config['min_rounds']}-{self.debate_config['max_rounds']}")
        
        # Initialize Hybrid Strategy (DOWN Framework, 2025)
        hybrid_config = self.debate_config.get('hybrid_strategy', {})
        self.hybrid_enabled = hybrid_config.get('enabled', True)
        self.hybrid_threshold = hybrid_config.get('default_threshold', 0.85)
        self.hybrid_strategy = HybridStrategy(threshold=self.hybrid_threshold)
        logger.info(f"Hybrid Strategy: {'ENABLED' if self.hybrid_enabled else 'DISABLED'} (threshold={self.hybrid_threshold})")
    
    def set_hybrid_threshold(self, threshold: float):
        """
        Set hybrid threshold (e.g., after tuning on dev set).
        
        Args:
            threshold: New threshold value (0.0 - 1.0)
        """
        self.hybrid_threshold = threshold
        self.hybrid_strategy = HybridStrategy(threshold=threshold)
        logger.info(f"Hybrid threshold updated to: {threshold}")
    
    async def close(self):
        """Close all resources."""
        if self.llm_client:
            await self.llm_client.close()
            
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def _replace_env_vars(self, config_str: str) -> str:
        """
        Replace ${ENV_VAR} placeholders trong config string với actual values.
        
        Args:
            config_str: Config string với placeholders
            
        Returns:
            Config string với env vars replaced
        """
        
        def replace_match(match):
            env_var = match.group(1)
            value = os.getenv(env_var)
            
            if value is None:
                logger.warning(f"Environment variable {env_var} not found, keeping placeholder")
                return match.group(0)
            
            return value
        
        # Replace all ${VAR_NAME} patterns
        return re.sub(r'\$\{([A-Z_0-9]+)\}', replace_match, config_str)
    
    def _init_debators(self) -> List[Debator]:
        """Initialize debators từ config."""
        
        debators = []
        
        for i, config in enumerate(self.models_config['debators']):
            # Skip nếu disabled
            if not config.get('enabled', True):
                continue
            
            # Generate name from model if not provided
            model_name = config.get('model', f'debator_{i+1}')
            
            # Determine debator class based on role/config
            # For now, we use GenericDebator for all as we moved logic to Prompt
            debator = GenericDebator(
                name=model_name,
                model_config=config,
                llm_client=self.llm_client
            )
            
            debators.append(debator)
            
        return debators

    def _init_judge(self) -> Judge:
        """Initialize judge từ config.
        
        Note: generate_xai=False for experiments (save tokens).
        For demo UI, create Judge with generate_xai=True.
        """
        
        judge_config = self.models_config['judge']
        
        return Judge(
            model_config=judge_config,
            llm_client=self.llm_client,
            generate_xai=False  # XAI disabled for experiments, enable for demo UI
        )
    
    def debate(
        self,
        claim: str,
        evidences: List[Evidence] = None,
        model_verdict: str = None,
        model_confidence: float = None
    ) -> FinalVerdict:
        """
        Run adaptive debate cho claim với Gold Evidence.
        
        Args:
            claim: Claim cần verify
            evidences: List of gold evidences from dataset
            model_verdict: Model's initial verdict (Support/Refute/NEI)
            model_confidence: Model's confidence (0-1)
            
        Returns:
            FinalVerdict với verdict, reasoning, rounds_used, etc.
        """
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting SYNC debate for claim: {claim[:100]}...")
        logger.info(f"{'='*80}")
        
        if not evidences:
            logger.warning("No evidences provided!")
            evidences = []
        
        round_num = 1
        debate_history = []
        stop_reason = ""
        max_rounds = self.debate_config.get('max_rounds')  # None = unlimited
        stop_on_consensus = self.debate_config.get('stop_on_consensus', True)
        max_safety_rounds = self.debate_config.get('max_safety_rounds', 15)
        
        # Metrics tracking (Dec 2025)
        consensus_round = None  # Round at which consensus was first reached
        tokens_per_round = []  # Tokens used per round
        
        # Early stop tracking (Dec 24, 2025 - align with scope)
        # Rule: stop when Unanimous_r AND majority_label_r == majority_label_{r-1}
        previous_majority_label = None
        
        # Determine effective max rounds
        # Fixed mode (3/5/7): use max_rounds, ignore consensus
        # Unlimited mode (null): use max_safety_rounds, stop on consensus
        if max_rounds is None:
            effective_max = max_safety_rounds
            mode_str = f"UNLIMITED (max safety={max_safety_rounds})"
        else:
            effective_max = max_rounds
            mode_str = f"FIXED {max_rounds} rounds"
        
        logger.info(f"📋 Debate Mode: {mode_str}, stop_on_consensus={stop_on_consensus}")
        
        # Removed confirmation round mechanism (Dec 23, 2025)
        
        # Main debate loop
        while True:
            max_str = str(max_rounds) if max_rounds else "∞"
            logger.info(f"\n--- ROUND {round_num}/{max_str} (DEBATE) ---")
            
            # Get arguments from all debators
            round_arguments = []
            prev_args = debate_history[-1] if debate_history else None
            
            for debator in self.debators:
                try:
                    argument = debator.argue(
                        claim=claim,
                        evidences=evidences,
                        round_num=round_num,
                        previous_arguments=prev_args,
                        model_verdict=model_verdict,
                        model_confidence=model_confidence,
                        max_rounds=max_rounds,
                    )
                    round_arguments.append(argument)
                    
                except Exception as e:
                    logger.error(f"Error from {debator.name}: {e}")
                    round_arguments.append(DebateArgument(
                        debator_name=debator.name,
                        role=debator.role,
                        round_num=round_num,
                        verdict="NEI",
                        confidence=0.5,
                        reasoning=f"Error: {str(e)}",
                        key_points=["Error occurred"],
                        evidence_citations=[]
                    ))
            
            # Add to history
            debate_history.append(round_arguments)
            
            # Log round summary
            self._log_round_summary(round_num, round_arguments)
            
            # Check for consensus with stability (Dec 24, 2025 - align with scope)
            # Rule: stop when Unanimous_r AND (r >= 2) AND majority_label_r == majority_label_{r-1}
            has_unanimous, current_majority_label = self._check_unanimous_and_majority(round_arguments)
            
            # Check stability condition
            is_stable = (previous_majority_label is not None and 
                        current_majority_label == previous_majority_label)
            
            # Full early stop condition: unanimous + stable (r >= 2 implicit via is_stable)
            should_early_stop = has_unanimous and is_stable
            
            if has_unanimous:
                # Track first consensus round (for metrics)
                if consensus_round is None:
                    consensus_round = round_num
                
                if stop_on_consensus:
                    if should_early_stop:
                        # Full condition met: unanimous + stable
                        stop_reason = "unanimous_stable_consensus"
                        logger.info(f"\n✅ Stable consensus at round {round_num}: {current_majority_label} (same as R{round_num-1})")
                        break
                    else:
                        # Unanimous but not stable yet (R1 or label changed)
                        logger.info(f"\n📊 Unanimous at R{round_num}: {current_majority_label} (need stability confirmation)")
                else:
                    # Fixed mode: log consensus but continue
                    logger.info(f"\n📊 Consensus at round {round_num}: {current_majority_label} (continuing to round {effective_max})")
            
            # Update previous majority for next round
            previous_majority_label = current_majority_label
            
            # Check max rounds limit
            if round_num >= effective_max:
                if max_rounds is None:
                    logger.info(f"\n⚠️ Reached safety limit ({effective_max}) without consensus")
                else:
                    logger.info(f"\n✅ Completed all {effective_max} fixed rounds")
                stop_reason = "max_rounds_reached"
                break
            
            round_num += 1
        
        # Final verdict: use majority vote (align with scope - Dec 24, 2025)
        logger.info(f"\n--- FINAL VERDICT (Majority Vote) ---")
        final_verdict = self._majority_vote(debate_history, stop_reason)
        
        # Add metrics to final_verdict
        final_verdict.consensus_round = consensus_round
        
        # Log final verdict
        self._log_final_verdict(final_verdict)
        
        return final_verdict
    
    def _check_unanimous_and_majority(
        self,
        round_arguments: List[DebateArgument]
    ) -> tuple[bool, str]:
        """
        Check unanimous agreement and compute majority label.
        
        Returns:
            (is_unanimous, majority_label)
            - is_unanimous: True if 3/3 agree
            - majority_label: the majority verdict (always computed for stability tracking)
        """
        verdicts = [arg.verdict for arg in round_arguments]
        verdict_counts = Counter(verdicts)
        
        # Majority label (most common verdict)
        majority_label = verdict_counts.most_common(1)[0][0]
        
        # Check 3/3 unanimous agreement
        if len(set(verdicts)) == 1:
            logger.info(f"✅ UNANIMOUS: All {len(verdicts)} agents agree on {majority_label}")
            return True, majority_label
        
        # No unanimous - log current distribution
        logger.info(f"No unanimous yet: {dict(verdict_counts)} (majority: {majority_label})")
        return False, majority_label
    
    def _majority_vote(
        self,
        debate_history: List[List[DebateArgument]],
        stop_reason: str
    ) -> FinalVerdict:
        """
        Simple majority vote from final round (NO LLM call).
        This replaces the old Judge LLM call.
        
        Special case: If 1-1-1 (no majority), return NEI.
        
        Args:
            debate_history: All rounds of debate
            stop_reason: Why debate stopped
            
        Returns:
            FinalVerdict based on majority vote
        """
        final_round = debate_history[-1]
        total_agents = len(final_round)
        
        # Count verdicts
        verdict_counts = Counter([arg.verdict for arg in final_round])
        
        # Get majority verdict
        majority_verdict, majority_count = verdict_counts.most_common(1)[0]
        
        # Handle 1-1-1 tie (no majority) -> Return NEI
        if majority_count == 1 and len(verdict_counts) == total_agents:
            logger.info(f"\u26a0\ufe0f No majority (1-1-1 tie) - defaulting to NEI")
            majority_verdict = "NEI"
            majority_count = 0
            avg_confidence = 0.5
            reasoning_summary = f"No majority consensus after {len(debate_history)} rounds. All 3 agents disagree (1-1-1). Defaulting to NEI."
        else:
            # Normal majority case (2-1 or 3-0)
            majority_confidences = [arg.confidence for arg in final_round if arg.verdict == majority_verdict]
            avg_confidence = sum(majority_confidences) / len(majority_confidences)
            majority_reasonings = [arg.reasoning for arg in final_round if arg.verdict == majority_verdict]
            reasoning_summary = f"Majority vote ({majority_count}/{total_agents}): {majority_reasonings[0][:200]}..."
        
        # Build round_1_verdicts
        round_1_verdicts = {}
        if len(debate_history) > 0:
            for arg in debate_history[0]:
                round_1_verdicts[arg.debator_name] = {
                    "verdict": arg.verdict,
                    "confidence": arg.confidence,
                    "reasoning": arg.reasoning,
                    "role": arg.role
                }
        
        # Build all_rounds_verdicts (Dec 24, 2025 - include debate interaction fields)
        all_rounds_verdicts = []
        for round_args in debate_history:
            round_data = {}
            for arg in round_args:
                round_data[arg.debator_name] = {
                    "verdict": arg.verdict,
                    "confidence": arg.confidence,
                    "reasoning": arg.reasoning,
                    "role": arg.role,
                    # XAI fields for report visualization
                    "key_points": arg.key_points or [],
                    "disagree_with": arg.disagree_with or [],
                    "disagree_reason": arg.disagree_reason or "",
                    "agree_with": arg.agree_with or [],
                    "changed": arg.changed or False,
                    "change_reason": arg.change_reason or ""
                }
            all_rounds_verdicts.append(round_data)
        
        logger.info(f"Majority vote result: {majority_verdict} ({majority_count}/{total_agents}, conf={avg_confidence:.2f})")
        
        return FinalVerdict(
            verdict=majority_verdict,
            confidence=avg_confidence,
            reasoning=reasoning_summary,
            evidence_summary="",
            rounds_used=len(debate_history),
            debator_agreements=dict(verdict_counts),
            early_stopped=(stop_reason == "unanimous_consensus"),
            stop_reason=stop_reason,
            mvp_agent="Majority",
            best_quote_from="Majority",
            decision_path="MAJORITY_VOTE",
            round_1_verdicts=round_1_verdicts,
            all_rounds_verdicts=all_rounds_verdicts
        )

    def _create_unanimous_verdict(
        self,
        debate_history: List[List[DebateArgument]],
        stop_reason: str
    ) -> FinalVerdict:
        """Create verdict automatically without calling Judge (Cost Saving)."""
        
        final_round = debate_history[-1]
        consensus_verdict = final_round[0].verdict
        
        # Average confidence
        avg_conf = sum(arg.confidence for arg in final_round) / len(final_round)
        
        # Select best reasoning (highest confidence agent)
        best_arg = max(final_round, key=lambda x: x.confidence)
        reasoning_summary = best_arg.reasoning
        
        # Collect evidence summary
        evidence_summary = ""
        for arg in final_round:
            if arg.key_points:
                evidence_summary += f"{arg.key_points[0]} "
        
        # Agreements dict
        agreements = {consensus_verdict: len(final_round)}
        
        # Round 1 verdicts (with full details for case study)
        round_1_verdicts = {}
        if len(debate_history) > 0:
            for arg in debate_history[0]:
                round_1_verdicts[arg.debator_name] = {
                    "verdict": arg.verdict,
                    "confidence": arg.confidence,
                    "reasoning": arg.reasoning,
                    "role": arg.role
                }
        
        # All rounds verdicts for metrics visualization (Dec 24, 2025 - include XAI fields)
        all_rounds_verdicts = []
        for round_args in debate_history:
            round_data = {}
            for arg in round_args:
                round_data[arg.debator_name] = {
                    "verdict": arg.verdict,
                    "confidence": arg.confidence,
                    "reasoning": arg.reasoning,
                    "role": arg.role,
                    # XAI fields for report visualization
                    "key_points": arg.key_points or [],
                    "disagree_with": arg.disagree_with or [],
                    "disagree_reason": arg.disagree_reason or "",
                    "agree_with": arg.agree_with or [],
                    "changed": arg.changed or False,
                    "change_reason": arg.change_reason or ""
                }
            all_rounds_verdicts.append(round_data)

        return FinalVerdict(
            verdict=consensus_verdict,
            confidence=avg_conf,
            reasoning=reasoning_summary.strip(),
            evidence_summary=evidence_summary.strip(),
            rounds_used=len(debate_history),
            debator_agreements=agreements,
            early_stopped=True,
            stop_reason=stop_reason,
            mvp_agent="ALL (Consensus)",
            round_1_verdicts=round_1_verdicts,
            all_rounds_verdicts=all_rounds_verdicts
        )

    def _log_round_summary(self, round_num: int, arguments: List[DebateArgument]):
        """Log summary của round."""
        
        logger.info(f"\nRound {round_num} Summary:")
        
        verdict_counts = Counter([arg.verdict for arg in arguments])
        
        for verdict, count in verdict_counts.items():
            logger.info(f"  {verdict}: {count}/{len(arguments)}")
        
        for arg in arguments:
            logger.info(f"  {arg.debator_name}: {arg.verdict} (conf={arg.confidence:.2f})")
    
    def _log_final_verdict(self, verdict: FinalVerdict):
        """Log final verdict."""
        
        logger.info(f"\n{'='*80}")
        logger.info(f"FINAL VERDICT")
        logger.info(f"{'='*80}")
        logger.info(f"Verdict: {verdict.verdict}")
        logger.info(f"Confidence: {verdict.confidence:.2f}")
        logger.info(f"Reasoning: {verdict.reasoning}")
        logger.info(f"Rounds used: {verdict.rounds_used}")
        logger.info(f"Early stopped: {verdict.early_stopped} ({verdict.stop_reason})")
        logger.info(f"Debator agreements: {verdict.debator_agreements}")
        logger.info(f"{'='*80}\n")
    
    def batch_debate(
        self,
        claims_with_evidences: List[Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> List[FinalVerdict]:
        """
        Run debate cho multiple claims.
        
        Args:
            claims_with_evidences: List of {"claim": str, "evidences": List[Evidence]}
            save_path: Optional path to save results
            
        Returns:
            List of FinalVerdict
        """
        
        verdicts = []
        
        for i, item in enumerate(claims_with_evidences, 1):
            logger.info(f"\n\n{'#'*80}")
            logger.info(f"CLAIM {i}/{len(claims_with_evidences)}")
            logger.info(f"{'#'*80}")
            
            try:
                verdict = self.debate(
                    claim=item['claim'],
                    evidences=item['evidences']
                )
                
                verdicts.append(verdict)
                
            except Exception as e:
                logger.error(f"Error processing claim {i}: {e}")
                # Add fallback verdict
                verdicts.append(FinalVerdict(
                    verdict="NOT_ENOUGH_INFO",
                    confidence=0.5,
                    reasoning=f"Error: {str(e)}",
                    evidence_summary="",
                    rounds_used=0,
                    debator_agreements={},
                    early_stopped=False,
                    stop_reason="error"
                ))
        
        # Save results if path provided
        if save_path:
            self._save_results(verdicts, save_path)
        
        return verdicts
    
    def _save_results(self, verdicts: List[FinalVerdict], save_path: str):
        """Save verdicts to JSON file."""
        
        output = []
        
        for verdict in verdicts:
            output.append({
                "verdict": verdict.verdict,
                "confidence": verdict.confidence,
                "reasoning": verdict.reasoning,
                "evidence_summary": verdict.evidence_summary,
                "rounds_used": verdict.rounds_used,
                "debator_agreements": verdict.debator_agreements,
                "early_stopped": verdict.early_stopped,
                "stop_reason": verdict.stop_reason
            })
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✅ Results saved to: {save_path}")
    
    # ========== ASYNC DEBATE METHOD (TIER 2 OPTIMIZATION) ==========
    
    async def debate_async(
        self,
        claim: str,
        evidences: List[Evidence] = None,
        model_verdict: str = None,
        model_confidence: float = None,
        model_probs: Dict[str, float] = None,
        progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ) -> FinalVerdict:
        """
        Async version of debate() - runs debators concurrently in each round.
        Reduces debate time from ~25-30s to ~10-15s per claim.
        
        Args:
            claim: Claim cần verify
            evidences: List of gold evidences from dataset
            model_verdict: Model's initial verdict (Support/Refute/NEI)
            model_confidence: Model's confidence (0-1)
            model_probs: Model's probability distribution {Support: x, Refute: y, NEI: z}
            
        Returns:
            FinalVerdict với verdict, reasoning, rounds_used, etc. + hybrid fields
        """
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting ASYNC debate for claim: {claim[:100]}...")
        logger.info(f"{'='*80}")

        if progress_cb:
            try:
                progress_cb("DEBATE_START", {"claim_preview": claim[:120]})
            except Exception:
                pass
        
        if not evidences:
            logger.warning("No evidences provided!")
            evidences = []
        
        round_num = 1
        debate_history = []
        stop_reason = ""
        max_rounds = self.debate_config.get('max_rounds')  # None = unlimited
        stop_on_consensus = self.debate_config.get('stop_on_consensus', True)
        max_safety_rounds = self.debate_config.get('max_safety_rounds', 15)
        
        # Metrics tracking (Dec 2025)
        consensus_round = None  # Round at which consensus was first reached
        
        # Early stop tracking (Dec 24, 2025 - align with scope)
        # Rule: stop when Unanimous_r AND majority_label_r == majority_label_{r-1}
        previous_majority_label = None
        
        # Determine effective max rounds
        if max_rounds is None:
            effective_max = max_safety_rounds
            mode_str = f"UNLIMITED (max safety={max_safety_rounds})"
        else:
            effective_max = max_rounds
            mode_str = f"FIXED {max_rounds} rounds"
        
        logger.info(f"📋 Debate Mode: {mode_str}, stop_on_consensus={stop_on_consensus}")
        
        # Main debate loop
        while True:
            max_str = str(max_rounds) if max_rounds else "∞"
            logger.info(f"\n--- ROUND {round_num}/{max_str} (ASYNC) ---")

            if progress_cb:
                try:
                    progress_cb("ROUND_START", {"round": round_num, "max_rounds": max_rounds or 'unlimited'})
                except Exception:
                    pass
            
            # Get arguments from all debators CONCURRENTLY
            prev_args = debate_history[-1] if debate_history else None
            
            # Create tasks for all debators
            tasks = []
            for debator in self.debators:
                task = debator.argue_async(
                    claim=claim,
                    evidences=evidences,
                    round_num=round_num,
                    previous_arguments=prev_args,
                    model_verdict=model_verdict,
                    model_confidence=model_confidence,
                    max_rounds=max_rounds
                )
                tasks.append(task)
            
            # Run all debators concurrently
            try:
                round_arguments = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle exceptions
                clean_arguments = []
                for i, result in enumerate(round_arguments):
                    if isinstance(result, Exception):
                        logger.error(f"Error from {self.debators[i].name}: {result}")
                        clean_arguments.append(DebateArgument(
                            debator_name=self.debators[i].name,
                            role=self.debators[i].role,
                            round_num=round_num,
                            verdict="NEI",
                            confidence=0.5,
                            reasoning=f"Error: {str(result)}",
                            key_points=["Error occurred"],
                            evidence_citations=[]
                        ))
                    else:
                        clean_arguments.append(result)
                
                round_arguments = clean_arguments
                
            except Exception as e:
                logger.error(f"Fatal error in async debate round: {e}")
                round_arguments = [
                    DebateArgument(
                        debator_name=debator.name,
                        role=debator.role,
                        round_num=round_num,
                        verdict="NEI",
                        confidence=0.5,
                        reasoning=f"Fatal error: {str(e)}",
                        key_points=["Error occurred"],
                        evidence_citations=[]
                    ) for debator in self.debators
                ]
            
            # Add to history
            debate_history.append(round_arguments)
            
            # Log round summary
            self._log_round_summary(round_num, round_arguments)

            if progress_cb:
                try:
                    counts = {}
                    for arg in round_arguments:
                        v = (arg.verdict or "").upper()
                        counts[v] = counts.get(v, 0) + 1
                    progress_cb("ROUND_DONE", {"round": round_num, "vote_counts": counts})
                except Exception:
                    pass
            
            # Check for consensus with stability (Dec 24, 2025 - align with scope)
            # Rule: stop when Unanimous_r AND (r >= 2) AND majority_label_r == majority_label_{r-1}
            has_unanimous, current_majority_label = self._check_unanimous_and_majority(round_arguments)
            
            # Check stability condition
            is_stable = (previous_majority_label is not None and 
                        current_majority_label == previous_majority_label)
            
            # Full early stop condition: unanimous + stable (r >= 2 implicit via is_stable)
            should_early_stop = has_unanimous and is_stable
            
            if has_unanimous:
                # Track first consensus round (for metrics)
                if consensus_round is None:
                    consensus_round = round_num
                
                if stop_on_consensus:
                    if should_early_stop:
                        # Full condition met: unanimous + stable
                        stop_reason = "unanimous_stable_consensus"
                        logger.info(f"\n✅ Stable consensus at round {round_num}: {current_majority_label} (same as R{round_num-1})")
                        break
                    else:
                        # Unanimous but not stable yet (R1 or label changed)
                        logger.info(f"\n📊 Unanimous at R{round_num}: {current_majority_label} (need stability confirmation)")
                else:
                    # Fixed mode: log consensus but continue
                    logger.info(f"\n📊 Consensus at round {round_num}: {current_majority_label} (continuing to round {effective_max})")
            
            # Update previous majority for next round
            previous_majority_label = current_majority_label
            
            # Check max rounds limit
            if round_num >= effective_max:
                if max_rounds is None:
                    logger.info(f"\n⚠️ Reached safety limit ({effective_max}) without consensus")
                else:
                    logger.info(f"\n✅ Completed all {effective_max} fixed rounds")
                stop_reason = "max_rounds_reached"
                break
            
            round_num += 1
        
        # Final verdict: use majority vote (align with scope - Dec 24, 2025)
        logger.info(f"\n--- FINAL VERDICT (Majority Vote) ---")
        final_verdict = self._majority_vote(debate_history, stop_reason)
        
        # Add metrics to final_verdict
        final_verdict.consensus_round = consensus_round

        if progress_cb:
            try:
                progress_cb(
                    "DEBATE_DONE",
                    {
                        "verdict": final_verdict.verdict,
                        "confidence": final_verdict.confidence,
                        "rounds_used": final_verdict.rounds_used,
                        "stop_reason": final_verdict.stop_reason,
                    },
                )
            except Exception:
                pass
        
        # Apply Hybrid Strategy (DOWN Framework, 2025)
        if self.hybrid_enabled and model_probs:
            # Calculate model confidence from probs
            model_conf = max(model_probs.values()) if model_probs else (model_confidence or 0.0)
            
            # Apply hybrid decision
            hybrid_result = self.hybrid_strategy.decide(
                model_verdict=model_verdict or "NEI",
                model_probs=model_probs,
                debate_verdict=final_verdict.verdict,
                debate_confidence=final_verdict.confidence
            )
            
            # Update FinalVerdict with hybrid fields
            final_verdict.hybrid_verdict = hybrid_result.final_verdict
            final_verdict.hybrid_source = hybrid_result.decision_source
            final_verdict.hybrid_threshold = hybrid_result.threshold_used
            final_verdict.model_confidence_used = hybrid_result.model_confidence
            
            logger.info(f"\n🎯 HYBRID DECISION: {hybrid_result.final_verdict} (source: {hybrid_result.decision_source})")
            logger.info(f"   Model conf: {hybrid_result.model_confidence:.1%}, Threshold: {hybrid_result.threshold_used:.0%}")
        else:
            # Hybrid disabled or no model_probs - use debate verdict
            final_verdict.hybrid_verdict = final_verdict.verdict
            final_verdict.hybrid_source = "DEBATE"
            final_verdict.hybrid_threshold = self.hybrid_threshold
            final_verdict.model_confidence_used = model_confidence
        
        # Log final verdict
        self._log_final_verdict(final_verdict)
        
        # NOTE: Do NOT close sessions here in pipeline mode! 
        # It breaks concurrent debates sharing the same LLMClient.
        # await self._cleanup_async_sessions()
        
        return final_verdict
    
    async def _cleanup_async_sessions(self):
        """Close all aiohttp sessions from debators and judge"""
        try:
            # Close debator LLM client sessions
            for debator in self.debators:
                if hasattr(debator, 'llm_client') and hasattr(debator.llm_client, 'close_session'):
                    await debator.llm_client.close_session()
            
            # Close judge LLM client session
            if hasattr(self.judge, 'llm_client') and hasattr(self.judge.llm_client, 'close_session'):
                await self.judge.llm_client.close_session()
        except Exception as e:
            logger.warning(f"Error during async session cleanup: {e}")
