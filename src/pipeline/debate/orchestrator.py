"""
Simplified 2-Round Debate Orchestrator.

Flow:
- Round 1: Independent reasoning (đọc & tự phán)
- Round 2: Debate + final commit (tranh luận & chốt kèo)
- Judge: Called ONCE after Round 2 to make final decision

Author: Lockdown
Date: Dec 01, 2025 (Simplified from 3-round system)
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
from .xai_debate import DebateXAI, generate_debate_xai

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
        """Initialize judge từ config."""
        
        judge_config = self.models_config['judge']
        
        return Judge(
            model_config=judge_config,
            llm_client=self.llm_client
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
        # NOTE: Judge no longer provides mid-debate guidance in 2-round system
        
        while round_num <= self.debate_config['max_rounds']:
            logger.info(f"\n--- ROUND {round_num} ---")
            
            # Get arguments từ tất cả debators
            round_arguments = []
            
            for i, debator in enumerate(self.debators):
                try:
                    # Pass previous round arguments nếu có
                    prev_args = debate_history[-1] if debate_history else None
                    
                    argument = debator.argue(
                        claim=claim,
                        evidences=evidences,
                        round_num=round_num,
                        previous_arguments=prev_args,
                        model_verdict=model_verdict,
                        model_confidence=model_confidence
                    )
                    
                    round_arguments.append(argument)
                    
                    # No delay needed - OpenRouter paid tier handles concurrent requests
                    # Async method runs all debators in parallel without delays
                    
                except Exception as e:
                    logger.error(f"Error from {debator.name}: {e}")
                    # Add fallback argument
                    round_arguments.append(DebateArgument(
                        debator_name=debator.name,
                        role=debator.role,
                        round_num=round_num,
                        verdict="NOT_ENOUGH_INFO",
                        confidence=0.5,
                        reasoning=f"Error: {str(e)}",
                        key_points=["Error occurred"],
                        evidence_citations=[]
                    ))
            
            # Add to history
            debate_history.append(round_arguments)
            
            # Log round summary
            self._log_round_summary(round_num, round_arguments)

            if progress_cb:
                try:
                    counts: Dict[str, int] = {}
                    for a in round_arguments:
                        v = (a.verdict or "").upper()
                        counts[v] = counts.get(v, 0) + 1
                    progress_cb(
                        "ROUND_DONE",
                        {
                            "round": round_num,
                            "vote_counts": counts,
                        },
                    )
                except Exception:
                    pass
            
            # Check early stopping conditions
            should_stop, stop_reason = self._check_early_stop(
                round_arguments, 
                debate_history, 
                round_num
            )
            
            if should_stop:
                logger.info(f"\n✅ Early stop triggered: {stop_reason}")
                # For Evaluation on ViFactCheck: ALWAYS proceed to Judge
                # We break the debate loop, but let the flow continue to Judge deliberation
                break
            
            if round_num >= self.debate_config['max_rounds']:
                logger.info(f"\n⚠️ Reached max rounds ({self.debate_config['max_rounds']})")
                stop_reason = "max_rounds_reached"
                break
            
            # No delay needed between rounds for paid tier
            
            round_num += 1
        
        # Judge makes final verdict (only when no unanimous consensus)
        logger.info(f"\n--- JUDGE DELIBERATION ---")
        
        # Log model verdict if provided
        if model_verdict and model_confidence:
            logger.info(f"Model verdict: {model_verdict} (confidence: {model_confidence:.2f})")
        
        # No delay needed before judge
        
        final_verdict = self.judge.decide(
            claim=claim,
            debate_history=debate_history,
            early_stopped=(stop_reason != "max_rounds_reached"),
            stop_reason=stop_reason,
            model_verdict=model_verdict,
            model_confidence=model_confidence,
            evidences=evidences
        )
        
        # Log final verdict
        self._log_final_verdict(final_verdict)
        
        return final_verdict
    
    def _check_early_stop(
        self,
        round_arguments: List[DebateArgument],
        debate_history: List[List[DebateArgument]],
        round_num: int
    ) -> tuple[bool, str]:
        """
        Check early stopping conditions.
        FORCE DISABLED for Research/Reporting purposes (Paper Mode).
        We want to run full rounds to measure convergence and stability.
        """
        
        # RESEARCH MODE: Always return False to force full debate rounds
        return False, "research_mode_force_full_rounds"

        # ORIGINAL LOGIC (Commented out for now)
        # # 1. UNANIMOUS AGREEMENT (3/3 agree)
        # verdicts = [arg.verdict for arg in round_arguments]
        # if len(set(verdicts)) == 1:
        #     return True, "unanimous_agreement"
            
        # # 2. STABLE MAJORITY
        # if len(debate_history) >= 2:
        #     # ... (logic omitted)
        #     pass

        # return False, ""

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
        
        # All rounds verdicts for metrics visualization
        all_rounds_verdicts = []
        for round_args in debate_history:
            round_data = {}
            for arg in round_args:
                round_data[arg.debator_name] = {
                    "verdict": arg.verdict,
                    "confidence": arg.confidence,
                    "reasoning": arg.reasoning,
                    "role": arg.role
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
        # NOTE: Judge no longer provides mid-debate guidance in 2-round system
        
        # Main debate loop
        while round_num <= self.debate_config['max_rounds']:
            logger.info(f"\n--- ROUND {round_num} (ASYNC) ---")

            if progress_cb:
                try:
                    progress_cb("ROUND_START", {"round": round_num})
                except Exception:
                    pass
            
            # Get arguments từ tất cả debators CONCURRENTLY
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
                    model_confidence=model_confidence
                    # NOTE: judge_reasoning removed - no mid-debate guidance in 2-round system
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
                        # Add fallback argument
                        clean_arguments.append(DebateArgument(
                            debator_name=self.debators[i].name,
                            role=self.debators[i].role,
                            round_num=round_num,
                            verdict="NOT_ENOUGH_INFO",
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
                # Fallback to all NOT_ENOUGH_INFO
                round_arguments = [
                    DebateArgument(
                        debator_name=debator.name,
                        role=debator.role,
                        round_num=round_num,
                        verdict="NOT_ENOUGH_INFO",
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
            
            # Check early stopping conditions
            should_stop, stop_reason = self._check_early_stop(
                round_arguments, 
                debate_history, 
                round_num
            )
            
            if should_stop:
                logger.info(f"\n✅ Early stop triggered: {stop_reason}")
                # For Evaluation: ALWAYS proceed to Judge
                break
            
            if round_num >= self.debate_config['max_rounds']:
                logger.info(f"\n⚠️ Reached max rounds ({self.debate_config['max_rounds']})")
                stop_reason = "max_rounds_reached"
                break
            
            # NOTE: No mid-debate Judge guidance in 2-round system
            # Judge only runs ONCE after Round 2
            
            round_num += 1
        
        # Judge makes final verdict (only when no unanimous consensus)
        logger.info(f"\n--- JUDGE DELIBERATION (ASYNC) ---")

        if progress_cb:
            try:
                progress_cb("JUDGE_START", {})
            except Exception:
                pass
        
        # Log model verdict if provided
        if model_verdict and model_confidence:
            logger.info(f"Model verdict: {model_verdict} (confidence: {model_confidence:.2f})")
        
        # Judge makes final decision based on R1 + R2 outputs
        final_verdict = await self.judge.decide_async(
            claim=claim,
            debate_history=debate_history,
            early_stopped=(stop_reason != "max_rounds_reached"),
            stop_reason=stop_reason,
            model_verdict=model_verdict,
            model_confidence=model_confidence,
            evidences=evidences
        )

        if progress_cb:
            try:
                progress_cb(
                    "JUDGE_DONE",
                    {
                        "verdict": getattr(final_verdict, "verdict", None),
                        "confidence": getattr(final_verdict, "confidence", None),
                        "rounds_used": getattr(final_verdict, "rounds_used", None),
                        "stop_reason": getattr(final_verdict, "stop_reason", None),
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
        
        # Generate XAI (Dec 2025)
        evidence_text = " ".join([e.text for e in evidences]) if evidences else ""
        xai_dict = generate_debate_xai(
            claim=claim,
            evidence=evidence_text,
            final_verdict=final_verdict,
            debate_history=debate_history
        )
        # Store XAI in final_verdict for downstream use
        final_verdict.xai_dict = xai_dict
        
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
