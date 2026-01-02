"""
Judge v2.0 - Real-time debate moderator with per-round evaluation.

Thiết kế mới (Dec 31, 2025):
- Judge được gọi sau MỖI round để đánh giá và quyết định
- Judge đưa ra reasoning ngắn cho mỗi round
- Judge quyết định early stop hay continue
- Judge đưa ra final verdict với explanation chi tiết

Author: Lockdown
Date: Dec 31, 2025
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import logging
import json
import re
import ast
import hashlib
import os
import random
import threading

try:
    from .debator import DebateArgument, Evidence
except ImportError:
    from debator import DebateArgument, Evidence

logger = logging.getLogger(__name__)


_PROMPT_LOG_LOCK = threading.Lock()
_PROMPT_LOGGED_ONCE = False


def _should_log_prompt() -> bool:
    """Sampling-based prompt diagnostics to avoid log spam in batch runs."""
    # Env toggles (keep simple; no new CLI flags needed)
    sample_rate_raw = os.getenv("JUDGE_PROMPT_LOG_SAMPLE_RATE", "0.01")
    first_only_raw = os.getenv("JUDGE_PROMPT_LOG_FIRST_ONLY", "0")

    try:
        sample_rate = float(sample_rate_raw)
    except Exception:
        sample_rate = 0.01
    sample_rate = max(0.0, min(1.0, sample_rate))

    first_only = str(first_only_raw).strip().lower() in {"1", "true", "yes"}
    if first_only:
        global _PROMPT_LOGGED_ONCE
        with _PROMPT_LOG_LOCK:
            if _PROMPT_LOGGED_ONCE:
                return False
            _PROMPT_LOGGED_ONCE = True
            return True

    if sample_rate <= 0.0:
        return False
    return random.random() < sample_rate


def _sanitize_reasoning_vi(text: str, max_words: int = 18) -> str:
    """Make judge_reasoning short, natural Vietnamese and avoid rule-like phrasing."""
    if not isinstance(text, str):
        return ""

    t = " ".join(text.replace("\n", " ").split()).strip()
    if not t:
        return ""

    # Remove mechanical/rule-like fragments (keep it lightweight)
    banned_phrases = [
        "quy tắc",
        "the rule",
        "protocol",
        "cần ít nhất",
        "ít nhất 2",
        "round 1",
        "round 2",
        "round",
        "kiểm tra tính ổn định",
        "theo quy định",
    ]
    lower = t.lower()
    if any(p in lower for p in banned_phrases):
        # Replace with a neutral natural fallback
        t = "Chưa đủ chắc để chốt, cần thêm phản biện."

    # Trim to max words
    words = t.split()
    if len(words) > max_words:
        t = " ".join(words[:max_words]).rstrip(" ,;:.-") + "."

    return t


@dataclass
class RoundEvaluation:
    """Judge evaluation sau mỗi round."""
    round_num: int
    consensus_status: str  # "UNANIMOUS" | "MAJORITY" | "SPLIT"
    majority_verdict: str  # Verdict của majority
    verdict_distribution: Dict[str, int]  # {verdict: count}
    judge_reasoning: str  # Reasoning ngắn của judge
    early_stop_decision: bool  # True = dừng, False = tiếp tục
    stability_check: bool  # True nếu verdict giống round trước
    confidence: float  # Locked to 0.5 (avoid subjective LLM confidence)


@dataclass
class FinalVerdict:
    """Final verdict từ Judge sau khi debate kết thúc."""
    verdict: str  # SUPPORTS, REFUTES, NOT_ENOUGH_INFO
    confidence: float  # Locked to 0.5 (avoid subjective LLM confidence)
    reasoning: str  # Detailed reasoning
    evidence_summary: str
    rounds_used: int
    debator_agreements: Dict[str, int]  # verdict -> count
    early_stopped: bool
    stop_reason: str
    decision_path: str  # CONSENSUS | MAJORITY | SPLIT_DECISION
    round_evaluations: List[RoundEvaluation]  # All round evaluations
    round_1_verdicts: Dict[str, Dict[str, Any]] = None
    all_rounds_verdicts: List[Dict[str, Dict[str, Any]]] = None
    # XAI fields
    xai_conflict_claim: str = None
    xai_conflict_evidence: str = None
    xai_natural_explanation: str = None


class JudgeV2:
    """
    Judge v2.0 - Real-time debate moderator.
    
    Workflow:
    1. evaluate_round() - Gọi sau mỗi round debate
    2. make_final_decision() - Gọi khi debate kết thúc
    """
    
    # Agent name mapping for readable output
    AGENT_NAME_MAP = {
        "Truth Seeker A": "Grok",
        "Truth Seeker B": "Gemini", 
        "Truth Seeker C": "GPT",
    }
    
    def __init__(self, model_config: Dict[str, Any], llm_client, generate_xai: bool = False):
        """
        Args:
            model_config: Config cho judge model
            llm_client: LLM client để gọi API
            generate_xai: If True, include XAI in final prompt
        """
        self.config = model_config
        self.llm_client = llm_client
        self.generate_xai = generate_xai
        self.round_evaluations: List[RoundEvaluation] = []
    
    def evaluate_round(
        self,
        claim: str,
        evidence_list: List[Evidence],
        round_num: int,
        round_arguments: List[DebateArgument],
        previous_majority_verdict: Optional[str] = None,
        max_rounds: int = 3
    ) -> RoundEvaluation:
        """
        Đánh giá một round debate và quyết định early stop.
        
        Args:
            claim: Claim cần kiểm tra
            evidence_list: List evidences
            round_num: Round hiện tại (1, 2, 3...)
            round_arguments: Arguments của các agents trong round này
            previous_majority_verdict: Majority verdict của round trước (for stability)
            max_rounds: Max rounds allowed
            
        Returns:
            RoundEvaluation với judge reasoning + early stop decision
        """
        
        # Analyze current round
        verdicts = [arg.verdict for arg in round_arguments]
        verdict_counts = {}
        for v in verdicts:
            verdict_counts[v] = verdict_counts.get(v, 0) + 1
        
        # Determine consensus status
        unique_verdicts = len(set(verdicts))
        if unique_verdicts == 1:
            consensus_status = "UNANIMOUS" 
            majority_verdict = verdicts[0]
        elif len(verdict_counts) == 2:
            consensus_status = "MAJORITY"
            majority_verdict = max(verdict_counts.items(), key=lambda x: x[1])[0]
        else:
            consensus_status = "SPLIT" 
            majority_verdict = max(verdict_counts.items(), key=lambda x: x[1])[0]
        
        # Check stability (majority verdict same as previous round)
        stability_check = (previous_majority_verdict is not None and 
                          majority_verdict == previous_majority_verdict)
        
        # Build prompt for judge reasoning
        prompt = self._build_round_evaluation_prompt(
            claim=claim,
            evidence_list=evidence_list,
            round_num=round_num,
            round_arguments=round_arguments,
            consensus_status=consensus_status,
            majority_verdict=majority_verdict,
            verdict_counts=verdict_counts,
            stability_check=stability_check,
            max_rounds=max_rounds
        )
        
        try:
            if _should_log_prompt():
                prompt_chars = len(prompt)
                prompt_sha1 = hashlib.sha1(prompt.encode('utf-8', errors='ignore')).hexdigest()[:10]
                prompt_tok_est = max(1, prompt_chars // 4)
                logger.warning(
                    f"[JudgeV2][R{round_num}] prompt chars={prompt_chars}, tok_est~{prompt_tok_est}, sha1={prompt_sha1}"
                )
                logger.warning(
                    f"[JudgeV2][R{round_num}] prompt head: {prompt[:200].replace(chr(10), ' ')}"
                )
                logger.warning(
                    f"[JudgeV2][R{round_num}] prompt tail: {prompt[-200:].replace(chr(10), ' ')}"
                )

            # Call judge LLM
            response = self.llm_client.generate(
                model=self.config['model'],
                prompt=prompt,
                api_key=self.config['api_key'],
                base_url=self.config['base_url'],
                temperature=self.config.get('temperature', 0.0),
                max_tokens=self.config.get('max_tokens', 400)  # Shorter for per-round
            )
            
            # Parse response
            evaluation = self._parse_round_evaluation(
                response=response,
                round_num=round_num,
                consensus_status=consensus_status,
                majority_verdict=majority_verdict,
                verdict_counts=verdict_counts,
                stability_check=stability_check
            )
            
            logger.info(f"Judge Round {round_num}: {evaluation.judge_reasoning}")
            logger.info(f"Early stop decision: {evaluation.early_stop_decision}")
            
            # Store evaluation
            self.round_evaluations.append(evaluation)
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Judge round evaluation error: {e}")
            
            # Fallback: rule-based early stop
            early_stop = (consensus_status == "UNANIMOUS" and 
                         stability_check and 
                         round_num >= 2)
            
            evaluation = RoundEvaluation(
                round_num=round_num,
                consensus_status=consensus_status,
                majority_verdict=majority_verdict,
                verdict_distribution=verdict_counts,
                judge_reasoning=f"Fallback: {consensus_status} consensus on {majority_verdict}",
                early_stop_decision=early_stop,
                stability_check=stability_check,
                confidence=0.5
            )
            
            self.round_evaluations.append(evaluation)
            return evaluation
    
    def _build_round_evaluation_prompt(
        self,
        claim: str,
        evidence_list: List[Evidence],
        round_num: int,
        round_arguments: List[DebateArgument],
        consensus_status: str,
        majority_verdict: str,
        verdict_counts: Dict[str, int],
        stability_check: bool,
        max_rounds: int
    ) -> str:
        """Build prompt for per-round evaluation."""
        
        prompt = f"""You are the AGGREGATION JUDGE for a multi-agent debate.

Your role is ONLY to apply the provided voting and early-stop rules to the given votes.
Do NOT fact-check the claim. Do NOT introduce new arguments. Do NOT override the rules.

**SESSION INFO:**
- Current Round: {round_num}/{max_rounds}
- Number of Experts: {len(round_arguments)}
"""

        prompt += f"\n\n**ROUND VOTES:**"
        for arg in round_arguments:
            agent_name = self.AGENT_NAME_MAP.get(arg.role, arg.debator_name)
            prompt += f"\n• {agent_name}: {arg.verdict}"
        
        # Show vote distribution and consensus status
        vote_summary = ", ".join([f"{v}: {c}" for v, c in verdict_counts.items()])
        prompt += f"\n\n**VOTE DISTRIBUTION:** {vote_summary}"
        prompt += f"\n**CONSENSUS STATUS:** {consensus_status}"
        
        if round_num > 1:
            prompt += f"\n**STABILITY:** {'Stable' if stability_check else 'Unstable'} (same majority verdict as previous round)"
        
        prompt += f"""

**YOUR TASKS:**
1. **Decide early stopping:** Continue debate OR stop with current results  
2. **Provide brief reasoning** (1 short sentence in Vietnamese)

**EARLY STOP RULES:**
- **STOP CONDITION**: UNANIMOUS (3/3 agree) AND STABLE (same majority verdict as previous round) AND round >= 2
- **CONTINUE CONDITIONS**: 
  - No unanimous consensus (2-1 or 1-1-1 split)
  - OR unanimous but unstable (majority verdict changed from previous round)
  - OR round 1 (need at least 2 rounds for stability check)
- **FORCE STOP**: Always stop if reached max round {max_rounds}

**CURRENT SITUATION:**
- Round: {round_num}/{max_rounds}
- Consensus: {consensus_status}
- Stability: {'Stable' if stability_check else 'Unstable'}
**OUTPUT JSON:**
{{
    "early_stop_decision": true/false,
    "judge_reasoning": "One short natural Vietnamese sentence (max 18 words)"
}}

**REASONING STYLE RULES (VERY IMPORTANT):**
- Do NOT mention rules, stability checks, or minimum rounds.
- Do NOT say "the rule says" / "the protocol says" / "need at least 2 rounds".
- Keep it natural Vietnamese, short, and factual.
- Only refer to what you observe (consensus / disagreement / strong conflict / unclear evidence).

**GOOD EXAMPLES (Vietnamese, natural & short):**
- "Các chuyên gia chưa thống nhất, cần thêm phản biện để rõ hơn."
- "Đồng thuận đã ổn định, có thể chốt kết luận."
- "Vẫn còn ý kiến trái chiều, chưa đủ chắc để dừng."
- "Tranh luận đã đủ vòng, chốt kết luận để tránh kéo dài."

"""
        
        return prompt
    
    def _parse_round_evaluation(
        self,
        response: str,
        round_num: int,
        consensus_status: str,
        majority_verdict: str,
        verdict_counts: Dict[str, int],
        stability_check: bool
    ) -> RoundEvaluation:
        """Parse judge response for round evaluation."""
        
        # Clean and extract JSON
        cleaned = re.sub(r'```(?:json)?', '', response, flags=re.IGNORECASE).strip()
        
        try:
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                raise ValueError("No JSON found")
                
            raw_early_stop = data.get("early_stop_decision", False)
            if isinstance(raw_early_stop, bool):
                early_stop = raw_early_stop
            elif isinstance(raw_early_stop, str):
                early_stop = raw_early_stop.strip().lower() in {"true", "1", "yes"}
            else:
                early_stop = False
            reasoning = _sanitize_reasoning_vi(data.get("judge_reasoning", ""))
            
        except Exception as e:
            logger.warning(f"Judge parsing error: {e}, using fallback")
            
            # Fallback rule-based decision
            early_stop = (consensus_status == "UNANIMOUS" and stability_check and round_num >= 2)
            if early_stop:
                reasoning = "Đồng thuận đã ổn định, có thể chốt kết luận."
            else:
                reasoning = "Chưa đủ chắc để dừng, tiếp tục phản biện."
        
        return RoundEvaluation(
            round_num=round_num,
            consensus_status=consensus_status,
            majority_verdict=majority_verdict,
            verdict_distribution=verdict_counts,
            judge_reasoning=reasoning,
            early_stop_decision=early_stop,
            stability_check=stability_check,
            confidence=0.5
        )
    
    def make_final_decision(
        self,
        claim: str,
        evidence_list: List[Evidence],
        debate_history: List[List[DebateArgument]],
        stop_reason: str
    ) -> FinalVerdict:
        """
        Đưa ra final verdict sau khi debate kết thúc.
        
        Args:
            claim: Original claim
            evidence_list: List evidences
            debate_history: All rounds of debate
            stop_reason: Why debate stopped
            
        Returns:
            FinalVerdict with detailed reasoning
        """
        
        if not debate_history:
            logger.error("No debate history provided")
            return self._fallback_final_verdict(claim, stop_reason)
        
        # Final round analysis
        final_round = debate_history[-1]
        final_verdicts = [arg.verdict for arg in final_round]
        final_counts = {}
        for v in final_verdicts:
            final_counts[v] = final_counts.get(v, 0) + 1
        
        # Determine decision path
        if len(set(final_verdicts)) == 1:
            decision_path = "CONSENSUS"
        elif len(final_counts) == len(final_round):  # 1-1-1 split
            decision_path = "SPLIT_DECISION"
        else:
            decision_path = "MAJORITY"
        
        # Build final prompt
        prompt = self._build_final_decision_prompt(
            claim=claim,
            evidence_list=evidence_list,
            debate_history=debate_history,
            round_evaluations=self.round_evaluations,
            final_counts=final_counts,
            decision_path=decision_path,
            stop_reason=stop_reason
        )
        
        try:
            if _should_log_prompt():
                prompt_chars = len(prompt)
                prompt_sha1 = hashlib.sha1(prompt.encode('utf-8', errors='ignore')).hexdigest()[:10]
                prompt_tok_est = max(1, prompt_chars // 4)
                logger.warning(
                    f"[JudgeV2][FINAL] prompt chars={prompt_chars}, tok_est~{prompt_tok_est}, sha1={prompt_sha1}"
                )
                logger.warning(
                    f"[JudgeV2][FINAL] prompt head: {prompt[:200].replace(chr(10), ' ')}"
                )
                logger.warning(
                    f"[JudgeV2][FINAL] prompt tail: {prompt[-200:].replace(chr(10), ' ')}"
                )

            # Call judge LLM for final decision
            response = self.llm_client.generate(
                model=self.config['model'],
                prompt=prompt,
                api_key=self.config['api_key'],
                base_url=self.config['base_url'],
                temperature=self.config.get('temperature', 0.0),
                max_tokens=self.config.get('max_tokens', 1200)
            )
            
            # Parse final verdict
            final_verdict = self._parse_final_decision(
                response=response,
                claim=claim,
                evidence_list=evidence_list,
                debate_history=debate_history,
                final_counts=final_counts,
                decision_path=decision_path,
                stop_reason=stop_reason
            )
            
            logger.info(f"Judge final verdict: {final_verdict.verdict}")
            
            return final_verdict
            
        except Exception as e:
            logger.error(f"Judge final decision error: {e}")
            return self._fallback_final_verdict(claim, stop_reason, final_counts, decision_path)
    
    def _build_final_decision_prompt(
        self,
        claim: str,
        evidence_list: List[Evidence],
        debate_history: List[List[DebateArgument]],
        round_evaluations: List[RoundEvaluation],
        final_counts: Dict[str, int],
        decision_path: str,
        stop_reason: str
    ) -> str:
        """Build prompt for final decision."""
        
        prompt = f"""You are the AGGREGATION JUDGE making the FINAL VERDICT for this multi-agent debate.

Your role is ONLY to apply the provided verdict rules to the final vote counts.
Do NOT fact-check the claim. Do NOT use external knowledge. Do NOT change the rules."""
        
        # Skip round-by-round progress to save tokens
        
        # Final round ultra minimal
        prompt += f"\n\n**FINAL VOTES:**"
        final_round = debate_history[-1]
        for arg in final_round:
            agent_name = self.AGENT_NAME_MAP.get(arg.role, arg.debator_name)
            # Just verdict, no reasoning or quotes
            prompt += f"\n• {agent_name}: {arg.verdict}"
        
        # Final vote summary
        vote_summary = ", ".join([f"{v}: {c}" for v, c in final_counts.items()])
        prompt += f"\n\n**FINAL VOTE COUNT:** {vote_summary}"
        
        # Decision rules in English
        prompt += f"""

**VERDICT RULES:**
1. **CONSENSUS (3/3):** Follow consensus
2. **MAJORITY (2/1):** Follow majority verdict
3. **SPLIT (1-1-1):** Return NEI for safety

**YOUR TASK:**
Make the final verdict with detailed reasoning in Vietnamese."""

        # Output format (keep minimal to reduce noise/tokens)
        prompt += """

**OUTPUT JSON:**
{
    "verdict": "SUPPORTED | REFUTED | NEI", 
    "reasoning": "Detailed 2-3 sentence reasoning in Vietnamese"
}"""
        
        return prompt
    
    def _parse_final_decision(
        self,
        response: str,
        claim: str,
        evidence_list: List[Evidence],
        debate_history: List[List[DebateArgument]],
        final_counts: Dict[str, int],
        decision_path: str,
        stop_reason: str
    ) -> FinalVerdict:
        """Parse judge final decision."""
        
        # Clean and extract JSON
        cleaned = re.sub(r'```(?:json)?', '', response, flags=re.IGNORECASE).strip()
        
        try:
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                raise ValueError("No JSON found")
            
            verdict = data.get("verdict", "NEI").upper()
            confidence = 0.5
            reasoning = data.get("reasoning", "No reasoning provided")
            
            # Extract XAI if present
            xai_data = data.get("xai", {})
            xai_conflict_claim = xai_data.get("conflict_claim", "") if isinstance(xai_data, dict) else ""
            xai_conflict_evidence = xai_data.get("conflict_evidence", "") if isinstance(xai_data, dict) else ""
            xai_natural_explanation = xai_data.get("natural_explanation_vi", "") if isinstance(xai_data, dict) else ""
            
        except Exception as e:
            logger.warning(f"Final decision parsing error: {e}, using fallback")
            
            # Fallback to majority vote (align with old system: 1-1-1 => NEI)
            if final_counts:
                total_votes = sum(final_counts.values())
                majority_verdict, majority_count = max(final_counts.items(), key=lambda x: x[1])
                # If every label appears exactly once => no majority
                if majority_count == 1 and len(final_counts) == total_votes:
                    verdict = "NEI"
                    confidence = 0.5
                    reasoning = "Fallback: No majority (1-1-1), defaulting to NEI"
                else:
                    verdict = majority_verdict
                    confidence = 0.5
                    reasoning = f"Fallback majority vote: {verdict}"
            else:
                verdict = "NEI"
                confidence = 0.5
                reasoning = "Fallback: No valid votes"
            
            xai_conflict_claim = ""
            xai_conflict_evidence = ""
            xai_natural_explanation = ""
        
        # Build structured data
        round_1_verdicts = {}
        if len(debate_history) > 0:
            for arg in debate_history[0]:
                round_1_verdicts[arg.debator_name] = {
                    "verdict": arg.verdict,
                    "confidence": arg.confidence,
                    "reasoning": arg.reasoning,
                    "role": arg.role
                }
        
        # Extract all rounds verdicts
        all_rounds_verdicts = []
        for round_args in debate_history:
            round_data = {}
            for arg in round_args:
                round_data[arg.debator_name] = {
                    "verdict": arg.verdict,
                    "confidence": arg.confidence,
                    "reasoning": arg.reasoning,
                    "key_points": arg.key_points or [],
                    "agree_with": getattr(arg, 'agree_with', []),
                    "disagree_with": getattr(arg, 'disagree_with', []),
                    "changed": getattr(arg, 'changed', False)
                }
            all_rounds_verdicts.append(round_data)
        
        return FinalVerdict(
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning,
            evidence_summary="Judge decision based on debate analysis",
            rounds_used=len(debate_history),
            debator_agreements=final_counts,
            early_stopped=(stop_reason != "max_rounds_reached"),
            stop_reason=stop_reason,
            decision_path=decision_path,
            round_evaluations=self.round_evaluations,
            round_1_verdicts=round_1_verdicts,
            all_rounds_verdicts=all_rounds_verdicts,
            xai_conflict_claim=xai_conflict_claim,
            xai_conflict_evidence=xai_conflict_evidence,
            xai_natural_explanation=xai_natural_explanation
        )
    
    def _fallback_final_verdict(
        self,
        claim: str,
        stop_reason: str,
        final_counts: Dict[str, int] = None,
        decision_path: str = "FALLBACK"
    ) -> FinalVerdict:
        """Fallback when judge fails."""
        
        if final_counts:
            verdict = max(final_counts.items(), key=lambda x: x[1])[0]
            confidence = 0.6
            reasoning = f"Fallback majority vote: {verdict}"
        else:
            verdict = "NEI"
            confidence = 0.5
            reasoning = "Fallback: Judge error, defaulting to NEI"
            final_counts = {"NEI": 1}
        
        return FinalVerdict(
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning,
            evidence_summary="Fallback decision due to judge error",
            rounds_used=len(self.round_evaluations),
            debator_agreements=final_counts,
            early_stopped=False,
            stop_reason=stop_reason,
            decision_path=decision_path,
            round_evaluations=self.round_evaluations,
            round_1_verdicts={},
            all_rounds_verdicts=[]
        )

    # Async versions for compatibility
    async def evaluate_round_async(self, *args, **kwargs) -> RoundEvaluation:
        """Async version of evaluate_round.""" 
        # For now, just call sync version
        # Can be properly implemented later if needed
        return self.evaluate_round(*args, **kwargs)
    
    async def make_final_decision_async(self, *args, **kwargs) -> FinalVerdict:
        """Async version of make_final_decision."""
        # For now, just call sync version  
        # Can be properly implemented later if needed
        return self.make_final_decision(*args, **kwargs)
