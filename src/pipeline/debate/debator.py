"""
Debator - Base class và implementation cho debate agents.

Author: Lockdown
Date: Nov 10, 2025
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import logging
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class Evidence:
    """Evidence data structure."""
    text: str
    source: str
    rank: int
    nli_score: Dict[str, float]  # {entailment, neutral, contradiction, other}
    relevance_score: float = 0.0  # Binary relevance score (0-1) from reranker
    evidence_type: str = "UNKNOWN"  # DIRECT, NUANCE, CONTEXT


@dataclass
class DebateArgument:
    """Argument từ một debator trong một round."""
    debator_name: str
    role: str  # Added role field for Judge awareness
    round_num: int
    verdict: str  # SUPPORTS, REFUTES, NOT_ENOUGH_INFO
    confidence: float  # 0.0 - 1.0
    reasoning: str
    key_points: List[str]
    evidence_citations: List[int]  # Indices of evidence used
    # Round 2 interaction fields (XAI)
    agree_with: Optional[List[str]] = None  # List of agent names agreed with
    agree_reason: Optional[str] = None
    disagree_with: Optional[List[str]] = None  # List of agent names disagreed with
    disagree_reason: Optional[str] = None
    changed: bool = False  # Whether verdict changed from R1
    change_reason: Optional[str] = None  # NEW_QUOTE, MISREAD, CONVINCED_BY_COLLEAGUE, NO_CHANGE


class Debator(ABC):
    """
    Base class cho debate agents.
    Mỗi debator phân tích evidence và argue for một verdict.
    """
    
    def __init__(self, name: str, model_config: Dict[str, Any], llm_client):
        """
        Args:
            name: Tên debator (e.g., "deepseek-v3")
            model_config: Config cho model (API key, temperature, etc.)
            llm_client: LLM client để gọi API
        """
        self.name = name
        self.config = model_config
        self.llm_client = llm_client
        self.role = model_config.get('role', 'General Public')  # Default if not specified

    @abstractmethod
    def argue(
        self,
        claim: str,
        evidences: List[Evidence],
        round_num: int,
        previous_arguments: Optional[List[DebateArgument]] = None,
        model_verdict: str = None,
        model_confidence: float = None,
        judge_verdict: str = None,
        judge_confidence: float = None,
        judge_reasoning: str = None
    ) -> DebateArgument:
        """
        Tạo argument cho claim based on evidences (Gold Evidence from dataset).
        
        Args:
            claim: Claim cần verify
            evidences: List of gold evidences from dataset
            round_num: Round hiện tại (1, 2, 3)
            previous_arguments: Arguments từ round trước (for rebuttal)
            model_verdict: Initial model verdict (Support/Refute/NEI)
            model_confidence: Model confidence (0-1)
            
        Returns:
            DebateArgument với verdict, confidence, reasoning
        """
        pass
    
    # Mapping from role config to display name
    ROLE_NAME_MAP = {
        "Truth Seeker A": "Grok",
        "Truth Seeker B": "Gemini",
        "Truth Seeker C": "GPT",
    }
    
    def _get_agent_name(self) -> str:
        """Get agent display name (Grok/Gemini/GPT)."""
        return self.ROLE_NAME_MAP.get(self.role, self.name)
    
    def _verdict_to_vietnamese(self, verdict: str) -> str:
        """Convert verdict to Vietnamese for XAI display."""
        verdict_map = {
            "SUPPORTED": "ĐỒNG TÌNH",
            "REFUTED": "BÁC BỎ",
            "NEI": "CHƯA ĐỦ THÔNG TIN",
            "NOT_ENOUGH_INFO": "CHƯA ĐỦ THÔNG TIN",
        }
        return verdict_map.get(verdict.upper(), verdict)

    def _clean_segmentation(self, text: str) -> str:
        """Remove word segmentation underscores from Vietnamese text."""
        if text:
            return text.replace("_", " ").replace("  ", " ")
        return text
    
    def _build_prompt(
        self,
        claim: str,
        evidences: List[Evidence],
        round_num: int,
        previous_arguments: Optional[List[DebateArgument]] = None,
        model_verdict: str = None,
        model_confidence: float = None,
        judge_verdict: str = None,
        judge_confidence: float = None,
        judge_reasoning: str = None,
        debate_history: Optional[List[List['DebateArgument']]] = None
    ) -> str:
        """Build prompt for debator - Clean, evidence-grounded prompts."""
        
        # Clean segmentation artifacts
        claim = self._clean_segmentation(claim)
        agent_name = self._get_agent_name()
        
        # Build evidence text
        evidence_text = ""
        if evidences and len(evidences) > 0:
            for i, ev in enumerate(evidences, 1):
                ev_text = self._clean_segmentation(ev.text)
                if ev_text:
                    evidence_text += f"- {ev_text}\n"
        else:
            evidence_text = "(No evidence provided)"
            
        # ===== ROUND 1: INDEPENDENT ANALYSIS =====
        if round_num == 1:
            prompt = f"""You are a FACT-CHECKER. Analyze independently.

**Claim:** "{claim}"
**Evidence:** {evidence_text}

**STEP 1: FULL CLAIM CHECK**
Break the claim into ALL parts. For EACH part, find evidence:
- Part covered by evidence? → Note the quote
- Part NOT in evidence? → Mark as "MISSING"
- Part CONTRADICTED by evidence? → Mark as "CONFLICT"

**STEP 2: DECIDE**

| Situation | Verdict |
|-----------|---------|
| ALL parts confirmed (paraphrasing OK) | **SUPPORTED** |
| ANY part CONTRADICTED (opposite fact, wrong description) | **REFUTED** |
| Key parts MISSING (no info in evidence) | **NEI** |

**CRITICAL RULES:**
1. **CHECK THE WHOLE CLAIM:** Don't just match what evidence says. Check if claim has EXTRA parts not in evidence.
   - Claim adds specific details (names, dates, places) not in evidence → **NEI**
2. **SILENCE ≠ CONTRADICTION:**
   - Evidence not mentioning X ≠ Evidence saying X is false → **NEI**
3. **MISREPRESENTATION = CONTRADICTION:**
   - Claim describes method/event A, evidence describes different method/event B → **REFUTED**

**CHỈ TRẢ VỀ JSON:**
{{
    "key_quote": "Copy EXACT text from evidence",
    "verdict": "SUPPORTED | REFUTED | NEI",
    "confidence": 0.0-1.0,
    "reasoning_vi": "Giải thích ngắn gọn bằng tiếng Việt"
}}"""
        
        # ===== ROUND 2: CROSS-EXAMINATION =====
        else:
            # Find own R1 verdict and reasoning
            own_r1_verdict = "N/A"
            own_r1_conf_str = ""
            own_r1_reason = ""
            if previous_arguments:
                for arg in previous_arguments:
                    if arg.debator_name == self.name:
                        own_r1_verdict = arg.verdict
                        own_r1_conf_str = f" ({arg.confidence:.0%})"
                        own_r1_reason = arg.reasoning[:150] + "..." if len(arg.reasoning) > 150 else arg.reasoning
                        break
            
            # Build own R1 quote
            own_r1_quote = ""
            if previous_arguments:
                for arg in previous_arguments:
                    if arg.debator_name == self.name and arg.key_points:
                        own_r1_quote = arg.key_points[0] if arg.key_points else ""
                        break
            
            # Build colleagues list with their verdicts
            colleagues_info = []
            if previous_arguments:
                for arg in previous_arguments:
                    if arg.debator_name != self.name:
                        colleague_name = self.ROLE_NAME_MAP.get(arg.role, arg.debator_name)
                        colleague_quote = arg.key_points[0] if arg.key_points else "(no quote)"
                        colleagues_info.append({
                            "name": colleague_name,
                            "verdict": arg.verdict,
                            "confidence": arg.confidence,
                            "quote": colleague_quote,
                            "reasoning": arg.reasoning[:100] + "..." if len(arg.reasoning) > 100 else arg.reasoning
                        })
            
            prompt = f"""**ROUND 2: CROSS-EXAMINATION & FINAL VERDICT**

**Claim:** "{claim}"
**Evidence:** {evidence_text}

**YOUR round 1 PREVIOUS ANALYSIS:**
- Verdict: {own_r1_verdict}{own_r1_conf_str}
- Your Quote: "{own_r1_quote}"
"""
            if own_r1_reason:
                prompt += f"- Reasoning: {own_r1_reason}\n"
            
            # Colleagues' analysis with quotes
            if colleagues_info:
                prompt += "\n**COLLEAGUES' round 1 VERDICTS:**\n"
                for col in colleagues_info:
                    prompt += f"- **{col['name']}**: {col['verdict']} ({col['confidence']:.0%})\n"
                    prompt += f"  Quote: \"{col['quote']}\"\n"
                    prompt += f"  Reasoning: {col['reasoning']}\n"
            prompt += """
---

**YOUR TASK:**
Write reasoning in Vietnamese (2-3 sentences), naturally mention: Who you agree/disagree with and why

**CRITICAL: CHALLENGE, DON'T CONFORM!**

**IF YOU ARE IN MINORITY (your verdict ≠ majority):**
→ You may be RIGHT! Attack the majority's reasoning:
  - Did they MISREAD the evidence?
  - Did they confuse SILENCE with CONTRADICTION?
  - Find the quote that PROVES them wrong!

**IF YOU ARE IN MAJORITY:**
→ Play Devil's Advocate! Challenge yourself:
  - Did the minority find a quote you MISSED?
  - Are you just agreeing because others agree?
  - Re-read the evidence with fresh eyes!

**DECISION RULE:**
- CHANGE only if: You find a NEW quote, or realize you MISREAD evidence
- STAY STUBBORN if: You are just outnumbered but your quote is correct

**OUTPUT (JSON):**
{
    "verdict": "SUPPORTED | REFUTED | NEI",
    "confidence": 0.0-1.0,
    "key_quote": "Quote from evidence",
    "reasoning_vi": "Giải thích bằng tiếng Việt, kết thúc bằng: Tôi giữ nguyên/thay đổi quan điểm, tuyên bố này Đúng/Sai/Thiếu thông tin."
}"""
        
        return prompt
    
    def _parse_response(self, response: str, round_num: int) -> DebateArgument:
        """Parse LLM response thành DebateArgument."""
        import json
        import re
        
        # Clean response - remove markdown code blocks if present
        cleaned = response.strip()
        if "```json" in cleaned:
            cleaned = re.sub(r'```json\s*', '', cleaned)
            cleaned = re.sub(r'```\s*$', '', cleaned)
        elif "```" in cleaned:
            cleaned = re.sub(r'```\s*', '', cleaned)
        
        # Strategy 1: Try to find JSON with balanced braces
        def find_json_with_balanced_braces(text):
            start = text.find('{')
            if start == -1:
                return None
            
            depth = 0
            for i, char in enumerate(text[start:], start):
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        return text[start:i+1]
            return None
        
        json_str = find_json_with_balanced_braces(cleaned)
        
        # Strategy 2: Regex fallback
        if not json_str:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
        
        # Strategy 3: Try to parse the entire cleaned response
        if not json_str:
            json_str = cleaned
        
        if not json_str or '{' not in json_str:
            logger.warning(f"{self.name}: Could not find JSON in response, using fallback")
            # Log more of the response for debugging
            logger.warning(f"{self.name} Raw response (first 300 chars): {response[:300]}")
            return DebateArgument(
                debator_name=self.name,
                role=self.role,
                round_num=round_num,
                verdict="NEI",
                confidence=0.5,
                reasoning="Failed to parse response - no JSON found",
                key_points=["Parse error"],
                evidence_citations=[]
            )
        
        # Try to parse JSON, with repair attempts
        data = None
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.debug(f"{self.name}: Initial JSON parse failed: {e}")
            # Try to repair common JSON issues
            try:
                repaired = json_str
                
                # Strategy: Find unquoted values after "key": and wrap them
                # Pattern: "key": value_without_quotes
                def fix_unquoted_field(text, field_name):
                    # Find the field and everything after it until next field or closing brace
                    pattern = rf'"{field_name}"\s*:\s*(?!")'  # Field followed by non-quote
                    match = re.search(pattern, text)
                    if match:
                        start_pos = match.end()
                        # Find where the value ends (next "field": or closing brace)
                        rest = text[start_pos:]
                        # Look for next field or end of object
                        end_match = re.search(r',\s*"[^"]+"\s*:|,?\s*}', rest)
                        if end_match:
                            value = rest[:end_match.start()].strip()
                            # Escape quotes and wrap
                            escaped = value.replace('\\', '\\\\').replace('"', '\\"')
                            new_text = text[:start_pos] + f'"{escaped}"' + rest[end_match.start():]
                            return new_text
                    return text
                
                # Fix common unquoted fields
                for field in ['reasoning', 'key_finding', 'evidence_quote']:
                    repaired = fix_unquoted_field(repaired, field)
                
                data = json.loads(repaired)
                logger.debug(f"{self.name}: JSON repaired successfully")
            except Exception as repair_error:
                logger.debug(f"{self.name}: JSON repair failed: {repair_error}")
                
                # Last resort: try to extract fields manually
                try:
                    verdict_match = re.search(r'"verdict"\s*:\s*"([^"]+)"', json_str)
                    conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', json_str)
                    # For reasoning, get everything after "reasoning": until next field
                    reasoning_match = re.search(r'"reasoning"\s*:\s*"?([^"]*?)(?:"|,\s*"|\s*})', json_str, re.DOTALL)
                    if not reasoning_match:
                        # Try final_statement
                        reasoning_match = re.search(r'"final_statement"\s*:\s*"?([^"]*?)(?:"|,\s*"|\s*})', json_str, re.DOTALL)
                    
                    if verdict_match:
                        data = {
                            "verdict": verdict_match.group(1),
                            "confidence": float(conf_match.group(1)) if conf_match else 0.7,
                            "reasoning": reasoning_match.group(1).strip() if reasoning_match else ""
                        }
                        logger.debug(f"{self.name}: Extracted fields manually")
                    else:
                        data = None
                except:
                    data = None
        
        if data is not None:
            # Debug: Log parsed JSON keys for Round 2+ to verify interaction fields
            if round_num >= 2:
                logger.debug(f"{self.name} R{round_num} JSON keys: {list(data.keys())}")
                logger.debug(f"{self.name} R{round_num} agree_with: {data.get('agree_with', 'NOT_FOUND')}")
                logger.debug(f"{self.name} R{round_num} disagree_with: {data.get('disagree_with', 'NOT_FOUND')}")
            
            # Normalize verdict format (handle variations + R3 uses "final_verdict")
            raw_verdict = data.get("verdict") or data.get("final_verdict") or "NEI"
            raw_verdict = raw_verdict.upper().strip()
            verdict_map = {
                "SUPPORTED": "SUPPORTED",
                "SUPPORTS": "SUPPORTED",
                "SUPPORT": "SUPPORTED",
                "ĐÚNG": "SUPPORTED",  # Vietnamese label
                "DUNG": "SUPPORTED",  # No diacritics
                "REFUTED": "REFUTED",
                "REFUTES": "REFUTED",
                "REFUTE": "REFUTED",
                "SAI": "REFUTED",  # Vietnamese label
                "NEI": "NEI",
                "NOT_ENOUGH_INFO": "NEI",
                "NOT ENOUGH INFO": "NEI",
                "INSUFFICIENT": "NEI",
                "THIẾU THÔNG TIN": "NEI",  # Vietnamese label
                "THIEU THONG TIN": "NEI",  # No diacritics
            }
            verdict = verdict_map.get(raw_verdict, "NEI")
            
            # Extract key_points - unified schema: key_quote
            key_points = data.get("key_points", [])
            if not key_points:
                key_quote = data.get("key_quote", "")
                if key_quote:
                    key_points = [str(key_quote)]
                else:
                    # Fallback to old schema fields
                    kp = data.get("evidence_quotes") or data.get("claim_components") or data.get("quote_comparison")
                    if kp and isinstance(kp, list):
                        key_points = kp[:3]
                    elif kp:
                        key_points = [str(kp)]
            
            # Extract reasoning - unified field: reasoning_vi
            reasoning = data.get("reasoning_vi") or data.get("reasoning") or data.get("final_reasoning_vi", "")
            
            # Extract Round 2 interaction fields (XAI)
            agree_with = data.get("agree_with", [])
            if isinstance(agree_with, str):
                agree_with = [agree_with] if agree_with else []
            agree_reason = data.get("agree_reason", "")
            
            disagree_with = data.get("disagree_with", [])
            if isinstance(disagree_with, str):
                disagree_with = [disagree_with] if disagree_with else []
            disagree_reason = data.get("disagree_reason", "")
            
            changed = data.get("changed", False)
            change_reason = data.get("change_reason", "")
            
            return DebateArgument(
                debator_name=self.name,
                role=self.role,
                round_num=round_num,
                verdict=verdict,
                confidence=float(data.get("confidence", 0.5)),
                reasoning=reasoning,
                key_points=key_points,
                evidence_citations=[],
                # XAI fields
                agree_with=agree_with if agree_with else None,
                agree_reason=agree_reason if agree_reason else None,
                disagree_with=disagree_with if disagree_with else None,
                disagree_reason=disagree_reason if disagree_reason else None,
                changed=changed,
                change_reason=change_reason if change_reason else None
            )
        
        # Fallback: extract verdict from raw text
        logger.warning(f"{self.name}: JSON parse failed, using fallback extraction")
        
        verdict = "NEI"
        confidence = 0.5
        reasoning = f"Parse error, extracted from text: {response[:200]}..."
        
        response_upper = response.upper()
        if "REFUTED" in response_upper or "REFUTE" in response_upper:
            verdict = "REFUTED"
            confidence = 0.7
        elif "SUPPORTED" in response_upper or "SUPPORT" in response_upper:
            verdict = "SUPPORTED"
            confidence = 0.7
        
        return DebateArgument(
            debator_name=self.name,
            role=self.role,
            round_num=round_num,
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning,
            key_points=["Parse error - fallback extraction"],
            evidence_citations=[]
        )


class GenericDebator(Debator):
    """
    Generic debator implementation cho tất cả LLMs.
    Dùng LLMClient để gọi API.
    """
    
    def argue(
        self, 
        claim: str, 
        evidences: List['Evidence'], 
        round_num: int,
        previous_arguments: List['DebateArgument'] = None,
        model_verdict: str = None,
        model_confidence: float = None,
        judge_verdict: str = None,
        judge_confidence: float = None,
        judge_reasoning: str = None
    ) -> DebateArgument:
        """
        Generate an argument based on role, claim, gold evidence, and round history.
        """
        
        prompt = self._build_prompt(
            claim, 
            evidences, 
            round_num, 
            previous_arguments,
            model_verdict,
            model_confidence,
            judge_verdict,
            judge_confidence,
            judge_reasoning
        )
        
        # Call LLM
        try:
            response = self.llm_client.generate(
                model=self.config['model'],
                prompt=prompt,
                api_key=self.config['api_key'],
                base_url=self.config['base_url'],
                temperature=self.config.get('temperature', 0.7),
                max_tokens=self.config.get('max_tokens', 1500)
            )
            
            # Parse response
            argument = self._parse_response(response, round_num)
            
            logger.info(f"{self.name} Round {round_num}: {argument.verdict} (conf={argument.confidence:.2f})")
            logger.info(f"  → Reasoning: {argument.reasoning}")
            
            return argument
            
        except Exception as e:
            logger.error(f"{self.name}: Error during argue: {e}")
            return DebateArgument(
                debator_name=self.name,
                role=self.role,  # Pass role
                round_num=round_num,
                verdict="NEI",  # Standardized to NEI (was NOT_ENOUGH_INFO)
                confidence=0.5,
                reasoning=f"Error: {str(e)}",
                key_points=["Error occurred"],
                evidence_citations=[]
            )
    
    async def argue_async(
        self,
        claim: str,
        evidences: List[Evidence],
        round_num: int,
        previous_arguments: Optional[List[DebateArgument]] = None,
        model_verdict: str = None,
        model_confidence: float = None,
        judge_verdict: str = None,
        judge_confidence: float = None,
        judge_reasoning: str = None,
        debate_history: Optional[List[List[DebateArgument]]] = None
    ) -> DebateArgument:
        """Async version of argue method with Gold Evidence."""
        
        # Build prompt with Gold Evidence only
        prompt = self._build_prompt(
            claim=claim,
            evidences=evidences,
            round_num=round_num,
            previous_arguments=previous_arguments,
            model_verdict=model_verdict,
            model_confidence=model_confidence,
            judge_verdict=judge_verdict,
            judge_confidence=judge_confidence,
            judge_reasoning=judge_reasoning,
            debate_history=debate_history
        )
        
        # Call LLM asynchronously with retry logic
        max_retries = 3
        response = None
        
        for attempt in range(max_retries):
            try:
                # On last retry, use simplified prompt (no previous arguments)
                current_prompt = prompt
                if attempt == max_retries - 1 and previous_arguments:
                    logger.warning(f"{self.name}: Retry {attempt+1} with simplified prompt")
                    current_prompt = self._build_prompt(
                        claim=claim,
                        evidences=evidences,
                        round_num=round_num,
                        previous_arguments=None,  # Remove previous arguments
                        model_verdict=model_verdict,
                        model_confidence=model_confidence,
                        judge_verdict=judge_verdict,
                        judge_confidence=judge_confidence,
                        judge_reasoning=judge_reasoning
                    )
                
                response = await self.llm_client.generate_async(
                    model=self.config['model'],
                    prompt=current_prompt,
                    api_key=self.config['api_key'],
                    base_url=self.config['base_url'],
                    temperature=self.config.get('temperature', 0.7),
                    max_tokens=self.config.get('max_tokens', 1500)
                )
                
                # Check for empty response
                if response and response.strip():
                    break  # Success, exit retry loop
                else:
                    logger.warning(f"{self.name}: Empty response on attempt {attempt+1}/{max_retries}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)  # Brief pause before retry
                        
            except Exception as e:
                logger.warning(f"{self.name}: Attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                else:
                    raise
        
        # Parse response
        if not response or not response.strip():
            logger.error(f"{self.name}: All retries failed, using NEI fallback")
            return DebateArgument(
                debator_name=self.name,
                role=self.role,
                round_num=round_num,
                verdict="NEI",
                confidence=0.5,
                reasoning="All retries failed - empty response",
                key_points=["Retry failed"],
                evidence_citations=[]
            )
        
        argument = self._parse_response(response, round_num)
        
        logger.info(f"{self.name} Round {round_num}: {argument.verdict} (conf={argument.confidence:.2f})")
        logger.info(f"  → Reasoning: {argument.reasoning}")
        
        return argument
