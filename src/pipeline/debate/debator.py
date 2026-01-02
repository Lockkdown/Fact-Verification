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
    # Structured output fields from prompts
    parts: Optional[List[Dict]] = None  # Round 1: parts analysis
    decision_change: Optional[str] = None  # Round 2+: MAINTAIN/CONSIDER_CHANGE/CHANGE
    changed_from: Optional[str] = None  # Round 2+: previous verdict if changed
    change_trigger: Optional[str] = None  # Round 2+: reason for change
    rebuttals: Optional[List[Dict]] = None  # Round 2+: rebuttal details
    key_parts_checked: Optional[List[str]] = None  # Round 2+: parts examined


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
        debate_history: Optional[List[List['DebateArgument']]] = None,
        max_rounds: int = None,
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
            
        # ===== ROUND 1: INDEPENDENT ANALYSIS (Parts-based) =====
        if round_num == 1:
            prompt = f"""You are an EVIDENCE-BASED FACT-CHECKER. Analyze independently. Use ONLY the provided evidence. Do NOT use outside knowledge.

**Claim:** "{claim}"
**Evidence:** {evidence_text}

**STEP 1: BREAK CLAIM INTO KEY PARTS**
Identify 2-5 KEY verifiable parts of the claim. 
(Key parts = core entities, time, location, quantity, or main event described by the claim)

For EACH part:
- Find a VERBATIM quote from evidence that covers it → status = COVERED
- If no quote found → status = MISSING, quote = NULL
- If evidence EXPLICITLY NEGATES the part (opposite value, mutually exclusive) → status = CONFLICT

**STEP 2: DECIDE**
- ALL key parts COVERED → **SUPPORTED**
- ANY key part CONFLICT → **REFUTED**
- Any key part MISSING → **NEI**

**RULES:**
- Quotes must be VERBATIM substrings copied from Evidence
- If you cannot find a verbatim quote, set quote = NULL
- CONFLICT only when evidence explicitly negates (not just missing info)

**Output JSON:**
{{
    "parts": [
        {{"part": "key part of claim", "status": "COVERED | MISSING | CONFLICT", "quote": "VERBATIM quote OR NULL"}}
    ],
    "verdict": "SUPPORTED | REFUTED | NEI",
    "reasoning_vi": "1-2 câu giải thích"
}}"""
        
        
        # ===== ROUND 2+: DEBATE (Evidence-grounded rebuttals) =====
        else:
            # Find own previous position (Fix 6: get quotes from parts or key_points)
            own_prev_verdict = "N/A"
            own_prev_quotes = []
            
            if previous_arguments:
                for arg in previous_arguments:
                    if arg.debator_name == self.name:
                        own_prev_verdict = arg.verdict
                        # Get all quotes from key_points (parser already extracts from parts)
                        if arg.key_points:
                            own_prev_quotes = [q for q in arg.key_points if q and str(q).upper() != "NULL"]
                        break
            
            # Show up to 2 quotes for better context (Fix R2)
            own_prev_quote_str = " | ".join(own_prev_quotes[:2]) if own_prev_quotes else "(no quote)"
            
            # Build other agents' positions
            other_agents = []
            has_disagreement = False
            if previous_arguments:
                for arg in previous_arguments:
                    if arg.debator_name != self.name:
                        agent_name = self.ROLE_NAME_MAP.get(arg.role, arg.debator_name)
                        # Get up to 2 non-null quotes (Fix R2)
                        quotes = [q for q in (arg.key_points or []) if q and str(q).upper() != "NULL"]
                        quote_str = " | ".join(quotes[:2]) if quotes else "(no quote)"
                        other_agents.append({
                            "name": agent_name,
                            "verdict": arg.verdict,
                            "quote": quote_str
                        })
                        if arg.verdict != own_prev_verdict:
                            has_disagreement = True
            
            # Build prompt - Evidence-grounded debate with rebuttals
            prompt = f"""**ROUND {round_num}: DEBATE**

**Claim:** "{claim}"
**Evidence:** {evidence_text}

---

**Your previous position:** {own_prev_verdict}
Your quote: "{own_prev_quote_str}"

**Other agents' positions:**
"""
            for agent in other_agents:
                prompt += f"- {agent['name']}: {agent['verdict']} — \"{agent['quote']}\"\n"
            
            prompt += f"""
---

**TASK:**
Write a short research-style rebuttal.
1. State which parts of the claim you focus on.
2. Compare your evidence against other agents' evidence (agree/disagree and why).
3. Update your stance if and only if the evidence requires it.

**Evidence & Update Guidance:**
- Use **verbatim quotes** copied from the Evidence.
- Treat other agents' quotes as claims to be evaluated: is the quote relevant, and does it directly support/refute the claim?
- **Maintain your verdict by default.** Change your verdict only if you introduce a **new counter_quote** (verbatim) that directly undermines your previous quote/interpretation on a specific claim part.
- If you cannot find a clear, direct supporting/refuting quote for SUPPORTED/REFUTED, consider using NEI and briefly explain what is missing.

**Output JSON:**
{{
    "key_parts_checked": ["key part 1...", "key part 2..."],
    "has_new_evidence": true,
    "has_strong_counterevidence": false,
    "rebuttals": [
        {{
            "agent": "Agent name",
            "their_verdict": "SUPPORTED | REFUTED | NEI",
            "their_quote": "their quote",
            "issue": "irrelevant | not_direct | wrong_context | misses_key_part | weak_contradiction | no_supporting_quote",
            "counter_quote": "VERBATIM quote OR NULL"
        }}
    ],
    "decision_change": "MAINTAIN | CONSIDER_CHANGE | CHANGE",
    "changed_from": "SUPPORTED | REFUTED | NEI | N/A",
    "change_trigger": "stronger_quote | missed_part | context_error | no_supporting_quote | N/A",
    "verdict": "SUPPORTED | REFUTED | NEI",
    "key_quote": "VERBATIM quote OR NULL",
    "reasoning_vi": "1-2 câu: giữ/đổi verdict vì parts + quote nào"
}}"""
        
        return prompt
    
    def _parse_response(self, response: str, round_num: int) -> DebateArgument:
        """Parse LLM response thành DebateArgument."""
        import json
        import re

        def _maybe_warn_parse(msg: str):
            # Rate-limit noisy parse warnings (avoid spamming console in large runs)
            cnt = getattr(self, "_json_parse_fail_count", 0) + 1
            setattr(self, "_json_parse_fail_count", cnt)
            if cnt <= 3 or cnt % 50 == 0:
                logger.warning(msg)

        def _fallback_from_text(text: str, reason: str) -> DebateArgument:
            verdict = "NEI"
            confidence = 0.5
            t = (text or "").upper()
            if "REFUTED" in t or "REFUTE" in t or "SAI" in t:
                verdict = "REFUTED"
            elif "SUPPORTED" in t or "SUPPORT" in t or "ĐÚNG" in t or "DUNG" in t:
                verdict = "SUPPORTED"
            
            # For Round 1, create basic parts field để UI hiển thị
            parts_fallback = None
            if round_num == 1:
                parts_fallback = [
                    {"part": "Parse error - không thể phân tích chi tiết", "status": "MISSING", "quote": "NULL"}
                ]
            
            return DebateArgument(
                debator_name=self.name,
                role=self.role,
                round_num=round_num,
                verdict=verdict,
                confidence=confidence,
                reasoning=reason,
                key_points=["Parse fallback"],
                evidence_citations=[],
                # Add missing structured fields for proper UI display
                parts=parts_fallback,
                decision_change=None,
                changed_from=None,
                change_trigger=None,
                rebuttals=None,
                key_parts_checked=None
            )
        
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
            _maybe_warn_parse(f"{self.name}: Could not find JSON in response, using fallback")
            _maybe_warn_parse(f"{self.name} Raw response (first 300 chars): {response[:300]}")
            return _fallback_from_text(
                response,
                reason=f"Failed to parse response - no JSON found (fallback from text). Snippet: {response[:200]}..."
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
            
            # Extract key_points - multiple sources (Dec 24, 2025 v2)
            key_points = data.get("key_points", [])
            
            # New R1 format: parts array with quotes
            if not key_points:
                parts = data.get("parts", [])
                if parts and isinstance(parts, list):
                    for part_data in parts:
                        if isinstance(part_data, dict):
                            quote = part_data.get("quote", "")
                            if quote and str(quote).upper() != "NULL":
                                key_points.append(str(quote))
            
            # R2+ format: single key_quote
            if not key_points:
                key_quote = data.get("key_quote", "")
                if key_quote and str(key_quote).upper() != "NULL":
                    key_points = [str(key_quote)]
            
            # Fallback to old schema fields
            if not key_points:
                kp = data.get("evidence_quotes") or data.get("claim_components") or data.get("quote_comparison")
                if kp and isinstance(kp, list):
                    key_points = kp[:3]
                elif kp:
                    key_points = [str(kp)]
            
            # Extract reasoning - unified field: reasoning_vi
            reasoning = data.get("reasoning_vi") or data.get("reasoning") or data.get("final_reasoning_vi", "")
            
            # Extract Round 2+ interaction fields (Dec 24, 2025 v2)
            agree_with = []
            disagree_with = []
            agree_reason = ""
            disagree_reason = ""
            
            # New format: rebuttals array (Dec 24, 2025 v2)
            rebuttals = data.get("rebuttals", [])
            if rebuttals and isinstance(rebuttals, list):
                for rebuttal in rebuttals:
                    if isinstance(rebuttal, dict):
                        agent_name = rebuttal.get("agent", "")
                        issue = rebuttal.get("issue", "")
                        counter_quote = rebuttal.get("counter_quote", "")
                        # All rebuttals are disagreements
                        if agent_name:
                            disagree_with.append(agent_name)
                            reason = f"Issue: {issue}"
                            if counter_quote and counter_quote.upper() != "NULL":
                                reason += f" | Counter: {counter_quote[:50]}..."
                            disagree_reason += f"{agent_name}: {reason}; "
            
            # Fallback: old other_agents format
            if not disagree_with:
                other_agents = data.get("other_agents", [])
                if other_agents and isinstance(other_agents, list):
                    for agent_data in other_agents:
                        if isinstance(agent_data, dict):
                            agent_name = agent_data.get("agent", "")
                            stance = agent_data.get("stance", "").upper()
                            reason = agent_data.get("reason", "")
                            if stance == "AGREE":
                                agree_with.append(agent_name)
                                if reason:
                                    agree_reason += f"{agent_name}: {reason}; "
                            elif stance == "DISAGREE":
                                disagree_with.append(agent_name)
                                if reason:
                                    disagree_reason += f"{agent_name}: {reason}; "
            
            # Fallback: legacy format
            if not agree_with and not disagree_with:
                old_agree = data.get("agree_with", [])
                if isinstance(old_agree, str):
                    agree_with = [old_agree] if old_agree else []
                elif isinstance(old_agree, list):
                    agree_with = old_agree
                agree_reason = data.get("agree_reason", "")
                
                old_disagree = data.get("disagree_with", [])
                if isinstance(old_disagree, str):
                    disagree_with = [old_disagree] if old_disagree else []
                elif isinstance(old_disagree, list):
                    disagree_with = old_disagree
                disagree_reason = data.get("disagree_reason", "")
            
            # Decision change tracking (Dec 24, 2025 v2)
            decision_change = data.get("decision_change", "").upper()
            # Backward-compatible:
            # - Old prompt: KEPT | CHANGED
            # - New prompt (stability): MAINTAIN | CONSIDER_CHANGE | CHANGE
            # - Legacy boolean: changed
            changed = decision_change in ["CHANGED", "CHANGE"] or data.get("changed", False)
            changed_from = data.get("changed_from", "")
            change_reason = f"Changed from {changed_from}" if changed and changed_from else ""
            
            # Extract structured output fields
            key_parts_checked = data.get("key_parts_checked", [])
            parts = data.get("parts", [])
            rebuttals = data.get("rebuttals", [])
            change_trigger = data.get("change_trigger", "")
            
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
                change_reason=change_reason if change_reason else None,
                # Structured output fields from prompts
                parts=parts if parts else None,
                decision_change=decision_change if decision_change else None,
                changed_from=changed_from if changed_from else None,
                change_trigger=change_trigger if change_trigger else None,
                rebuttals=rebuttals if rebuttals else None,
                key_parts_checked=key_parts_checked if key_parts_checked else None
            )
        
        # Fallback: extract verdict from raw text
        _maybe_warn_parse(f"{self.name}: JSON parse failed, using fallback extraction")
        return _fallback_from_text(
            response,
            reason=f"Parse error, extracted from text. Snippet: {response[:200]}..."
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
        judge_reasoning: str = None,
        max_rounds: int = None,
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
            judge_reasoning,
            max_rounds=max_rounds,
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
        debate_history: Optional[List[List[DebateArgument]]] = None,
        max_rounds: int = None,
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
            debate_history=debate_history,
            max_rounds=max_rounds,
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
