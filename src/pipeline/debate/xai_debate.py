"""
XAI Module for Debate (Slow Path)
Generates structured explanations with 4 fields (consistent with PhoBERT XAI):
- Relationship
- Natural explanation

This module extracts structured XAI from LLM debate reasoning,
ensuring output format matches PhoBERT XAI for pipeline consistency.

Author: Lockdown
Date: Dec 7, 2025
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Try to import PyVi for Vietnamese word segmentation
try:
    from pyvi import ViTokenizer
    PYVI_AVAILABLE = True
except ImportError:
    PYVI_AVAILABLE = False


@dataclass
class DebateXAIOutput:
    """Structured XAI output from Debate (matches PhoBERT XAI format)."""
    relationship: str
    natural_explanation: str
    claim_conflict_word: str = ""
    evidence_conflict_word: str = ""
    # Debate-specific fields
    confidence: float = 0.0
    reasoning_source: str = "judge"  # judge / majority / agent
    debate_summary: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "relationship": self.relationship,
            "natural_explanation": self.natural_explanation,
            "claim_conflict_word": self.claim_conflict_word,
            "evidence_conflict_word": self.evidence_conflict_word,
            "confidence": self.confidence,
            "reasoning_source": self.reasoning_source,
            "debate_summary": self.debate_summary
        }


class DebateXAI:
    """
    XAI generator for Debate (Slow Path).
    Extracts structured explanations from LLM debate reasoning.
    
    Output format is consistent with PhoBERT XAI for unified pipeline.
    """
    
    # Verdict mapping from Debate format to standard format
    VERDICT_MAP = {
        "SUPPORTED": "SUPPORTS",
        "SUPPORTS": "SUPPORTS",
        "REFUTED": "REFUTES",
        "REFUTES": "REFUTES",
        "NEI": "NEI",
        "NOT_ENOUGH_INFO": "NEI",
    }
    
    def __init__(self):
        """Initialize DebateXAI."""
        pass
    
    def generate_xai(
        self,
        claim: str,
        evidence: str,
        final_verdict: Any,  # FinalVerdict object from Judge
        debate_history: List[List[Any]] = None,  # List of DebateArgument rounds
    ) -> Dict[str, Any]:
        """
        Generate XAI with 4 fields from Debate output.
        
        Args:
            claim: Original claim text
            evidence: Evidence text
            final_verdict: FinalVerdict object from Judge
            debate_history: Full debate history (optional, for richer XAI)
            
        Returns:
            Dictionary with XAI fields:
            - relationship
            - natural_explanation
            - claim_conflict_word (for REFUTES)
            - evidence_conflict_word (for REFUTES)
        """
        # Extract from FinalVerdict
        verdict = self._normalize_verdict(final_verdict.verdict)
        reasoning = final_verdict.reasoning or ""
        evidence_summary = final_verdict.evidence_summary or ""
        confidence = final_verdict.confidence or 0.0

        claim_conflict = final_verdict.xai_conflict_claim or ""
        evidence_conflict = final_verdict.xai_conflict_evidence or ""
        judge_natural_explanation = getattr(final_verdict, 'xai_natural_explanation', "") or ""

        # Fallback: Extract conflict from reasoning (legacy mode) if missing
        if verdict == "REFUTES" and (not claim_conflict or not evidence_conflict):
            claim_conflict, evidence_conflict = self._find_conflict_from_reasoning(
                claim, evidence, reasoning
            )
        
        # 4. Relationship = normalized verdict
        relationship = verdict
        
        # 5. Generate natural explanation
        # Prefer Judge's LLM-generated explanation if available
        if judge_natural_explanation:
            natural_explanation = judge_natural_explanation
        else:
            natural_explanation = self._generate_natural_explanation(
                claim_highlight="",
                evidence_highlight="",
                relationship=relationship,
                reasoning=reasoning,
                claim_conflict=claim_conflict,
                evidence_conflict=evidence_conflict
            )
        
        # 6. Create debate summary (for additional context)
        debate_summary = self._create_debate_summary(final_verdict, debate_history)
        
        # Determine reasoning source based on decision path
        decision_path = getattr(final_verdict, 'decision_path', 'MAJORITY_VOTE')
        if decision_path == "MAJORITY_VOTE":
            reasoning_source = "majority_vote"
        elif "CONSENSUS" in str(decision_path):
            reasoning_source = "consensus"
        else:
            reasoning_source = "debate"
        
        return {
            "relationship": relationship,
            "natural_explanation": natural_explanation,
            "claim_conflict_word": claim_conflict,
            "evidence_conflict_word": evidence_conflict,
            "confidence": confidence,
            "reasoning_source": reasoning_source,
            "debate_summary": debate_summary,
            # For compatibility with PhoBERT XAI
            "similarity_score": confidence
        }
    
    def _normalize_verdict(self, verdict: str) -> str:
        """Normalize verdict to standard format."""
        if not verdict:
            return "NEI"
        return self.VERDICT_MAP.get(verdict.upper(), "NEI")
    
    def _extract_claim_highlight(self, claim: str, reasoning: str) -> str:
        """
        Extract key phrase from claim based on debate reasoning.
        
        Strategy:
        1. Look for quoted claim text in reasoning
        2. Fall back to PyVi-based extraction (like PhoBERT)
        """
        # Try to find quoted text from claim in reasoning
        quoted = self._find_quoted_text(reasoning)
        for quote in quoted:
            # Check if quote is from claim (significant overlap)
            if self._text_overlap(quote, claim) > 0.5:
                return quote
        
        # Fall back to middle portion extraction (like PhoBERT)
        return self._extract_middle_portion(claim)
    
    def _extract_evidence_highlight(
        self, 
        evidence: str, 
        reasoning: str,
        evidence_summary: str
    ) -> str:
        """
        Extract key evidence phrase from debate reasoning.
        
        Strategy:
        1. Use evidence_summary if available
        2. Look for quoted evidence in reasoning
        3. Fall back to first sentence of evidence
        """
        # If evidence_summary is good, use it
        if evidence_summary and len(evidence_summary) > 20:
            # Clean up
            clean_summary = evidence_summary.strip()
            if len(clean_summary) <= 200:
                return clean_summary
        
        # Try to find quoted evidence in reasoning
        quoted = self._find_quoted_text(reasoning)
        for quote in quoted:
            if self._text_overlap(quote, evidence) > 0.3:
                return quote
        
        # Fall back to first sentence
        sentences = self._split_sentences(evidence)
        if sentences:
            return sentences[0][:200]
        
        return evidence[:200] if evidence else ""
    
    def _find_conflict_from_reasoning(
        self,
        claim: str,
        evidence: str,
        reasoning: str
    ) -> Tuple[str, str]:
        """
        Extract conflict words from debate reasoning.
        
        Strategy:
        1. Look for quoted conflicts in reasoning
        2. Fall back to word-level diff (more reliable)
        """
        # Try to find quoted text pairs in reasoning
        quoted = self._find_quoted_text(reasoning)
        
        # If we have 2+ quotes, check if they're from claim/evidence
        if len(quoted) >= 2:
            claim_quote = None
            evidence_quote = None
            
            for q in quoted:
                q_lower = q.lower()
                if self._text_overlap(q, claim) > 0.3 and not claim_quote:
                    claim_quote = q
                elif self._text_overlap(q, evidence) > 0.3 and not evidence_quote:
                    evidence_quote = q
            
            if claim_quote and evidence_quote:
                return claim_quote, evidence_quote
        
        # Primary method: word-level diff (more reliable like PhoBERT)
        return self._find_word_level_conflict(claim, evidence)
    
    def _find_word_level_conflict(
        self,
        claim: str,
        evidence: str
    ) -> Tuple[str, str]:
        """Find conflicting words using word diff (like PhoBERT)."""
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()
        
        # Check common conflict patterns first (like PhoBERT)
        conflict_patterns = [
            # Ordinal number conflicts
            (['thứ hai', 'thứ 2'], ['đầu tiên', 'thứ nhất', 'thứ 1'], 'ordinal'),
            (['thứ ba', 'thứ 3'], ['đầu tiên', 'thứ hai', 'thứ 2'], 'ordinal'),
            (['đầu tiên', 'thứ nhất'], ['thứ hai', 'thứ 2', 'thứ ba'], 'ordinal'),
            
            # Tense/status conflicts
            (['đã ban hành', 'đã thông qua'], ['đang xem xét', 'dự thảo', 'chưa ban hành'], 'status'),
            (['đã'], ['đang', 'chưa', 'sẽ'], 'tense'),
            
            # Quantity conflicts
            (['tăng'], ['giảm', 'không đổi'], 'direction'),
            (['giảm'], ['tăng', 'không đổi'], 'direction'),
        ]
        
        for claim_patterns, evidence_patterns, _ in conflict_patterns:
            for cp in claim_patterns:
                if cp in claim_lower:
                    for ep in evidence_patterns:
                        if ep in evidence_lower:
                            return cp, ep
        
        # Fall back to word-level diff
        def normalize(text: str) -> set:
            if PYVI_AVAILABLE:
                segmented = ViTokenizer.tokenize(text.lower())
                words = segmented.split()
            else:
                text = re.sub(r'[^\w\s]', ' ', text.lower())
                words = text.split()
            
            stopwords = {'là', 'và', 'của', 'có', 'được', 'trong', 'để', 'với', 
                        'cho', 'này', 'từ', 'các', 'một', 'những', 'đã', 'đang',
                        'sẽ', 'không', 'cũng', 'như', 'theo', 'về', 'trên', 'khi'}
            return set(w for w in words if w not in stopwords and len(w) > 1)
        
        claim_words = normalize(claim)
        evidence_words = normalize(evidence)
        
        claim_only = list(claim_words - evidence_words)
        evidence_only = list(evidence_words - claim_words)
        
        # Priority: years > numbers > longest word
        def get_best(words):
            years = [w for w in words if re.match(r'^20\d{2}$|^19\d{2}$', w)]
            if years:
                return years[0].replace('_', ' ')
            nums = [w for w in words if any(c.isdigit() for c in w)]
            if nums:
                return nums[0].replace('_', ' ')
            if words:
                return max(words, key=len).replace('_', ' ')
            return ""
        
        return get_best(claim_only), get_best(evidence_only)
    
    def _generate_natural_explanation(
        self,
        claim_highlight: str,
        evidence_highlight: str,
        relationship: str,
        reasoning: str,
        claim_conflict: str = "",
        evidence_conflict: str = ""
    ) -> str:
        """
        Generate natural language explanation.
        
        For Debate, we can use the LLM's reasoning directly,
        but format it consistently with PhoBERT template.
        """
        # For REFUTES with explicit conflict
        if relationship == "REFUTES" and claim_conflict and evidence_conflict:
            return (
                f"Claim nói '{claim_conflict}' nhưng Evidence nói '{evidence_conflict}'. "
                f"Đây là mâu thuẫn trực tiếp."
            )
        
        # Extract key reasoning (first 1-2 sentences)
        if reasoning:
            sentences = self._split_sentences(reasoning)
            if sentences:
                key_reasoning = ' '.join(sentences[:2])
                if len(key_reasoning) <= 200:
                    return key_reasoning
                return key_reasoning[:200] + "..."
        
        # Fall back to template (like PhoBERT)
        templates = {
            "SUPPORTS": f"Bằng chứng khẳng định '{evidence_highlight[:100]}...', ủng hộ tuyên bố.",
            "REFUTES": f"Bằng chứng mâu thuẫn với tuyên bố về '{claim_highlight[:50]}'.",
            "NEI": f"Bằng chứng không cung cấp đủ thông tin để xác minh tuyên bố."
        }
        
        return templates.get(relationship, "Không đủ thông tin để giải thích.")
    
    def _create_debate_summary(
        self,
        final_verdict: Any,
        debate_history: List[List[Any]] = None
    ) -> str:
        """Create a brief summary of the debate process."""
        summary_parts = []
        
        # Verdict info
        verdict = final_verdict.verdict
        confidence = final_verdict.confidence
        rounds = final_verdict.rounds_used
        
        summary_parts.append(f"Verdict: {verdict} ({confidence:.0%} confidence)")
        summary_parts.append(f"Rounds: {rounds}")
        
        # Agreement info
        if final_verdict.debator_agreements:
            agreements = final_verdict.debator_agreements
            summary_parts.append(f"Agent votes: {agreements}")
        
        # Decision path
        if final_verdict.decision_path:
            summary_parts.append(f"Decision: {final_verdict.decision_path}")
        
        return " | ".join(summary_parts)
    
    # === Helper Methods ===
    
    def _find_quoted_text(self, text: str) -> List[str]:
        """Find all quoted text in a string."""
        patterns = [
            r'"([^"]+)"',  # Double quotes
            r"'([^']+)'",  # Single quotes
            r'「([^」]+)」',  # Asian quotes
            r'"([^"]+)"',  # Smart quotes
        ]
        
        quoted = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            quoted.extend(matches)
        
        return quoted
    
    def _text_overlap(self, text1: str, text2: str) -> float:
        """Calculate word overlap ratio between two texts."""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        smaller = min(len(words1), len(words2))
        
        return intersection / smaller
    
    def _extract_middle_portion(self, text: str, min_tokens: int = 4, max_tokens: int = 8) -> str:
        """Extract middle portion of text (like PhoBERT)."""
        if PYVI_AVAILABLE:
            segmented = ViTokenizer.tokenize(text)
            tokens = segmented.split()
        else:
            tokens = text.split()
        
        # Remove punctuation tokens
        tokens = [t for t in tokens if t not in ['.', ',', '!', '?', ';', ':']]
        
        if len(tokens) <= max_tokens:
            return text.rstrip('.,!?;:')
        
        # Skip common first tokens
        skip_first = {'việt_nam', 'chính_phủ', 'bộ', 'ông', 'bà'}
        start_idx = 1 if tokens and tokens[0].lower() in skip_first else 0
        
        end_idx = min(start_idx + max_tokens, len(tokens))
        
        highlight = ' '.join(tokens[start_idx:end_idx])
        highlight = highlight.replace('_', ' ')
        
        return highlight.rstrip('.,!?;:')
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if not text:
            return []
        
        # Simple sentence splitting
        sentences = re.split(r'[.!?]\s+', text)
        return [s.strip() for s in sentences if s.strip()]


def generate_debate_xai(
    claim: str,
    evidence: str,
    final_verdict: Any,
    debate_history: List[List[Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate Debate XAI.
    
    Args:
        claim: Claim text
        evidence: Evidence text
        final_verdict: FinalVerdict from Judge
        debate_history: Optional debate history
        
    Returns:
        XAI dict with same format as PhoBERT XAI
    """
    xai = DebateXAI()
    return xai.generate_xai(claim, evidence, final_verdict, debate_history)
