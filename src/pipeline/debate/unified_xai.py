"""
Unified XAI Module - Consistent XAI format for both Fast Path and Slow Path

This module provides a unified interface for XAI (Explainable AI) outputs
from both PhoBERT (Fast Path) and Debate (Slow Path).

Author: Lockdown
Date: Dec 11, 2025
"""

from dataclasses import dataclass, field
from typing import Dict, Any
import json


@dataclass
class UnifiedXAI:
    """
    Unified XAI structure for both Fast Path (PhoBERT) and Slow Path (Debate).
    
    Based on:
    - Atanasova et al. (2020): Extractive explanations
    - JustiLM (2023): Atomic justification format
    """
    
    # === Required Fields ===
    relationship: str              # SUPPORTS | REFUTES | NEI
    natural_explanation: str       # Natural language explanation (Vietnamese)
    
    # === Bonus for REFUTES ===
    conflict_claim: str = ""       # Conflicting word/phrase in claim
    conflict_evidence: str = ""    # Conflicting word/phrase in evidence
    
    # === Metadata ===
    source: str = "UNKNOWN"        # FAST_PATH | SLOW_PATH
    confidence: float = 0.0        # Confidence score (0-1)
    
    # === Additional Context ===
    debate_summary: str = ""       # Summary of debate (Slow Path only)
    similarity_score: float = 0.0  # Semantic similarity (Fast Path only)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "relationship": self.relationship,
            "natural_explanation": self.natural_explanation,
            "conflict_claim": self.conflict_claim,
            "conflict_evidence": self.conflict_evidence,
            "source": self.source,
            "confidence": self.confidence,
            "debate_summary": self.debate_summary,
            "similarity_score": self.similarity_score,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
    
    def to_html(self) -> str:
        """Generate HTML visualization of XAI."""
        # Relationship colors
        rel_colors = {
            "SUPPORTS": "#90EE90",  # Light green
            "REFUTES": "#FFB6C1",   # Light red
            "NEI": "#D3D3D3",       # Light gray
        }
        rel_color = rel_colors.get(self.relationship, "#FFFFFF")
        
        # Build conflict section (only for REFUTES)
        conflict_html = ""
        if self.relationship == "REFUTES" and self.conflict_claim and self.conflict_evidence:
            conflict_html = f"""
            <div style="margin: 10px 0; padding: 10px; background-color: #fff3cd; border-radius: 5px; border-left: 4px solid #ffc107;">
                <strong>⚡ Key Conflict:</strong>
                <span style="background-color: #FF6B6B; padding: 2px 6px; border-radius: 3px; color: white;">"{self.conflict_claim}"</span>
                <span style="margin: 0 8px;">⚔️</span>
                <span style="background-color: #4ECDC4; padding: 2px 6px; border-radius: 3px; color: white;">"{self.conflict_evidence}"</span>
            </div>
            """
        
        html = f"""
        <div style="border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 8px; font-family: sans-serif;">
            {conflict_html}
            <div style="margin-bottom: 10px;">
                <strong>🏷️ Relationship:</strong>
                <span style="background-color: {rel_color}; padding: 4px 8px; border-radius: 4px; font-weight: bold;">
                    {self.relationship}
                </span>
                <span style="color: #666; margin-left: 10px;">({self.confidence:.0%} confidence)</span>
            </div>
            <div style="margin-bottom: 10px; background-color: #f8f9fa; padding: 10px; border-left: 3px solid #333; border-radius: 4px;">
                <strong>💬 Explanation:</strong> {self.natural_explanation}
            </div>
            <div style="font-size: 12px; color: #888;">
                Source: {self.source}
            </div>
        </div>
        """
        return html
    
    def _highlight_text(self, text: str, color: str = "#FFD700") -> str:
        """Highlight text with background color."""
        if not text:
            return ""
        return f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px;">{text}</span>'


def create_unified_xai(
    xai_dict: Dict[str, Any],
    source: str = "UNKNOWN"
) -> UnifiedXAI:
    """
    Create UnifiedXAI from a dictionary (from PhoBERT or Debate XAI).
    
    Args:
        xai_dict: XAI dictionary from either path
        source: "FAST_PATH" or "SLOW_PATH"
        
    Returns:
        UnifiedXAI instance
    """
    # Normalize relationship field name
    relationship = xai_dict.get("relationship") or xai_dict.get("verdict", "NEI")
    
    # Normalize relationship value
    rel_map = {
        "SUPPORTED": "SUPPORTS",
        "SUPPORTS": "SUPPORTS",
        "REFUTED": "REFUTES",
        "REFUTES": "REFUTES",
        "NEI": "NEI",
        "NOT_ENOUGH_INFO": "NEI",
    }
    relationship = rel_map.get(relationship.upper(), "NEI") if relationship else "NEI"
    
    # Normalize conflict field names (PhoBERT uses _word suffix)
    conflict_claim = (
        xai_dict.get("conflict_claim") or 
        xai_dict.get("claim_conflict_word") or 
        ""
    )
    conflict_evidence = (
        xai_dict.get("conflict_evidence") or 
        xai_dict.get("evidence_conflict_word") or 
        ""
    )
    
    return UnifiedXAI(
        relationship=relationship,
        natural_explanation=xai_dict.get("natural_explanation", ""),
        conflict_claim=conflict_claim,
        conflict_evidence=conflict_evidence,
        source=source,
        confidence=xai_dict.get("confidence", 0.0),
        debate_summary=xai_dict.get("debate_summary", ""),
        similarity_score=xai_dict.get("similarity_score", 0.0),
    )


def from_phobert_xai(xai_dict: Dict[str, Any]) -> UnifiedXAI:
    """Create UnifiedXAI from PhoBERT XAI output."""
    return create_unified_xai(xai_dict, source="FAST_PATH")


def from_debate_xai(xai_dict: Dict[str, Any]) -> UnifiedXAI:
    """Create UnifiedXAI from Debate XAI output."""
    return create_unified_xai(xai_dict, source="SLOW_PATH")
