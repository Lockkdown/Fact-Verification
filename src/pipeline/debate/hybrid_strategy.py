"""
Hybrid Strategy - Confidence-aware decision between Model and Debate.

Based on research:
- DOWN framework (2025): Debate Only When Necessary
- iMAD (2025): Intelligent Multi-Agent Debate

Author: Lockdown
Date: Dec 01, 2025
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    """Result từ hybrid decision."""
    final_verdict: str  # SUPPORTED, REFUTED, NOT_ENOUGH_INFO
    decision_source: str  # MODEL_HIGH_CONF, DEBATE, MODEL_TIEBREAK
    model_verdict: str
    model_confidence: float
    debate_verdict: str
    debate_confidence: float
    threshold_used: float
    reasoning: str


class HybridStrategy:
    """
    Hybrid strategy để quyết định giữa Model và Debate verdict.
    
    Logic:
    - Model confidence >= threshold: Trust Model
    - Model confidence < threshold: Trust Debate
    
    Threshold có thể được tune trên dev set.
    """
    
    def __init__(self, threshold: float = 0.85):
        """
        Args:
            threshold: Confidence threshold để trust model (default: 0.85 từ research)
        """
        self.threshold = threshold
        logger.info(f"HybridStrategy initialized with threshold={threshold}")
    
    def decide(
        self,
        model_verdict: str,
        model_probs: Dict[str, float],
        debate_verdict: str,
        debate_confidence: float
    ) -> HybridResult:
        """
        Quyết định final verdict dựa trên model confidence.
        
        Args:
            model_verdict: Model's predicted verdict
            model_probs: Model's probability distribution {Support: 0.x, Refute: 0.x, NEI: 0.x}
            debate_verdict: Debate's final verdict
            debate_confidence: Debate's confidence (0-1)
            
        Returns:
            HybridResult với final decision và metadata
        """
        # Get max probability as confidence
        model_confidence = max(model_probs.values()) if model_probs else 0.0
        
        # Normalize verdicts
        model_verdict_norm = self._normalize_verdict(model_verdict)
        debate_verdict_norm = self._normalize_verdict(debate_verdict)
        
        # Decision logic
        if model_confidence >= self.threshold:
            final_verdict = model_verdict_norm
            decision_source = "MODEL_HIGH_CONF"
            reasoning = f"Model confidence {model_confidence:.0%} >= threshold {self.threshold:.0%}, trusting model."
        else:
            final_verdict = debate_verdict_norm
            decision_source = "DEBATE"
            reasoning = f"Model confidence {model_confidence:.0%} < threshold {self.threshold:.0%}, trusting debate."
        
        return HybridResult(
            final_verdict=final_verdict,
            decision_source=decision_source,
            model_verdict=model_verdict_norm,
            model_confidence=model_confidence,
            debate_verdict=debate_verdict_norm,
            debate_confidence=debate_confidence,
            threshold_used=self.threshold,
            reasoning=reasoning
        )
    
    def _normalize_verdict(self, verdict: str) -> str:
        """Normalize verdict to standard format."""
        if verdict is None:
            return "NOT_ENOUGH_INFO"
        
        verdict_upper = verdict.upper().strip()
        
        if verdict_upper in ["SUPPORTED", "SUPPORT", "SUPPORTS"]:
            return "SUPPORTED"
        elif verdict_upper in ["REFUTED", "REFUTE", "REFUTES"]:
            return "REFUTED"
        else:
            return "NOT_ENOUGH_INFO"
    
    @staticmethod
    def tune_threshold(
        results: List[Dict[str, Any]],
        thresholds: List[float] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Tune threshold trên dev set.
        
        Args:
            results: List of {
                "model_probs": {...},
                "model_verdict": str,
                "debate_verdict": str,
                "gold_label": str
            }
            thresholds: List of thresholds to try (default: [0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
            
        Returns:
            (best_threshold, accuracy_by_threshold)
        """
        if thresholds is None:
            thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        
        accuracy_by_threshold = {}
        
        for threshold in thresholds:
            strategy = HybridStrategy(threshold=threshold)
            correct = 0
            total = len(results)
            
            for sample in results:
                model_probs = sample.get("model_probs", sample.get("verdict_3label_probs", {}))
                
                # Normalize probs keys
                if "Support" in model_probs:
                    model_probs = {
                        "SUPPORTED": model_probs.get("Support", 0),
                        "REFUTED": model_probs.get("Refute", 0),
                        "NOT_ENOUGH_INFO": model_probs.get("NEI", 0)
                    }
                
                result = strategy.decide(
                    model_verdict=sample.get("model_verdict", "NEI"),
                    model_probs=model_probs,
                    debate_verdict=sample.get("debate_verdict", sample.get("final_verdict", "NEI")),
                    debate_confidence=sample.get("debate_confidence", 0.5)
                )
                
                gold = strategy._normalize_verdict(sample.get("gold_label", "NEI"))
                if result.final_verdict == gold:
                    correct += 1
            
            accuracy = correct / total if total > 0 else 0
            accuracy_by_threshold[threshold] = accuracy
            logger.info(f"Threshold {threshold}: {accuracy:.2%}")
        
        # Find best threshold
        best_threshold = max(accuracy_by_threshold, key=accuracy_by_threshold.get)
        logger.info(f"Best threshold: {best_threshold} with accuracy {accuracy_by_threshold[best_threshold]:.2%}")
        
        return best_threshold, accuracy_by_threshold
    
    @staticmethod
    def analyze_by_confidence_bucket(
        results: List[Dict[str, Any]],
        buckets: List[Tuple[float, float]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Phân tích accuracy theo confidence bucket.
        
        Args:
            results: List of samples với model_probs, model_verdict, debate_verdict, gold_label
            buckets: List of (min, max) tuples (default: [(0, 0.7), (0.7, 0.9), (0.9, 1.0)])
            
        Returns:
            {
                "0.0-0.7": {"n": 20, "model_acc": 0.65, "debate_acc": 0.78},
                "0.7-0.9": {...},
                ...
            }
        """
        if buckets is None:
            buckets = [(0.0, 0.70), (0.70, 0.90), (0.90, 1.01)]
        
        analysis = {}
        strategy = HybridStrategy()
        
        for bucket_min, bucket_max in buckets:
            bucket_name = f"{bucket_min:.1f}-{bucket_max:.1f}"
            bucket_samples = []
            
            for sample in results:
                model_probs = sample.get("model_probs", sample.get("verdict_3label_probs", {}))
                if "Support" in model_probs:
                    max_conf = max(model_probs.values())
                else:
                    max_conf = max(model_probs.values()) if model_probs else 0
                
                if bucket_min <= max_conf < bucket_max:
                    bucket_samples.append(sample)
            
            if not bucket_samples:
                analysis[bucket_name] = {"n": 0, "model_acc": 0, "debate_acc": 0}
                continue
            
            # Calculate accuracies
            model_correct = 0
            debate_correct = 0
            
            for sample in bucket_samples:
                gold = strategy._normalize_verdict(sample.get("gold_label", "NEI"))
                model_v = strategy._normalize_verdict(sample.get("model_verdict", "NEI"))
                debate_v = strategy._normalize_verdict(
                    sample.get("debate_verdict", sample.get("final_verdict", "NEI"))
                )
                
                if model_v == gold:
                    model_correct += 1
                if debate_v == gold:
                    debate_correct += 1
            
            n = len(bucket_samples)
            analysis[bucket_name] = {
                "n": n,
                "model_acc": model_correct / n,
                "debate_acc": debate_correct / n,
                "better": "MODEL" if model_correct >= debate_correct else "DEBATE"
            }
        
        return analysis


def print_analysis_report(analysis: Dict[str, Dict[str, float]], overall_results: Dict[str, float] = None):
    """Print formatted analysis report."""
    print("\n" + "="*70)
    print("HYBRID STRATEGY ANALYSIS REPORT")
    print("="*70)
    
    print("\n📊 Accuracy by Confidence Bucket:")
    print("-"*70)
    print(f"{'Bucket':<15} {'N':<8} {'Model Acc':<12} {'Debate Acc':<12} {'Winner':<10}")
    print("-"*70)
    
    for bucket, data in analysis.items():
        print(f"{bucket:<15} {data['n']:<8} {data['model_acc']:.1%}{'':>5} {data['debate_acc']:.1%}{'':>5} {data.get('better', '-'):<10}")
    
    print("-"*70)
    
    if overall_results:
        print("\n📈 Overall Results:")
        print(f"  Model Accuracy:  {overall_results.get('model_acc', 0):.2%}")
        print(f"  Debate Accuracy: {overall_results.get('debate_acc', 0):.2%}")
        print(f"  Hybrid Accuracy: {overall_results.get('hybrid_acc', 0):.2%}")
        print(f"  Threshold Used:  {overall_results.get('threshold', 0.85)}")
    
    print("="*70 + "\n")
