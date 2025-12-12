"""
XAI Metrics - Quantitative Evaluation for Explainable AI

Metrics based on:
- ERASER Benchmark (DeYoung et al., 2019)
- G-Eval Framework (Liu et al., 2023)
- Fact-Checking XAI Research (2024-2025)

Author: Lockdown
Date: Nov 28, 2025
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DebateXAIMetrics:
    """Metrics for Debate quality and reasoning."""
    # Consensus metrics
    round_1_agreement: float = 0.0    # % agreement in round 1
    final_agreement: float = 0.0      # % agreement at end
    consensus_shift: int = 0          # 0=easy, 1=moderate, 2=dramatic
    
    # Verdict consistency
    verdict_reasoning_match: float = 0.0  # Does reasoning match verdict?
    
    # Drama index
    num_flips: int = 0                # How many agents changed their mind
    rounds_used: int = 1              # How many rounds needed
    
    # NEW: Advanced XAI metrics
    evidence_utilization: float = 0.0     # % evidence được sử dụng trong reasoning
    reasoning_coherence: float = 0.0      # Reasoning có mạch lạc không (0-1)
    argument_diversity: float = 0.0       # Các agent có đưa ra argument khác nhau không
    faithfulness_score: float = 0.0       # Explanation có phản ánh đúng decision không
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_1_agreement": round(self.round_1_agreement, 4),
            "final_agreement": round(self.final_agreement, 4),
            "consensus_shift": self.consensus_shift,
            "verdict_reasoning_match": round(self.verdict_reasoning_match, 4),
            "num_flips": self.num_flips,
            "rounds_used": self.rounds_used,
            "evidence_utilization": round(self.evidence_utilization, 4),
            "reasoning_coherence": round(self.reasoning_coherence, 4),
            "argument_diversity": round(self.argument_diversity, 4),
            "faithfulness_score": round(self.faithfulness_score, 4)
        }


@dataclass 
class XAISampleMetrics:
    """XAI metrics for a single sample (Debate only - Gold Evidence mode)."""
    sample_id: str
    debate_metrics: DebateXAIMetrics = field(default_factory=DebateXAIMetrics)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "debate": self.debate_metrics.to_dict()
        }


# =============================================================================
# TEXT UTILITIES
# =============================================================================

def tokenize_vietnamese(text: str) -> List[str]:
    """Simple Vietnamese tokenizer (word-level)."""
    if not text:
        return []
    # Remove punctuation and lowercase
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # Split by whitespace
    tokens = text.split()
    # Filter empty tokens
    return [t for t in tokens if t.strip()]


# =============================================================================
# DEBATE METRICS CALCULATOR
# =============================================================================

def calculate_agreement_ratio(verdicts: List[str]) -> float:
    """Calculate agreement ratio from list of verdicts."""
    if not verdicts:
        return 0.0
    
    verdict_counts = Counter(verdicts)
    max_count = max(verdict_counts.values())
    return max_count / len(verdicts)


def calculate_verdict_reasoning_match(verdict: str, reasoning: str) -> float:
    """
    Check if reasoning text matches the verdict.
    
    Simple heuristic: Look for keywords that match verdict.
    """
    if not reasoning:
        return 0.0
    
    reasoning_lower = reasoning.lower()
    verdict_upper = verdict.upper()
    
    # Keywords for each verdict
    support_keywords = ["đúng", "chính xác", "xác nhận", "support", "true", "correct", "ủng hộ"]
    refute_keywords = ["sai", "mâu thuẫn", "phản bác", "refute", "false", "incorrect", "bác bỏ", "không đúng"]
    nei_keywords = ["không đủ", "thiếu", "không rõ", "nei", "không có", "chưa thể", "không thể xác định"]
    
    if verdict_upper in ["SUPPORTED", "SUPPORT"]:
        matches = sum(1 for kw in support_keywords if kw in reasoning_lower)
        return min(1.0, matches / 2)  # Need at least 2 keywords for full score
    elif verdict_upper in ["REFUTED", "REFUTE"]:
        matches = sum(1 for kw in refute_keywords if kw in reasoning_lower)
        return min(1.0, matches / 2)
    elif verdict_upper in ["NEI", "NOT_ENOUGH_INFO"]:
        matches = sum(1 for kw in nei_keywords if kw in reasoning_lower)
        return min(1.0, matches / 2)
    
    return 0.5  # Unknown verdict


def calculate_consensus_shift(round_1_agreement: float, final_agreement: float, rounds_used: int) -> int:
    """
    Calculate consensus shift level (drama index).
    
    Returns:
        0: Easy (high agreement from start)
        1: Moderate (some disagreement resolved)
        2: Dramatic (major shifts or prolonged debate)
    """
    if round_1_agreement >= 1.0:
        return 0  # Perfect agreement from start
    elif round_1_agreement >= 0.66 and rounds_used <= 2:
        return 1  # Moderate disagreement, resolved quickly
    else:
        return 2  # Dramatic debate


def calculate_evidence_utilization(
    reasoning: str,
    extracted_evidences: List[Dict[str, Any]]
) -> float:
    """
    Calculate Evidence Utilization Rate.
    
    Measures what percentage of extracted evidence is actually referenced in reasoning.
    High utilization = reasoning uses the provided evidence well.
    
    Args:
        reasoning: The reasoning text from agent/judge
        extracted_evidences: List of evidence dicts with quotes
        
    Returns:
        Utilization rate (0-1)
    """
    if not reasoning or not extracted_evidences:
        return 0.0
    
    reasoning_tokens = set(tokenize_vietnamese(reasoning))
    
    utilized_count = 0
    for evidence in extracted_evidences:
        quote = evidence.get("quote", evidence.get("text", ""))
        if not quote:
            continue
        
        evidence_tokens = set(tokenize_vietnamese(quote))
        
        # Check if significant portion of evidence appears in reasoning
        if evidence_tokens:
            overlap = len(reasoning_tokens & evidence_tokens) / len(evidence_tokens)
            if overlap >= 0.3:  # At least 30% of evidence tokens appear in reasoning
                utilized_count += 1
    
    return utilized_count / len(extracted_evidences) if extracted_evidences else 0.0


def calculate_reasoning_coherence(reasoning: str) -> float:
    """
    Calculate Reasoning Coherence Score.
    
    Measures how well-structured and coherent the reasoning is.
    Based on presence of logical connectors and structure.
    
    Args:
        reasoning: The reasoning text
        
    Returns:
        Coherence score (0-1)
    """
    if not reasoning:
        return 0.0
    
    reasoning_lower = reasoning.lower()
    
    # Logical connectors (Vietnamese + English)
    connectors = [
        # Cause-effect
        "vì vậy", "do đó", "therefore", "because", "bởi vì", "nên",
        # Contrast
        "tuy nhiên", "nhưng", "however", "but", "mặc dù", "although",
        # Addition
        "ngoài ra", "thêm vào đó", "moreover", "furthermore", "hơn nữa",
        # Conclusion
        "kết luận", "tóm lại", "in conclusion", "finally", "cuối cùng",
        # Evidence reference
        "theo", "dựa trên", "based on", "according to", "cho thấy"
    ]
    
    # Count connectors
    connector_count = sum(1 for c in connectors if c in reasoning_lower)
    
    # Check for structure (numbered points, bullets)
    has_structure = bool(re.search(r'(\d+\.|•|-|\*)\s', reasoning))
    
    # Calculate score
    base_score = min(1.0, connector_count / 3)  # Need ~3 connectors for full score
    structure_bonus = 0.2 if has_structure else 0.0
    
    return min(1.0, base_score + structure_bonus)


def calculate_argument_diversity(reasonings: List[str]) -> float:
    """
    Calculate Argument Diversity Score.
    
    Measures how diverse the arguments from different agents are.
    High diversity = agents bring different perspectives.
    
    Args:
        reasonings: List of reasoning texts from different agents
        
    Returns:
        Diversity score (0-1)
    """
    if not reasonings or len(reasonings) < 2:
        return 0.0
    
    # Tokenize all reasonings
    token_sets = [set(tokenize_vietnamese(r)) for r in reasonings if r]
    
    if len(token_sets) < 2:
        return 0.0
    
    # Calculate pairwise Jaccard distances (1 - similarity = diversity)
    diversities = []
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            if token_sets[i] and token_sets[j]:
                intersection = len(token_sets[i] & token_sets[j])
                union = len(token_sets[i] | token_sets[j])
                similarity = intersection / union if union > 0 else 0
                diversities.append(1 - similarity)
    
    return np.mean(diversities) if diversities else 0.0


def calculate_faithfulness_score(
    verdict: str,
    reasoning: str,
    verdict_correct: bool
) -> float:
    """
    Calculate Faithfulness Score.
    
    Measures whether the explanation (reasoning) faithfully reflects the decision process.
    High faithfulness = reasoning accurately explains why the verdict was chosen.
    
    Args:
        verdict: The final verdict
        reasoning: The explanation/reasoning
        verdict_correct: Whether the verdict was correct
        
    Returns:
        Faithfulness score (0-1)
    """
    if not reasoning:
        return 0.0
    
    # Base: verdict-reasoning match
    base_match = calculate_verdict_reasoning_match(verdict, reasoning)
    
    # Coherence factor
    coherence = calculate_reasoning_coherence(reasoning)
    
    # Correctness weight (faithful explanations should lead to correct verdicts)
    correctness_weight = 1.0 if verdict_correct else 0.7
    
    # Combined score
    faithfulness = (base_match * 0.5 + coherence * 0.3) * correctness_weight + 0.2 * correctness_weight
    
    return min(1.0, faithfulness)


def calculate_debate_metrics(
    round_1_verdicts: Dict[str, Dict[str, Any]],
    final_verdict: str,
    final_reasoning: str,
    rounds_used: int,
    extracted_evidences: List[Dict[str, Any]] = None,
    all_reasonings: List[str] = None,
    verdict_correct: bool = True
) -> DebateXAIMetrics:
    """
    Calculate all Debate XAI metrics.
    
    Args:
        round_1_verdicts: Dict of {agent_name: {verdict, confidence, reasoning}}
        final_verdict: Final verdict from Judge
        final_reasoning: Final reasoning from Judge
        rounds_used: Number of rounds used
        extracted_evidences: Evidence from Hunter (for utilization calc)
        all_reasonings: All reasoning texts from agents (for diversity calc)
        verdict_correct: Whether the final verdict was correct
        
    Returns:
        DebateXAIMetrics object
    """
    if not round_1_verdicts:
        return DebateXAIMetrics(rounds_used=rounds_used)
    
    # Extract round 1 verdicts
    r1_verdicts = [v.get("verdict", "NEI") for v in round_1_verdicts.values()]
    r1_agreement = calculate_agreement_ratio(r1_verdicts)
    
    # Final agreement (assume all agree with judge)
    final_agreement = 1.0 if rounds_used == 1 else 0.66  # Simplified
    
    # Consensus shift
    consensus_shift = calculate_consensus_shift(r1_agreement, final_agreement, rounds_used)
    
    # Verdict-reasoning match
    verdict_match = calculate_verdict_reasoning_match(final_verdict, final_reasoning)
    
    # Count flips (simplified: if round 1 majority != final, count as flip)
    r1_majority = Counter(r1_verdicts).most_common(1)[0][0] if r1_verdicts else "NEI"
    num_flips = 1 if r1_majority.upper() != final_verdict.upper() else 0
    
    # NEW: Advanced XAI metrics
    evidence_util = calculate_evidence_utilization(final_reasoning, extracted_evidences or [])
    reasoning_coherence = calculate_reasoning_coherence(final_reasoning)
    
    # Argument diversity from all agent reasonings
    if all_reasonings:
        arg_diversity = calculate_argument_diversity(all_reasonings)
    else:
        # Extract from round_1_verdicts if available
        r1_reasonings = [v.get("reasoning", "") for v in round_1_verdicts.values()]
        arg_diversity = calculate_argument_diversity(r1_reasonings)
    
    faithfulness = calculate_faithfulness_score(final_verdict, final_reasoning, verdict_correct)
    
    return DebateXAIMetrics(
        round_1_agreement=r1_agreement,
        final_agreement=final_agreement,
        consensus_shift=consensus_shift,
        verdict_reasoning_match=verdict_match,
        num_flips=num_flips,
        rounds_used=rounds_used,
        evidence_utilization=evidence_util,
        reasoning_coherence=reasoning_coherence,
        argument_diversity=arg_diversity,
        faithfulness_score=faithfulness
    )


# =============================================================================
# AGGREGATE METRICS
# =============================================================================

@dataclass
class XAIAggregateMetrics:
    """Aggregated XAI metrics across all samples (Debate only - Gold Evidence mode)."""
    total_samples: int = 0
    
    # Debate aggregates
    avg_round_1_agreement: float = 0.0
    consensus_distribution: Dict[int, int] = field(default_factory=dict)  # {0: count, 1: count, 2: count}
    avg_verdict_match: float = 0.0
    flip_rate: float = 0.0
    avg_rounds: float = 0.0
    avg_evidence_utilization: float = 0.0
    avg_reasoning_coherence: float = 0.0
    avg_argument_diversity: float = 0.0
    avg_faithfulness: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "debate": {
                "avg_round_1_agreement": round(self.avg_round_1_agreement, 4),
                "consensus_distribution": self.consensus_distribution,
                "avg_verdict_match": round(self.avg_verdict_match, 4),
                "flip_rate": round(self.flip_rate, 4),
                "avg_rounds": round(self.avg_rounds, 2),
                "avg_evidence_utilization": round(self.avg_evidence_utilization, 4),
                "avg_reasoning_coherence": round(self.avg_reasoning_coherence, 4),
                "avg_argument_diversity": round(self.avg_argument_diversity, 4),
                "avg_faithfulness": round(self.avg_faithfulness, 4)
            }
        }


def aggregate_xai_metrics(samples: List[XAISampleMetrics]) -> XAIAggregateMetrics:
    """Aggregate XAI metrics across all samples (Debate only)."""
    if not samples:
        return XAIAggregateMetrics()
    
    n = len(samples)
    
    # Debate aggregates only
    avg_r1_agreement = np.mean([s.debate_metrics.round_1_agreement for s in samples])
    consensus_dist = Counter([s.debate_metrics.consensus_shift for s in samples])
    avg_verdict_match = np.mean([s.debate_metrics.verdict_reasoning_match for s in samples])
    flip_rate = sum(1 for s in samples if s.debate_metrics.num_flips > 0) / n
    avg_rounds = np.mean([s.debate_metrics.rounds_used for s in samples])
    avg_evidence_util = np.mean([s.debate_metrics.evidence_utilization for s in samples])
    avg_coherence = np.mean([s.debate_metrics.reasoning_coherence for s in samples])
    avg_diversity = np.mean([s.debate_metrics.argument_diversity for s in samples])
    avg_faithfulness = np.mean([s.debate_metrics.faithfulness_score for s in samples])
    
    return XAIAggregateMetrics(
        total_samples=n,
        avg_round_1_agreement=avg_r1_agreement,
        consensus_distribution=dict(consensus_dist),
        avg_verdict_match=avg_verdict_match,
        flip_rate=flip_rate,
        avg_rounds=avg_rounds,
        avg_evidence_utilization=avg_evidence_util,
        avg_reasoning_coherence=avg_coherence,
        avg_argument_diversity=avg_diversity,
        avg_faithfulness=avg_faithfulness
    )


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_consensus_distribution(metrics: XAIAggregateMetrics, output_path: str):
    """Plot debate drama/consensus distribution pie chart."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    labels = ['Easy (Instant)', 'Moderate (Some Debate)', 'Dramatic (Major Shifts)']
    sizes = [
        metrics.consensus_distribution.get(0, 0),
        metrics.consensus_distribution.get(1, 0),
        metrics.consensus_distribution.get(2, 0)
    ]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    explode = (0.02, 0.05, 0.08)
    
    # Handle case where all sizes are 0
    if sum(sizes) == 0:
        sizes = [1, 0, 0]  # Default to easy
    
    # Filter out zero values to avoid cluttered labels
    non_zero = [(l, s, c, e) for l, s, c, e in zip(labels, sizes, colors, explode) if s > 0]
    if non_zero:
        labels_nz, sizes_nz, colors_nz, explode_nz = zip(*non_zero)
    else:
        labels_nz, sizes_nz, colors_nz, explode_nz = labels[:1], [1], colors[:1], explode[:1]
    
    wedges, texts, autotexts = ax.pie(
        sizes_nz, explode=explode_nz, labels=labels_nz, colors=colors_nz,
        autopct='%1.1f%%', shadow=True, startangle=90,
        textprops={'fontsize': 11},
        pctdistance=0.6,  # Move percentage closer to center
        labeldistance=1.15  # Move labels further out
    )
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    
    ax.set_title('Debate "Drama" Index\n(Consensus Shift Distribution)', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved consensus distribution chart to {output_path}")
    return fig


def plot_xai_radar(metrics: XAIAggregateMetrics, output_path: str):
    """Plot XAI quality radar chart (Debate metrics only)."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    categories = [
        'Verdict\nMatch',
        'Evidence\nUtilization', 
        'Reasoning\nCoherence',
        'Round 1\nAgreement',
        'Debate\nEfficiency'
    ]
    
    # Normalize values to 0-1 (max_rounds = 2 in simplified system)
    max_rounds = 2
    values = [
        metrics.avg_verdict_match,  # Reasoning quality
        metrics.avg_evidence_utilization,  # Evidence usage
        metrics.avg_reasoning_coherence,  # Coherence
        metrics.avg_round_1_agreement,  # Initial consensus
        1.0 - (metrics.avg_rounds - 1) / (max_rounds - 1) if max_rounds > 1 else 1.0  # Efficiency
    ]
    
    # Close the radar chart
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    ax.fill(angles, values, color='#3498db', alpha=0.25)
    ax.plot(angles, values, color='#2980b9', linewidth=2, marker='o', markersize=8)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Debate XAI Quality Radar\n(Gold Evidence Mode)', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved XAI radar chart to {output_path}")
    return fig


def plot_xai_comprehensive(metrics: XAIAggregateMetrics, output_path: str):
    """
    Plot comprehensive XAI metrics dashboard (Debate only - Gold Evidence mode).
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # --- Subplot 1: Debate XAI Metrics ---
    ax1 = axes[0, 0]
    debate_labels = ['Verdict\nMatch', 'Evidence\nUtil.', 'Coherence', 'Diversity', 'Faithful.']
    debate_values = [
        metrics.avg_verdict_match,
        metrics.avg_evidence_utilization,
        metrics.avg_reasoning_coherence,
        metrics.avg_argument_diversity,
        metrics.avg_faithfulness
    ]
    colors1 = ['#27ae60', '#3498db', '#9b59b6', '#e67e22', '#e74c3c']
    
    bars1 = ax1.bar(debate_labels, debate_values, color=colors1, edgecolor='black', linewidth=1.2)
    for bar, val in zip(bars1, debate_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Debate: Reasoning Quality Metrics', fontsize=12, fontweight='bold')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(axis='y', alpha=0.3)
    
    # --- Subplot 2: Consensus Metrics ---
    ax2 = axes[0, 1]
    # max_rounds = 2 in simplified system
    max_rounds = 2
    consensus_labels = ['R1\nAgreement', 'Flip\nRate', 'Avg\nRounds']
    consensus_values = [
        metrics.avg_round_1_agreement,
        metrics.flip_rate,
        metrics.avg_rounds / max_rounds  # Normalize to 0-1
    ]
    colors2 = ['#2ecc71', '#e74c3c', '#3498db']
    
    bars2 = ax2.bar(consensus_labels, consensus_values, color=colors2, edgecolor='black', linewidth=1.2)
    for bar, val in zip(bars2, consensus_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title('Debate: Consensus Metrics', fontsize=12, fontweight='bold')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)
    
    # --- Subplot 3: XAI Radar ---
    ax3 = axes[1, 0]
    ax3.remove()
    ax3 = fig.add_subplot(2, 2, 3, polar=True)
    
    radar_categories = [
        'Verdict Match', 'Coherence', 'Diversity',
        'Faithfulness', 'Evidence Util.', 'R1 Agreement'
    ]
    radar_values = [
        metrics.avg_verdict_match,
        metrics.avg_reasoning_coherence,
        metrics.avg_argument_diversity,
        metrics.avg_faithfulness,
        metrics.avg_evidence_utilization,
        metrics.avg_round_1_agreement
    ]
    
    # Close the radar
    radar_values += radar_values[:1]
    angles = np.linspace(0, 2 * np.pi, len(radar_categories), endpoint=False).tolist()
    angles += angles[:1]
    
    ax3.fill(angles, radar_values, color='#2ecc71', alpha=0.25)
    ax3.plot(angles, radar_values, color='#27ae60', linewidth=2, marker='o', markersize=8)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(radar_categories, fontsize=9)
    ax3.set_ylim(0, 1)
    ax3.set_title('XAI Quality Radar (Gold Evidence Mode)', fontsize=12, fontweight='bold', pad=15)
    
    # --- Subplot 4: Summary Table ---
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary table
    table_data = [
        ['Metric Category', 'Score', 'Interpretation'],
        ['─' * 15, '─' * 8, '─' * 20],
        ['Verdict Match', f'{metrics.avg_verdict_match:.1%}', 'Reasoning-verdict consistency'],
        ['Faithfulness', f'{metrics.avg_faithfulness:.1%}', 'Explanation accuracy'],
        ['Reasoning Coherence', f'{metrics.avg_reasoning_coherence:.1%}', 'Logic structure'],
        ['Argument Diversity', f'{metrics.avg_argument_diversity:.1%}', 'Multi-perspective'],
        ['Evidence Utilization', f'{metrics.avg_evidence_utilization:.1%}', 'Evidence usage'],
        ['R1 Agreement', f'{metrics.avg_round_1_agreement:.1%}', 'Initial consensus'],
        ['─' * 15, '─' * 8, '─' * 20],
        ['Overall XAI Score', f'{np.mean([metrics.avg_verdict_match, metrics.avg_faithfulness, metrics.avg_reasoning_coherence, metrics.avg_evidence_utilization]):.1%}', 'Weighted average']
    ]
    
    table_text = '\n'.join([f'{row[0]:<20} {row[1]:<10} {row[2]}' for row in table_data])
    ax4.text(0.1, 0.9, table_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax4.set_title('XAI Metrics Summary (Gold Evidence Mode)', fontsize=12, fontweight='bold')
    
    plt.suptitle('Comprehensive XAI Evaluation Dashboard', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved comprehensive XAI chart to {output_path}")
    return fig


def plot_faithfulness_analysis(metrics: XAIAggregateMetrics, output_path: str):
    """
    Plot faithfulness analysis - key XAI metric (Debate only).
    Shows relationship between explanation quality and correctness.
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Faithfulness Components
    components = ['Verdict-Reasoning\nMatch', 'Reasoning\nCoherence', 'Evidence\nUtilization', 'Combined\nFaithfulness']
    values = [
        metrics.avg_verdict_match,
        metrics.avg_reasoning_coherence,
        metrics.avg_evidence_utilization,
        metrics.avg_faithfulness
    ]
    colors = ['#3498db', '#9b59b6', '#f39c12', '#27ae60']
    
    bars = ax.bar(components, values, color=colors, edgecolor='black', linewidth=1.2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Faithfulness Analysis (Gold Evidence Mode)', fontsize=14, fontweight='bold')
    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good threshold (70%)')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Acceptable (50%)')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Add explanation text
    explanation = (
        "Faithfulness measures how well the explanation reflects the actual decision process.\n"
        "High faithfulness = explanation accurately represents why the verdict was given."
    )
    ax.text(0.5, -0.12, explanation, transform=ax.transAxes, fontsize=9,
            ha='center', style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved faithfulness analysis chart to {output_path}")
    return fig


def generate_xai_report(
    samples: List[XAISampleMetrics],
    output_dir: str
) -> Dict[str, Any]:
    """
    Generate full XAI report with metrics and charts.
    
    Args:
        samples: List of XAI metrics per sample
        output_dir: Directory to save charts
        
    Returns:
        Dict with aggregate metrics and chart paths
    """
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Aggregate metrics
    agg_metrics = aggregate_xai_metrics(samples)
    
    # Generate charts
    charts = {}
    
    try:
        hunter_chart = output_path / "xai_hunter_performance.png"
        plot_hunter_performance(agg_metrics, str(hunter_chart))
        charts["hunter_performance"] = str(hunter_chart)
    except Exception as e:
        logger.warning(f"Could not generate Hunter chart: {e}")
    
    try:
        consensus_chart = output_path / "xai_consensus_distribution.png"
        plot_consensus_distribution(agg_metrics, str(consensus_chart))
        charts["consensus_distribution"] = str(consensus_chart)
    except Exception as e:
        logger.warning(f"Could not generate consensus chart: {e}")
    
    try:
        radar_chart = output_path / "xai_quality_radar.png"
        plot_xai_radar(agg_metrics, str(radar_chart))
        charts["quality_radar"] = str(radar_chart)
    except Exception as e:
        logger.warning(f"Could not generate radar chart: {e}")
    
    # NEW: Comprehensive XAI Dashboard
    try:
        comprehensive_chart = output_path / "xai_comprehensive_dashboard.png"
        plot_xai_comprehensive(agg_metrics, str(comprehensive_chart))
        charts["comprehensive_dashboard"] = str(comprehensive_chart)
    except Exception as e:
        logger.warning(f"Could not generate comprehensive XAI chart: {e}")
    
    # NEW: Faithfulness Analysis
    try:
        faithfulness_chart = output_path / "xai_faithfulness_analysis.png"
        plot_faithfulness_analysis(agg_metrics, str(faithfulness_chart))
        charts["faithfulness_analysis"] = str(faithfulness_chart)
    except Exception as e:
        logger.warning(f"Could not generate faithfulness chart: {e}")
    
    # Save metrics JSON
    metrics_file = output_path / "xai_metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump({
            "aggregate": agg_metrics.to_dict(),
            "samples": [s.to_dict() for s in samples]
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved XAI metrics to {metrics_file}")
    
    return {
        "aggregate": agg_metrics.to_dict(),
        "charts": charts,
        "metrics_file": str(metrics_file)
    }
