"""
Debate Metrics & Visualization Module

Cung cấp các công cụ để:
1. Track metrics qua các vòng debate
2. Visualize hiệu suất (Quantitative Analysis)
3. Case Study Analysis (Qualitative Analysis)

Author: Lockdown
Date: Nov 27, 2025
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RoundMetrics:
    """Metrics cho một round debate."""
    round_num: int
    verdicts: Dict[str, str]  # {debator_name: verdict}
    confidences: Dict[str, float]  # {debator_name: confidence}
    reasonings: Dict[str, str] = field(default_factory=dict)  # {debator_name: reasoning}
    roles: Dict[str, str] = field(default_factory=dict)  # {debator_name: role}
    agreement_ratio: float = 0.0  # Tỷ lệ đồng thuận (0-1)
    majority_verdict: str = ""
    verdict_changed_from_prev: bool = False
    # XAI Interaction fields (Round 2+)
    agree_with: Dict[str, List[str]] = field(default_factory=dict)  # {debator: [agents agreed with]}
    agree_reasons: Dict[str, str] = field(default_factory=dict)
    disagree_with: Dict[str, List[str]] = field(default_factory=dict)  # {debator: [agents disagreed with]}
    disagree_reasons: Dict[str, str] = field(default_factory=dict)
    changed: Dict[str, bool] = field(default_factory=dict)  # {debator: changed?}
    change_reasons: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DebateSampleMetrics:
    """Metrics cho một sample qua toàn bộ debate."""
    sample_id: str
    statement: str
    gold_label: str
    model_verdict: str
    model_confidence: float
    final_verdict: str
    final_confidence: float
    
    rounds_used: int
    early_stopped: bool
    stop_reason: str
    mvp_agent: str = "Unknown"
    
    round_metrics: List[RoundMetrics] = field(default_factory=list)
    
    # Hybrid Strategy fields (DOWN Framework, 2025)
    hybrid_verdict: str = None
    hybrid_source: str = None  # MODEL_HIGH_CONF | DEBATE
    hybrid_threshold: float = 0.85
    
    # Derived metrics
    model_correct: bool = False
    final_correct: bool = False
    hybrid_correct: bool = False  # Hybrid Strategy correct
    debate_flipped: bool = False  # Debate đã thay đổi verdict so với model
    debate_fixed: bool = False    # Debate sửa được lỗi của model
    debate_broke: bool = False    # Debate làm sai kết quả đúng của model
    hybrid_rescued: bool = False  # Hybrid cứu được sample mà Debate làm sai
    
    def __post_init__(self):
        self.model_correct = (self.model_verdict == self.gold_label)
        self.final_correct = (self.final_verdict == self.gold_label)
        self.debate_flipped = (self.model_verdict != self.final_verdict)
        self.debate_fixed = (not self.model_correct and self.final_correct)
        self.debate_broke = (self.model_correct and not self.final_correct)
        # Hybrid metrics
        if self.hybrid_verdict:
            self.hybrid_correct = (self.hybrid_verdict == self.gold_label)
            self.hybrid_rescued = (not self.final_correct and self.hybrid_correct)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['round_metrics'] = [rm.to_dict() for rm in self.round_metrics]
        return d


class DebateMetricsTracker:
    """
    Tracker để thu thập và phân tích metrics từ debate sessions.
    """
    
    def __init__(self):
        self.samples: List[DebateSampleMetrics] = []
    
    def add_sample(self, sample: DebateSampleMetrics):
        """Thêm một sample đã debate."""
        self.samples.append(sample)
    
    def add_from_result(
        self,
        sample_id: str,
        statement: str,
        gold_label: str,
        pipeline_result: Dict[str, Any],
        debate_history: Optional[List[List[Any]]] = None
    ):
        """
        Thêm sample từ kết quả pipeline.
        
        Args:
            sample_id: ID của sample
            statement: Statement/claim
            gold_label: Ground truth label
            pipeline_result: Kết quả từ ViFactCheckPipeline.predict()
            debate_history: Optional debate history từ orchestrator
        """
        round_metrics = []
        
        # Parse round metrics từ debate_history nếu có
        if debate_history:
            for round_num, round_args in enumerate(debate_history, 1):
                verdicts = {arg.debator_name: arg.verdict for arg in round_args}
                confidences = {arg.debator_name: arg.confidence for arg in round_args}
                # ✅ FIX: Extract reasonings and roles for XAI display
                reasonings = {arg.debator_name: arg.reasoning for arg in round_args if hasattr(arg, 'reasoning') and arg.reasoning}
                roles = {arg.debator_name: arg.role for arg in round_args if hasattr(arg, 'role') and arg.role}
                
                verdict_counts = Counter(verdicts.values())
                majority_verdict = verdict_counts.most_common(1)[0][0]
                agreement_ratio = max(verdict_counts.values()) / len(verdicts)
                
                # Check if verdict changed from previous round
                changed = False
                if round_num > 1 and len(round_metrics) > 0:
                    prev_verdicts = round_metrics[-1].verdicts
                    changed = any(
                        verdicts.get(name) != prev_verdicts.get(name)
                        for name in verdicts
                    )
                
                round_metrics.append(RoundMetrics(
                    round_num=round_num,
                    verdicts=verdicts,
                    confidences=confidences,
                    reasonings=reasonings,  # ✅ FIX: Now includes XAI
                    roles=roles,            # ✅ FIX: Now includes roles
                    agreement_ratio=agreement_ratio,
                    majority_verdict=majority_verdict,
                    verdict_changed_from_prev=changed
                ))
        
        sample = DebateSampleMetrics(
            sample_id=sample_id,
            statement=statement[:200],
            gold_label=gold_label,
            model_verdict=pipeline_result.get('model_verdict', 'NEI'),
            model_confidence=pipeline_result.get('model_confidence', 0.0),
            final_verdict=pipeline_result.get('final_verdict', 'NEI'),
            final_confidence=pipeline_result.get('debate_confidence', 0.0),
            rounds_used=len(round_metrics) if round_metrics else 1,
            early_stopped=pipeline_result.get('early_stopped', False),
            stop_reason=pipeline_result.get('stop_reason', ''),
            round_metrics=round_metrics,
            # Hybrid Strategy fields (DOWN Framework, 2025)
            hybrid_verdict=pipeline_result.get('hybrid_verdict'),
            hybrid_source=pipeline_result.get('hybrid_source'),
            hybrid_threshold=pipeline_result.get('hybrid_threshold', 0.85)
        )
        
        self.samples.append(sample)
    
    def get_summary(self) -> Dict[str, Any]:
        """Tính toán summary statistics."""
        if not self.samples:
            return {}
        
        total = len(self.samples)
        
        # Accuracy metrics
        model_correct = sum(1 for s in self.samples if s.model_correct)
        final_correct = sum(1 for s in self.samples if s.final_correct)
        
        # Hybrid Strategy metrics (DOWN Framework, 2025)
        hybrid_correct = sum(1 for s in self.samples if s.hybrid_correct)
        hybrid_rescued = sum(1 for s in self.samples if s.hybrid_rescued)
        hybrid_from_model = sum(1 for s in self.samples if s.hybrid_source == "MODEL_HIGH_CONF")
        hybrid_from_debate = sum(1 for s in self.samples if s.hybrid_source == "DEBATE")
        
        # Debate impact
        flipped = sum(1 for s in self.samples if s.debate_flipped)
        fixed = sum(1 for s in self.samples if s.debate_fixed)
        broke = sum(1 for s in self.samples if s.debate_broke)
        
        # Round distribution
        round_counts = Counter(s.rounds_used for s in self.samples)
        
        # Early stop reasons
        stop_reasons = Counter(s.stop_reason for s in self.samples if s.early_stopped)
        
        return {
            "total_samples": total,
            "model_accuracy": model_correct / total,
            "final_accuracy": final_correct / total,  # Debate only
            "hybrid_accuracy": hybrid_correct / total,  # Hybrid Strategy
            "debate_gain": (final_correct - model_correct) / total,
            "debate_gain_absolute": final_correct - model_correct,
            "hybrid_gain": (hybrid_correct - model_correct) / total,
            "hybrid_gain_absolute": hybrid_correct - model_correct,
            
            # Hybrid distribution
            "hybrid_from_model": hybrid_from_model,
            "hybrid_from_debate": hybrid_from_debate,
            "hybrid_rescued_count": hybrid_rescued,  # Samples Hybrid cứu được từ Debate sai
            
            "flipped_count": flipped,
            "flipped_ratio": flipped / total,
            "fixed_count": fixed,
            "fixed_ratio": fixed / total,
            "broke_count": broke,
            "broke_ratio": broke / total,
            
            "round_distribution": dict(round_counts),
            "avg_rounds": np.mean([s.rounds_used for s in self.samples]),
            
            "early_stop_reasons": dict(stop_reasons),
            "early_stop_ratio": sum(1 for s in self.samples if s.early_stopped) / total
        }
    
    def get_case_studies(self, n: int = 5) -> Dict[str, List[DebateSampleMetrics]]:
        """
        Lấy các case study điển hình.
        
        Returns:
            Dict với các loại case study:
            - "fixed": Debate sửa được lỗi của model
            - "broke": Debate làm sai kết quả đúng
            - "high_disagreement": Có nhiều bất đồng giữa debators
        """
        fixed_cases = [s for s in self.samples if s.debate_fixed]
        broke_cases = [s for s in self.samples if s.debate_broke]
        
        # High disagreement: samples có agreement_ratio thấp ở round cuối
        high_disagreement = []
        for s in self.samples:
            if s.round_metrics:
                last_round = s.round_metrics[-1]
                if last_round.agreement_ratio < 0.67:  # < 2/3 đồng thuận
                    high_disagreement.append(s)
        
        return {
            "fixed": fixed_cases[:n],
            "broke": broke_cases[:n],
            "high_disagreement": high_disagreement[:n]
        }
    
    def plot_round_distribution(self, save_path: Optional[str] = None, max_rounds: int = 2) -> plt.Figure:
        """
        Vẽ biểu đồ phân bố số mẫu có dữ liệu tại mỗi round.
        
        Args:
            max_rounds: Maximum number of rounds (default 2 for simplified system)
        """
        total_samples = len(self.samples)
        
        # Count samples that have data at each round (dynamic based on max_rounds)
        round_counts = {r: 0 for r in range(1, max_rounds + 1)}
        for s in self.samples:
            for rm in s.round_metrics:
                if rm.round_num in round_counts:
                    round_counts[rm.round_num] += 1
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Dynamic rounds based on max_rounds
        all_rounds = list(range(1, max_rounds + 1))
        counts = [round_counts.get(r, 0) for r in all_rounds]
        colors = ['#2ecc71', '#f39c12', '#e74c3c'][:max_rounds]
        
        bars = ax.bar(all_rounds, counts, color=colors, edgecolor='black', linewidth=1.2)
        
        # Add value labels on bars
        max_count = max(counts) if max(counts) > 0 else 1
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            pct = count / total_samples * 100 if total_samples > 0 else 0
            label = f'{count}\n({pct:.1f}%)'
            
            # Position label above bar with enough margin
            y_pos = height + max_count * 0.05
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos, label,
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Debate Round', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_title('Samples Processed at Each Round', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(all_rounds)
        ax.set_xticklabels([f'Round {r}' for r in all_rounds])
        
        # Set y-limit with extra margin for labels
        ax.set_ylim(0, max_count * 1.35)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved round distribution plot to {save_path}")
        
        return fig
    
    def plot_accuracy_by_round(self, save_path: Optional[str] = None, max_rounds: int = 2) -> plt.Figure:
        """
        Vẽ biểu đồ accuracy của Majority Vote tại mỗi round.
        
        Args:
            max_rounds: Maximum number of rounds (default 2 for simplified system)
        """
        # Calculate majority vote accuracy at each round (dynamic)
        round_stats = {r: {'correct': 0, 'total': 0} for r in range(1, max_rounds + 1)}
        
        for s in self.samples:
            gold = s.gold_label
            for rm in s.round_metrics:
                r = rm.round_num
                if r in round_stats:
                    round_stats[r]['total'] += 1
                    # Check if majority verdict matches gold
                    majority = rm.majority_verdict
                    if majority:
                        # Normalize verdict
                        if majority.upper() in ['SUPPORTED', 'SUPPORT']:
                            m_norm = 'Support'
                        elif majority.upper() in ['REFUTED', 'REFUTE']:
                            m_norm = 'Refute'
                        else:
                            m_norm = 'NOT_ENOUGH_INFO'
                        if m_norm == gold:
                            round_stats[r]['correct'] += 1
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Dynamic rounds based on max_rounds
        all_rounds = list(range(1, max_rounds + 1))
        round_acc = []
        sample_counts = []
        
        for r in all_rounds:
            if round_stats[r]['total'] > 0:
                round_acc.append(round_stats[r]['correct'] / round_stats[r]['total'])
                sample_counts.append(round_stats[r]['total'])
            else:
                round_acc.append(0)
                sample_counts.append(0)
        
        x = np.arange(len(all_rounds))
        colors = ['#2ecc71', '#f39c12', '#e74c3c'][:max_rounds]
        
        bars = ax.bar(x, round_acc, color=colors, edgecolor='black', linewidth=1.2)
        
        ax.set_xlabel('Debate Round', fontsize=12)
        ax.set_ylabel('Majority Vote Accuracy', fontsize=12)
        ax.set_title('Majority Vote Accuracy at Each Round', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Round {r}' for r in all_rounds])
        ax.set_ylim(0, 1.25)  # Extra space for labels
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels with sample count
        for bar, acc, count in zip(bars, round_acc, sample_counts):
            height = bar.get_height()
            label = f'{acc:.1%}\n(n={count})'
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.03,
                   label, ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved accuracy by round plot to {save_path}")
        
        return fig
    
    def plot_accuracy_progression(self, save_path: Optional[str] = None, max_rounds: int = 2) -> plt.Figure:
        """
        Vẽ biểu đồ tiến triển accuracy qua từng round.
        
        Args:
            max_rounds: Maximum number of rounds (default 2 for simplified system)
        """
        # Calculate accuracy at each round (dynamic)
        round_accuracies = {r: {'correct': 0, 'total': 0} for r in range(1, max_rounds + 1)}
        model_correct_count = 0
        final_correct_count = 0
        total_samples = len(self.samples)
        
        for s in self.samples:
            gold = s.gold_label
            
            # Model accuracy
            if s.model_correct:
                model_correct_count += 1
            
            # Final accuracy
            if s.final_correct:
                final_correct_count += 1
            
            # Round-by-round majority vote accuracy
            for rm in s.round_metrics:
                r_num = rm.round_num
                if r_num in round_accuracies:
                    round_accuracies[r_num]['total'] += 1
                    # Check if majority verdict matches gold
                    if rm.majority_verdict.upper() in ['SUPPORTED', 'SUPPORT']:
                        majority_normalized = 'Support'
                    elif rm.majority_verdict.upper() in ['REFUTED', 'REFUTE']:
                        majority_normalized = 'Refute'
                    else:
                        majority_normalized = 'NOT_ENOUGH_INFO'
                    
                    if majority_normalized == gold:
                        round_accuracies[r_num]['correct'] += 1
        
        # Build data for plotting
        labels = ['Model\n(Baseline)']
        accuracies = [model_correct_count / total_samples if total_samples > 0 else 0]
        
        for r in range(1, max_rounds + 1):
            if round_accuracies[r]['total'] > 0:
                acc = round_accuracies[r]['correct'] / round_accuracies[r]['total']
                labels.append(f'Round {r}\n(Majority)')
                accuracies.append(acc)
        
        labels.append('Final\n(Judge)')
        accuracies.append(final_correct_count / total_samples if total_samples > 0 else 0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_pos = range(len(labels))
        
        # Line plot with markers
        ax.plot(x_pos, accuracies, marker='o', linewidth=3, markersize=12, color='#2ecc71', label='Accuracy')
        ax.fill_between(x_pos, 0, accuracies, alpha=0.2, color='#2ecc71')
        
        # Add value labels
        for i, (label, acc) in enumerate(zip(labels, accuracies)):
            ax.text(i, acc + 0.03, f'{acc:.1%}', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold')
        
        # Highlight improvement from Model to Final
        if len(accuracies) >= 2:
            improvement = accuracies[-1] - accuracies[0]
            mid_x = (len(labels) - 1) / 2
            mid_y = (accuracies[0] + accuracies[-1]) / 2
            
            color = '#27ae60' if improvement >= 0 else '#e74c3c'
            ax.annotate('', xy=(len(labels)-1, accuracies[-1]), xytext=(0, accuracies[0]),
                       arrowprops=dict(arrowstyle='<->', color=color, lw=2))
            ax.text(mid_x, mid_y + 0.05, f'{improvement:+.1%}', 
                   fontsize=14, fontweight='bold', color=color, ha='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Formatting
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Accuracy Progression Across Debate Rounds', fontsize=14, fontweight='bold', pad=15)
        ax.set_ylim(0, 1.25)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved accuracy progression plot to {save_path}")
        
        return fig
    
    def plot_debate_impact(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Vẽ biểu đồ tác động của debate (Fixed vs Broke).
        """
        summary = self.get_summary()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = ['Model\nCorrect', 'Debate\nFixed', 'Debate\nBroke', 'Final\nCorrect']
        values = [
            summary['model_accuracy'],
            summary['fixed_ratio'],
            summary['broke_ratio'],
            summary['final_accuracy']
        ]
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        
        bars = ax.bar(categories, values, color=colors)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val*100:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Ratio', fontsize=12)
        ax.set_title('Debate Impact Analysis', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.0)
        
        # Add debate gain annotation
        gain = summary['debate_gain']
        gain_text = f"Debate Gain: {'+' if gain >= 0 else ''}{gain*100:.2f}%"
        ax.text(0.5, 0.95, gain_text, transform=ax.transAxes, 
                fontsize=14, fontweight='bold', ha='center',
                color='green' if gain >= 0 else 'red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved debate impact plot to {save_path}")
        
        return fig
    
    def plot_consensus_heatmap(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Vẽ heatmap ma trận đồng thuận giữa các cặp agent.
        Cho thấy agent nào thường đồng ý với nhau.
        """
        # Collect all agent names
        agent_names = set()
        for s in self.samples:
            for rm in s.round_metrics:
                agent_names.update(rm.verdicts.keys())
        
        agent_names = sorted(list(agent_names))
        n_agents = len(agent_names)
        
        if n_agents < 2:
            logger.warning("Not enough agents for consensus heatmap")
            return None
        
        # Build agreement matrix
        agreement_matrix = np.zeros((n_agents, n_agents))
        count_matrix = np.zeros((n_agents, n_agents))
        
        for s in self.samples:
            for rm in s.round_metrics:
                verdicts = rm.verdicts
                for i, agent_i in enumerate(agent_names):
                    for j, agent_j in enumerate(agent_names):
                        if agent_i in verdicts and agent_j in verdicts:
                            count_matrix[i, j] += 1
                            if verdicts[agent_i] == verdicts[agent_j]:
                                agreement_matrix[i, j] += 1
        
        # Calculate agreement ratio
        with np.errstate(divide='ignore', invalid='ignore'):
            agreement_ratio = np.where(count_matrix > 0, 
                                       agreement_matrix / count_matrix, 0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Shorten agent names for display
        short_names = [name.split('/')[-1][:12] if '/' in name else name[:12] 
                      for name in agent_names]
        
        # Create heatmap
        im = ax.imshow(agreement_ratio, cmap='RdYlGn', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Agreement Ratio', rotation=-90, va="bottom", fontsize=11)
        
        # Set ticks
        ax.set_xticks(np.arange(n_agents))
        ax.set_yticks(np.arange(n_agents))
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(short_names, fontsize=10)
        
        # Add text annotations
        for i in range(n_agents):
            for j in range(n_agents):
                text = ax.text(j, i, f'{agreement_ratio[i, j]:.2f}',
                              ha='center', va='center', fontsize=11, fontweight='bold',
                              color='white' if agreement_ratio[i, j] < 0.5 else 'black')
        
        ax.set_title('Inter-Agent Agreement Matrix', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Agent', fontsize=12)
        ax.set_ylabel('Agent', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved consensus heatmap to {save_path}")
        
        return fig
    
    def plot_confidence_calibration(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Vẽ scatter plot confidence vs accuracy để đánh giá calibration.
        Agent có confidence cao có thực sự chính xác hơn không?
        """
        # Collect confidence and correctness data
        data_points = []  # (confidence, is_correct, agent_name)
        
        for s in self.samples:
            gold = s.gold_label
            for rm in s.round_metrics:
                for agent_name, verdict in rm.verdicts.items():
                    conf = rm.confidences.get(agent_name, 0.5)
                    # Normalize verdict
                    if verdict.upper() in ['SUPPORTED', 'SUPPORT']:
                        v_norm = 'Support'
                    elif verdict.upper() in ['REFUTED', 'REFUTE']:
                        v_norm = 'Refute'
                    else:
                        v_norm = 'NOT_ENOUGH_INFO'
                    
                    is_correct = 1 if v_norm == gold else 0
                    data_points.append((conf, is_correct, agent_name))
        
        if not data_points:
            logger.warning("No data for confidence calibration plot")
            return None
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # --- Subplot 1: Binned Calibration Curve ---
        ax1 = axes[0]
        
        # Bin confidences
        bins = np.linspace(0, 1, 11)  # 10 bins
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_accuracies = []
        bin_counts = []
        
        for i in range(len(bins) - 1):
            bin_data = [d for d in data_points if bins[i] <= d[0] < bins[i+1]]
            if bin_data:
                acc = sum(d[1] for d in bin_data) / len(bin_data)
                bin_accuracies.append(acc)
                bin_counts.append(len(bin_data))
            else:
                bin_accuracies.append(None)
                bin_counts.append(0)
        
        # Plot calibration curve
        valid_centers = [c for c, a in zip(bin_centers, bin_accuracies) if a is not None]
        valid_accs = [a for a in bin_accuracies if a is not None]
        
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.7)
        ax1.plot(valid_centers, valid_accs, 'o-', color='#3498db', linewidth=2, 
                markersize=10, label='Actual Calibration')
        ax1.fill_between(valid_centers, valid_accs, [c for c in valid_centers], 
                        alpha=0.2, color='#3498db')
        
        ax1.set_xlabel('Confidence', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Confidence Calibration Curve', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=10)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.grid(alpha=0.3)
        
        # --- Subplot 2: Per-Agent Calibration ---
        ax2 = axes[1]
        
        # Group by agent
        agent_data = {}
        for conf, is_correct, agent in data_points:
            short_name = agent.split('/')[-1][:15] if '/' in agent else agent[:15]
            if short_name not in agent_data:
                agent_data[short_name] = {'confs': [], 'corrects': []}
            agent_data[short_name]['confs'].append(conf)
            agent_data[short_name]['corrects'].append(is_correct)
        
        # Calculate per-agent stats
        agents = []
        avg_confs = []
        accuracies = []
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
        
        for i, (agent, data) in enumerate(agent_data.items()):
            agents.append(agent)
            avg_confs.append(np.mean(data['confs']))
            accuracies.append(np.mean(data['corrects']))
        
        # Scatter plot with better label positioning
        for i, (agent, conf, acc) in enumerate(zip(agents, avg_confs, accuracies)):
            ax2.scatter(conf, acc, s=200, c=colors[i % len(colors)], 
                       label=agent, edgecolors='black', linewidth=1.5)
        
        # Add legend instead of overlapping annotations
        ax2.legend(loc='lower right', fontsize=9, framealpha=0.9)
        
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax2.set_xlabel('Average Confidence', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Per-Agent Calibration', fontsize=14, fontweight='bold')
        ax2.set_xlim(0.4, 1.05)
        ax2.set_ylim(0.4, 1.05)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved confidence calibration plot to {save_path}")
        
        return fig
    
    def plot_verdict_flow(self, save_path: Optional[str] = None, max_rounds: int = 2) -> plt.Figure:
        """
        Vẽ stacked bar chart hiển thị phân bố verdict qua các giai đoạn.
        
        Args:
            max_rounds: Maximum number of rounds (default 2 for simplified system)
        """
        # Collect verdict distributions at each stage (dynamic)
        stages = ['Model'] + [f'Round {r}' for r in range(1, max_rounds + 1)] + ['Final']
        verdict_types = ['Support', 'Refute', 'NEI']
        
        # Initialize counts
        stage_counts = {stage: {'Support': 0, 'Refute': 0, 'NEI': 0} for stage in stages}
        
        for s in self.samples:
            # Model verdict
            model_v = self._normalize_verdict(s.model_verdict)
            stage_counts['Model'][model_v] += 1
            
            # Round verdicts (majority)
            for rm in s.round_metrics:
                round_key = f'Round {rm.round_num}'
                if round_key in stage_counts:
                    majority_v = self._normalize_verdict(rm.majority_verdict)
                    stage_counts[round_key][majority_v] += 1
            
            # Final verdict
            final_v = self._normalize_verdict(s.final_verdict)
            stage_counts['Final'][final_v] += 1
        
        # Filter out empty stages
        active_stages = [s for s in stages if sum(stage_counts[s].values()) > 0]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(active_stages))
        width = 0.6
        
        # Colors for each verdict type
        colors = {'Support': '#2ecc71', 'Refute': '#e74c3c', 'NEI': '#f39c12'}
        
        # Create stacked bars
        bottom = np.zeros(len(active_stages))
        
        for verdict_type in verdict_types:
            counts = [stage_counts[s][verdict_type] for s in active_stages]
            bars = ax.bar(x, counts, width, label=verdict_type, bottom=bottom, 
                         color=colors[verdict_type], edgecolor='black', linewidth=0.5)
            
            # Add count labels
            for i, (bar, count) in enumerate(zip(bars, counts)):
                if count > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, bottom[i] + height/2,
                           f'{count}', ha='center', va='center', fontsize=10, 
                           fontweight='bold', color='white')
            
            bottom += counts
        
        ax.set_xlabel('Stage', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_title('Verdict Distribution Flow Across Stages', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(active_stages, fontsize=11)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved verdict flow plot to {save_path}")
        
        return fig
    
    def _normalize_verdict(self, verdict: str) -> str:
        """Normalize verdict to standard format."""
        if not verdict:
            return 'NEI'
        v_upper = verdict.upper()
        if v_upper in ['SUPPORTED', 'SUPPORT']:
            return 'Support'
        elif v_upper in ['REFUTED', 'REFUTE']:
            return 'Refute'
        else:
            return 'NEI'
    
    def plot_agent_performance(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Vẽ bar chart so sánh hiệu suất của từng agent.
        Bao gồm: Accuracy, Avg Confidence, Win Rate (khi có bất đồng).
        """
        # Collect per-agent stats
        agent_stats = {}
        
        for s in self.samples:
            gold = s.gold_label
            for rm in s.round_metrics:
                for agent_name, verdict in rm.verdicts.items():
                    short_name = agent_name.split('/')[-1][:15] if '/' in agent_name else agent_name[:15]
                    
                    if short_name not in agent_stats:
                        agent_stats[short_name] = {
                            'correct': 0, 'total': 0,
                            'confidences': [],
                            'wins': 0, 'disagreements': 0
                        }
                    
                    # Accuracy
                    v_norm = self._normalize_verdict(verdict)
                    agent_stats[short_name]['total'] += 1
                    if v_norm == gold:
                        agent_stats[short_name]['correct'] += 1
                    
                    # Confidence
                    conf = rm.confidences.get(agent_name, 0.5)
                    agent_stats[short_name]['confidences'].append(conf)
                    
                    # Win rate (when there's disagreement and agent is correct)
                    if rm.agreement_ratio < 1.0:  # There's disagreement
                        agent_stats[short_name]['disagreements'] += 1
                        if v_norm == gold:
                            agent_stats[short_name]['wins'] += 1
        
        if not agent_stats:
            logger.warning("No agent data for performance plot")
            return None
        
        # Calculate metrics
        agents = list(agent_stats.keys())
        accuracies = [agent_stats[a]['correct'] / agent_stats[a]['total'] 
                     if agent_stats[a]['total'] > 0 else 0 for a in agents]
        avg_confs = [np.mean(agent_stats[a]['confidences']) 
                    if agent_stats[a]['confidences'] else 0 for a in agents]
        win_rates = [agent_stats[a]['wins'] / agent_stats[a]['disagreements'] 
                    if agent_stats[a]['disagreements'] > 0 else 0 for a in agents]
        
        # Create figure with more width for agent names
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
        x = np.arange(len(agents))
        
        # --- Subplot 1: Accuracy ---
        ax1 = axes[0]
        bars1 = ax1.bar(x, accuracies, color=[colors[i % len(colors)] for i in range(len(agents))],
                       edgecolor='black', linewidth=1.2)
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Agent Accuracy', fontsize=13, fontweight='bold')
        ax1.set_ylim(0, 1.2)
        ax1.set_xticks(x)
        ax1.set_xticklabels(agents, rotation=45, ha='right', fontsize=9)
        ax1.grid(axis='y', alpha=0.3)
        
        # --- Subplot 2: Average Confidence ---
        ax2 = axes[1]
        bars2 = ax2.bar(x, avg_confs, color=[colors[i % len(colors)] for i in range(len(agents))],
                       edgecolor='black', linewidth=1.2)
        for bar, conf in zip(bars2, avg_confs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{conf:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Avg Confidence', fontsize=12)
        ax2.set_title('Agent Confidence', fontsize=13, fontweight='bold')
        ax2.set_ylim(0, 1.2)
        ax2.set_xticks(x)
        ax2.set_xticklabels(agents, rotation=45, ha='right', fontsize=9)
        ax2.grid(axis='y', alpha=0.3)
        
        # --- Subplot 3: Win Rate (when disagreement) ---
        ax3 = axes[2]
        bars3 = ax3.bar(x, win_rates, color=[colors[i % len(colors)] for i in range(len(agents))],
                       edgecolor='black', linewidth=1.2)
        for bar, wr in zip(bars3, win_rates):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{wr:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax3.set_ylabel('Win Rate', fontsize=12)
        ax3.set_title('Win Rate (When Disagreement)', fontsize=13, fontweight='bold')
        ax3.set_ylim(0, 1.2)
        ax3.set_xticks(x)
        ax3.set_xticklabels(agents, rotation=45, ha='right', fontsize=9)
        ax3.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Agent Performance Comparison', fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved agent performance plot to {save_path}")
        
        return fig
    
    def plot_error_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Vẽ pie chart và bar chart phân tích lỗi.
        Phân loại: Fixed, Broke, Consistent Correct, Consistent Wrong.
        """
        # Count categories
        fixed = sum(1 for s in self.samples if s.debate_fixed)
        broke = sum(1 for s in self.samples if s.debate_broke)
        consistent_correct = sum(1 for s in self.samples if s.model_correct and s.final_correct)
        consistent_wrong = sum(1 for s in self.samples if not s.model_correct and not s.final_correct)
        
        total = len(self.samples)
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        categories = ['Consistent\nCorrect', 'Fixed by\nDebate', 'Broke by\nDebate', 'Consistent\nWrong']
        counts = [consistent_correct, fixed, broke, consistent_wrong]
        colors = ['#27ae60', '#3498db', '#e74c3c', '#95a5a6']
        
        # --- Subplot 1: Pie Chart ---
        ax1 = axes[0]
        
        # Filter out zero values for pie chart
        non_zero_cats = [(c, cnt, col) for c, cnt, col in zip(categories, counts, colors) if cnt > 0]
        if non_zero_cats:
            pie_cats, pie_counts, pie_colors = zip(*non_zero_cats)
            wedges, texts, autotexts = ax1.pie(pie_counts, labels=pie_cats, colors=pie_colors,
                                               autopct='%1.1f%%', startangle=90,
                                               explode=[0.02] * len(pie_counts),
                                               textprops={'fontsize': 11})
            for autotext in autotexts:
                autotext.set_fontweight('bold')
        
        ax1.set_title('Error Distribution', fontsize=14, fontweight='bold')
        
        # --- Subplot 2: Bar Chart with counts ---
        ax2 = axes[1]
        
        x = np.arange(len(categories))
        bars = ax2.bar(x, counts, color=colors, edgecolor='black', linewidth=1.2)
        
        for bar, count in zip(bars, counts):
            pct = count / total * 100 if total > 0 else 0
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{count}\n({pct:.1f}%)', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold')
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories, fontsize=10)
        ax2.set_ylabel('Number of Samples', fontsize=12)
        ax2.set_title('Error Analysis by Category', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, max(counts) * 1.25 if counts else 1)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add summary text
        summary_text = f"Total: {total} | Model Acc: {(consistent_correct + broke)/total*100:.1f}% | Final Acc: {(consistent_correct + fixed)/total*100:.1f}%"
        fig.text(0.5, 0.02, summary_text, ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved error analysis plot to {save_path}")
        
        return fig
    
    def _format_debator_table(self, rm: RoundMetrics) -> List[str]:
        """Format debator details as a table for case study report."""
        lines = []
        lines.append(f"\n  📋 Round {rm.round_num} Details:")
        lines.append(f"  {'─' * 76}")
        lines.append(f"  | {'Agent':<20} | {'Role':<18} | {'Verdict':<10} | {'Conf':<6} |")
        lines.append(f"  {'─' * 76}")
        
        for debator_name in rm.verdicts.keys():
            verdict = rm.verdicts.get(debator_name, "N/A")
            conf = rm.confidences.get(debator_name, 0.0)
            role = rm.roles.get(debator_name, "Unknown")
            # Shorten name for display
            short_name = debator_name[:20] if len(debator_name) > 20 else debator_name
            short_role = role[:18] if len(role) > 18 else role
            lines.append(f"  | {short_name:<20} | {short_role:<18} | {verdict:<10} | {conf:<6.2f} |")
        
        lines.append(f"  {'─' * 76}")
        lines.append(f"  Agreement: {rm.agreement_ratio:.0%} | Majority: {rm.majority_verdict}")
        
        # Show interaction for Round 2+ (XAI)
        if rm.round_num >= 2 and (rm.agree_with or rm.disagree_with):
            lines.append(f"\n  🤝 Interaction (Round {rm.round_num}):")
            for debator_name in rm.verdicts.keys():
                short_name = debator_name.split('/')[-1][:15] if '/' in debator_name else debator_name[:15]
                
                # Show agree
                agree_list = rm.agree_with.get(debator_name, [])
                agree_reason = rm.agree_reasons.get(debator_name, "")
                if agree_list:
                    lines.append(f"    ✅ {short_name} ĐỒNG Ý với: {', '.join(agree_list)}")
                    if agree_reason:
                        lines.append(f"       → {agree_reason[:100]}...")
                
                # Show disagree
                disagree_list = rm.disagree_with.get(debator_name, [])
                disagree_reason = rm.disagree_reasons.get(debator_name, "")
                if disagree_list:
                    lines.append(f"    ❌ {short_name} PHẢN ĐỐI: {', '.join(disagree_list)}")
                    if disagree_reason:
                        lines.append(f"       → {disagree_reason[:100]}...")
                
                # Show if changed
                changed = rm.changed.get(debator_name, False)
                change_reason = rm.change_reasons.get(debator_name, "")
                if changed:
                    lines.append(f"    🔄 {short_name} ĐỔI Ý: {change_reason}")
            lines.append("")
        
        # Add FULL reasonings if available (No truncation)
        if rm.reasonings:
            lines.append(f"\n  💬 Reasoning (Full):")
            for debator_name, reasoning in rm.reasonings.items():
                short_name = debator_name.split('/')[-1][:15] if '/' in debator_name else debator_name[:15]
                lines.append(f"    • {short_name}:")
                # Add indentation for reasoning text
                wrapped_reasoning = "\n".join([f"      {line}" for line in reasoning.split('\n')])
                lines.append(f"{wrapped_reasoning}")
                lines.append("")  # Empty line for separation
        
        return lines

    def generate_case_study_report(self, output_path: str):
        """
        Tạo báo cáo case study chi tiết với thông tin đầy đủ của từng debator.
        """
        case_studies = self.get_case_studies(n=3)
        
        report = []
        report.append("=" * 80)
        report.append("📊 DEBATE CASE STUDY REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {Path(output_path).stem}")
        report.append("")
        
        # Fixed cases
        report.append("## 1. ✅ DEBATE FIXED (Model sai → Debate sửa đúng)")
        report.append("-" * 60)
        if not case_studies['fixed']:
            report.append("(Không có case nào)")
        for i, case in enumerate(case_studies['fixed'], 1):
            report.append(f"\n### Case {i}")
            report.append(f"📝 Statement: {case.statement}")
            report.append(f"🎯 Gold Label: {case.gold_label}")
            report.append(f"🤖 Model: {case.model_verdict} (conf={case.model_confidence:.2f}) ❌")
            report.append(f"⚖️  Final:  {case.final_verdict} (conf={case.final_confidence:.2f}) ✅")
            report.append(f"🔄 Rounds: {case.rounds_used} | MVP: {case.mvp_agent}")
            if case.round_metrics:
                for rm in case.round_metrics:
                    report.extend(self._format_debator_table(rm))
        
        report.append("\n")
        
        # Broke cases
        report.append("## 2. ❌ DEBATE BROKE (Model đúng → Debate làm sai)")
        report.append("-" * 60)
        if not case_studies['broke']:
            report.append("(Không có case nào)")
        for i, case in enumerate(case_studies['broke'], 1):
            report.append(f"\n### Case {i}")
            report.append(f"📝 Statement: {case.statement}")
            report.append(f"🎯 Gold Label: {case.gold_label}")
            report.append(f"🤖 Model: {case.model_verdict} (conf={case.model_confidence:.2f}) ✅")
            report.append(f"⚖️  Final:  {case.final_verdict} (conf={case.final_confidence:.2f}) ❌")
            report.append(f"🔄 Rounds: {case.rounds_used} | MVP: {case.mvp_agent}")
            if case.round_metrics:
                for rm in case.round_metrics:
                    report.extend(self._format_debator_table(rm))
        
        report.append("\n")
        
        # High disagreement
        report.append("## 3. ⚠️ HIGH DISAGREEMENT (Debators bất đồng cao)")
        report.append("-" * 60)
        if not case_studies['high_disagreement']:
            report.append("(Không có case nào)")
        for i, case in enumerate(case_studies['high_disagreement'], 1):
            report.append(f"\n### Case {i}")
            report.append(f"📝 Statement: {case.statement}")
            report.append(f"🎯 Gold Label: {case.gold_label}")
            report.append(f"🤖 Model: {case.model_verdict} | ⚖️ Final: {case.final_verdict}")
            report.append(f"🔄 Rounds: {case.rounds_used}")
            if case.round_metrics:
                for rm in case.round_metrics:
                    report.extend(self._format_debator_table(rm))
        
        # Write to file
        report_text = "\n".join(report)
        Path(output_path).write_text(report_text, encoding='utf-8')
        logger.info(f"Saved case study report to {output_path}")
        
        return report_text
    
    def save_metrics(self, output_path: str):
        """Lưu tất cả metrics ra JSON."""
        data = {
            "summary": self.get_summary(),
            "samples": [s.to_dict() for s in self.samples]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved metrics to {output_path}")
    
    def load_metrics(self, input_path: str):
        """Load metrics từ JSON."""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.samples = []
        for s_dict in data.get('samples', []):
            round_metrics = [
                RoundMetrics(**rm) for rm in s_dict.pop('round_metrics', [])
            ]
            sample = DebateSampleMetrics(**s_dict, round_metrics=round_metrics)
            self.samples.append(sample)
        
        logger.info(f"Loaded {len(self.samples)} samples from {input_path}")
