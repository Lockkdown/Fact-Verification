"""
Hybrid Strategy Analysis - Post-hoc Simulation & Visualization

Dựa trên DOWN Framework (2025): "Debate Only When Necessary"
ArXiv: https://arxiv.org/abs/2504.05047

Script này load kết quả debate đã chạy và giả lập Hybrid Strategy với các threshold khác nhau.

Outputs:
1. accuracy_vs_cost.png - Trade-off giữa Accuracy và % Skipped Debate
2. correction_regression_matrix.png - Ma trận sửa sai (Correction vs Regression)
3. confidence_bucket_analysis.png - So sánh Model vs Debate theo confidence bucket
4. hybrid_analysis_report.json - Báo cáo chi tiết

Author: Lockdown
Date: Dec 03, 2025
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HybridSimulationResult:
    """Kết quả giả lập Hybrid Strategy tại một threshold."""
    threshold: float
    
    # Accuracy metrics
    model_accuracy: float
    debate_accuracy: float
    hybrid_accuracy: float
    
    # Skip metrics
    skip_count: int
    skip_ratio: float
    debate_count: int
    
    # Correction/Regression analysis
    correction_count: int  # Model Sai → Debate Đúng (khi debate được trigger)
    regression_count: int  # Model Đúng → Debate Sai (khi debate được trigger)
    hybrid_rescued_count: int  # Hybrid cứu được (Model đúng, Debate sai, Hybrid chọn Model)
    
    # Cost estimation (relative)
    relative_cost: float  # 1.0 = Full Debate, lower = cheaper


class HybridAnalyzer:
    """
    Phân tích Hybrid Strategy từ kết quả debate đã chạy.
    
    Input: debate_metrics_test.json (chứa model_confidence, model_verdict, final_verdict)
    Output: Biểu đồ và báo cáo cho thesis/paper
    """
    
    def __init__(self, results_path: str):
        """
        Args:
            results_path: Path to debate_metrics_test.json
        """
        self.results_path = Path(results_path)
        self.samples = []
        self.summary = {}
        self._load_results()
    
    def _load_results(self):
        """Load kết quả từ JSON file."""
        with open(self.results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Support both 'samples' (debate_metrics) and 'results' (vifactcheck_test_results) formats
        self.samples = data.get('samples', data.get('results', []))
        self.summary = data.get('summary', {})
        
        logger.info(f"Loaded {len(self.samples)} samples from {self.results_path}")
    
    def _normalize_verdict(self, verdict: str) -> str:
        """Normalize verdict to standard format."""
        if not verdict:
            return 'NEI'
        v_upper = verdict.upper().strip()
        if v_upper in ['SUPPORTED', 'SUPPORT', 'SUPPORTS']:
            return 'Support'
        elif v_upper in ['REFUTED', 'REFUTE', 'REFUTES']:
            return 'Refute'
        else:
            return 'NEI'
    
    def simulate_hybrid(self, threshold: float) -> HybridSimulationResult:
        """
        Giả lập Hybrid Strategy với một threshold cụ thể.
        
        Logic:
        - model_confidence >= threshold → Trust Model (skip debate)
        - model_confidence < threshold → Trust Debate
        """
        model_correct = 0
        debate_correct = 0
        hybrid_correct = 0
        
        skip_count = 0
        correction_count = 0
        regression_count = 0
        hybrid_rescued_count = 0
        
        for sample in self.samples:
            gold = self._normalize_verdict(sample.get('gold_label', ''))
            model_v = self._normalize_verdict(sample.get('model_verdict', ''))
            debate_v = self._normalize_verdict(sample.get('final_verdict', ''))
            
            # Get model confidence from verdict_3label_probs (max prob) or model_confidence
            model_conf = sample.get('model_confidence', 0.0)
            if model_conf == 0.0:
                probs = sample.get('verdict_3label_probs', {})
                if probs:
                    model_conf = max(probs.values()) if probs.values() else 0.0
            
            # Model accuracy
            is_model_correct = (model_v == gold)
            if is_model_correct:
                model_correct += 1
            
            # Debate accuracy
            is_debate_correct = (debate_v == gold)
            if is_debate_correct:
                debate_correct += 1
            
            # Hybrid decision
            if model_conf >= threshold:
                # Trust Model (skip debate)
                hybrid_v = model_v
                skip_count += 1
                
                # Check if hybrid rescued (Model đúng, Debate sai)
                if is_model_correct and not is_debate_correct:
                    hybrid_rescued_count += 1
            else:
                # Trust Debate
                hybrid_v = debate_v
                
                # Correction: Model sai → Debate đúng
                if not is_model_correct and is_debate_correct:
                    correction_count += 1
                
                # Regression: Model đúng → Debate sai
                if is_model_correct and not is_debate_correct:
                    regression_count += 1
            
            # Hybrid accuracy
            if hybrid_v == gold:
                hybrid_correct += 1
        
        total = len(self.samples)
        debate_count = total - skip_count
        
        return HybridSimulationResult(
            threshold=threshold,
            model_accuracy=model_correct / total if total > 0 else 0,
            debate_accuracy=debate_correct / total if total > 0 else 0,
            hybrid_accuracy=hybrid_correct / total if total > 0 else 0,
            skip_count=skip_count,
            skip_ratio=skip_count / total if total > 0 else 0,
            debate_count=debate_count,
            correction_count=correction_count,
            regression_count=regression_count,
            hybrid_rescued_count=hybrid_rescued_count,
            relative_cost=debate_count / total if total > 0 else 1.0
        )
    
    def simulate_all_thresholds(
        self, 
        thresholds: List[float] = None
    ) -> List[HybridSimulationResult]:
        """
        Chạy simulation với nhiều threshold.
        
        Args:
            thresholds: List các threshold để test (default: 0.5 đến 0.99)
        """
        if thresholds is None:
            thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
        
        results = []
        for t in thresholds:
            result = self.simulate_hybrid(t)
            results.append(result)
            logger.info(f"Threshold {t:.2f}: Hybrid Acc={result.hybrid_accuracy:.2%}, "
                       f"Skip={result.skip_ratio:.1%}, Rescued={result.hybrid_rescued_count}")
        
        return results
    
    def analyze_by_confidence_bucket(
        self,
        buckets: List[Tuple[float, float]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Phân tích accuracy theo confidence bucket.
        
        Returns:
            {
                "0.5-0.6": {"n": 20, "model_acc": 0.65, "debate_acc": 0.78, "better": "DEBATE"},
                ...
            }
        """
        if buckets is None:
            buckets = [
                (0.0, 0.50), (0.50, 0.60), (0.60, 0.70), 
                (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)
            ]
        
        analysis = {}
        
        for bucket_min, bucket_max in buckets:
            bucket_name = f"{bucket_min:.1f}-{bucket_max:.1f}"
            bucket_samples = []
            
            for sample in self.samples:
                # Get model confidence from verdict_3label_probs or model_confidence
                conf = sample.get('model_confidence', 0.0)
                if conf == 0.0:
                    probs = sample.get('verdict_3label_probs', {})
                    if probs:
                        conf = max(probs.values()) if probs.values() else 0.0
                
                if bucket_min <= conf < bucket_max:
                    bucket_samples.append(sample)
            
            if not bucket_samples:
                analysis[bucket_name] = {
                    "n": 0, "model_acc": 0, "debate_acc": 0, 
                    "better": "-", "diff": 0
                }
                continue
            
            model_correct = 0
            debate_correct = 0
            
            for sample in bucket_samples:
                gold = self._normalize_verdict(sample.get('gold_label', ''))
                model_v = self._normalize_verdict(sample.get('model_verdict', ''))
                debate_v = self._normalize_verdict(sample.get('final_verdict', ''))
                
                if model_v == gold:
                    model_correct += 1
                if debate_v == gold:
                    debate_correct += 1
            
            n = len(bucket_samples)
            model_acc = model_correct / n
            debate_acc = debate_correct / n
            
            analysis[bucket_name] = {
                "n": n,
                "model_acc": model_acc,
                "debate_acc": debate_acc,
                "better": "MODEL" if model_acc >= debate_acc else "DEBATE",
                "diff": model_acc - debate_acc
            }
        
        return analysis
    
    def get_correction_regression_matrix(self) -> Dict[str, int]:
        """
        Tính ma trận Correction/Regression cho Full Debate.
        
        Returns:
            {
                "both_correct": N,      # Model đúng, Debate đúng
                "both_wrong": N,        # Model sai, Debate sai
                "correction": N,        # Model sai → Debate đúng (GOOD)
                "regression": N,        # Model đúng → Debate sai (BAD)
            }
        """
        matrix = {
            "both_correct": 0,
            "both_wrong": 0,
            "correction": 0,
            "regression": 0
        }
        
        for sample in self.samples:
            gold = self._normalize_verdict(sample.get('gold_label', ''))
            model_v = self._normalize_verdict(sample.get('model_verdict', ''))
            debate_v = self._normalize_verdict(sample.get('final_verdict', ''))
            
            is_model_correct = (model_v == gold)
            is_debate_correct = (debate_v == gold)
            
            if is_model_correct and is_debate_correct:
                matrix["both_correct"] += 1
            elif not is_model_correct and not is_debate_correct:
                matrix["both_wrong"] += 1
            elif not is_model_correct and is_debate_correct:
                matrix["correction"] += 1
            else:  # is_model_correct and not is_debate_correct
                matrix["regression"] += 1
        
        return matrix
    
    # ==================== VISUALIZATION ====================
    
    def plot_accuracy_vs_cost(
        self, 
        results: List[HybridSimulationResult],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Vẽ biểu đồ Trade-off: Accuracy vs. Cost (% Debate Triggered).
        
        Tương tự Figure 3 trong paper DOWN.
        """
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        thresholds = [r.threshold for r in results]
        hybrid_accs = [r.hybrid_accuracy * 100 for r in results]
        skip_ratios = [r.skip_ratio * 100 for r in results]
        debate_ratios = [(1 - r.skip_ratio) * 100 for r in results]
        
        # Baseline lines
        model_acc = results[0].model_accuracy * 100
        debate_acc = results[0].debate_accuracy * 100
        
        # Primary axis: Accuracy
        color_hybrid = '#2ecc71'
        color_model = '#3498db'
        color_debate = '#e74c3c'
        
        ax1.axhline(y=model_acc, color=color_model, linestyle='--', linewidth=2, 
                   label=f'Model Only ({model_acc:.1f}%)')
        ax1.axhline(y=debate_acc, color=color_debate, linestyle='--', linewidth=2,
                   label=f'Full Debate ({debate_acc:.1f}%)')
        
        line1, = ax1.plot(thresholds, hybrid_accs, 'o-', color=color_hybrid, 
                         linewidth=3, markersize=10, label='Hybrid Accuracy')
        
        ax1.set_xlabel('Confidence Threshold', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold', color='black')
        ax1.tick_params(axis='y', labelcolor='black', labelsize=12)
        ax1.tick_params(axis='x', labelsize=12)
        ax1.set_ylim(min(hybrid_accs) - 5, max(hybrid_accs) + 5)
        ax1.grid(alpha=0.3, linestyle='--')
        
        # Secondary axis: Skip Ratio (Cost Savings)
        ax2 = ax1.twinx()
        color_skip = '#9b59b6'
        
        line2, = ax2.plot(thresholds, skip_ratios, 's--', color=color_skip,
                         linewidth=2, markersize=8, label='% Skipped (Cost Saved)')
        
        ax2.set_ylabel('% Debate Skipped (Cost Saved)', fontsize=14, 
                      fontweight='bold', color=color_skip)
        ax2.tick_params(axis='y', labelcolor=color_skip, labelsize=12)
        ax2.set_ylim(0, 100)
        
        # Find optimal threshold (highest hybrid accuracy)
        best_idx = np.argmax(hybrid_accs)
        best_threshold = thresholds[best_idx]
        best_acc = hybrid_accs[best_idx]
        best_skip = skip_ratios[best_idx]
        
        # Highlight optimal point
        ax1.scatter([best_threshold], [best_acc], s=300, c='gold', 
                   edgecolors='black', linewidth=2, zorder=5, marker='*')
        
        # Smart annotation positioning - place at bottom right to avoid overlap
        # Always place annotation at bottom-right corner of the plot
        text_x = 0.95  # Near right edge
        text_y = 0.15  # Near bottom
        
        ax1.annotate(
            f'★ Optimal: θ = {best_threshold}\nAcc = {best_acc:.1f}% | Skip = {best_skip:.0f}%',
            xy=(best_threshold, best_acc), 
            xytext=(text_x, text_y),
            textcoords='axes fraction',
            fontsize=10, fontweight='bold', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fffacd', 
                     edgecolor='#ffd700', linewidth=2, alpha=0.95),
            arrowprops=dict(arrowstyle='->', color='#333', lw=1.5,
                           connectionstyle='arc3,rad=0.2')
        )
        
        # Combined legend - place at upper left, away from annotation
        ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
        
        plt.title('Hybrid Strategy: Accuracy vs. Cost Trade-off\n(DOWN Framework Analysis)', 
                 fontsize=16, fontweight='bold', pad=15)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved accuracy vs cost plot to {save_path}")
        
        return fig
    
    def plot_correction_regression_matrix(
        self,
        matrix: Dict[str, int],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Vẽ ma trận Correction vs Regression.
        
        Hiển thị:
        - Both Correct (Model ✓, Debate ✓)
        - Correction (Model ✗ → Debate ✓) - GOOD
        - Regression (Model ✓ → Debate ✗) - BAD
        - Both Wrong (Model ✗, Debate ✗)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # --- Left: Confusion Matrix Style ---
        ax1 = axes[0]
        
        # Create 2x2 matrix
        data = np.array([
            [matrix['both_correct'], matrix['regression']],
            [matrix['correction'], matrix['both_wrong']]
        ])
        
        total = sum(matrix.values())
        data_pct = data / total * 100 if total > 0 else data * 0
        
        # Colors: Green for good, Red for bad
        colors = np.array([
            ['#2ecc71', '#e74c3c'],  # Both correct (good), Regression (bad)
            ['#27ae60', '#95a5a6']   # Correction (good), Both wrong (neutral)
        ])
        
        for i in range(2):
            for j in range(2):
                ax1.add_patch(plt.Rectangle((j, 1-i), 1, 1, 
                             facecolor=colors[i, j], edgecolor='white', linewidth=3))
                ax1.text(j + 0.5, 1.5 - i, f'{data[i, j]}\n({data_pct[i, j]:.1f}%)',
                        ha='center', va='center', fontsize=16, fontweight='bold',
                        color='white' if colors[i, j] != '#95a5a6' else 'black')
        
        ax1.set_xlim(0, 2)
        ax1.set_ylim(0, 2)
        ax1.set_xticks([0.5, 1.5])
        ax1.set_xticklabels(['Debate ✓', 'Debate ✗'], fontsize=13, fontweight='bold')
        ax1.set_yticks([0.5, 1.5])
        ax1.set_yticklabels(['Model ✗', 'Model ✓'], fontsize=13, fontweight='bold')
        ax1.set_xlabel('Debate Result', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Model Result', fontsize=14, fontweight='bold')
        ax1.set_title('Correction vs Regression Matrix', fontsize=15, fontweight='bold')
        
        # Add labels
        ax1.text(0.5, 2.15, 'Correction\n(GOOD)', ha='center', fontsize=10, color='#27ae60', fontweight='bold')
        ax1.text(1.5, 2.15, 'Regression\n(BAD)', ha='center', fontsize=10, color='#e74c3c', fontweight='bold')
        
        # --- Right: Bar Chart ---
        ax2 = axes[1]
        
        categories = ['Both\nCorrect', 'Correction\n(M✗→D✓)', 'Regression\n(M✓→D✗)', 'Both\nWrong']
        values = [matrix['both_correct'], matrix['correction'], matrix['regression'], matrix['both_wrong']]
        colors_bar = ['#2ecc71', '#27ae60', '#e74c3c', '#95a5a6']
        
        bars = ax2.bar(categories, values, color=colors_bar, edgecolor='black', linewidth=1.5)
        
        for bar, val in zip(bars, values):
            pct = val / total * 100 if total > 0 else 0
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val}\n({pct:.1f}%)', ha='center', va='bottom', 
                    fontsize=12, fontweight='bold')
        
        ax2.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
        ax2.set_title('Debate Impact Distribution', fontsize=15, fontweight='bold')
        ax2.set_ylim(0, max(values) * 1.25)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add insight text
        net_gain = matrix['correction'] - matrix['regression']
        insight = f"Net Debate Gain: {'+' if net_gain >= 0 else ''}{net_gain} samples"
        ax2.text(0.5, 0.95, insight, transform=ax2.transAxes, fontsize=13,
                fontweight='bold', ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved correction/regression matrix to {save_path}")
        
        return fig
    
    def plot_confidence_bucket_analysis(
        self,
        bucket_analysis: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Vẽ biểu đồ so sánh Model vs Debate accuracy theo confidence bucket.
        
        Chứng minh: Ở vùng confidence cao, Model >= Debate → Không cần debate.
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        buckets = list(bucket_analysis.keys())
        model_accs = [bucket_analysis[b]['model_acc'] * 100 for b in buckets]
        debate_accs = [bucket_analysis[b]['debate_acc'] * 100 for b in buckets]
        sample_counts = [bucket_analysis[b]['n'] for b in buckets]
        
        x = np.arange(len(buckets))
        width = 0.35
        
        # Bars
        bars1 = ax.bar(x - width/2, model_accs, width, label='Model', 
                      color='#3498db', edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x + width/2, debate_accs, width, label='Debate',
                      color='#e74c3c', edgecolor='black', linewidth=1.2)
        
        # Add value labels (only for buckets with samples)
        for bar, acc, n in zip(bars1, model_accs, sample_counts):
            if n > 0:  # Only show label if bucket has samples
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        for bar, acc, n in zip(bars2, debate_accs, sample_counts):
            if n > 0:  # Only show label if bucket has samples
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Highlight "Model wins" regions
        for i, bucket in enumerate(buckets):
            if bucket_analysis[bucket]['better'] == 'MODEL' and bucket_analysis[bucket]['n'] > 0:
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.15, color='#3498db')
        
        ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_title('Model vs Debate Accuracy by Confidence Bucket\n(Blue shading = Model wins → Skip Debate)', 
                    fontsize=15, fontweight='bold', pad=15)
        ax.set_xticks(x)
        
        # Create custom x-tick labels with n= included
        xlabels = [f'{bucket}\n(n={n})' for bucket, n in zip(buckets, sample_counts)]
        ax.set_xticklabels(xlabels, fontsize=10)
        
        ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
        ax.set_ylim(0, 115)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved confidence bucket analysis to {save_path}")
        
        return fig
    
    def generate_full_report(
        self,
        output_dir: str,
        thresholds: List[float] = None
    ) -> Dict[str, Any]:
        """
        Chạy toàn bộ phân tích và xuất báo cáo + biểu đồ.
        
        Args:
            output_dir: Thư mục output
            thresholds: List threshold để test
            
        Returns:
            Dict chứa toàn bộ kết quả phân tích
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Simulate all thresholds
        print("\n" + "="*70)
        print("🔬 HYBRID STRATEGY ANALYSIS (DOWN Framework)")
        print("="*70)
        
        print("\n📊 Simulating Hybrid Strategy with different thresholds...")
        simulation_results = self.simulate_all_thresholds(thresholds)
        
        # 2. Find optimal threshold
        best_result = max(simulation_results, key=lambda r: r.hybrid_accuracy)
        print(f"\n⭐ Optimal Threshold: {best_result.threshold}")
        print(f"   - Hybrid Accuracy: {best_result.hybrid_accuracy:.2%}")
        print(f"   - Skip Ratio: {best_result.skip_ratio:.1%}")
        print(f"   - Rescued Samples: {best_result.hybrid_rescued_count}")
        
        # 3. Correction/Regression matrix
        print("\n📈 Analyzing Correction vs Regression...")
        matrix = self.get_correction_regression_matrix()
        print(f"   - Both Correct: {matrix['both_correct']}")
        print(f"   - Correction (M✗→D✓): {matrix['correction']} (GOOD)")
        print(f"   - Regression (M✓→D✗): {matrix['regression']} (BAD)")
        print(f"   - Both Wrong: {matrix['both_wrong']}")
        
        # 4. Confidence bucket analysis
        print("\n📉 Analyzing by Confidence Bucket...")
        bucket_analysis = self.analyze_by_confidence_bucket()
        for bucket, data in bucket_analysis.items():
            if data['n'] > 0:
                print(f"   [{bucket}] n={data['n']:3d} | Model: {data['model_acc']:.1%} | "
                      f"Debate: {data['debate_acc']:.1%} | Winner: {data['better']}")
        
        # 5. Generate plots
        print("\n🎨 Generating visualizations...")
        
        # Plot 1: Accuracy vs Cost
        self.plot_accuracy_vs_cost(
            simulation_results,
            save_path=str(output_path / "hybrid_accuracy_vs_cost.png")
        )
        
        # Plot 2: Correction/Regression Matrix
        self.plot_correction_regression_matrix(
            matrix,
            save_path=str(output_path / "hybrid_correction_regression.png")
        )
        
        # Plot 3: Confidence Bucket Analysis
        self.plot_confidence_bucket_analysis(
            bucket_analysis,
            save_path=str(output_path / "hybrid_confidence_buckets.png")
        )
        
        # 6. Compile report
        report = {
            "analysis_type": "Hybrid Strategy Simulation (DOWN Framework)",
            "source_file": str(self.results_path),
            "total_samples": len(self.samples),
            "baseline": {
                "model_accuracy": simulation_results[0].model_accuracy,
                "debate_accuracy": simulation_results[0].debate_accuracy
            },
            "optimal_threshold": {
                "threshold": best_result.threshold,
                "hybrid_accuracy": best_result.hybrid_accuracy,
                "skip_ratio": best_result.skip_ratio,
                "debate_count": best_result.debate_count,
                "rescued_count": best_result.hybrid_rescued_count,
                "relative_cost": best_result.relative_cost
            },
            "correction_regression_matrix": matrix,
            "confidence_bucket_analysis": bucket_analysis,
            "all_threshold_results": [asdict(r) for r in simulation_results]
        }
        
        # Save report
        report_path = output_path / "hybrid_analysis_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Report saved to: {report_path}")
        print("="*70)
        
        # Summary comparison
        print("\n📋 SUMMARY COMPARISON:")
        print("-"*50)
        print(f"{'Method':<20} {'Accuracy':<12} {'Cost':<12}")
        print("-"*50)
        print(f"{'Model Only':<20} {simulation_results[0].model_accuracy:.2%}{'':>5} {'0%':>10}")
        print(f"{'Full Debate':<20} {simulation_results[0].debate_accuracy:.2%}{'':>5} {'100%':>10}")
        print(f"{'Hybrid (θ={best_result.threshold})':<20} {best_result.hybrid_accuracy:.2%}{'':>5} "
              f"{best_result.relative_cost:.0%}{'':>5}")
        print("-"*50)
        
        gain_vs_model = (best_result.hybrid_accuracy - simulation_results[0].model_accuracy) * 100
        gain_vs_debate = (best_result.hybrid_accuracy - simulation_results[0].debate_accuracy) * 100
        cost_saved = (1 - best_result.relative_cost) * 100
        
        print(f"\n🎯 Hybrid Gains:")
        print(f"   vs Model: {'+' if gain_vs_model >= 0 else ''}{gain_vs_model:.1f}% accuracy")
        print(f"   vs Debate: {'+' if gain_vs_debate >= 0 else ''}{gain_vs_debate:.1f}% accuracy")
        print(f"   Cost Saved: {cost_saved:.0f}%")
        print("="*70 + "\n")
        
        return report


def main():
    """Entry point for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid Strategy Analysis (DOWN Framework)")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="results/vifactcheck/test/debate_analysis/debate_metrics_test.json",
        help="Path to debate_metrics JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results/vifactcheck/test/hybrid_analysis",
        help="Output directory for reports and plots"
    )
    parser.add_argument(
        "--thresholds", "-t",
        type=float,
        nargs="+",
        default=None,
        help="List of thresholds to test (default: 0.5 to 0.99)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run analysis
    analyzer = HybridAnalyzer(args.input)
    report = analyzer.generate_full_report(args.output, args.thresholds)
    
    return report


if __name__ == "__main__":
    main()
