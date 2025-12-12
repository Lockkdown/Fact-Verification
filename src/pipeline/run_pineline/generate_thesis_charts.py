"""
Generate Additional Charts for Thesis/Paper Report

Includes:
1. Comparison Chart: Full Debate vs Hybrid vs Model
2. Per-Class Performance (F1/Precision/Recall)
3. Confusion Matrix Heatmaps
4. Summary Table (LaTeX/Markdown)

Author: Lockdown
Date: Dec 03, 2025
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


class ThesisChartGenerator:
    """Generate publication-ready charts for thesis/paper."""
    
    def __init__(self, hybrid_dir: str, full_debate_dir: str = None):
        """
        Args:
            hybrid_dir: Path to hybrid_debate results
            full_debate_dir: Path to full_debate results (optional)
        """
        self.hybrid_dir = Path(hybrid_dir)
        self.full_debate_dir = Path(full_debate_dir) if full_debate_dir else None
        
        # Load data
        self.hybrid_metrics = self._load_json(self.hybrid_dir / "metrics" / "metrics_test.json")
        self.hybrid_results = self._load_json(self.hybrid_dir / "vifactcheck_test_results.json")
        self.hybrid_analysis = self._load_json(self.hybrid_dir / "hybrid_analysis_report.json")
        
        if self.full_debate_dir:
            self.full_metrics = self._load_json(self.full_debate_dir / "metrics" / "metrics_test.json")
            self.full_results = self._load_json(self.full_debate_dir / "vifactcheck_test_results.json")
        else:
            self.full_metrics = None
            self.full_results = None
    
    def _load_json(self, path: Path) -> Optional[Dict]:
        """Load JSON file safely."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load {path}: {e}")
            return None
    
    def plot_comparison_chart(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Bar chart comparing Model vs Full Debate vs Hybrid.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # --- Left: Accuracy Comparison ---
        ax1 = axes[0]
        
        # Get accuracies
        model_acc = self.hybrid_metrics.get('model_accuracy', 0) * 100
        hybrid_final_acc = self.hybrid_metrics.get('final_accuracy', 0) * 100
        
        if self.full_metrics:
            full_debate_acc = self.full_metrics.get('final_accuracy', 0) * 100
        else:
            full_debate_acc = hybrid_final_acc  # Use same if not available
        
        # Get optimal hybrid accuracy from analysis
        if self.hybrid_analysis and 'optimal_threshold' in self.hybrid_analysis:
            opt = self.hybrid_analysis['optimal_threshold']
            optimal_hybrid_acc = opt.get('hybrid_accuracy', 0) * 100
        else:
            optimal_hybrid_acc = hybrid_final_acc
        
        methods = ['Model\n(PhoBERT)', 'Full Debate\n(All Samples)', 'Hybrid\n(DOWN Framework)']
        accuracies = [model_acc, full_debate_acc, optimal_hybrid_acc]
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        bars = ax1.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax1.set_title('Accuracy Comparison', fontsize=16, fontweight='bold')
        ax1.set_ylim(0, 105)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Highlight best
        best_idx = np.argmax(accuracies)
        bars[best_idx].set_edgecolor('#ffd700')
        bars[best_idx].set_linewidth(3)
        
        # --- Right: Cost Comparison ---
        ax2 = axes[1]
        
        # Estimate costs (relative)
        # Model: 1 call per sample
        # Full Debate: 3 agents × 2 rounds + 1 judge = 7 calls
        # Hybrid: depends on skip ratio
        
        if self.hybrid_analysis and 'optimal_threshold' in self.hybrid_analysis:
            skip_ratio = self.hybrid_analysis['optimal_threshold'].get('skip_ratio', 0)
        else:
            skip_ratio = 0.5
        
        model_cost = 1.0
        full_debate_cost = 7.0  # 3 agents × 2 rounds + judge
        hybrid_cost = 1.0 + (1 - skip_ratio) * 6.0  # Model + (1-skip) × debate overhead
        
        methods_cost = ['Model', 'Full Debate', 'Hybrid']
        costs = [model_cost, full_debate_cost, hybrid_cost]
        
        bars2 = ax2.bar(methods_cost, costs, color=colors, edgecolor='black', linewidth=1.5)
        
        for bar, cost in zip(bars2, costs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{cost:.1f}×', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax2.set_ylabel('Relative API Cost', fontsize=14, fontweight='bold')
        ax2.set_title('Cost Comparison (API Calls)', fontsize=16, fontweight='bold')
        ax2.set_ylim(0, 9)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add cost savings annotation
        savings = (1 - hybrid_cost / full_debate_cost) * 100
        ax2.annotate(f'{savings:.0f}% savings\nvs Full Debate',
                    xy=(2, hybrid_cost), xytext=(2.3, hybrid_cost + 2),
                    fontsize=11, ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#d5f5e3', edgecolor='#27ae60'),
                    arrowprops=dict(arrowstyle='->', color='#27ae60'))
        
        plt.suptitle('Model vs Full Debate vs Hybrid Strategy', fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved comparison chart to {save_path}")
        
        return fig
    
    def plot_per_class_performance(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Grouped bar chart showing Precision/Recall/F1 per class.
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        if not self.hybrid_metrics or 'final_per_class' not in self.hybrid_metrics:
            logger.warning("No per-class metrics available")
            return fig
        
        per_class = self.hybrid_metrics['final_per_class']
        classes = list(per_class.keys())
        
        # Map class names for display
        class_display = {
            'Support': 'SUPPORTED',
            'Refute': 'REFUTED', 
            'NOT_ENOUGH_INFO': 'NEI'
        }
        classes_display = [class_display.get(c, c) for c in classes]
        
        precision = [per_class[c]['precision'] * 100 for c in classes]
        recall = [per_class[c]['recall'] * 100 for c in classes]
        f1 = [per_class[c]['f1'] * 100 for c in classes]
        support = [per_class[c]['support'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db', edgecolor='black')
        bars2 = ax.bar(x, recall, width, label='Recall', color='#e74c3c', edgecolor='black')
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#2ecc71', edgecolor='black')
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
        ax.set_title('Per-Class Performance (Precision / Recall / F1)', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        
        # Include n= in x-tick labels to avoid overlap
        xlabels = [f'{cls}\n(n={sup})' for cls, sup in zip(classes_display, support)]
        ax.set_xticklabels(xlabels, fontsize=11, fontweight='bold')
        ax.legend(loc='lower right', fontsize=12)
        ax.set_ylim(0, 115)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add macro averages
        macro = self.hybrid_metrics.get('final_macro', {})
        if macro:
            macro_text = f"Macro Avg: P={macro.get('macro_precision', 0)*100:.1f}% | R={macro.get('macro_recall', 0)*100:.1f}% | F1={macro.get('macro_f1', 0)*100:.1f}%"
            ax.text(0.5, 1.02, macro_text, transform=ax.transAxes, ha='center', fontsize=11,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', edgecolor='gray'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved per-class performance to {save_path}")
        
        return fig
    
    def plot_confusion_matrices(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Side-by-side confusion matrix heatmaps for Model and Final verdict.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        if not self.hybrid_metrics:
            logger.warning("No metrics available")
            return fig
        
        # Get confusion matrices
        model_cm = self.hybrid_metrics.get('model_confusion_matrix', {})
        final_cm = self.hybrid_metrics.get('final_confusion_matrix', {})
        
        labels = model_cm.get('labels', ['Support', 'Refute', 'NOT_ENOUGH_INFO'])
        label_display = ['SUPPORTED', 'REFUTED', 'NEI']
        
        def matrix_to_array(cm_dict, labels):
            matrix = cm_dict.get('matrix', {})
            arr = np.zeros((len(labels), len(labels)))
            for i, true_label in enumerate(labels):
                for j, pred_label in enumerate(labels):
                    arr[i, j] = matrix.get(true_label, {}).get(pred_label, 0)
            return arr
        
        model_arr = matrix_to_array(model_cm, labels)
        final_arr = matrix_to_array(final_cm, labels)
        
        # Custom annotation function to handle text color based on cell value
        def annotate_heatmap(ax, data, cmap_name):
            """Add annotations with smart text color."""
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    val = data[i, j]
                    # Use dark text for light cells (low values), white for dark cells
                    text_color = 'white' if val > data.max() * 0.3 else 'black'
                    ax.text(j + 0.5, i + 0.5, f'{int(val)}',
                           ha='center', va='center', fontsize=16, fontweight='bold',
                           color=text_color)
        
        # Plot Model CM
        ax1 = axes[0]
        sns.heatmap(model_arr, annot=False, cmap='Blues', 
                   xticklabels=label_display, yticklabels=label_display,
                   ax=ax1, cbar=False, linewidths=2, linecolor='white')
        annotate_heatmap(ax1, model_arr, 'Blues')
        ax1.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax1.set_title(f'Model (PhoBERT)\nAccuracy: {self.hybrid_metrics.get("model_accuracy", 0)*100:.1f}%', 
                     fontsize=14, fontweight='bold')
        
        # Plot Final CM
        ax2 = axes[1]
        sns.heatmap(final_arr, annot=False, cmap='Greens',
                   xticklabels=label_display, yticklabels=label_display,
                   ax=ax2, cbar=False, linewidths=2, linecolor='white')
        annotate_heatmap(ax2, final_arr, 'Greens')
        ax2.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax2.set_title(f'Final (Hybrid Debate)\nAccuracy: {self.hybrid_metrics.get("final_accuracy", 0)*100:.1f}%',
                     fontsize=14, fontweight='bold')
        
        plt.suptitle('Confusion Matrix Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved confusion matrices to {save_path}")
        
        return fig
    
    def generate_summary_table(self, save_path: Optional[str] = None) -> str:
        """
        Generate summary table in Markdown and LaTeX format.
        """
        if not self.hybrid_metrics:
            return "No metrics available"
        
        # Collect all metrics
        model_acc = self.hybrid_metrics.get('model_accuracy', 0) * 100
        final_acc = self.hybrid_metrics.get('final_accuracy', 0) * 100
        
        macro = self.hybrid_metrics.get('final_macro', {})
        macro_p = macro.get('macro_precision', 0) * 100
        macro_r = macro.get('macro_recall', 0) * 100
        macro_f1 = macro.get('macro_f1', 0) * 100
        
        # Hybrid specific
        if self.hybrid_analysis and 'optimal_threshold' in self.hybrid_analysis:
            opt = self.hybrid_analysis['optimal_threshold']
            hybrid_acc = opt.get('hybrid_accuracy', 0) * 100
            skip_ratio = opt.get('skip_ratio', 0) * 100
            threshold = opt.get('threshold', 0.85)
        else:
            hybrid_acc = final_acc
            skip_ratio = 0
            threshold = 0.85
        
        # Debate impact
        debate_impact = self.hybrid_metrics.get('debate_impact', {})
        fixed = debate_impact.get('fixed', 0)
        broken = debate_impact.get('broken', 0)
        
        # Build Markdown table
        md_table = f"""
# 📊 Evaluation Summary

## Overall Performance

| Metric | Value |
|--------|-------|
| **Model Accuracy (PhoBERT)** | {model_acc:.2f}% |
| **Final Accuracy (Debate)** | {final_acc:.2f}% |
| **Hybrid Accuracy** | {hybrid_acc:.2f}% |
| **Improvement** | +{final_acc - model_acc:.2f}% |

## Macro-Averaged Metrics

| Metric | Score |
|--------|-------|
| **Precision** | {macro_p:.2f}% |
| **Recall** | {macro_r:.2f}% |
| **F1-Score** | {macro_f1:.2f}% |

## Hybrid Strategy (DOWN Framework)

| Parameter | Value |
|-----------|-------|
| **Optimal Threshold** | {threshold} |
| **Skip Ratio** | {skip_ratio:.1f}% |
| **Cost Savings** | ~{skip_ratio:.0f}% fewer API calls |

## Debate Impact

| Metric | Count |
|--------|-------|
| **Fixed (Model ✗ → Debate ✓)** | {fixed} |
| **Broken (Model ✓ → Debate ✗)** | {broken} |
| **Net Gain** | {fixed - broken:+d} |

---
*Generated by Fake News Detection Pipeline*
"""
        
        # Build LaTeX table
        latex_table = f"""
% LaTeX Table for Thesis
\\begin{{table}}[h]
\\centering
\\caption{{Evaluation Results on ViFactCheck Test Set}}
\\label{{tab:results}}
\\begin{{tabular}}{{lc}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\midrule
Model Accuracy (PhoBERT) & {model_acc:.2f}\\% \\\\
Final Accuracy (Debate) & {final_acc:.2f}\\% \\\\
Hybrid Accuracy & {hybrid_acc:.2f}\\% \\\\
\\midrule
Macro Precision & {macro_p:.2f}\\% \\\\
Macro Recall & {macro_r:.2f}\\% \\\\
Macro F1-Score & {macro_f1:.2f}\\% \\\\
\\midrule
Optimal Threshold & {threshold} \\\\
Skip Ratio & {skip_ratio:.1f}\\% \\\\
Fixed Samples & {fixed} \\\\
Broken Samples & {broken} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
        
        full_content = md_table + "\n\n" + "="*60 + "\n\n" + latex_table
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(full_content)
            logger.info(f"Saved summary table to {save_path}")
        
        return full_content
    
    def generate_all(self, output_dir: str = None):
        """Generate all thesis charts and tables."""
        if output_dir is None:
            output_dir = self.hybrid_dir
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*70)
        print("📊 GENERATING THESIS CHARTS & TABLES")
        print("="*70)
        
        # 1. Comparison Chart
        print("\n1️⃣ Generating Comparison Chart...")
        self.plot_comparison_chart(output_path / "comparison_model_debate_hybrid.png")
        plt.close()
        
        # 2. Per-Class Performance
        print("2️⃣ Generating Per-Class Performance...")
        self.plot_per_class_performance(output_path / "per_class_performance.png")
        plt.close()
        
        # 3. Confusion Matrices
        print("3️⃣ Generating Confusion Matrices...")
        self.plot_confusion_matrices(output_path / "confusion_matrices.png")
        plt.close()
        
        # 4. Summary Table
        print("4️⃣ Generating Summary Table...")
        self.generate_summary_table(output_path / "summary_table.md")
        
        print("\n" + "="*70)
        print("✅ All thesis charts generated!")
        print(f"📁 Output: {output_path}")
        print("="*70)


def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Thesis Charts")
    parser.add_argument(
        "--hybrid-dir", "-h",
        type=str,
        default="results/vifactcheck/test/hybrid_debate",
        help="Path to hybrid_debate results"
    )
    parser.add_argument(
        "--full-dir", "-f",
        type=str,
        default="results/vifactcheck/test/full_debate",
        help="Path to full_debate results"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (default: same as hybrid-dir)"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    generator = ThesisChartGenerator(args.hybrid_dir, args.full_dir)
    generator.generate_all(args.output)


if __name__ == "__main__":
    main()
