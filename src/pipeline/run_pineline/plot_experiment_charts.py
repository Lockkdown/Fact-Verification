"""
Plot Experiment Charts for Scope-aligned Experiments (Dec 2025)

Charts aligned with docs/SOME_METRICS&CHARTS.md:
1. Main Results Table (9 settings)
2. Fixed K improves? (Line chart: X=K, Y=Macro-F1)
3. Early stopping saves cost? (Histogram: stopping round distribution)
4. Quality vs Cost trade-off (Scatter: X=Avg Rounds, Y=Macro-F1)
5. Hybrid threshold sweep (Dual-axis: X=threshold, Y=Macro-F1 + Avg Rounds)

Usage:
    python plot_experiment_charts.py --results-dir results/experiments/ --output-dir charts/
    python plot_experiment_charts.py --demo  # Run with mock data for testing
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
import seaborn as sns


@dataclass
class ExperimentResult:
    """Single experiment result (1 of 9 settings)."""
    setting_name: str
    accuracy: float
    macro_f1: float
    avg_rounds: float
    model_accuracy: Optional[float] = None
    avg_tokens: Optional[float] = None
    latency_p95: Optional[float] = None
    routed_to_debate_pct: Optional[float] = None  # Only for Hybrid
    rounds_distribution: Dict[int, int] = field(default_factory=dict)
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    n_samples: int = 0
    std_accuracy: Optional[float] = None  # For multiple runs
    std_macro_f1: Optional[float] = None


def load_experiment_results(results_dir: str, split: str = "test") -> List[ExperimentResult]:
    """
    Load experiment results from directory.
    
    Supports two structures:
    
    1) Flat structure (for aggregated results):
        results_dir/
            phobert_baseline/metrics.json
            full_fixed_k3/metrics.json
            ...
    
    2) Nested structure from eval_vifactcheck_pipeline.py:
        results_dir/
            {split}/
                full_debate/
                    fixed_k3/metrics/metrics_{split}.json
                    earlystop_max7/metrics/metrics_{split}.json
                hybrid_debate/
                    fixed_k3/metrics/metrics_{split}.json
                    ...
    """
    results = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"⚠️ Results directory not found: {results_dir}")
        return results
    
    # Try nested structure first (from eval script)
    split_dir = results_path / split
    if split_dir.exists():
        for debate_mode in ["full_debate", "hybrid_debate"]:
            mode_dir = split_dir / debate_mode
            if not mode_dir.exists():
                continue
            
            for config_dir in mode_dir.iterdir():
                if not config_dir.is_dir():
                    continue
                
                # Look for metrics file in metrics/ subfolder
                metrics_file = config_dir / "metrics" / f"metrics_{split}.json"
                if not metrics_file.exists():
                    # Fallback to direct metrics.json
                    metrics_file = config_dir / "metrics.json"
                
                if metrics_file.exists():
                    with open(metrics_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Try to infer rounds distribution + avg rounds from round_by_round_accuracy
                    rounds_dist = data.get('rounds_distribution', {})
                    if not rounds_dist and isinstance(data.get('round_by_round_accuracy'), dict):
                        rounds_dist = data['round_by_round_accuracy'].get('rounds_distribution', {}) or {}

                    def _compute_avg_rounds(dist: Dict[Any, Any]) -> float:
                        if not dist:
                            return 0.0
                        total = 0
                        weighted = 0
                        for k, v in dist.items():
                            try:
                                rk = int(k)
                            except Exception:
                                continue
                            try:
                                cnt = int(v)
                            except Exception:
                                continue
                            total += cnt
                            weighted += rk * cnt
                        return (weighted / total) if total > 0 else 0.0

                    avg_rounds_val = data.get('avg_rounds_used', data.get('avg_rounds', None))
                    if avg_rounds_val is None or avg_rounds_val == 0:
                        avg_rounds_val = _compute_avg_rounds(rounds_dist)

                    # n_samples fallback: some metrics files only store totals under debate_impact or round_by_round_accuracy
                    n_samples_val = data.get('total_samples', 0)
                    if not n_samples_val and isinstance(data.get('debate_impact'), dict):
                        n_samples_val = data['debate_impact'].get('total_analyzed', 0) or 0
                    if not n_samples_val and isinstance(data.get('round_by_round_accuracy'), dict):
                        n_samples_val = data['round_by_round_accuracy'].get('final_total', 0) or 0
                    
                    # Build setting name: e.g., "Full Fixed K=3" or "Hybrid EarlyStop max7"
                    mode_name = "Full" if debate_mode == "full_debate" else "Hybrid"
                    config_name = config_dir.name  # e.g., "fixed_k3", "earlystop_max7"
                    
                    # Parse config name to display format
                    if config_name.startswith("fixed_k"):
                        k = config_name.replace("fixed_k", "")
                        setting_name = f"{mode_name} Fixed K={k}"
                    elif config_name.startswith("earlystop_"):
                        suffix = config_name.replace("earlystop_", "")
                        setting_name = f"{mode_name} EarlyStop {suffix}"
                    else:
                        setting_name = f"{mode_name} {config_name}"
                    
                    result = ExperimentResult(
                        setting_name=setting_name,
                        model_accuracy=data.get('model_accuracy', None),
                        accuracy=data.get('final_accuracy', data.get('accuracy', 0)),
                        macro_f1=data.get('final_macro', {}).get('macro_f1', 0),
                        avg_rounds=float(avg_rounds_val) if avg_rounds_val is not None else 0.0,
                        avg_tokens=data.get('avg_tokens', None),
                        latency_p95=data.get('latency_p95', None),
                        routed_to_debate_pct=data.get('routed_to_debate_pct', None),
                        rounds_distribution=rounds_dist,
                        per_class_f1=data.get('per_class_f1', {}),
                        n_samples=int(n_samples_val) if n_samples_val else 0,
                        std_accuracy=data.get('std_accuracy', None),
                        std_macro_f1=data.get('std_macro_f1', None)
                    )
                    results.append(result)
    
    # Fallback to flat structure
    if not results:
        for subdir in results_path.iterdir():
            if subdir.is_dir():
                metrics_file = subdir / "metrics.json"
                if metrics_file.exists():
                    with open(metrics_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    result = ExperimentResult(
                        setting_name=subdir.name,
                        accuracy=data.get('final_accuracy', data.get('accuracy', 0)),
                        macro_f1=data.get('final_macro', {}).get('macro_f1', 0),
                        avg_rounds=data.get('avg_rounds_used', data.get('avg_rounds', 1)),
                        avg_tokens=data.get('avg_tokens', None),
                        latency_p95=data.get('latency_p95', None),
                        routed_to_debate_pct=data.get('routed_to_debate_pct', None),
                        rounds_distribution=data.get('rounds_distribution', {}),
                        per_class_f1=data.get('per_class_f1', {}),
                        n_samples=data.get('total_samples', 0),
                        std_accuracy=data.get('std_accuracy', None),
                        std_macro_f1=data.get('std_macro_f1', None),
                    )
                    results.append(result)
    
    return results


def generate_mock_data() -> List[ExperimentResult]:
    """Generate mock data for testing charts."""
    np.random.seed(42)
    
    mock_results = [
        # Baseline
        ExperimentResult(
            setting_name="PhoBERT Baseline",
            accuracy=0.8321,
            macro_f1=0.8250,
            avg_rounds=0,  # No debate
            n_samples=1447,
        ),
        # Full Debate Fixed K
        ExperimentResult(
            setting_name="Full Fixed K=3",
            accuracy=0.8456,
            macro_f1=0.8380,
            avg_rounds=3.0,
            rounds_distribution={3: 1447},
            n_samples=1447,
        ),
        ExperimentResult(
            setting_name="Full Fixed K=5",
            accuracy=0.8612,
            macro_f1=0.8540,
            avg_rounds=5.0,
            rounds_distribution={5: 1447},
            n_samples=1447,
        ),
        ExperimentResult(
            setting_name="Full Fixed K=7",
            accuracy=0.8694,
            macro_f1=0.8620,
            avg_rounds=7.0,
            rounds_distribution={7: 1447},
            n_samples=1447,
        ),
        # Full Debate EarlyStop
        ExperimentResult(
            setting_name="Full EarlyStop max7",
            accuracy=0.8650,
            macro_f1=0.8580,
            avg_rounds=4.2,
            rounds_distribution={2: 320, 3: 450, 4: 280, 5: 180, 6: 120, 7: 97},
            n_samples=1447,
        ),
        # Hybrid Fixed K
        ExperimentResult(
            setting_name="Hybrid Fixed K=3",
            accuracy=0.8720,
            macro_f1=0.8650,
            avg_rounds=0.72,  # 24% routed * 3 rounds
            routed_to_debate_pct=24.0,
            rounds_distribution={3: 347},
            n_samples=1447,
        ),
        ExperimentResult(
            setting_name="Hybrid Fixed K=5",
            accuracy=0.8755,
            macro_f1=0.8690,
            avg_rounds=1.2,  # 24% routed * 5 rounds
            routed_to_debate_pct=24.0,
            rounds_distribution={5: 347},
            n_samples=1447,
        ),
        ExperimentResult(
            setting_name="Hybrid Fixed K=7",
            accuracy=0.8777,
            macro_f1=0.8710,
            avg_rounds=1.68,  # 24% routed * 7 rounds
            routed_to_debate_pct=24.0,
            rounds_distribution={7: 347},
            n_samples=1447,
        ),
        # Hybrid EarlyStop
        ExperimentResult(
            setting_name="Hybrid EarlyStop max7",
            accuracy=0.8790,
            macro_f1=0.8725,
            avg_rounds=1.01,  # 24% routed * 4.2 avg rounds
            routed_to_debate_pct=24.0,
            rounds_distribution={2: 77, 3: 108, 4: 67, 5: 43, 6: 29, 7: 23},
            n_samples=1447,
        ),
    ]
    
    return mock_results


# =============================================================================
# CHART 1: Main Results Table
# =============================================================================
def plot_main_results_table(
    results: List[ExperimentResult],
    output_file: str,
    title: str = "Main Results"
):
    """
    Create main results table as a figure.
    
    Columns: Setting | Accuracy | Macro-F1 | Avg Rounds | % Routed (Hybrid)
    """
    # Prepare data
    data = []
    for r in results:
        routed = f"{r.routed_to_debate_pct:.1f}%" if r.routed_to_debate_pct else "-"
        data.append([
            r.setting_name,
            f"{r.accuracy:.2%}",
            f"{r.macro_f1:.4f}",
            f"{r.avg_rounds:.2f}" if r.avg_rounds > 0 else "-",
            routed
        ])
    
    # Sort by setting name for consistent order
    # Order: Baseline, Full Fixed K=3/5/7, Full ES, Hybrid Fixed K=3/5/7, Hybrid ES
    order_map = {
        "PhoBERT Baseline": 0,
        "Full Fixed K=3": 1, "Full Fixed K=5": 2, "Full Fixed K=7": 3,
        "Full EarlyStop max7": 4,
        "Hybrid Fixed K=3": 5, "Hybrid Fixed K=5": 6, "Hybrid Fixed K=7": 7,
        "Hybrid EarlyStop max7": 8,
    }
    data.sort(key=lambda x: order_map.get(x[0], 99))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    columns = ['Setting', 'Accuracy', 'Macro-F1', 'Avg Rounds', '% Routed']
    
    # Create table
    table = ax.table(
        cellText=data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colColours=['#2c3e50'] * 5
    )
    
    # Style
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Header style
    for i in range(len(columns)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Highlight best accuracy row
    best_acc_idx = max(range(len(data)), key=lambda i: float(data[i][1].strip('%')) / 100)
    for i in range(len(columns)):
        table[(best_acc_idx + 1, i)].set_facecolor('#d5f5e3')
    
    # Dynamic title (avoid hardcoded "9 settings")
    plt.title(f"{title} ({len(results)} Settings)", fontsize=16, fontweight='bold', pad=20)
    # Reserve top space for legend
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved main results table: {output_file}")


# =============================================================================
# CHART 2: Fixed K improves? (Line chart)
# =============================================================================
def plot_fixed_k_comparison(
    results: List[ExperimentResult],
    output_file: str,
    title: str = "Does Increasing K Improve Quality?"
):
    """
    Line chart: X = K (3, 5, 7), Y = Macro-F1
    Shows both Full Debate and Hybrid (optional)
    """
    # Filter Full Fixed K results
    full_fixed = [(r, int(r.setting_name.split('=')[1])) for r in results 
                  if 'Full Fixed K=' in r.setting_name]
    full_fixed.sort(key=lambda x: x[1])
    
    # Filter Hybrid Fixed K results
    hybrid_fixed = [(r, int(r.setting_name.split('=')[1])) for r in results 
                    if 'Hybrid Fixed K=' in r.setting_name]
    hybrid_fixed.sort(key=lambda x: x[1])
    
    if not full_fixed:
        print("⚠️ No Full Fixed K results found, skipping Fixed K chart")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Full Debate
    k_values = [x[1] for x in full_fixed]
    f1_values = [x[0].macro_f1 for x in full_fixed]
    ax.plot(k_values, f1_values, marker='o', linewidth=3, markersize=12, 
            color='#e74c3c', label='Full Debate')
    
    # Add value labels
    for k, f1 in zip(k_values, f1_values):
        ax.text(k, f1 + 0.005, f'{f1:.4f}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    # Plot Hybrid if available
    if hybrid_fixed:
        k_values_h = [x[1] for x in hybrid_fixed]
        f1_values_h = [x[0].macro_f1 for x in hybrid_fixed]
        ax.plot(k_values_h, f1_values_h, marker='s', linewidth=3, markersize=12, 
                color='#3498db', label='Hybrid')
        for k, f1 in zip(k_values_h, f1_values_h):
            ax.text(k, f1 + 0.005, f'{f1:.4f}', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold', color='#3498db')
    
    # Formatting
    ax.set_xlabel('Number of Rounds (K)', fontsize=14)
    ax.set_ylabel('Macro-F1', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks([3, 5, 7])
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Set y-axis limits based on data range
    all_f1 = f1_values + (f1_values_h if hybrid_fixed else [])
    y_min = min(all_f1) - 0.02
    y_max = max(all_f1) + 0.02
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved Fixed K comparison chart: {output_file}")


# =============================================================================
# CHART 3: Early stopping saves cost? (Histogram)
# =============================================================================
def plot_early_stopping_distribution(
    results: List[ExperimentResult],
    output_file: str,
    title: str = "Early Stopping: Rounds Distribution"
):
    """
    Histogram showing distribution of stopping rounds for EarlyStop settings.
    Shows Avg Rounds and p95 Rounds below chart.
    """
    # Find EarlyStop results
    es_results = [r for r in results if 'EarlyStop' in r.setting_name]
    
    if not es_results:
        print("⚠️ No EarlyStop results found, skipping distribution chart")
        return
    
    fig, axes = plt.subplots(1, len(es_results), figsize=(7 * len(es_results), 6))
    if len(es_results) == 1:
        axes = [axes]
    
    # Dynamic colors (avoid truncation when >2 settings)
    colors = plt.cm.Set2(np.linspace(0.2, 0.9, len(es_results)))
    
    for idx, r in enumerate(es_results):
        ax = axes[idx]
        color = colors[idx]
        dist = r.rounds_distribution
        if not dist:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Convert keys to int and sort
        rounds = sorted([int(k) for k in dist.keys()])
        counts = [dist.get(r, dist.get(str(r), 0)) for r in rounds]
        total = sum(counts)
        
        # Bar chart
        bars = ax.bar([f'R{r}' for r in rounds], counts, color=color, alpha=0.7, edgecolor='black')
        
        # Add labels
        for bar, count in zip(bars, counts):
            pct = count / total * 100 if total > 0 else 0
            ax.text(bar.get_x() + bar.get_width()/2, count + total * 0.02,
                    f'{count}\n({pct:.1f}%)', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
        
        # Calculate stats
        avg_rounds = r.avg_rounds
        
        # Calculate p95
        cumsum = 0
        p95_round = rounds[-1]
        for rd, cnt in zip(rounds, counts):
            cumsum += cnt
            if cumsum / total >= 0.95:
                p95_round = rd
                break
        
        # Add stats below
        stats_text = f"Avg Rounds: {avg_rounds:.2f} | p95: Round {p95_round}"
        ax.text(0.5, -0.12, stats_text, ha='center', va='top', 
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(r.setting_name, fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(counts) * 1.25 if counts else 1)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.suptitle("EarlyStop: Rounds Used Distribution", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved early stopping distribution chart: {output_file}")


# =============================================================================
# CHART 4: Quality vs Cost trade-off (Scatter)
# =============================================================================
def plot_quality_vs_cost(
    results: List[ExperimentResult],
    output_file: str,
    title: str = "Quality vs Cost Trade-off"
):
    """
    Scatter plot: X = Avg Rounds (cost proxy), Y = Macro-F1 (quality)
    Each point = 1 setting, labeled.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color mapping
    colors = {
        'PhoBERT': '#95a5a6',
        'Full Fixed': '#e74c3c',
        'Full EarlyStop': '#c0392b',
        'Hybrid Fixed': '#3498db',
        'Hybrid EarlyStop': '#2980b9',
    }
    
    markers = {
        'PhoBERT': 'D',
        'Full Fixed': 'o',
        'Full EarlyStop': 's',
        'Hybrid Fixed': '^',
        'Hybrid EarlyStop': 'v',
    }
    
    plotted_cats = set()
    for r in results:
        # Determine category
        if 'PhoBERT' in r.setting_name:
            cat = 'PhoBERT'
        elif 'Full Fixed' in r.setting_name:
            cat = 'Full Fixed'
        elif 'Full EarlyStop' in r.setting_name:
            cat = 'Full EarlyStop'
        elif 'Hybrid Fixed' in r.setting_name:
            cat = 'Hybrid Fixed'
        else:
            cat = 'Hybrid EarlyStop'

        label = cat if cat not in plotted_cats else None
        ax.scatter(
            r.avg_rounds,
            r.macro_f1,
            c=colors[cat],
            marker=markers[cat],
            s=200,
            edgecolors='black',
            linewidths=1.5,
            alpha=0.85,
            label=label,
            zorder=3,
        )
        plotted_cats.add(cat)

        # Shorter label for readability in small-N quick runs
        short_name = r.setting_name
        if 'EarlyStop' in short_name and 'k' in short_name.lower():
            # e.g., "Full EarlyStop k3" -> "k3"
            import re
            m = re.search(r'k(\d+)', short_name.lower())
            if m:
                short_name = f"k{m.group(1)}"

        # Use offset-points (screen space) instead of data offsets to avoid labels going out-of-bounds
        # Also adapt direction: rightmost points (e.g., k7) label to the left.
        dx, dy = (10, 10)
        if r.avg_rounds >= max([rr.avg_rounds for rr in results]) - 1e-6:
            dx = -28
        ax.annotate(
            short_name,
            (r.avg_rounds, r.macro_f1),
            textcoords='offset points',
            xytext=(dx, dy),
            fontsize=10,
            ha='left',
            va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#444'),
            zorder=4,
        )
    
    # Formatting
    ax.set_xlabel('Average Rounds (Cost Proxy)', fontsize=14)
    ax.set_ylabel('Macro-F1 (Quality)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved quality vs cost chart: {output_file}")


# =============================================================================
# CHART 5: Hybrid threshold sweep (Dual-axis)
# =============================================================================
def plot_threshold_sweep(
    threshold_data: List[Dict],
    output_file: str,
    title: str = "Hybrid Threshold Sweep"
):
    """
    Dual-axis line chart: X = threshold, Y1 = Macro-F1, Y2 = Avg Rounds
    
    threshold_data format:
    [
        {"threshold": 0.1, "macro_f1": 0.85, "avg_rounds": 5.0},
        {"threshold": 0.2, "macro_f1": 0.86, "avg_rounds": 4.5},
        ...
    ]
    """
    if not threshold_data:
        print("⚠️ No threshold sweep data, skipping threshold sweep chart")
        return
    
    thresholds = [d["threshold"] for d in threshold_data]
    f1_scores = [d["macro_f1"] for d in threshold_data]
    avg_rounds = [d["avg_rounds"] for d in threshold_data]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Left Y-axis: Macro-F1
    color1 = '#2ecc71'
    ax1.set_xlabel('Confidence Threshold (t)', fontsize=14)
    ax1.set_ylabel('Macro-F1', fontsize=14, color=color1)
    line1, = ax1.plot(thresholds, f1_scores, marker='o', linewidth=3, markersize=10, 
                      color=color1, label='Macro-F1')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(min(f1_scores) - 0.02, max(f1_scores) + 0.02)
    
    # Right Y-axis: Avg Rounds
    ax2 = ax1.twinx()
    color2 = '#e74c3c'
    ax2.set_ylabel('Avg Rounds', fontsize=14, color=color2)
    line2, = ax2.plot(thresholds, avg_rounds, marker='s', linewidth=3, markersize=10, 
                      color=color2, linestyle='--', label='Avg Rounds')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, max(avg_rounds) * 1.1)
    
    # Legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=12)
    
    # Highlight optimal threshold (example: t=0.85)
    optimal_t = 0.85
    if optimal_t in thresholds:
        idx = thresholds.index(optimal_t)
        ax1.axvline(x=optimal_t, color='gray', linestyle=':', alpha=0.7)
        ax1.annotate(f'Optimal t={optimal_t}', xy=(optimal_t, f1_scores[idx]),
                     xytext=(optimal_t + 0.05, f1_scores[idx] + 0.01),
                     fontsize=11, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='gray'))
    
    ax1.set_title(title, fontsize=16, fontweight='bold')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_xticks(thresholds)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved threshold sweep chart: {output_file}")


# =============================================================================
# CHART 6 (Nice-to-have): Bucket analysis by PhoBERT confidence
# =============================================================================
def plot_bucket_analysis(
    bucket_data: List[Dict],
    output_file: str,
    title: str = "Performance by PhoBERT Confidence Bucket"
):
    """
    Bar chart: X = confidence buckets, Y = Macro-F1 for PhoBERT/Debate/Hybrid
    
    bucket_data format:
    [
        {"bucket": "0.0-0.5", "phobert_f1": 0.65, "debate_f1": 0.75, "hybrid_f1": 0.78},
        ...
    ]
    """
    if not bucket_data:
        print("⚠️ No bucket data, skipping bucket analysis chart")
        return
    
    buckets = [d["bucket"] for d in bucket_data]
    phobert_f1 = [d["phobert_f1"] for d in bucket_data]
    debate_f1 = [d["debate_f1"] for d in bucket_data]
    hybrid_f1 = [d.get("hybrid_f1", d["debate_f1"]) for d in bucket_data]
    
    x = np.arange(len(buckets))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    bars1 = ax.bar(x - width, phobert_f1, width, label='PhoBERT', color='#95a5a6', alpha=0.8)
    bars2 = ax.bar(x, debate_f1, width, label='Debate (K=7)', color='#e74c3c', alpha=0.8)
    bars3 = ax.bar(x + width, hybrid_f1, width, label='Hybrid', color='#3498db', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('PhoBERT Confidence Bucket', fontsize=14)
    ax.set_ylabel('Macro-F1', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved bucket analysis chart: {output_file}")


# =============================================================================
# NEW: Round-by-Round Accuracy (Dec 24, 2025)
# =============================================================================
def plot_round_by_round_accuracy_v2(
    metrics_file: str,
    results_file: str,
    split_name: str,
    output_file: str
):
    """
    Vẽ biểu đồ round-by-round accuracy RÕ RÀNG cho early-stop debates.
    
    Strategy: CHỈ hiển thị Carry-Forward Accuracy (all samples) - không hiển thị subset
    để tránh hiểu nhầm "R1 tốt nhất".
    
    Args:
        metrics_file: Path to metrics_{split}.json
        results_file: Path to {split}_results.json
        split_name: Split name (dev/test) for title
        output_file: Output PNG path
    """
    import json
    from collections import Counter
    
    # Load data
    with open(metrics_file, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        results = data['results'] if isinstance(data, dict) and 'results' in data else data
    
    def _norm(label):
        if not isinstance(label, str):
            label = str(label)
        label = label.upper().strip()
        if label in ["SUPPORT", "SUPPORTS", "SUPPORTED", "0"]:
            return "Support"
        if label in ["REFUTE", "REFUTES", "REFUTED", "1"]:
            return "Refute"
        if label in ["NEI", "NOT_ENOUGH_INFO", "NOT ENOUGH INFO", "2", "UNVERIFIED"]:
            return "NOT_ENOUGH_INFO"
        return label.title() if label else "NOT_ENOUGH_INFO"
    
    def _majority(verdicts):
        c = Counter(verdicts)
        return c.most_common(1)[0][0] if c else "NOT_ENOUGH_INFO"
    
    # Detect available rounds
    max_round = 0
    for r in results:
        debate_res = r.get("debate_result", {}) if isinstance(r.get("debate_result"), dict) else {}
        all_rounds = debate_res.get("all_rounds_verdicts", [])
        if isinstance(all_rounds, list):
            max_round = max(max_round, len(all_rounds))
    
    if max_round == 0:
        print("⚠️  No round data found, skipping round-by-round chart")
        return
    
    # Compute carry-forward accuracy per round
    total_valid = 0
    correct_by_round = {r: 0 for r in range(1, max_round + 1)}
    
    for r in results:
        gold = _norm(r.get("gold_label", ""))
        if not gold or r.get("final_verdict") == "ERROR" or r.get("model_verdict") == "ERROR":
            continue
        
        debate_res = r.get("debate_result", {}) if isinstance(r.get("debate_result"), dict) else {}
        all_rounds = debate_res.get("all_rounds_verdicts", [])
        if not isinstance(all_rounds, list) or not all_rounds:
            continue
        
        total_valid += 1
        
        # Compute majority verdict per round (carry forward if stopped early)
        majority_per_round = []
        for round_data in all_rounds:
            if not isinstance(round_data, dict):
                majority_per_round.append("NOT_ENOUGH_INFO")
                continue
            vs = [_norm(v.get("verdict", "NOT_ENOUGH_INFO")) for v in round_data.values() if isinstance(v, dict)]
            majority_per_round.append(_majority(vs))
        
        # Carry forward: each round uses the last available verdict
        for round_num in range(1, max_round + 1):
            idx = min(round_num, len(majority_per_round)) - 1
            pred = majority_per_round[idx]
            if pred == gold:
                correct_by_round[round_num] += 1
    
    if total_valid == 0:
        print("⚠️  No valid samples, skipping round-by-round chart")
        return
    
    # Compute accuracy per round
    rounds = list(range(1, max_round + 1))
    accuracies = [correct_by_round[r] / total_valid for r in rounds]
    
    # Get model and final accuracy for comparison
    model_acc = metrics.get("model_accuracy", 0.0)
    final_acc = metrics.get("final_accuracy", 0.0)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Baseline and Final as reference lines
    ax.axhline(y=model_acc, color='#3498db', linestyle='--', linewidth=2, label=f'Model Baseline ({model_acc:.1%})')
    ax.axhline(y=final_acc, color='#e74c3c', linestyle='--', linewidth=2, label=f'Final (Judge) ({final_acc:.1%})')
    
    # Bars for each round
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(rounds)))
    bars = ax.bar([f"R{r}" for r in rounds], accuracies, color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{acc:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Annotation
    ax.text(0.02, 0.98, f'N = {total_valid} samples (carry-forward logic)',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    
    ax.set_xlabel('Debate Round', fontsize=13)
    ax.set_ylabel('Accuracy (All Samples)', fontsize=13)
    ax.set_title(f'Round-by-Round Accuracy - Carry-Forward Strategy ({split_name.capitalize()})',
                 fontsize=15, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved round-by-round accuracy chart (v2): {output_file}")


# =============================================================================
# NEW: Consensus Evolution Chart (no selection bias)
# =============================================================================
def plot_consensus_evolution(results_file: str, split_name: str, output_file: str):
    """
    Plot consensus evolution: % samples reaching unanimous (3/3) verdict per round.
    This metric is NOT affected by selection bias because it's computed on ALL samples.
    """
    # Load results
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get('results', data) if isinstance(data, dict) else data
    if not isinstance(results, list):
        print("⚠️  Invalid results format")
        return
    
    # Find max rounds
    max_round = 1
    for r in results:
        debate_res = r.get("debate_result", {}) if isinstance(r.get("debate_result"), dict) else {}
        all_rounds = debate_res.get("all_rounds_verdicts", [])
        if isinstance(all_rounds, list):
            max_round = max(max_round, len(all_rounds))
    
    # Count unanimous per round using carry-forward (no selection bias)
    unanimous_by_round = {rn: 0 for rn in range(1, max_round + 1)}
    total_valid = 0
    
    for r in results:
        debate_res = r.get("debate_result", {}) if isinstance(r.get("debate_result"), dict) else {}
        all_rounds = debate_res.get("all_rounds_verdicts", [])
        
        if not isinstance(all_rounds, list) or not all_rounds:
            continue
        
        total_valid += 1
        
        # Carry-forward: for rounds beyond early stop, reuse the last available round
        last_round_data = all_rounds[-1] if all_rounds else {}
        for round_num in range(1, max_round + 1):
            if round_num <= len(all_rounds):
                round_data = all_rounds[round_num - 1]
            else:
                round_data = last_round_data
            
            if isinstance(round_data, dict):
                verdicts = [v.get("verdict", "").upper() for v in round_data.values() if isinstance(v, dict)]
                # Unanimous = all 3 same verdict
                if len(verdicts) >= 3 and len(set(verdicts)) == 1:
                    unanimous_by_round[round_num] += 1
    
    if total_valid == 0:
        print("⚠️  No valid samples, skipping consensus evolution chart")
        return
    
    # Compute percentages
    rounds = list(range(1, max_round + 1))
    consensus_pct = [unanimous_by_round[r] / total_valid * 100 for r in rounds]
    
    # Create line chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(rounds, consensus_pct, marker='o', linewidth=2.5, markersize=10, 
            color='#2ecc71', label='Unanimous (3/3)')
    
    # Add value labels
    for r, pct in zip(rounds, consensus_pct):
        ax.text(r, pct + 2, f'{pct:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Annotation
    ax.text(0.02, 0.98, f'N = {total_valid} samples (carry-forward, no selection bias)',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))
    
    ax.set_xlabel('Debate Round', fontsize=13)
    ax.set_ylabel('% Samples with Unanimous Verdict', fontsize=13)
    ax.set_title(f'Consensus Evolution - Debate Convergence ({split_name.capitalize()})',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(rounds)
    ax.set_xticklabels([f'R{r}' for r in rounds])
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved consensus evolution chart: {output_file}")


# =============================================================================
# NEW: Compute Hybrid Cost Metrics (Dec 25, 2025)
# =============================================================================
def compute_hybrid_cost_metrics(results_file: str) -> Dict[str, Any]:
    """
    Compute cost metrics for Hybrid strategy from vifactcheck_test_results.json.
    
    Returns:
        Dict with:
        - total_samples: int
        - skipped_count: int (Fast Path)
        - debate_count: int (Slow Path)
        - skip_ratio: float (% samples skipped)
        - debate_ratio: float (% samples routed to debate)
        - relative_cost: float (debate_ratio, cost relative to full debate = 1.0)
        - threshold: float (threshold used)
    """
    import json
    from pathlib import Path
    
    results_path = Path(results_file)
    if not results_path.exists():
        print(f"⚠️ Results file not found: {results_file}")
        return {}
    
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get('results', data) if isinstance(data, dict) else data
    if not isinstance(results, list):
        print("⚠️ Invalid results format")
        return {}
    
    total_samples = len(results)
    skipped_count = 0
    debate_count = 0
    threshold_used = None
    
    for r in results:
        hybrid_info = r.get('hybrid_info', {})
        if isinstance(hybrid_info, dict):
            skipped = hybrid_info.get('skipped', False)
            if skipped:
                skipped_count += 1
            else:
                debate_count += 1
            
            # Extract threshold (should be same for all samples)
            if threshold_used is None:
                threshold_used = hybrid_info.get('threshold', None)
    
    skip_ratio = skipped_count / total_samples if total_samples > 0 else 0.0
    debate_ratio = debate_count / total_samples if total_samples > 0 else 0.0
    relative_cost = debate_ratio  # Cost relative to full debate (1.0)
    
    metrics = {
        'total_samples': total_samples,
        'skipped_count': skipped_count,
        'debate_count': debate_count,
        'skip_ratio': skip_ratio,
        'debate_ratio': debate_ratio,
        'relative_cost': relative_cost,
        'threshold': threshold_used
    }
    
    return metrics


def print_hybrid_cost_summary(results_dir: str, split: str = 'test', configs: List[str] = None):
    """
    Print cost summary for hybrid configurations.
    
    Args:
        results_dir: Base results directory (e.g., 'results/vifactcheck')
        split: 'test' or 'dev'
        configs: List of config names (e.g., ['earlystop_k3', 'earlystop_k5', 'earlystop_k7'])
    """
    from pathlib import Path
    
    results_path = Path(results_dir)
    hybrid_dir = results_path / split / 'hybrid_debate'
    
    if not hybrid_dir.exists():
        print(f"⚠️ Hybrid debate directory not found: {hybrid_dir}")
        return
    
    if configs is None:
        configs = ['earlystop_k3', 'earlystop_k5', 'earlystop_k7']
    
    print("\n" + "="*80)
    print(f"HYBRID COST METRICS - {split.upper()} SET")
    print("="*80)
    print(f"{'Config':<20} {'Total':<10} {'Skip':<10} {'Debate':<10} {'Skip %':<12} {'Cost %':<12} {'Threshold':<10}")
    print("-"*80)
    
    for config_name in configs:
        config_dir = hybrid_dir / config_name
        results_file = config_dir / f'vifactcheck_{split}_results.json'
        
        if not results_file.exists():
            print(f"{config_name:<20} File not found")
            continue
        
        metrics = compute_hybrid_cost_metrics(str(results_file))
        
        if not metrics:
            print(f"{config_name:<20} Error computing metrics")
            continue
        
        print(f"{config_name:<20} "
              f"{metrics['total_samples']:<10} "
              f"{metrics['skipped_count']:<10} "
              f"{metrics['debate_count']:<10} "
              f"{metrics['skip_ratio']*100:>10.2f}% "
              f"{metrics['relative_cost']*100:>10.2f}% "
              f"{metrics['threshold'] if metrics['threshold'] else 'N/A':<10}")
    
    print("="*80)
    print("\n📊 Cost Interpretation:")
    print("  - Skip %: Percentage of samples using Fast Path (model confidence high)")
    print("  - Cost %: Percentage of samples requiring debate (Slow Path)")
    print("  - Relative to Full Debate = 100% (all samples debated)")
    print()


# =============================================================================
# NEW: Verdict Flip Rate Chart (no selection bias)
# =============================================================================
def plot_verdict_flip_rate(results_file: str, split_name: str, output_file: str):
    """
    Plot verdict flip rate: % samples where majority verdict changed between rounds.
    Shows debate has actual impact on decisions.
    """
    # Load results
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get('results', data) if isinstance(data, dict) else data
    if not isinstance(results, list):
        print("⚠️  Invalid results format")
        return
    
    def _norm(label):
        if not isinstance(label, str):
            label = str(label)
        label = label.upper().strip()
        if label in ["SUPPORT", "SUPPORTS", "SUPPORTED", "0"]:
            return "SUPPORTED"
        if label in ["REFUTE", "REFUTES", "REFUTED", "1"]:
            return "REFUTED"
        return "NEI"
    
    def _majority(verdicts):
        from collections import Counter
        c = Counter(verdicts)
        return c.most_common(1)[0][0] if c else "NEI"
    
    # Find max rounds
    max_round = 1
    for r in results:
        debate_res = r.get("debate_result", {}) if isinstance(r.get("debate_result"), dict) else {}
        all_rounds = debate_res.get("all_rounds_verdicts", [])
        if isinstance(all_rounds, list):
            max_round = max(max_round, len(all_rounds))
    
    if max_round < 2:
        print("⚠️  Need at least 2 rounds for flip rate chart")
        return
    
    # Count flips per transition using carry-forward (no selection bias)
    transitions = [f"R{r}→R{r+1}" for r in range(1, max_round)]
    flip_counts = {t: 0 for t in transitions}
    total_valid = 0
    
    for r in results:
        debate_res = r.get("debate_result", {}) if isinstance(r.get("debate_result"), dict) else {}
        all_rounds = debate_res.get("all_rounds_verdicts", [])
        
        if not isinstance(all_rounds, list) or not all_rounds:
            continue
        
        total_valid += 1
        
        # Majority per available round
        majorities = []
        for round_data in all_rounds:
            if isinstance(round_data, dict):
                vs = [_norm(v.get("verdict", "NEI")) for v in round_data.values() if isinstance(v, dict)]
                majorities.append(_majority(vs))
            else:
                majorities.append("NEI")
        
        # Carry-forward to max_round so every transition uses the same denominator
        if len(majorities) < max_round:
            majorities.extend([majorities[-1]] * (max_round - len(majorities)))
        
        # Check flips across all transitions
        for i in range(max_round - 1):
            trans = f"R{i+1}→R{i+2}"
            if majorities[i] != majorities[i+1]:
                flip_counts[trans] += 1
    
    if total_valid == 0:
        print("⚠️  No valid samples, skipping flip rate chart")
        return
    
    # Compute percentages
    flip_pct = [flip_counts[t] / total_valid * 100 for t in transitions]
    
    # Create bar chart
    # Wider figure to accommodate long transition labels (R1→R2, ...)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.Oranges(np.linspace(0.4, 0.8, len(transitions)))
    bars = ax.bar(transitions, flip_pct, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, pct in zip(bars, flip_pct):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Annotation
    ax.text(0.02, 0.98, f'N = {total_valid} samples (carry-forward)',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    
    ax.set_xlabel('Round Transition', fontsize=13)
    ax.set_ylabel('% Samples with Verdict Change', fontsize=13)
    ax.set_title(f'Verdict Flip Rate - Debate Impact ({split_name.capitalize()})',
                 fontsize=14, fontweight='bold')
    # Avoid singular y-lims when all values are 0
    max_val = max(flip_pct) if flip_pct else 0
    if max_val <= 0:
        ax.set_ylim(0, 5)
    else:
        ax.set_ylim(0, max_val * 1.3)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Improve readability & avoid tight_layout warning
    ax.tick_params(axis='x', labelrotation=20)

    plt.tight_layout(pad=1.2)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved verdict flip rate chart: {output_file}")


# =============================================================================
# NEW: max_K Ablation Chart (replaces Fixed-K comparison)
# =============================================================================
def plot_maxk_ablation(results: List[ExperimentResult], output_file: str):
    """
    Plot max_K ablation: compare EarlyStop with different max_K values (3, 5, 7).
    Replaces the old Fixed-K comparison chart.
    """
    # Filter for EarlyStop settings only
    earlystop_results = [r for r in results if 'earlystop' in r.setting_name.lower()]
    
    if len(earlystop_results) < 2:
        print("⚠️  Need at least 2 EarlyStop settings for max_K ablation")
        return
    
    # Sort by max_K (extract from setting name)
    def extract_k(setting_name):
        import re
        match = re.search(r'k(\d+)', setting_name.lower())
        return int(match.group(1)) if match else 0
    
    earlystop_results.sort(key=lambda r: extract_k(r.setting_name))
    
    # Prepare data
    settings = [r.setting_name for r in earlystop_results]
    model_acc_pct = [((r.model_accuracy or 0.0) * 100) for r in earlystop_results]
    final_acc_pct = [r.accuracy * 100 for r in earlystop_results]
    macro_f1_pct = [r.macro_f1 * 100 for r in earlystop_results]
    avg_rounds = [r.avg_rounds for r in earlystop_results]
    
    # Create figure with 2 y-axes
    # Wider figure to accommodate legends + gain annotations
    fig, ax1 = plt.subplots(figsize=(12, 6.8))
    
    x = np.arange(len(settings))
    width = 0.35
    
    # Bars for model vs final accuracy + macro-F1
    width = 0.25
    bars0 = ax1.bar(x - width, model_acc_pct, width, label='Model Acc', color='#95a5a6', alpha=0.85)
    bars1 = ax1.bar(x, final_acc_pct, width, label='Final Acc (Debate)', color='#3498db', alpha=0.85)
    bars2 = ax1.bar(x + width, macro_f1_pct, width, label='Final Macro-F1', color='#2ecc71', alpha=0.85)
    
    ax1.set_xlabel('EarlyStop Setting (max_K)', fontsize=12)
    ax1.set_ylabel('Quality (%)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'max_K={extract_k(s)}' for s in settings], fontsize=11)
    ax1.set_ylim(0, 100)
    
    # Add value labels
    for bar in bars0:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
    
    # Second y-axis for avg rounds
    ax2 = ax1.twinx()
    ax2.plot(
        x,
        avg_rounds,
        marker='s',
        color='#e74c3c',
        linewidth=2,
        markersize=8,
        label='Avg Rounds',
        alpha=0.9,
        zorder=1,
    )
    ax2.set_ylabel('Avg Rounds', fontsize=12, color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    # Place Avg Rounds legend above plot (avoid covering data)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.0, 1.18), fontsize=10, frameon=True)
    
    # Gain annotations: FinalAcc - ModelAcc
    gains = [fa - ma for fa, ma in zip(final_acc_pct, model_acc_pct)]
    for i, g in enumerate(gains):
        y_pos = 98.0
        if i == len(gains) - 1:
            y_pos = 99.0
        ax1.text(
            x[i],
            y_pos,
            f"Gain: {g:+.1f}",
            ha='center',
            va='top',
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#444'),
            zorder=10,
        )

    # Use a figure-level title + legend to avoid overlap/cropping across backends
    fig.suptitle('max_K Ablation: Quality vs Cost (EarlyStop)', fontsize=14, fontweight='bold', y=0.98)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Figure-level legend placed BELOW the title
    handles = [bars0[0], bars1[0], bars2[0]]
    labels = ['Model Acc', 'Final Acc (Debate)', 'Final Macro-F1']
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.93), ncol=3, fontsize=10, frameon=True)

    # Reserve top space for title + legend
    # Reserve more top space for title + 2 legends
    fig.subplots_adjust(top=0.74)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved max_K ablation chart: {output_file}")


# =============================================================================
# MAIN: Generate all charts
# =============================================================================
def generate_all_charts(
    results: List[ExperimentResult],
    output_dir: str,
    threshold_data: Optional[List[Dict]] = None,
    bucket_data: Optional[List[Dict]] = None
):
    """Generate all charts for the experiment."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎨 Generating charts to: {output_dir}\n")
    
    # 0. Export summary JSON for easy reference
    summary_data = []
    for r in results:
        summary_data.append({
            "setting": r.setting_name,
            "accuracy": r.accuracy,
            "macro_f1": r.macro_f1,
            "avg_rounds": r.avg_rounds,
            "routed_to_debate_pct": r.routed_to_debate_pct,
            "n_samples": r.n_samples,
        })
    
    summary_file = output_path / "comparison_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"📋 Saved comparison summary: {summary_file}")
    
    # 1. Main Results Table
    plot_main_results_table(results, str(output_path / "01_main_results_table.png"))
    
    # 2. max_K Ablation (replaces Fixed-K comparison)
    plot_maxk_ablation(results, str(output_path / "02_maxk_ablation.png"))
    
    # 3. Early stopping distribution
    plot_early_stopping_distribution(results, str(output_path / "03_early_stopping_dist.png"))
    
    # 4. Quality vs Cost
    plot_quality_vs_cost(results, str(output_path / "04_quality_vs_cost.png"))
    
    # 5. Threshold sweep
    plot_threshold_sweep(threshold_data or [], str(output_path / "05_threshold_sweep.png"))
    
    # 6. Bucket analysis (nice-to-have)
    plot_bucket_analysis(bucket_data or [], str(output_path / "06_bucket_analysis.png"))
    
    # Note: Consensus Evolution and Verdict Flip Rate charts require results_file path
    # They should be called separately with: plot_consensus_evolution(results_file, split, output)
    #                                        plot_verdict_flip_rate(results_file, split, output)
    
    print(f"\n✅ All charts generated in: {output_dir}")
    print(f"📊 Total settings compared: {len(results)}")
    print(f"💡 For Consensus Evolution & Verdict Flip Rate, call:")
    print(f"   plot_consensus_evolution(results_file, split_name, output_file)")
    print(f"   plot_verdict_flip_rate(results_file, split_name, output_file)")


def main():
    parser = argparse.ArgumentParser(description="Generate experiment charts")
    parser.add_argument("--results-dir", type=str, help="Directory containing experiment results (e.g., results/vifactcheck)")
    parser.add_argument("--split", type=str, default="test", help="Split to load results from (dev/test)")
    parser.add_argument("--output-dir", type=str, default="charts/experiments", help="Output directory for charts")
    parser.add_argument("--demo", action="store_true", help="Run with mock data for testing")
    
    args = parser.parse_args()
    
    if args.demo:
        print("🧪 Running in DEMO mode with mock data...")
        results = generate_mock_data()
    elif args.results_dir:
        print(f"📂 Loading results from: {args.results_dir} (split={args.split})")
        results = load_experiment_results(args.results_dir, split=args.split)
        if not results:
            print("❌ No results found. Use --demo to test with mock data.")
            return
        print(f"✅ Loaded {len(results)} experiment results")
    else:
        print("❌ Please provide --results-dir or use --demo")
        return
    
    generate_all_charts(results, args.output_dir)


if __name__ == "__main__":
    main()
