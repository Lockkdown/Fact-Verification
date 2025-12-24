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
    title: str = "Main Results (9 Settings)"
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
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
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
    
    colors = ['#9b59b6', '#1abc9c']
    
    for idx, (r, ax, color) in enumerate(zip(es_results, axes, colors)):
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
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
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
        
        ax.scatter(r.avg_rounds, r.macro_f1, 
                   c=colors[cat], marker=markers[cat], s=200, 
                   edgecolors='black', linewidths=1.5, alpha=0.8,
                   label=cat if cat not in [l.get_label() for l in ax.collections] else '')
        
        # Label point
        # Offset for readability
        offset_x = 0.15
        offset_y = 0.003
        ax.annotate(r.setting_name.replace(' ', '\n'), 
                    (r.avg_rounds + offset_x, r.macro_f1 + offset_y),
                    fontsize=9, ha='left')
    
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
        print("⚠️ No threshold sweep data, generating mock data")
        # Mock data for testing
        threshold_data = [
            {"threshold": t, "macro_f1": 0.82 + 0.06 * (1 - t), "avg_rounds": 7 * (1 - t)}
            for t in np.arange(0.1, 1.0, 0.1)
        ]
    
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
        print("⚠️ No bucket data, generating mock data")
        bucket_data = [
            {"bucket": "0.0-0.5", "phobert_f1": 0.48, "debate_f1": 0.65, "hybrid_f1": 0.65},
            {"bucket": "0.5-0.6", "phobert_f1": 0.40, "debate_f1": 0.83, "hybrid_f1": 0.83},
            {"bucket": "0.6-0.7", "phobert_f1": 0.63, "debate_f1": 0.78, "hybrid_f1": 0.78},
            {"bucket": "0.7-0.8", "phobert_f1": 0.58, "debate_f1": 0.73, "hybrid_f1": 0.73},
            {"bucket": "0.8-0.9", "phobert_f1": 0.75, "debate_f1": 0.79, "hybrid_f1": 0.77},
            {"bucket": "0.9-1.0", "phobert_f1": 0.93, "debate_f1": 0.93, "hybrid_f1": 0.93},
        ]
    
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
    
    # 2. Fixed K comparison
    plot_fixed_k_comparison(results, str(output_path / "02_fixed_k_comparison.png"))
    
    # 3. Early stopping distribution
    plot_early_stopping_distribution(results, str(output_path / "03_early_stopping_dist.png"))
    
    # 4. Quality vs Cost
    plot_quality_vs_cost(results, str(output_path / "04_quality_vs_cost.png"))
    
    # 5. Threshold sweep
    plot_threshold_sweep(threshold_data or [], str(output_path / "05_threshold_sweep.png"))
    
    # 6. Bucket analysis (nice-to-have)
    plot_bucket_analysis(bucket_data or [], str(output_path / "06_bucket_analysis.png"))
    
    print(f"\n✅ All charts generated in: {output_dir}")
    print(f"📊 Total settings compared: {len(results)}")


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
