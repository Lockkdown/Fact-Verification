"""
Plot Metrics from ViFactCheck Evaluation

Creates visualizations:
- Confusion Matrix (heatmap with manual annotations)
- F1 Score per Class (bar chart)
- Precision/Recall per Class (grouped bar chart)
- Debate Impact (bar chart)
- Model vs Final Comparison (grouped bar chart)
- Debate Rounds Distribution (bar chart)
- Debator Performance (bar chart)

Usage:
    Called from test_vifactcheck_pipeline.py with --full-report
    Or standalone: python plot_metrics.py metrics_dev.json --split dev --output-dir charts/
"""

import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix


def load_metrics(json_file: str) -> dict:
    """Load metrics from JSON file"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_confusion_matrix(cm_data: dict, split_name: str, output_file: str):
    """
    Plot confusion matrix with manual text annotations (fix for invisible numbers).
    
    Args:
        cm_data: Confusion matrix data
        split_name: Split name (Dev/Test) for title
        output_file: Output file path
    """
    labels = cm_data["labels"]
    cm = cm_data["matrix"]
    
    # Convert to numpy array
    matrix = np.array([[cm[true_label][pred_label] for pred_label in labels] for true_label in labels])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap WITHOUT annotations first
    sns.heatmap(
        matrix,
        annot=False,  # CRITICAL: Turn off automatic annotations
        fmt='d',
        cmap='Blues',
        xticklabels=[l.replace("_", "\n") for l in labels],
        yticklabels=[l.replace("_", "\n") for l in labels],
        cbar_kws={'label': 'Count'},
        ax=ax,
        linewidths=2,
        linecolor='white',
        square=True
    )
    
    # Manually add text annotations to ensure visibility
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            value = matrix[i][j]
            # Dynamic color: white text for dark cells, black for light cells
            text_color = 'white' if matrix[i][j] > matrix.max() / 2 else 'black'
            ax.text(
                j + 0.5, i + 0.5,  # Center of cell
                f'{value}',
                ha='center', va='center',
                fontsize=14, fontweight='bold',
                color=text_color
            )
    
    # Labels
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title(f'Confusion Matrix ({split_name.capitalize()})', fontsize=16, fontweight='bold', pad=15)
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # Accuracy info (robust to 0 samples, and placed above to avoid overlap with X label)
    total = matrix.sum()
    if total > 0:
        accuracy = float(np.trace(matrix) / total)
        acc_str = f'{accuracy:.2%}'
    else:
        accuracy = 0.0
        acc_str = 'N/A'
    info_text = f'Total: {total:,} samples | Accuracy: {acc_str}'
    ax.text(
        0.5, 1.08, info_text,
        transform=ax.transAxes,
        ha='center', va='bottom',
        fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    )
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    
    print(f"📊 Saved confusion matrix: {output_file}")


def plot_f1_per_class(per_class_data: dict, split_name: str, output_file: str):
    """
    Plot F1 score per class as bar chart.
    
    Args:
        per_class_data: Per-class metrics
        split_name: Split name (Dev/Test) for title
        output_file: Output file path
    """
    labels = list(per_class_data.keys())
    f1_scores = [per_class_data[label]["f1"] for label in labels]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot bars
    colors = ['#3498db', '#e74c3c', '#f39c12']
    bars = plt.bar(range(len(labels)), f1_scores, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars (fixed position)
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        plt.text(
            i,  # X position centered on bar
            score + 0.02,  # Y position above bar
            f'{score:.3f}',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )
    
    # Formatting
    plt.xticks(range(len(labels)), [l.replace("_", "\n") for l in labels], fontsize=11)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title(f'F1 Score per Class ({split_name.capitalize()})', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.05)  # Extra space for labels
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved F1 bar chart: {output_file}")


def plot_precision_recall_per_class(per_class_data: dict, split_name: str, output_file: str):
    """
    Plot precision and recall per class as grouped bar chart (fixed text positions).
    
    Args:
        per_class_data: Per-class metrics
        split_name: Split name (Dev/Test) for title
        output_file: Output file path
    """
    labels = list(per_class_data.keys())
    precisions = [per_class_data[label]["precision"] for label in labels]
    recalls = [per_class_data[label]["recall"] for label in labels]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(labels))
    width = 0.35
    
    # Plot bars
    bars1 = ax.bar(x - width/2, precisions, width, label='Precision', color='#3498db', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, recalls, width, label='Recall', color='#e74c3c', alpha=0.7, edgecolor='black')
    
    # Add value labels (fixed position)
    for i, (p, r) in enumerate(zip(precisions, recalls)):
        # Precision label
        ax.text(
            i - width/2,  # X position centered on precision bar
            p + 0.02,  # Y position above bar
            f'{p:.3f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
        # Recall label
        ax.text(
            i + width/2,  # X position centered on recall bar
            r + 0.02,  # Y position above bar
            f'{r:.3f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=11)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Precision & Recall per Class ({split_name.capitalize()})', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)  # Extra space for labels
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved Precision/Recall chart: {output_file}")


def plot_debate_impact(debate_data: dict, split_name: str, output_file: str):
    """
    Plot debate impact analysis as bar chart (fixed text positions).
    
    Args:
        debate_data: Debate impact data
        split_name: Split name (Dev/Test) for title
        output_file: Output file path
    """
    if not debate_data:
        print("⚠️  No debate data available, skipping debate impact plot")
        return
    
    categories = ['Kept\nCorrect', 'Fixed\n(✓)', 'Broken\n(✗)', 'Kept\nIncorrect']
    values = [
        debate_data['kept_correct'],
        debate_data['fixed'],
        debate_data['broken'],
        debate_data['kept_incorrect']
    ]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels (fixed position)
    for i, (bar, value) in enumerate(zip(bars, values)):
        # Avoid division by zero
        if debate_data["total_analyzed"] > 0:
            label = f'{value}\n({value/debate_data["total_analyzed"]*100:.1f}%)'
        else:
            label = f'{value}'
        
        ax.text(
            i,  # X position centered on bar
            value + 0.3,  # Y position above bar
            label,
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )
    
    # Formatting
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Debate System Impact Analysis ({split_name.capitalize()})', fontsize=14, fontweight='bold')
    max_val = max(values) if values else 1
    ax.set_ylim(0, max(max_val * 1.15, 1))  # Extra space for labels, minimum 1
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved debate impact chart: {output_file}")


def plot_model_vs_final_comparison(metrics: dict, split_name: str, output_file: str):
    """
    Plot comparison of model vs final metrics (Accuracy, Precision, Recall, F1).
    
    Args:
        metrics: Full metrics dict
        split_name: Split name (Dev/Test) for title
        output_file: Output file path
    """
    # Extract metrics
    model_acc = metrics["model_accuracy"]
    final_acc = metrics["final_accuracy"]
    model_macro = metrics["model_macro"]
    final_macro = metrics["final_macro"]
    
    categories = ['Accuracy', 'Precision', 'Recall', 'F1']
    model_values = [
        model_acc,  # Already in decimal format (0.0-1.0)
        model_macro["macro_precision"],
        model_macro["macro_recall"],
        model_macro["macro_f1"]
    ]
    final_values = [
        final_acc,  # Already in decimal format (0.0-1.0)
        final_macro["macro_precision"],
        final_macro["macro_recall"],
        final_macro["macro_f1"]
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, model_values, width, label='Model (pre-debate)', color='#3498db', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, final_values, width, label='Final (post-debate)', color='#2ecc71', alpha=0.7, edgecolor='black')
    
    # Add value labels (fixed position)
    for i, (m, f) in enumerate(zip(model_values, final_values)):
        # Model label
        ax.text(
            i - width/2,
            m + 0.02,
            f'{m:.3f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
        # Final label
        ax.text(
            i + width/2,
            f + 0.02,
            f'{f:.3f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Model vs Final Metrics Comparison ({split_name.capitalize()})', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved model vs final comparison: {output_file}")


def plot_debate_rounds_distribution(debate_rounds: dict, split_name: str, output_file: str):
    """
    Plot distribution of debate rounds used.
    
    Args:
        debate_rounds: Dict with round counts {1: count, 2: count, 3: count}
        split_name: Split name (Dev/Test) for title
        output_file: Output file path
    """
    if not debate_rounds:
        print("⚠️  No debate rounds data available, skipping rounds distribution plot")
        return
    
    rounds = sorted(debate_rounds.keys())
    counts = [debate_rounds[r] for r in rounds]
    total = sum(counts)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar([f'Round {r}' for r in rounds], counts, color='#9b59b6', alpha=0.7, edgecolor='black')
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(
            i,
            count + (total * 0.02),
            f'{count}\n({count/total*100:.1f}%)',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )
    
    # Formatting
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Debate Rounds Distribution ({split_name.capitalize()})', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(counts) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved debate rounds distribution: {output_file}")


def plot_debator_performance(debator_performance: dict, split_name: str, output_file: str):
    """
    Plot individual debator accuracy/F1.
    
    Args:
        debator_performance: Dict with debator metrics {model_name: {acc, f1, ...}}
        split_name: Split name (Dev/Test) for title
        output_file: Output file path
    """
    if not debator_performance:
        print("⚠️  No debator performance data available, skipping debator performance plot")
        return
    
    models = list(debator_performance.keys())
    accuracies = [debator_performance[m]['accuracy'] for m in models]
    f1_scores = [debator_performance[m]['f1'] for m in models]
    
    # Shorten model names for display
    display_names = []
    for m in models:
        m_lower = m.lower()
        if 'gpt-5-mini' in m_lower or 'gpt5' in m_lower:
            display_names.append('GPT-5 Mini\n(Logical Critic)')
        elif 'gemini-2.5' in m_lower or 'gemini-2.0' in m_lower:
            display_names.append('Gemini 2.5\n(Data Analyst)')
        elif 'qwen3-235b' in m_lower or 'qwen3' in m_lower:
            display_names.append('Qwen3 235B\n(Context Guardian)')
        elif 'qwen-2.5-72b' in m_lower:
            display_names.append('Qwen 2.5\n72B')
        elif 'qwen' in m_lower:
            display_names.append('Qwen\n(other)')
        elif 'llama-3.3' in m_lower:
            display_names.append('Llama 3.3\n70B')
        elif 'llama' in m_lower:
            display_names.append('Llama\n(other)')
        elif 'deepseek' in m_lower:
            display_names.append('DeepSeek V3\n(Judge)')
        elif 'gpt-4o-mini' in m_lower:
            display_names.append('GPT-4o Mini')
        elif 'gpt' in m_lower:
            display_names.append('GPT\n(other)')
        else:
            # Use last part of model name
            short_name = m.split('/')[-1][:15] if '/' in m else m[:15]
            display_names.append(short_name)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#e67e22', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, f1_scores, width, label='Macro-F1', color='#16a085', alpha=0.7, edgecolor='black')
    
    # Add value labels
    for i, (acc, f1) in enumerate(zip(accuracies, f1_scores)):
        ax.text(
            i - width/2,
            acc + 0.02,
            f'{acc:.3f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
        ax.text(
            i + width/2,
            f1 + 0.02,
            f'{f1:.3f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=10)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Individual Debator Performance ({split_name.capitalize()})', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved debator performance chart: {output_file}")


def plot_classification_report(per_class_metrics: dict, split_name: str, output_file: str, title_prefix: str = "Final"):
    """
    Plot Classification Report as a table with Precision, Recall, F1, Support.
    
    Args:
        per_class_metrics: Dict with per-class metrics {label: {precision, recall, f1, support}}
        split_name: Split name (Dev/Test) for title
        output_file: Output file path
        title_prefix: "Model" or "Final" for title
    """
    if not per_class_metrics:
        print(f"⚠️  No per-class metrics available, skipping classification report plot")
        return
    
    labels = ["Support", "Refute", "NOT_ENOUGH_INFO"]
    
    # Prepare data
    data = []
    for label in labels:
        if label in per_class_metrics:
            m = per_class_metrics[label]
            data.append([
                label,
                f"{m.get('precision', 0):.4f}",
                f"{m.get('recall', 0):.4f}",
                f"{m.get('f1', 0):.4f}",
                str(m.get('support', 0))
            ])
        else:
            data.append([label, "N/A", "N/A", "N/A", "0"])
    
    # Calculate macro averages
    precisions = [per_class_metrics.get(l, {}).get('precision', 0) for l in labels]
    recalls = [per_class_metrics.get(l, {}).get('recall', 0) for l in labels]
    f1s = [per_class_metrics.get(l, {}).get('f1', 0) for l in labels]
    total_support = sum(per_class_metrics.get(l, {}).get('support', 0) for l in labels)
    
    data.append([
        "Macro Avg",
        f"{sum(precisions)/len(precisions):.4f}",
        f"{sum(recalls)/len(recalls):.4f}",
        f"{sum(f1s)/len(f1s):.4f}",
        str(total_support)
    ])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    # Create table
    columns = ['Label', 'Precision', 'Recall', 'F1-Score', 'Support']
    table = ax.table(
        cellText=data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colColours=['#3498db'] * 5
    )
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Color header
    for i in range(len(columns)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Color macro avg row
    for i in range(len(columns)):
        table[(len(data), i)].set_facecolor('#ecf0f1')
        table[(len(data), i)].set_text_props(fontweight='bold')
    
    plt.title(f'{title_prefix} Classification Report ({split_name.capitalize()})', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved classification report: {output_file}")


def plot_consensus_progression(consensus_data: dict, split_name: str, output_file: str):
    """
    TIER 1: Plot consensus scores progression from Round 1 to Final.
    
    Args:
        consensus_data: Consensus metrics dict
        split_name: Split name (Dev/Test) for title
        output_file: Output file path
    """
    if not consensus_data:
        print("⚠️  No consensus data available, skipping consensus progression plot")
        return
    
    rounds = ['Round 1', 'Final']
    consensus_scores = [
        consensus_data.get('round_1_avg_consensus', 0),
        consensus_data.get('final_avg_consensus', 0)
    ]
    
    # Insert Round 2 if available
    if consensus_data.get('round_2_avg_consensus') is not None:
        rounds.insert(1, 'Round 2')
        consensus_scores.insert(1, consensus_data['round_2_avg_consensus'])

    unanimous_rates = [
        consensus_data.get('round_1_unanimous_rate', 0),
        consensus_data.get('final_unanimous_rate', 0)
    ]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Average consensus score
    ax1.plot(rounds, consensus_scores, marker='o', linewidth=3, markersize=10, color='#3498db')
    ax1.fill_between(range(len(rounds)), 0, consensus_scores, alpha=0.3, color='#3498db')
    
    for i, (r, score) in enumerate(zip(rounds, consensus_scores)):
        ax1.text(i, score + 0.02, f'{score:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('Average Consensus Score', fontsize=12)
    ax1.set_title(f'Consensus Progression ({split_name.capitalize()})', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_xticks(range(len(rounds)))
    ax1.set_xticklabels(rounds)
    
    # Right: Unanimous rate
    bars = ax2.bar(rounds, unanimous_rates, color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
    
    for bar, rate in zip(bars, unanimous_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, rate + 0.02, 
                f'{rate:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('Unanimous Agreement Rate', fontsize=12)
    ax2.set_title(f'Unanimous Decisions ({split_name.capitalize()})', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved consensus progression chart: {output_file}")


def plot_inter_agent_agreement_matrix(agreement_data: dict, split_name: str, output_file: str):
    """
    TIER 1: Plot pairwise agreement between debators as heatmap.
    
    Args:
        agreement_data: Inter-agent agreement dict
        split_name: Split name (Dev/Test) for title
        output_file: Output file path
    """
    if not agreement_data or not agreement_data.get("agreement_matrix"):
        print("⚠️  No inter-agent agreement data available, skipping agreement matrix plot")
        return
    
    agreement_matrix = agreement_data["agreement_matrix"]
    
    # Get model names from agreement data (safer than parsing pair keys)
    if "model_names" in agreement_data:
        models = agreement_data["model_names"]
    else:
        # Fallback: Extract from pair keys (legacy compatibility)
        models = set()
        for pair_key in agreement_matrix.keys():
            # Parse by finding all possible splits
            parts = pair_key.split('-')
            if len(parts) == 2:
                models.add(parts[0])
                models.add(parts[1])
            else:
                # Complex case: try to reconstruct from known patterns
                # Just use the pair_key as-is for display
                models.add(pair_key)
        models = sorted(list(models))
    
    n = len(models)
    
    # Build symmetric matrix
    matrix = np.ones((n, n))
    for i, model_i in enumerate(models):
        for j, model_j in enumerate(models):
            if i != j:
                # Create sorted pair key (same logic as evaluate_results.py)
                pair_key = tuple(sorted([model_i, model_j]))
                key = f"{pair_key[0]}-{pair_key[1]}"
                if key in agreement_matrix:
                    matrix[i, j] = agreement_matrix[key]
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    
    # Add text annotations
    for i in range(n):
        for j in range(n):
            if i == j:
                text = 'Self'
                color = 'gray'
            else:
                text = f'{matrix[i, j]:.2%}'
                color = 'white' if matrix[i, j] < 0.5 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=11, fontweight='bold')
    
    # Labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([m[:15] for m in models], rotation=45, ha='right')
    ax.set_yticklabels([m[:15] for m in models])
    ax.set_xlabel('Debator', fontsize=12, fontweight='bold')
    ax.set_ylabel('Debator', fontsize=12, fontweight='bold')
    ax.set_title(f'Inter-Agent Agreement Matrix ({split_name.capitalize()})', fontsize=14, fontweight='bold', pad=15)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Agreement Rate', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved inter-agent agreement matrix: {output_file}")


def plot_round_by_round_accuracy(round_accuracy_data: dict, model_accuracy: float, split_name: str, output_file: str):
    """
    TIER 1: Plot accuracy improvement across debate rounds.
    
    Args:
        round_accuracy_data: Round-by-round accuracy dict
        model_accuracy: Baseline model accuracy (pre-debate)
        split_name: Split name (Dev/Test) for title
        output_file: Output file path
    """
    if not round_accuracy_data:
        print("⚠️  No round-by-round accuracy data available, skipping round accuracy plot")
        return
    
    # Build data
    rounds = ['Model\n(Baseline)']
    accuracies = [model_accuracy / 100]  # Convert % to decimal
    
    if 'round_1_accuracy' in round_accuracy_data:
        rounds.append('Round 1\n(Majority)')
        accuracies.append(round_accuracy_data['round_1_accuracy'])
    
    if 'round_2_accuracy' in round_accuracy_data:
        rounds.append('Round 2\n(Majority)')
        accuracies.append(round_accuracy_data['round_2_accuracy'])
    
    if 'final_accuracy' in round_accuracy_data:
        rounds.append('Final\n(Judge)')
        accuracies.append(round_accuracy_data['final_accuracy'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Line plot
    x_pos = range(len(rounds))
    ax.plot(x_pos, accuracies, marker='o', linewidth=3, markersize=12, color='#2ecc71', label='Accuracy')
    ax.fill_between(x_pos, 0, accuracies, alpha=0.2, color='#2ecc71')
    
    # Add value labels
    for i, (r, acc) in enumerate(zip(rounds, accuracies)):
        ax.text(i, acc + 0.015, f'{acc:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Highlight improvement
    if len(accuracies) > 1:
        improvement = accuracies[-1] - accuracies[0]
        ax.annotate('', xy=(len(rounds)-1, accuracies[-1]), xytext=(0, accuracies[0]),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        mid_x = (len(rounds)-1) / 2
        mid_y = (accuracies[0] + accuracies[-1]) / 2
        ax.text(mid_x + 0.2, mid_y, f'+{improvement:.1%}', 
               fontsize=12, fontweight='bold', color='red',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(rounds, fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Accuracy Progression Across Rounds ({split_name.capitalize()})', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.25)  # Increased from 1.15 for labels
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved round-by-round accuracy chart: {output_file}")


def plot_rounds_distribution(round_accuracy_data: dict, split_name: str, output_file: str):
    """
    TIER 1: Plot adaptive stop distribution - how many samples stopped at each round.
    
    Shows:
    - Pie chart of rounds distribution (R1, R2, R3)
    - Helps evaluate efficiency of adaptive debate logic
    
    Args:
        round_accuracy_data: Round-by-round accuracy dict containing "rounds_distribution"
        split_name: Split name (Dev/Test) for title
        output_file: Output file path
    """
    if not round_accuracy_data or "rounds_distribution" not in round_accuracy_data:
        print("⚠️  No rounds distribution data available, skipping distribution plot")
        return
    
    rounds_dist = round_accuracy_data["rounds_distribution"]
    if not rounds_dist:
        print("⚠️  Empty rounds distribution, skipping distribution plot")
        return
    
    # Check for Forced Full Rounds (Single round with 100%)
    is_forced_mode = (len(rounds_dist) == 1) and (3 in rounds_dist or "3" in rounds_dist)
    
    # Prepare data ensuring all rounds 1, 2, 3 are present
    labels = []
    sizes = []
    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
    explode = []
    
    # Force display of rounds 1, 2, 3 even if count is 0
    all_rounds = [1, 2, 3]
    
    for round_num in all_rounds:
        # Get count (handle string/int keys)
        count = 0
        if round_num in rounds_dist:
            count = rounds_dist[round_num]
        elif str(round_num) in rounds_dist:
            count = rounds_dist[str(round_num)]
            
        labels.append(f'Round {round_num}')
        sizes.append(count)
        
        # Explode logic (only relevant for pie chart, which we might hide)
        is_min_round = (count > 0 and round_num == min([r for r in all_rounds if (str(r) in rounds_dist or r in rounds_dist)]))
        explode.append(0.05 if is_min_round else 0)
    
    total = sum(sizes)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    if is_forced_mode:
        # SPECIAL MODE: Forced Full Rounds
        # Still show pie chart but with a watermark-like overlay explaining why
        wedges, texts, autotexts = ax1.pie(
            sizes, 
            labels=labels, 
            colors=colors,
            explode=explode,
            autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*total)})' if pct > 0 else '',
            startangle=90,
            textprops={'fontsize': 11}
        )
        
        # Overlay text
        ax1.text(0, 0, "RESEARCH MODE\n(Full 3 Rounds)", 
                ha='center', va='center', fontsize=14, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#e74c3c', alpha=0.8))
        
        ax1.set_title(f'Adaptive Stop Distribution ({split_name.capitalize()})', 
                      fontsize=14, fontweight='bold')
    else:
        # Normal Mode
        # Filter out zero-sized slices for Pie Chart to avoid ugly rendering
        pie_sizes = []
        pie_labels = []
        pie_colors = []
        pie_explode = []
        
        for s, l, c, e in zip(sizes, labels, colors, explode):
            if s > 0:
                pie_sizes.append(s)
                pie_labels.append(l)
                pie_colors.append(c)
                pie_explode.append(e)
        
        if pie_sizes:
            wedges, texts, autotexts = ax1.pie(
                pie_sizes, 
                labels=pie_labels, 
                colors=pie_colors,
                explode=pie_explode,
                autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*total)})',
                startangle=90,
                textprops={'fontsize': 11}
            )
        ax1.set_title(f'Adaptive Stop Distribution ({split_name.capitalize()})', 
                      fontsize=14, fontweight='bold')
    
    # Bar chart with more details - ALWAYS SHOW ALL 3 ROUNDS
    x_pos = range(len(labels))
    bars = ax2.bar(x_pos, sizes, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, count in zip(bars, sizes):
        height = bar.get_height()
        pct = count / total * 100 if total > 0 else 0
        
        # Only show label if bar has height (or show 0 at bottom)
        label_y = height + (max(sizes)*0.05) if max(sizes) > 0 else 0.5
        
        ax2.text(bar.get_x() + bar.get_width()/2, label_y, 
                f'{count}\n({pct:.1f}%)', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title(f'Samples per Round ({split_name.capitalize()})', fontsize=14, fontweight='bold')
    
    # Increase Y-limit to avoid text clipping with title
    # Add 40% extra space on top
    top_margin = max(sizes) * 1.4 if max(sizes) > 0 else 10
    ax2.set_ylim(0, top_margin)
    
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add summary text
    if not is_forced_mode:
        # Calculate early stop percentage (rounds < max_round)
        max_round = max(all_rounds)  # = 3
        early_stop_pct = sum(v for k, v in rounds_dist.items() if int(k) < max_round) / total * 100 if total > 0 else 0
        fig.text(0.5, 0.02, 
                 f'Early Stop Rate: {early_stop_pct:.1f}% | Total Samples: {total}',
                 ha='center', fontsize=12, style='italic',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    else:
         fig.text(0.5, 0.02, 
                 f'Total Samples Analyzed: {total} (Full 3 Rounds)',
                 ha='center', fontsize=12, style='italic',
                 bbox=dict(boxstyle='round', facecolor='#fce5cd', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved rounds distribution chart: {output_file}")


def plot_reliability_diagram(calibration_data: dict, split_name: str, output_file: str):
    """
    TIER 2: Plot reliability diagram showing confidence vs accuracy.
    
    Shows calibration quality:
    - Perfect calibration = diagonal line
    - Gap between curve and diagonal = miscalibration (ECE)
    
    If temperature != 1.0, shows 2 subplots (before/after).
    
    Args:
        calibration_data: Calibration metrics dict with "raw" and "calibrated"
        split_name: Split name (Dev/Test) for title
        output_file: Output file path
    """
    if not calibration_data or "raw" not in calibration_data:
        print("⚠️  No calibration data available, skipping reliability diagram")
        return
    
    T = calibration_data.get("temperature", 1.0)
    
    # Decide: 1 or 2 subplots
    if T != 1.0 and "calibrated" in calibration_data:
        # Two subplots: before and after
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        axes = [ax1, ax2]
        data_list = [calibration_data["raw"], calibration_data["calibrated"]]
        titles = [f"Before Calibration (T=1.0)", f"After Calibration (T={T:.3f})"]
    else:
        # Single subplot: raw only
        fig, ax = plt.subplots(figsize=(8, 7))
        axes = [ax]
        data_list = [calibration_data["raw"]]
        titles = [f"Model Calibration (T=1.0)"]
    
    for ax, data, title in zip(axes, data_list, titles):
        bins = data.get("bins", [])
        
        # Extract bin data (skip empty bins)
        bin_confidences = []
        bin_accuracies = []
        bin_sizes = []
        
        for b in bins:
            if b["bin_size"] > 0:
                bin_confidences.append(b["bin_confidence"])
                bin_accuracies.append(b["bin_accuracy"])
                bin_sizes.append(b["bin_size"])
        
        if not bin_confidences:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=14)
            continue
        
        # Normalize bin sizes for plotting
        total_samples = sum(bin_sizes)
        bin_sizes_norm = [s / total_samples * 100 for s in bin_sizes]  # As percentages
        
        # Plot perfect calibration line (diagonal)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.7)
        
        # Plot actual calibration curve
        ax.plot(bin_confidences, bin_accuracies, 'o-', linewidth=3, markersize=8, 
               color='#e74c3c', label='Model', alpha=0.8)
        
        # Fill gap area (ECE visualization)
        ax.fill_between(bin_confidences, bin_accuracies, bin_confidences, 
                       alpha=0.2, color='red', label='Calibration Gap')
        
        # Add bin size as scatter point size (optional visual aid)
        scatter = ax.scatter(bin_confidences, bin_accuracies, s=[s*10 for s in bin_sizes_norm], 
                           alpha=0.3, color='#e74c3c')
        
        # Annotations
        ece = data.get("ece", 0)
        brier = data.get("brier", 0)
        avg_conf = data.get("avg_confidence", 0)
        avg_acc = data.get("avg_accuracy", 0)
        
        textstr = f'ECE: {ece:.4f}\nBrier: {brier:.4f}\nAvg Conf: {avg_conf:.3f}\nAvg Acc: {avg_acc:.3f}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Formatting
        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'{title} ({split_name.capitalize()})', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved reliability diagram: {output_file}")


def plot_mvp_distribution(df: pd.DataFrame, split_name: str, output_file: Path):
    """Pie Chart of MVP Agents."""
    if "debate_result" not in df.columns:
        return
        
    # Extract MVP agents
    def get_mvp(row):
        if not isinstance(row["debate_result"], dict) or not row["debate_result"].get("metrics"):
            return None
        mvp = row["debate_result"]["metrics"].get("mvp_agent", "Unknown")
        return mvp if mvp and mvp != "Unknown" else None
        
    mvps = df.apply(get_mvp, axis=1).dropna()
    
    if mvps.empty:
        return
        
    counts = mvps.value_counts()
    
    plt.figure(figsize=(10, 8))
    plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=140, 
            colors=sns.color_palette("pastel"))
    
    plt.title(f"Judge's MVP Agent Distribution - {split_name}\n(Who provided the winning argument?)")
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"📊 Saved MVP distribution chart: {output_file}")


def plot_confusion_matrix_comparison(df: pd.DataFrame, split_name: str, output_file: Path):
    """Side-by-side Confusion Matrices: PhoBERT vs Final System."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    labels = ["Support", "Refute", "NOT_ENOUGH_INFO"]
    
    # Filter valid labels
    valid_df = df[df["gold_label"].isin(labels) & 
                  df["model_verdict"].isin(labels) & 
                  df["final_verdict"].isin(labels)]
    
    if valid_df.empty:
        return

    # Helper function to add annotations with dynamic color (white on dark, black on light)
    def add_annotations(ax, cm):
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                value = cm[i][j]
                # Use white text on dark cells, black on light cells
                text_color = 'white' if value > cm.max() / 2 else 'black'
                ax.text(j + 0.5, i + 0.5, f'{value}',
                       ha='center', va='center',
                       fontsize=16, fontweight='bold',
                       color=text_color)
    
    # 1. Model CM (Blues colormap)
    cm_model = confusion_matrix(valid_df["gold_label"], valid_df["model_verdict"], labels=labels)
    sns.heatmap(cm_model, annot=False, cmap='Blues', ax=axes[0],
                xticklabels=labels, yticklabels=labels, cbar=False,
                linewidths=2, linecolor='white', square=True)
    add_annotations(axes[0], cm_model)
    axes[0].set_title(f"Baseline (PhoBERT) - {split_name}", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Predicted", fontsize=12)
    axes[0].set_ylabel("True Label", fontsize=12)
    
    # 2. Final CM (Greens colormap)
    cm_final = confusion_matrix(valid_df["gold_label"], valid_df["final_verdict"], labels=labels)
    sns.heatmap(cm_final, annot=False, cmap='Greens', ax=axes[1],
                xticklabels=labels, yticklabels=labels, cbar=False,
                linewidths=2, linecolor='white', square=True)
    add_annotations(axes[1], cm_final)
    axes[1].set_title(f"Final System (Debate) - {split_name}", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Predicted", fontsize=12)
    axes[1].set_ylabel("True Label", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"📊 Saved CM comparison chart: {output_file}")


def generate_all_plots(metrics: dict, results: list, split_name: str, output_dir: Path):
    """
    Generate all plots for a given split.
    
    Args:
        metrics: Metrics dictionary
        results: Raw list of sample results
        split_name: Split name (dev/test)
        output_dir: Output directory for plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # === NEW ADVANCED PLOTS (Require raw data) ===
    
    # Plot 0: MVP Agent Distribution (XAI)
    plot_mvp_distribution(df, split_name, output_dir / "mvp_distribution.png")
    
    # Plot 0.5: Confusion Matrix Comparison (Model vs Final)
    plot_confusion_matrix_comparison(df, split_name, output_dir / "confusion_matrix_comparison.png")
    
    # === EXISTING PLOTS (Use metrics) ===
    
    # Plot 1: Confusion Matrix (Final)
    plot_confusion_matrix(
        metrics["final_confusion_matrix"],
        split_name,
        output_dir / "confusion_matrix_final.png"
    )
    
    # Plot 2: F1 per Class (Final)
    plot_f1_per_class(
        metrics["final_per_class"],
        split_name,
        output_dir / "f1_per_class_final.png"
    )
    
    # Plot 3: Precision/Recall per Class (Final)
    plot_precision_recall_per_class(
        metrics["final_per_class"],
        split_name,
        output_dir / "precision_recall_final.png"
    )
    
    # Plot 3.5: Classification Reports (Model & Final)
    if metrics.get("model_per_class"):
        plot_classification_report(
            metrics["model_per_class"],
            split_name,
            output_dir / "classification_report_model.png",
            title_prefix="Model (PhoBERT)"
        )
    
    plot_classification_report(
        metrics["final_per_class"],
        split_name,
        output_dir / "classification_report_final.png",
        title_prefix="Final (Debate)"
    )
    
    # Plot 4: Model vs Final Comparison
    plot_model_vs_final_comparison(
        metrics,
        split_name,
        output_dir / "model_vs_final_comparison.png"
    )
    
    # Plot 5: Debate Impact (if available)
    if metrics.get("debate_impact"):
        plot_debate_impact(
            metrics["debate_impact"],
            split_name,
            output_dir / "debate_impact.png"
        )
    
    # Plot 6: Debate Rounds Distribution (if available)
    if metrics.get("debate_rounds_distribution"):
        plot_debate_rounds_distribution(
            metrics["debate_rounds_distribution"],
            split_name,
            output_dir / "debate_rounds_distribution.png"
        )
    
    # Plot 7: Debator Performance (if available)
    if metrics.get("debator_performance"):
        plot_debator_performance(
            metrics["debator_performance"],
            split_name,
            output_dir / "debator_performance.png"
        )
    
    # TIER 1 Plot 1: Consensus Progression
    if metrics.get("consensus_scores"):
        plot_consensus_progression(
            metrics["consensus_scores"],
            split_name,
            output_dir / "tier1_consensus_progression.png"
        )
    
    # TIER 1 Plot 2: Inter-Agent Agreement Matrix
    if metrics.get("inter_agent_agreement"):
        plot_inter_agent_agreement_matrix(
            metrics["inter_agent_agreement"],
            split_name,
            output_dir / "tier1_inter_agent_agreement.png"
        )
    
    # TIER 1 Plot 3: Round-by-Round Accuracy
    if metrics.get("round_by_round_accuracy"):
        plot_round_by_round_accuracy(
            metrics["round_by_round_accuracy"],
            metrics["model_accuracy"],
            split_name,
            output_dir / "tier1_round_accuracy.png"
        )
    
    # TIER 1 Plot 4: Adaptive Stop Distribution
    if metrics.get("round_by_round_accuracy"):
        plot_rounds_distribution(
            metrics["round_by_round_accuracy"],
            split_name,
            output_dir / "tier1_rounds_distribution.png"
        )
    
    # TIER 2 Plot: Reliability Diagram
    if metrics.get("model_calibration"):
        plot_reliability_diagram(
            metrics["model_calibration"],
            split_name,
            output_dir / "tier2_reliability_diagram.png"
        )
    
    print(f"\n✅ All plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Plot ViFactCheck Evaluation Metrics")
    parser.add_argument("metrics_file", type=str, help="Path to metrics JSON file")
    parser.add_argument("--split", type=str, default="dev", help="Split name (dev/test) for titles")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Load metrics
    print(f"📂 Loading metrics from: {args.metrics_file}")
    metrics = load_metrics(args.metrics_file)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    
    # Generate all plots
    generate_all_plots(metrics, args.split, output_dir)


if __name__ == "__main__":
    main()
