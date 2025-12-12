"""
Generate Judge Performance Metrics from existing debate results.
No need to re-run full debate - just post-hoc analysis.

Author: Lockdown
Date: Dec 5, 2025
"""

import json
import argparse
from pathlib import Path
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def normalize_verdict(verdict: str) -> str:
    """Normalize verdict to standard format."""
    if not verdict:
        return "NEI"
    v = verdict.upper().strip()
    if v in ["SUPPORTED", "SUPPORT"]:
        return "Support"
    elif v in ["REFUTED", "REFUTE"]:
        return "Refute"
    else:
        return "NEI"


def get_majority_verdict(round_verdicts: dict) -> str:
    """Get majority verdict from debators."""
    if not round_verdicts:
        return None
    
    verdicts = [normalize_verdict(v.get("verdict", "")) for v in round_verdicts.values()]
    counter = Counter(verdicts)
    majority = counter.most_common(1)[0][0]
    return majority


def analyze_judge_performance(results_path: str, output_path: str = None):
    """
    Analyze Judge performance from debate results.
    
    Args:
        results_path: Path to vifactcheck_test_results.json
        output_path: Path to save judge metrics (optional)
    """
    
    print(f"Loading results from: {results_path}")
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get("results", [])
    print(f"Total samples: {len(results)}")
    
    # Collect data
    gold_labels = []
    judge_verdicts = []
    majority_verdicts = []
    decision_paths = []
    
    # Override analysis
    override_cases = {
        "total": 0,
        "correct_override": 0,  # Judge override majority and was correct
        "wrong_override": 0,    # Judge override majority and was wrong
        "follow_majority_correct": 0,
        "follow_majority_wrong": 0
    }
    
    # Consensus analysis
    consensus_cases = {
        "total": 0,
        "correct": 0,
        "wrong": 0
    }
    
    for r in results:
        gold = normalize_verdict(r.get("gold_label", ""))
        judge = normalize_verdict(r.get("final_verdict", ""))
        
        gold_labels.append(gold)
        judge_verdicts.append(judge)
        
        # Get debate result info
        debate_result = r.get("debate_result", {})
        
        # Get decision path if available
        decision_path = debate_result.get("decision_path", "UNKNOWN")
        if decision_path == "UNKNOWN":
            # Infer from debator_agreements
            agreements = debate_result.get("metrics", {}).get("debator_agreements", {})
            if agreements:
                max_count = max(agreements.values())
                if max_count == 3:
                    decision_path = "CONSENSUS"
                elif max_count == 2:
                    decision_path = "MAJORITY"
                else:
                    decision_path = "SPLIT"
        decision_paths.append(decision_path)
        
        # Get majority verdict from Round 2 (or Round 1 if R2 not available)
        all_rounds = debate_result.get("all_rounds_verdicts", [])
        if all_rounds and len(all_rounds) >= 2:
            round_2_verdicts = all_rounds[-1]  # Last round
            majority = get_majority_verdict(round_2_verdicts)
        elif debate_result.get("round_1_verdicts"):
            majority = get_majority_verdict(debate_result["round_1_verdicts"])
        else:
            majority = judge  # Fallback
        
        majority_verdicts.append(majority)
        
        # Analyze override behavior
        judge_correct = (judge == gold)
        majority_correct = (majority == gold)
        judge_followed_majority = (judge == majority)
        
        if decision_path == "CONSENSUS":
            consensus_cases["total"] += 1
            if judge_correct:
                consensus_cases["correct"] += 1
            else:
                consensus_cases["wrong"] += 1
        
        if not judge_followed_majority:
            override_cases["total"] += 1
            if judge_correct:
                override_cases["correct_override"] += 1
            else:
                override_cases["wrong_override"] += 1
        else:
            if judge_correct:
                override_cases["follow_majority_correct"] += 1
            else:
                override_cases["follow_majority_wrong"] += 1
    
    # Calculate metrics
    labels = ["Support", "Refute", "NEI"]
    
    # Judge accuracy
    judge_accuracy = accuracy_score(gold_labels, judge_verdicts)
    
    # Judge per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        gold_labels, judge_verdicts, labels=labels, zero_division=0
    )
    
    judge_per_class = {}
    for i, label in enumerate(labels):
        judge_per_class[label] = {
            "precision": round(precision[i], 4),
            "recall": round(recall[i], 4),
            "f1": round(f1[i], 4),
            "support": int(support[i])
        }
    
    # Macro metrics
    macro_precision = round(precision.mean(), 4)
    macro_recall = round(recall.mean(), 4)
    macro_f1 = round(f1.mean(), 4)
    
    # Confusion matrix
    cm = confusion_matrix(gold_labels, judge_verdicts, labels=labels)
    cm_dict = {
        "labels": labels,
        "matrix": {
            labels[i]: {labels[j]: int(cm[i][j]) for j in range(len(labels))}
            for i in range(len(labels))
        }
    }
    
    # Decision path distribution
    path_counter = Counter(decision_paths)
    decision_path_dist = {
        path: {
            "count": count,
            "ratio": round(count / len(results) * 100, 2)
        }
        for path, count in path_counter.items()
    }
    
    # Build output
    judge_metrics = {
        "judge_accuracy": round(judge_accuracy, 4),
        "judge_per_class": judge_per_class,
        "judge_macro": {
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1
        },
        "judge_confusion_matrix": cm_dict,
        "decision_path_distribution": decision_path_dist,
        "consensus_analysis": {
            "total_consensus": consensus_cases["total"],
            "consensus_correct": consensus_cases["correct"],
            "consensus_wrong": consensus_cases["wrong"],
            "consensus_accuracy": round(
                consensus_cases["correct"] / consensus_cases["total"] * 100, 2
            ) if consensus_cases["total"] > 0 else 0
        },
        "override_analysis": {
            "total_overrides": override_cases["total"],
            "correct_overrides": override_cases["correct_override"],
            "wrong_overrides": override_cases["wrong_override"],
            "override_success_rate": round(
                override_cases["correct_override"] / override_cases["total"] * 100, 2
            ) if override_cases["total"] > 0 else 0,
            "follow_majority_correct": override_cases["follow_majority_correct"],
            "follow_majority_wrong": override_cases["follow_majority_wrong"]
        },
        "total_samples": len(results)
    }
    
    # Print summary
    print("\n" + "="*60)
    print("JUDGE PERFORMANCE METRICS")
    print("="*60)
    print(f"\n📊 Judge Accuracy: {judge_accuracy:.2%}")
    print(f"📊 Judge Macro F1: {macro_f1:.4f}")
    print(f"\n📋 Per-class F1:")
    for label, metrics in judge_per_class.items():
        print(f"   - {label}: {metrics['f1']:.4f} (P={metrics['precision']:.4f}, R={metrics['recall']:.4f})")
    
    print(f"\n🎯 Decision Path Distribution:")
    for path, info in decision_path_dist.items():
        print(f"   - {path}: {info['count']} ({info['ratio']}%)")
    
    print(f"\n🤝 Consensus Analysis:")
    print(f"   - Total consensus cases: {consensus_cases['total']}")
    print(f"   - Consensus accuracy: {judge_metrics['consensus_analysis']['consensus_accuracy']}%")
    
    print(f"\n⚖️ Override Analysis:")
    print(f"   - Total overrides: {override_cases['total']}")
    if override_cases["total"] > 0:
        print(f"   - Override success rate: {judge_metrics['override_analysis']['override_success_rate']}%")
    
    # Save to file
    if output_path:
        output_file = Path(output_path)
    else:
        output_file = Path(results_path).parent / "metrics" / "judge_metrics.json"
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(judge_metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Judge metrics saved to: {output_file}")
    
    return judge_metrics


def main():
    parser = argparse.ArgumentParser(description="Generate Judge Performance Metrics")
    parser.add_argument(
        "--results", "-r",
        type=str,
        default="results/vifactcheck/test/full_debate/vifactcheck_test_results.json",
        help="Path to debate results JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for judge metrics (optional)"
    )
    
    args = parser.parse_args()
    
    analyze_judge_performance(args.results, args.output)


if __name__ == "__main__":
    main()
