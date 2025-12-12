"""
Evaluate Results from ViFactCheck Pipeline Test

Computes detailed metrics:
- Accuracy, Macro-F1, F1 per class
- Precision, Recall per class
- Confusion Matrix
- Debate Impact Analysis

Usage:
    python evaluate_results.py vifactcheck_dev_results.json
    python evaluate_results.py vifactcheck_test_results.json --save-metrics metrics_dev.json
"""

import json
import argparse
from typing import Dict, List, Any
from collections import defaultdict
import sys

# Import calibration module
try:
    from calibration_3label import compute_calibration_metrics
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from calibration_3label import compute_calibration_metrics


def normalize_label(label: str) -> str:
    """
    Normalize label to standard format (Title Case).
    Handles: SUPPORTS/Support/0, REFUTES/Refute/1, NEI/NOT_ENOUGH_INFO/2
    """
    if not isinstance(label, str):
        label = str(label)
    
    label = label.upper().strip()
    
    if label in ["SUPPORT", "SUPPORTS", "SUPPORTED", "0"]:
        return "Support"
    elif label in ["REFUTE", "REFUTES", "REFUTED", "1"]:
        return "Refute"
    elif label in ["NEI", "NOT_ENOUGH_INFO", "NOT ENOUGH INFO", "NOT_ENOUGH_INFO", "2", "UNVERIFIED"]:
        return "NOT_ENOUGH_INFO"
    
    return label.title() if label else "NOT_ENOUGH_INFO"


def load_results(json_file: str) -> Dict[str, Any]:
    """Load results from JSON file"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def compute_confusion_matrix(results: List[Dict[str, Any]], verdict_key: str = "final_verdict") -> Dict[str, Any]:
    """
    Compute confusion matrix for 3-label classification.
    
    Args:
        results: List of result dicts
        verdict_key: 'model_verdict' or 'final_verdict'
    
    Returns:
        Dict with confusion matrix and label order
    """
    labels = ["Support", "Refute", "NOT_ENOUGH_INFO"]
    
    # Initialize confusion matrix
    cm = {true_label: {pred_label: 0 for pred_label in labels} for true_label in labels}
    
    # Populate confusion matrix
    for r in results:
        true_label = r["gold_label"]
        pred_label = r[verdict_key]
        
        # Skip errors
        if pred_label == "ERROR":
            continue
        
        if true_label in labels and pred_label in labels:
            cm[true_label][pred_label] += 1
    
    return {
        "labels": labels,
        "matrix": cm
    }


def compute_metrics_per_class(cm_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Compute precision, recall, F1 per class from confusion matrix.
    
    Returns:
        Dict[label] -> {precision, recall, f1, support}
    """
    labels = cm_data["labels"]
    cm = cm_data["matrix"]
    
    metrics = {}
    
    for label in labels:
        # True Positives: cm[label][label]
        tp = cm[label][label]
        
        # False Positives: sum of column for label (excluding TP)
        fp = sum(cm[other][label] for other in labels if other != label)
        
        # False Negatives: sum of row for label (excluding TP)
        fn = sum(cm[label][other] for other in labels if other != label)
        
        # Support: total true instances of label
        support = sum(cm[label].values())
        
        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1 = 2 * (Precision * Recall) / (Precision + Recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support
        }
    
    return metrics


def compute_macro_metrics(metrics_per_class: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Compute macro-averaged metrics"""
    num_classes = len(metrics_per_class)
    
    macro_precision = sum(m["precision"] for m in metrics_per_class.values()) / num_classes
    macro_recall = sum(m["recall"] for m in metrics_per_class.values()) / num_classes
    macro_f1 = sum(m["f1"] for m in metrics_per_class.values()) / num_classes
    
    return {
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4)
    }


def analyze_hybrid_strategy(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze Hybrid Strategy (DOWN Framework) statistics.
    
    Returns:
        Dict with skip/debate counts and accuracy breakdown
    """
    total = 0
    skipped = 0
    debated = 0
    skipped_correct = 0
    debated_correct = 0
    threshold = 0.85  # Default
    
    for r in results:
        if r.get("model_verdict") == "ERROR" or r.get("final_verdict") == "ERROR":
            continue
        
        total += 1
        hybrid_info = r.get("hybrid_info", {})
        
        if hybrid_info.get("skipped", False) or hybrid_info.get("source") == "MODEL_HIGH_CONF":
            skipped += 1
            threshold = hybrid_info.get("threshold", 0.85)
            if r.get("final_correct", False):
                skipped_correct += 1
        elif r.get("debate_info") or r.get("debate_result"):
            debated += 1
            if r.get("final_correct", False):
                debated_correct += 1
        else:
            # No debate info and no hybrid skip = model only (no debate enabled)
            skipped += 1
            if r.get("final_correct", False):
                skipped_correct += 1
    
    return {
        "total": total,
        "threshold": threshold,
        "skipped": skipped,
        "skipped_ratio": round(skipped / total * 100, 2) if total > 0 else 0.0,
        "skipped_accuracy": round(skipped_correct / skipped * 100, 2) if skipped > 0 else 0.0,
        "debated": debated,
        "debated_ratio": round(debated / total * 100, 2) if total > 0 else 0.0,
        "debated_accuracy": round(debated_correct / debated * 100, 2) if debated > 0 else 0.0,
        "cost_saved": round(skipped / total * 100, 2) if total > 0 else 0.0
    }


def analyze_debate_impact(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze debate system impact on accuracy.
    
    Returns:
        Dict with:
        - kept_correct: model đúng, final đúng
        - kept_incorrect: model sai, final sai
        - fixed: model sai → final đúng
        - broken: model đúng → final sai
    """
    kept_correct = 0
    kept_incorrect = 0
    fixed = 0
    broken = 0
    
    for r in results:
        if r["model_verdict"] == "ERROR" or r["final_verdict"] == "ERROR":
            continue
        
        model_correct = r["model_correct"]
        final_correct = r["final_correct"]
        
        if model_correct and final_correct:
            kept_correct += 1
        elif not model_correct and not final_correct:
            kept_incorrect += 1
        elif not model_correct and final_correct:
            fixed += 1
        elif model_correct and not final_correct:
            broken += 1
    
    total = kept_correct + kept_incorrect + fixed + broken
    
    return {
        "total_analyzed": total,
        "kept_correct": kept_correct,
        "kept_incorrect": kept_incorrect,
        "fixed": fixed,
        "broken": broken,
        "fix_rate": round(fixed / total * 100, 2) if total > 0 else 0.0,
        "break_rate": round(broken / total * 100, 2) if total > 0 else 0.0
    }


def compute_debate_rounds_distribution(results: List[Dict[str, Any]]) -> Dict[int, int]:
    """
    Compute distribution of debate rounds used.
    
    Returns:
        Dict[round_number] -> count
    """
    rounds_count = defaultdict(int)
    
    for r in results:
        if r.get("debate_result") and r["debate_result"].get("rounds_used"):
            rounds = r["debate_result"]["rounds_used"]
            rounds_count[rounds] += 1
    
    return dict(rounds_count) if rounds_count else {}


def compute_debator_performance(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Compute individual debator performance (accuracy, F1).
    
    Each debator's Round 1 verdict is compared against gold label.
    
    Returns:
        Dict[model_name] -> {accuracy, f1, precision, recall}
    """
    debator_verdicts = defaultdict(lambda: {"correct": 0, "total": 0, "by_label": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})})
    
    for r in results:
        if not r.get("debate_result") or not r["debate_result"].get("round_1_verdicts"):
            continue
        
        gold = r["gold_label"]
        round_1 = r["debate_result"]["round_1_verdicts"]
        
        for model_name, verdict_data in round_1.items():
            verdict = normalize_label(verdict_data["verdict"])
            debator_verdicts[model_name]["total"] += 1
            
            if verdict == gold:
                debator_verdicts[model_name]["correct"] += 1
                debator_verdicts[model_name]["by_label"][gold]["tp"] += 1
            else:
                debator_verdicts[model_name]["by_label"][verdict]["fp"] += 1
                debator_verdicts[model_name]["by_label"][gold]["fn"] += 1
    
    # Compute metrics
    performance = {}
    labels = ["Support", "Refute", "NOT_ENOUGH_INFO"]
    
    for model_name, data in debator_verdicts.items():
        acc = data["correct"] / data["total"] if data["total"] > 0 else 0.0
        
        # Compute F1 per class, then macro-average
        f1_scores = []
        precisions = []
        recalls = []
        
        for label in labels:
            tp = data["by_label"][label]["tp"]
            fp = data["by_label"][label]["fp"]
            fn = data["by_label"][label]["fn"]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            f1_scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)
        
        performance[model_name] = {
            "accuracy": round(acc, 4),
            "f1": round(sum(f1_scores) / len(f1_scores), 4),
            "precision": round(sum(precisions) / len(precisions), 4),
            "recall": round(sum(recalls) / len(recalls), 4),
            "total_samples": data["total"]
        }
    
    return performance


def compute_consensus_scores(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    TIER 1 METRIC: Compute consensus/convergence scores for debate system.
    
    Measures agreement between debators at each round and final.
    
    Returns:
        {
            "round_1_consensus": 0.75,  # 75% claims with unanimous agreement
            "final_consensus": 0.90,    # 90% claims with unanimous agreement
            "consensus_trend": +0.15,   # Improvement from R1 to final
            "per_claim_consensus": [...] # Individual scores
        }
    """
    from collections import Counter
    
    round_1_scores = []
    round_2_scores = []
    final_scores = []
    
    for r in results:
        debate_res = r.get("debate_result", {})
        
        # METHOD 1: Use 'round_metrics' (Rich data from new logs)
        if debate_res and "round_metrics" in debate_res:
            metrics = debate_res["round_metrics"]
            if len(metrics) > 0:
                # Round 1 is always index 0
                round_1_scores.append(metrics[0].get("agreement_ratio", 0))
                # Final round is always the last one
                final_scores.append(metrics[-1].get("agreement_ratio", 0))
                
                # Round 2 if available
                if len(metrics) >= 2:
                    round_2_scores.append(metrics[1].get("agreement_ratio", 0))
            continue

        # METHOD 2: Legacy logic
        if not debate_res or not debate_res.get("round_1_verdicts"):
            continue
        
        # Round 1 consensus
        round_1_verdicts = [normalize_label(v["verdict"]) for v in debate_res["round_1_verdicts"].values()]
        if round_1_verdicts:
            most_common_count = Counter(round_1_verdicts).most_common(1)[0][1]
            r1_consensus = most_common_count / len(round_1_verdicts)
            round_1_scores.append(r1_consensus)
        
        # Final consensus (check if all agreed at final round)
        debators_final = debate_res.get("debator_verdicts", {})
        
        if debators_final:
            final_verdicts = [normalize_label(v) for v in debators_final.values()]
            most_common_final = Counter(final_verdicts).most_common(1)[0][1]
            final_consensus = most_common_final / len(final_verdicts)
            final_scores.append(final_consensus)
    
    if not round_1_scores:
        return {}
    
    avg_r1 = sum(round_1_scores) / len(round_1_scores)
    avg_r2 = sum(round_2_scores) / len(round_2_scores) if round_2_scores else 0.0
    avg_final = sum(final_scores) / len(final_scores) if final_scores else avg_r1
    
    # Unanimous rate (perfect consensus = 1.0)
    unanimous_r1 = sum(1 for s in round_1_scores if s == 1.0) / len(round_1_scores)
    unanimous_final = sum(1 for s in final_scores if s == 1.0) / len(final_scores) if final_scores else unanimous_r1
    
    return {
        "round_1_avg_consensus": round(avg_r1, 4),
        "round_2_avg_consensus": round(avg_r2, 4) if round_2_scores else None,
        "final_avg_consensus": round(avg_final, 4),
        "consensus_improvement": round(avg_final - avg_r1, 4),
        "round_1_unanimous_rate": round(unanimous_r1, 4),
        "final_unanimous_rate": round(unanimous_final, 4),
        "total_claims_analyzed": len(round_1_scores)
    }


def compute_inter_agent_agreement(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    TIER 1 METRIC: Compute pairwise agreement between debators.
    
    Returns agreement matrix showing how often each pair of debators agree.
    
    Returns:
        {
            "agreement_matrix": {
                "llama-gemini": 0.85,
                "llama-gpt": 0.80,
                "gemini-gpt": 0.78
            },
            "avg_pairwise_agreement": 0.81
        }
    """
    from collections import defaultdict
    
    # Track agreements for each pair
    pair_agreements = defaultdict(lambda: {"agree": 0, "total": 0})
    model_names = set()
    
    for r in results:
        if not r.get("debate_result") or not r["debate_result"].get("round_1_verdicts"):
            continue
        
        verdicts = r["debate_result"]["round_1_verdicts"]
        models = list(verdicts.keys())
        model_names.update(models)
        
        # Check all pairs
        for i, model_a in enumerate(models):
            for model_b in models[i+1:]:
                # Create consistent pair key (alphabetical order)
                pair_key = tuple(sorted([
                    model_a.split('/')[-1][:10],  # Shorten names
                    model_b.split('/')[-1][:10]
                ]))
                pair_key_str = f"{pair_key[0]}-{pair_key[1]}"
                
                verdict_a = normalize_label(verdicts[model_a]["verdict"])
                verdict_b = normalize_label(verdicts[model_b]["verdict"])
                
                pair_agreements[pair_key_str]["total"] += 1
                if verdict_a == verdict_b:
                    pair_agreements[pair_key_str]["agree"] += 1
    
    # Compute agreement rates
    agreement_matrix = {}
    for pair, counts in pair_agreements.items():
        if counts["total"] > 0:
            agreement_matrix[pair] = round(counts["agree"] / counts["total"], 4)
    
    # Average across all pairs
    avg_agreement = round(sum(agreement_matrix.values()) / len(agreement_matrix), 4) if agreement_matrix else 0.0
    
    # Extract unique shortened model names for plotting
    shortened_models = sorted(list(set([
        model.split('/')[-1][:10] for model in model_names
    ])))
    
    return {
        "agreement_matrix": agreement_matrix,
        "avg_pairwise_agreement": avg_agreement,
        "total_pairs": len(agreement_matrix),
        "model_names": shortened_models  # For plot parsing
    }


def compute_round_by_round_accuracy(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    TIER 1 METRIC: Track accuracy improvement across debate rounds.
    
    Computes accuracy of majority vote at Round 1, Round 2 vs Final (Judge).
    Uses 'round_metrics' if available (new format), otherwise falls back to limited data.
    
    Returns:
        {
            "round_1_accuracy": 0.70,
            "round_2_accuracy": 0.75,
            "final_accuracy": 0.85,
            "improvement_r1_to_final": 0.15,
            "rounds_distribution": {1: 50, 2: 30, 3: 20}
        }
    """
    from collections import Counter
    
    round_accuracies = defaultdict(lambda: {"correct": 0, "total": 0})
    rounds_distribution = defaultdict(int)
    
    for r in results:
        gold = r["gold_label"]
        debate_res = r.get("debate_result", {})
        
        # Track rounds used distribution
        if debate_res and debate_res.get("metrics"):
            rounds_used = debate_res["metrics"].get("rounds_used", 1)
            rounds_distribution[rounds_used] += 1
        
        # METHOD 1: Use 'round_metrics' list (Rich data from new logs)
        if debate_res and "round_metrics" in debate_res:
            for round_data in debate_res["round_metrics"]:
                r_num = round_data["round_num"]
                majority_v = normalize_label(round_data.get("majority_verdict", "NEI"))
                
                round_key = f"round_{r_num}"
                round_accuracies[round_key]["total"] += 1
                if majority_v == gold:
                    round_accuracies[round_key]["correct"] += 1
                    
        # METHOD 2: Fallback to legacy fields (round_1_verdicts) if round_metrics missing
        elif debate_res and debate_res.get("round_1_verdicts"):
            round_1_verdicts = [normalize_label(v["verdict"]) for v in debate_res["round_1_verdicts"].values()]
            majority_r1 = Counter(round_1_verdicts).most_common(1)[0][0]
            
            round_accuracies["round_1"]["total"] += 1
            if majority_r1 == gold:
                round_accuracies["round_1"]["correct"] += 1
        
        # Final (Judge verdict) - Always available
        if r.get("final_verdict") and r["final_verdict"] != "ERROR":
            final_v = normalize_label(r["final_verdict"])
            round_accuracies["final"]["total"] += 1
            if final_v == gold:
                round_accuracies["final"]["correct"] += 1
    
    # Compute accuracy per round
    result = {}
    for round_name, counts in round_accuracies.items():
        if counts["total"] > 0:
            acc = counts["correct"] / counts["total"]
            result[f"{round_name}_accuracy"] = round(acc, 4)
            result[f"{round_name}_correct"] = counts["correct"]
            result[f"{round_name}_total"] = counts["total"]
    
    # Add rounds distribution
    result["rounds_distribution"] = dict(rounds_distribution)
    
    # Compute improvement
    if "round_1_accuracy" in result and "final_accuracy" in result:
        result["improvement_r1_to_final"] = round(
            result["final_accuracy"] - result["round_1_accuracy"], 4
        )
    
    return result


def print_metrics_report(metrics: Dict[str, Any]):
    """Pretty print metrics report"""
    print("\n" + "="*80)
    print("EVALUATION METRICS REPORT".center(80))
    print("="*80)
    
    # Overall accuracy
    print("\n📊 Overall Metrics:")
    print(f"  Model Accuracy (pre-debate):  {metrics['model_accuracy']:.2f}%")
    print(f"  Final Accuracy (post-debate): {metrics['final_accuracy']:.2f}%")
    
    # Macro metrics
    print(f"\n📈 Macro-Averaged Metrics (Model):")
    model_macro = metrics["model_macro"]
    print(f"  Macro-Precision: {model_macro['macro_precision']:.4f}")
    print(f"  Macro-Recall:    {model_macro['macro_recall']:.4f}")
    print(f"  Macro-F1:        {model_macro['macro_f1']:.4f}")
    
    print(f"\n📈 Macro-Averaged Metrics (Final):")
    final_macro = metrics["final_macro"]
    print(f"  Macro-Precision: {final_macro['macro_precision']:.4f}")
    print(f"  Macro-Recall:    {final_macro['macro_recall']:.4f}")
    print(f"  Macro-F1:        {final_macro['macro_f1']:.4f}")
    
    # Per-class metrics (Final)
    print(f"\n📋 Per-Class Metrics (Final):")
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 70)
    for label, m in metrics["final_per_class"].items():
        print(f"{label:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10}")
    
    # Confusion Matrix (Final)
    print(f"\n🔢 Confusion Matrix (Final):")
    cm = metrics["final_confusion_matrix"]["matrix"]
    labels = metrics["final_confusion_matrix"]["labels"]
    
    # Header
    header_text = "True \\ Pred"
    print(f"{header_text:<20}", end="")
    for label in labels:
        print(f"{label[:10]:>12}", end="")
    print()
    print("-" * (20 + 12 * len(labels)))
    
    # Rows
    for true_label in labels:
        print(f"{true_label:<20}", end="")
        for pred_label in labels:
            count = cm[true_label][pred_label]
            print(f"{count:>12}", end="")
        print()
    
    # Hybrid Strategy Stats (DOWN Framework)
    if metrics.get("hybrid_strategy"):
        hybrid = metrics["hybrid_strategy"]
        if hybrid.get("debated", 0) > 0:  # Only show if hybrid mode was active
            print(f"\n🚀 Hybrid Strategy (DOWN Framework):")
            print(f"  Threshold:          {hybrid['threshold']}")
            print(f"  Skipped (high conf): {hybrid['skipped']}/{hybrid['total']} ({hybrid['skipped_ratio']:.1f}%)")
            print(f"  Debated (low conf):  {hybrid['debated']}/{hybrid['total']} ({hybrid['debated_ratio']:.1f}%)")
            print(f"  Skipped Accuracy:   {hybrid['skipped_accuracy']:.1f}%")
            print(f"  Debated Accuracy:   {hybrid['debated_accuracy']:.1f}%")
            print(f"  💰 Cost Saved:       ~{hybrid['cost_saved']:.0f}%")
    
    # Debate impact
    if metrics.get("debate_impact"):
        debate = metrics["debate_impact"]
        print(f"\n💬 Debate Impact Analysis:")
        print(f"  Total analyzed:     {debate['total_analyzed']}")
        print(f"  Kept correct:       {debate['kept_correct']} (model ✓ → final ✓)")
        print(f"  Fixed:              {debate['fixed']} (model ✗ → final ✓) - {debate['fix_rate']:.2f}%")
        print(f"  Broken:             {debate['broken']} (model ✓ → final ✗) - {debate['break_rate']:.2f}%")
        print(f"  Kept incorrect:     {debate['kept_incorrect']} (model ✗ → final ✗)")
    
    # TIER 1: Consensus Scores
    if metrics.get("consensus_scores"):
        consensus = metrics["consensus_scores"]
        print(f"\n🤝 TIER 1 - Consensus Metrics:")
        print(f"  Round 1 Avg Consensus:    {consensus.get('round_1_avg_consensus', 0):.4f}")
        print(f"  Final Avg Consensus:      {consensus.get('final_avg_consensus', 0):.4f}")
        print(f"  Consensus Improvement:    {consensus.get('consensus_improvement', 0):+.4f}")
        print(f"  Round 1 Unanimous Rate:   {consensus.get('round_1_unanimous_rate', 0):.2%}")
        print(f"  Final Unanimous Rate:     {consensus.get('final_unanimous_rate', 0):.2%}")
    
    # TIER 1: Inter-Agent Agreement
    if metrics.get("inter_agent_agreement"):
        agreement = metrics["inter_agent_agreement"]
        print(f"\n🔗 TIER 1 - Inter-Agent Agreement:")
        print(f"  Average Pairwise Agreement: {agreement.get('avg_pairwise_agreement', 0):.2%}")
        if agreement.get("agreement_matrix"):
            print(f"  Agreement Matrix:")
            for pair, rate in sorted(agreement["agreement_matrix"].items()):
                print(f"    {pair:<30}: {rate:.2%}")
    
    # TIER 1: Round-by-Round Accuracy
    if metrics.get("round_by_round_accuracy"):
        round_acc = metrics["round_by_round_accuracy"]
        print(f"\n📊 TIER 1 - Round-by-Round Accuracy:")
        if "round_1_accuracy" in round_acc:
            print(f"  Round 1 Majority Vote:    {round_acc['round_1_accuracy']:.2%} ({round_acc['round_1_correct']}/{round_acc['round_1_total']})")
        if "round_2_accuracy" in round_acc:
            print(f"  Round 2 Majority Vote:    {round_acc['round_2_accuracy']:.2%} ({round_acc['round_2_correct']}/{round_acc['round_2_total']})")
        if "final_accuracy" in round_acc:
            print(f"  Final (Judge):            {round_acc['final_accuracy']:.2%} ({round_acc['final_correct']}/{round_acc['final_total']})")
        if "improvement_r1_to_final" in round_acc:
            print(f"  Improvement (R1→Final):   {round_acc['improvement_r1_to_final']:+.2%}")
    
    # TIER 2: Model Calibration
    if metrics.get("model_calibration"):
        calib = metrics["model_calibration"]
        print(f"\n🎯 TIER 2 - Model Calibration (3-Label):")
        print(f"  Samples analyzed:         {calib.get('n_samples', 0)}")
        
        if calib.get("raw"):
            raw = calib["raw"]
            print(f"\n  Before Calibration (T=1.0):")
            print(f"    ECE:                    {raw['ece']:.4f}")
            print(f"    Brier Score:            {raw['brier']:.4f}")
            print(f"    Avg Confidence:         {raw['avg_confidence']:.4f}")
            print(f"    Avg Conf (Correct):     {raw['avg_confidence_correct']:.4f}")
            print(f"    Avg Conf (Incorrect):   {raw['avg_confidence_incorrect']:.4f}")
        
        if calib.get("calibrated") and calib.get("temperature", 1.0) != 1.0:
            cal = calib["calibrated"]
            T = calib["temperature"]
            print(f"\n  After Calibration (T={T:.4f}):")
            print(f"    ECE:                    {cal['ece']:.4f} ({(cal['ece'] - raw['ece']) / raw['ece']:+.1%})")
            print(f"    Brier Score:            {cal['brier']:.4f} ({(cal['brier'] - raw['brier']) / raw['brier']:+.1%})")
            print(f"    Avg Confidence:         {cal['avg_confidence']:.4f}")
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ViFactCheck Pipeline Results")
    parser.add_argument("results_file", type=str, help="Path to results JSON file")
    parser.add_argument("--save-metrics", type=str, default=None, help="Save metrics to JSON file")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for calibration (1.0 = no calibration)")
    
    args = parser.parse_args()
    
    # Load results
    print(f"📂 Loading results from: {args.results_file}")
    data = load_results(args.results_file)
    results = data["results"]
    
    # Compute model metrics (pre-debate)
    model_cm = compute_confusion_matrix(results, verdict_key="model_verdict")
    model_per_class = compute_metrics_per_class(model_cm)
    model_macro = compute_macro_metrics(model_per_class)
    
    # Compute final metrics (post-debate)
    final_cm = compute_confusion_matrix(results, verdict_key="final_verdict")
    final_per_class = compute_metrics_per_class(final_cm)
    final_macro = compute_macro_metrics(final_per_class)
    
    # Auto-detect if debate is enabled by checking if any sample has debate_info with data
    # Note: debate_info can be None (no debate), {} (debate failed), or dict with data (debate OK)
    debate_enabled = any(
        r.get("debate_info") and 
        isinstance(r.get("debate_info"), dict) and 
        r["debate_info"].get("verdict")  # Has actual debate verdict
        for r in results
    )
    
    # Debate impact (always compute if debate caused any changes)
    debate_impact = analyze_debate_impact(results) if debate_enabled else None
    
    # Hybrid Strategy stats (DOWN Framework)
    hybrid_stats = analyze_hybrid_strategy(results)
    
    # Debate rounds distribution
    debate_rounds = compute_debate_rounds_distribution(results) if debate_enabled else {}
    
    # Debator performance
    debator_perf = compute_debator_performance(results) if debate_enabled else {}
    
    # TIER 1 METRICS
    consensus_scores = compute_consensus_scores(results) if debate_enabled else {}
    inter_agent_agreement = compute_inter_agent_agreement(results) if debate_enabled else {}
    round_accuracy = compute_round_by_round_accuracy(results) if debate_enabled else {}
    
    # TIER 2: Model Calibration
    model_calibration = compute_calibration_metrics(results, T=args.temperature, n_bins=10)
    
    # Compile metrics
    metrics = {
        "model_accuracy": data.get("model_accuracy", 0.0),
        "final_accuracy": data.get("final_accuracy", 0.0),
        "model_confusion_matrix": model_cm,
        "final_confusion_matrix": final_cm,
        "model_per_class": model_per_class,
        "final_per_class": final_per_class,
        "model_macro": model_macro,
        "final_macro": final_macro,
        "debate_impact": debate_impact,
        "hybrid_strategy": hybrid_stats,
        "debate_rounds_distribution": debate_rounds,
        "debator_performance": debator_perf,
        # TIER 1 metrics
        "consensus_scores": consensus_scores,
        "inter_agent_agreement": inter_agent_agreement,
        "round_by_round_accuracy": round_accuracy,
        # TIER 2 metrics
        "model_calibration": model_calibration
    }
    
    # Print report
    print_metrics_report(metrics)
    
    # Save metrics if requested
    if args.save_metrics:
        with open(args.save_metrics, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"💾 Metrics saved to: {args.save_metrics}")
    
    print("✅ Evaluation complete!")


if __name__ == "__main__":
    main()
