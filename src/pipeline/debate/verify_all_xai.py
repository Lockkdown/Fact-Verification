"""
Verify XAI results for all 4 files (test/dev, hybrid/full).
Checks:
1. XAI presence for fast path (phobert_xai) and slow path (debate_result.xai)
2. Metrics unchanged
3. XAI field completeness
"""

import json
from pathlib import Path

# Expected metrics (from original evaluation)
EXPECTED_METRICS = {
    "test/hybrid_debate": {"model_accuracy": 0.8321, "final_accuracy": 0.8777},
    "test/full_debate": {"model_accuracy": 0.8321, "final_accuracy": 0.8694},
    "dev/hybrid_debate": {"model_accuracy": 0.8327, "final_accuracy": 0.8712},
    "dev/full_debate": {"model_accuracy": 0.8327, "final_accuracy": 0.8615},
}

def check_file(file_path: Path, file_key: str):
    """Check a single results file."""
    print(f"\n{'='*60}")
    print(f"📁 {file_key}")
    print(f"{'='*60}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get("results", [])
    total = len(results)
    
    # Count XAI presence
    fast_path_count = 0
    fast_path_with_xai = 0
    slow_path_count = 0
    slow_path_with_xai = 0
    
    # Check XAI field completeness
    xai_fields_missing = {
        "relationship": 0,
        "natural_explanation": 0
    }
    
    for result in results:
        if result.get("debate_result") is not None:
            # Slow path (went through debate)
            slow_path_count += 1
            xai = result["debate_result"].get("xai")
            if xai:
                slow_path_with_xai += 1
                # Check fields
                for field in xai_fields_missing:
                    if not xai.get(field):
                        xai_fields_missing[field] += 1
        else:
            # Fast path (PhoBERT only)
            fast_path_count += 1
            xai = result.get("phobert_xai")
            if xai:
                fast_path_with_xai += 1
                # Check fields
                for field in xai_fields_missing:
                    if not xai.get(field):
                        xai_fields_missing[field] += 1
    
    # Print results
    print(f"\n📊 Sample Distribution:")
    print(f"   Total samples: {total}")
    print(f"   Fast path (PhoBERT): {fast_path_count}")
    print(f"   Slow path (Debate): {slow_path_count}")
    
    print(f"\n✅ XAI Coverage:")
    if fast_path_count > 0:
        pct = fast_path_with_xai / fast_path_count * 100
        status = "✅" if pct == 100 else "⚠️"
        print(f"   {status} Fast path XAI: {fast_path_with_xai}/{fast_path_count} ({pct:.1f}%)")
    
    if slow_path_count > 0:
        pct = slow_path_with_xai / slow_path_count * 100
        status = "✅" if pct == 100 else "⚠️"
        print(f"   {status} Slow path XAI: {slow_path_with_xai}/{slow_path_count} ({pct:.1f}%)")
    
    # Check metrics
    model_acc = data.get("model_accuracy", 0)
    final_acc = data.get("final_accuracy", 0)
    expected = EXPECTED_METRICS.get(file_key, {})
    
    print(f"\n📈 Metrics Verification:")
    
    # Model accuracy check (with tolerance)
    exp_model = expected.get("model_accuracy", 0)
    model_match = abs(model_acc - exp_model) < 0.001
    status = "✅" if model_match else "❌"
    print(f"   {status} model_accuracy: {model_acc:.4f} (expected ~{exp_model:.4f})")
    
    # Final accuracy check
    exp_final = expected.get("final_accuracy", 0)
    final_match = abs(final_acc - exp_final) < 0.001
    status = "✅" if final_match else "❌"
    print(f"   {status} final_accuracy: {final_acc:.4f} (expected ~{exp_final:.4f})")
    
    # XAI field completeness
    print(f"\n📝 XAI Field Completeness:")
    all_complete = True
    for field, missing in xai_fields_missing.items():
        if missing > 0:
            all_complete = False
            print(f"   ⚠️ {field}: {missing} samples missing")
        else:
            print(f"   ✅ {field}: Complete")
    
    # Sample XAI preview
    print(f"\n🔍 Sample XAI Preview:")
    
    # Show one fast path sample (if exists)
    for result in results:
        if result.get("phobert_xai"):
            xai = result["phobert_xai"]
            print(f"\n   [FAST PATH - PhoBERT]")
            print(f"   relationship: {xai.get('relationship', 'N/A')}")
            print(f"   source: {xai.get('source', 'N/A')}")
            break
    
    # Show one slow path sample
    for result in results:
        debate_result = result.get("debate_result")
        if debate_result and debate_result.get("xai"):
            xai = debate_result["xai"]
            print(f"\n   [SLOW PATH - Judge LLM]")
            print(f"   relationship: {xai.get('relationship', 'N/A')}")
            print(f"   source: {xai.get('source', 'N/A')}")
            break
    
    return {
        "file_key": file_key,
        "total": total,
        "fast_path": fast_path_count,
        "fast_path_xai": fast_path_with_xai,
        "slow_path": slow_path_count,
        "slow_path_xai": slow_path_with_xai,
        "metrics_ok": model_match and final_match,
        "xai_complete": all_complete
    }


def main():
    print("🔍 XAI VERIFICATION FOR ALL RESULT FILES")
    print("=" * 60)
    
    results_dir = Path(__file__).parent.parent / "results" / "vifactcheck"
    
    files = [
        ("test/hybrid_debate", results_dir / "test" / "hybrid_debate" / "vifactcheck_test_results.json"),
        ("test/full_debate", results_dir / "test" / "full_debate" / "vifactcheck_test_results.json"),
        ("dev/hybrid_debate", results_dir / "dev" / "hybrid_debate" / "vifactcheck_dev_results.json"),
        ("dev/full_debate", results_dir / "dev" / "full_debate" / "vifactcheck_dev_results.json"),
    ]
    
    all_results = []
    for file_key, file_path in files:
        if file_path.exists():
            result = check_file(file_path, file_key)
            all_results.append(result)
        else:
            print(f"\n❌ File not found: {file_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    
    total_samples = 0
    total_xai = 0
    all_metrics_ok = True
    
    print("\n| File | Samples | Fast XAI | Slow XAI | Metrics |")
    print("|------|---------|----------|----------|---------|")
    
    for r in all_results:
        total_samples += r["total"]
        total_xai += r["fast_path_xai"] + r["slow_path_xai"]
        if not r["metrics_ok"]:
            all_metrics_ok = False
        
        fast_str = f"{r['fast_path_xai']}/{r['fast_path']}" if r['fast_path'] > 0 else "N/A"
        slow_str = f"{r['slow_path_xai']}/{r['slow_path']}" if r['slow_path'] > 0 else "N/A"
        metrics_str = "✅" if r["metrics_ok"] else "❌"
        
        print(f"| {r['file_key']:18} | {r['total']:7} | {fast_str:8} | {slow_str:8} | {metrics_str:7} |")
    
    print(f"\n📊 Total: {total_samples} samples, {total_xai} with XAI")
    
    if all_metrics_ok:
        print("\n✅ ALL METRICS UNCHANGED - XAI post-processing successful!")
    else:
        print("\n❌ SOME METRICS CHANGED - Please investigate!")


if __name__ == "__main__":
    main()
