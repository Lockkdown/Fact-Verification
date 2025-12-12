"""Quick script to check XAI results in hybrid_debate file."""
import json

# Load data
data = json.load(open('results/vifactcheck/test/hybrid_debate/vifactcheck_test_results.json', encoding='utf-8'))

# Split by path
fast_path = [r for r in data['results'] if r.get('debate_result') is None]
slow_path = [r for r in data['results'] if r.get('debate_result') is not None]

print("=" * 60)
print("HYBRID DEBATE TEST RESULTS - XAI CHECK")
print("=" * 60)
print(f"Total samples: {len(data['results'])}")
print(f"Fast Path (PhoBERT only): {len(fast_path)}")
print(f"Slow Path (with Debate): {len(slow_path)}")
print()

# Check Slow Path XAI
xai_count = sum(1 for r in slow_path if r.get('debate_result', {}).get('xai'))
print(f"Slow Path with XAI: {xai_count}/{len(slow_path)}")

# Verdict distribution
verdicts = {}
for r in slow_path:
    v = r['debate_result']['verdict']
    verdicts[v] = verdicts.get(v, 0) + 1
print(f"Slow Path Verdict Distribution: {verdicts}")

# Check XAI field completeness
xai_fields = ['relationship', 'natural_explanation', 'conflict_claim', 'conflict_evidence']
missing = {f: 0 for f in xai_fields}
for r in slow_path:
    xai = r['debate_result'].get('xai', {})
    for f in xai_fields:
        if not xai.get(f):
            missing[f] += 1

print(f"\nMissing XAI fields (expected for conflict_* when not REFUTED):")
for k, v in missing.items():
    if v > 0:
        print(f"  - {k}: {v} samples")

# Sample XAI for each verdict type
print("\n" + "=" * 60)
print("SAMPLE XAI BY VERDICT TYPE")
print("=" * 60)

for verdict_type in ['SUPPORTED', 'REFUTED', 'NEI']:
    samples = [r for r in slow_path if r['debate_result']['verdict'] == verdict_type]
    if samples:
        s = samples[0]
        xai = s['debate_result'].get('xai', {})
        print(f"\n--- {verdict_type} ({len(samples)} samples) ---")
        print(f"Claim: {s['statement'][:60]}...")
        print(f"Evidence: {s['evidence'][:60]}...")
        print(f"XAI:")
        print(f"  relationship: {xai.get('relationship', 'N/A')}")
        print(f"  natural_explanation: {xai.get('natural_explanation', 'N/A')[:80]}...")
        if verdict_type == 'REFUTED':
            print(f"  conflict_claim: {xai.get('conflict_claim', 'N/A')}")
            print(f"  conflict_evidence: {xai.get('conflict_evidence', 'N/A')}")

# Verify metrics unchanged
print("\n" + "=" * 60)
print("METRICS VERIFICATION")
print("=" * 60)
print(f"model_accuracy: {data['model_accuracy']:.4f} (83.21%)")
print(f"final_accuracy: {data['final_accuracy']:.4f} (87.77%)")
