"""
Test XAI consistency between PhoBERT (fast path) and Judge LLM (slow path).
Generates XAI for a few sample claims and compares output structure.
"""

import json
import sys
from pathlib import Path
import argparse
import asyncio
import os
import re
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / '.env')

TEST_SAMPLES = [
    {
        "id": "test_support",
        "statement": "Việt_Nam đã giành huy_chương vàng tại SEA Games 32 .",
        "evidence": "Đoàn thể_thao Việt_Nam đã xuất_sắc giành được huy_chương vàng tại SEA Games 32 tổ_chức tại Campuchia .",
        "verdict": "Support",
        "expected_relationship": "SUPPORTS"
    },
    {
        "id": "test_refute",
        "statement": "Giá xăng tăng 500 đồng / lít vào ngày 15 / 3 .",
        "evidence": "Giá xăng giảm 200 đồng / lít kể từ 15 / 3 theo quyết_định của liên Bộ Công_Thương - Tài_chính .",
        "verdict": "Refute",
        "expected_relationship": "REFUTES"
    },
    {
        "id": "test_nei",
        "statement": "Ông Nguyễn_Văn_A được bổ_nhiệm làm Giám_đốc Sở Y_tế tỉnh Bình_Dương .",
        "evidence": "UBND tỉnh Bình_Dương vừa công_bố quyết_định bổ_nhiệm lãnh_đạo các sở , ngành .",
        "verdict": "NEI",
        "expected_relationship": "NEI"
    }
]


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("_", " ")).strip()


def _label_to_rel(label: Any) -> Optional[str]:
    if label is None:
        return None
    try:
        label_int = int(label)
    except Exception:
        return None
    if label_int == 0:
        return "SUPPORTS"
    if label_int == 1:
        return "REFUTES"
    if label_int == 2:
        return "NEI"
    return None


def load_samples_from_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    if not path.exists():
        return samples

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue

            rel = _label_to_rel(r.get("label"))
            if not rel:
                continue

            samples.append(
                {
                    "id": str(r.get("sample_id") or r.get("id") or len(samples)),
                    "statement": r.get("statement", ""),
                    "evidence": r.get("evidence", ""),
                    "expected_relationship": rel,
                }
            )
            if len(samples) >= limit:
                break

    return samples


def _check_fast_xai(sample: Dict[str, Any], xai: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    rel = (xai.get("relationship") or "").upper()
    expl = xai.get("natural_explanation") or ""

    if len(_norm_text(expl)) < 10:
        issues.append("FAST_EMPTY_OR_SHORT_EXPLANATION")

    if "_" in str(expl):
        issues.append("FAST_UNDERSCORE_IN_EXPLANATION")

    cc = _norm_text(xai.get("claim_conflict_word") or "")
    ce = _norm_text(xai.get("evidence_conflict_word") or "")

    claim_text = _norm_text(sample.get("statement") or "").lower()
    evidence_text = _norm_text(sample.get("evidence") or "").lower()

    if rel != "REFUTES":
        if cc or ce:
            issues.append("FAST_NON_REFUTES_HAS_CONFLICT")
    else:
        if not cc or not ce:
            issues.append("FAST_REFUTES_MISSING_CONFLICT")
        else:
            if cc.lower() not in claim_text:
                issues.append("FAST_CONFLICT_CLAIM_NOT_IN_CLAIM")
            if ce.lower() not in evidence_text:
                issues.append("FAST_CONFLICT_EVIDENCE_NOT_IN_EVIDENCE")

    return issues


def _check_judge_xai(sample: Dict[str, Any], xai: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    rel = (sample.get("expected_relationship") or "").upper()
    expl = xai.get("natural_explanation_vi") or ""

    if len(_norm_text(expl)) < 10:
        issues.append("SLOW_EMPTY_OR_SHORT_EXPLANATION")

    if "_" in str(expl):
        issues.append("SLOW_UNDERSCORE_IN_EXPLANATION")

    cc = _norm_text(xai.get("conflict_claim") or "")
    ce = _norm_text(xai.get("conflict_evidence") or "")

    claim_text = _norm_text(sample.get("statement") or "").lower()
    evidence_text = _norm_text(sample.get("evidence") or "").lower()

    if rel != "REFUTES":
        if cc or ce:
            issues.append("SLOW_NON_REFUTES_HAS_CONFLICT")
    else:
        if not cc or not ce:
            issues.append("SLOW_REFUTES_MISSING_CONFLICT")
        else:
            if cc.lower() not in claim_text:
                issues.append("SLOW_CONFLICT_CLAIM_NOT_IN_CLAIM")
            if ce.lower() not in evidence_text:
                issues.append("SLOW_CONFLICT_EVIDENCE_NOT_IN_EVIDENCE")

    return issues


def test_phobert_xai(
    samples: List[Dict[str, Any]],
    verbose: bool = False,
    max_print: int = 10,
    full_text: bool = False,
    quote_max_len: int = 0,
):
    """Test PhoBERT XAI generation."""
    print("=" * 60)
    print("TESTING PHOBERT XAI (Fast Path)")
    print("=" * 60)
    
    try:
        from src.pipeline.fact_checking.xai_phobert import load_xai_model
        
        model_path = project_root / "results/fact_checking/pyvi/checkpoints/best_model_pyvi.pt"
        print(f"Loading model from: {model_path}")
        
        xai_module = load_xai_model(str(model_path))
        print("✅ PhoBERT model loaded successfully")
        
        results = []
        for idx, sample in enumerate(samples, 1):
            if verbose and idx <= max_print:
                print(f"\n--- Sample: {sample['id']} ---")
                print(f"Statement: {sample['statement']}")
                print(f"Evidence: {sample['evidence'][:60]}...")
                print(f"Expected: {sample['expected_relationship']}")
            
            xai = xai_module.generate_xai(
                claim=_norm_text(sample['statement']),
                evidence=_norm_text(sample['evidence']),
                model_verdict=sample['expected_relationship']
            )
            
            if verbose and idx <= max_print:
                print(f"\nXAI Output:")
                print(f"  relationship: {xai.get('relationship', 'N/A')}")
                nat = xai.get('natural_explanation', 'N/A')
                if quote_max_len and isinstance(nat, str):
                    nat = re.sub(
                        r"\(Trích:\s*'(.*?)'\)",
                        lambda m: f"(Trích: '{m.group(1)[:quote_max_len]}...')" if len(m.group(1)) > quote_max_len else m.group(0),
                        nat,
                        flags=re.DOTALL,
                    )
                if full_text:
                    print(f"  natural_explanation: {nat}")
                else:
                    print(f"  natural_explanation: {str(nat)[:80]}...")

                if sample['expected_relationship'] == "REFUTES":
                    print(f"  claim_conflict_word: {xai.get('claim_conflict_word', 'N/A')}")
                    print(f"  evidence_conflict_word: {xai.get('evidence_conflict_word', 'N/A')}")
            
            results.append({
                "sample_id": sample['id'],
                "xai": xai,
                "source": "PHOBERT"
            })
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_judge_xai(samples: List[Dict[str, Any]], max_concurrency: int = 5):
    """Test Judge LLM XAI generation."""
    print("\n" + "=" * 60)
    print("TESTING JUDGE LLM XAI (Slow Path)")
    print("=" * 60)

    from src.pipeline.debate.llm_client import LLMClient
    
    XAI_PROMPT = """**TASK:** Generate XAI (Explainable AI) fields for a fact-checking verdict.

**CONTEXT:**
- Claim: "{claim}"
- Evidence: "{evidence}"
- Verdict: {verdict}

**YOUR JOB:** Based on the verdict above, generate structured XAI fields.

**OUTPUT (JSON only, no markdown):**
{{
    "conflict_claim": "Nếu REFUTED: từ/cụm trong claim gây mâu thuẫn. Để trống '' nếu SUPPORTED/NEI",
    "conflict_evidence": "Nếu REFUTED: từ/cụm trong evidence mâu thuẫn với claim. Để trống '' nếu SUPPORTED/NEI",
    "natural_explanation_vi": "Giải thích 1-2 câu ngắn gọn bằng tiếng Việt"
}}

**RULES:**
- conflict_*: Chỉ điền nếu verdict là REFUTED, để trống "" nếu SUPPORTED/NEI
- conflict_claim: phải là từ/cụm từ XUẤT HIỆN NGUYÊN VĂN trong claim
- conflict_evidence: phải là từ/cụm từ XUẤT HIỆN NGUYÊN VĂN trong evidence
- Tuyệt đối KHÔNG bịa thêm fact/entity/number ngoài claim/evidence
- natural_explanation_vi: Giải thích ngắn gọn, dễ hiểu

**OUTPUT JSON:**"""

    async def generate_xai(client: LLMClient, sample: Dict[str, Any], sem: asyncio.Semaphore):
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        
        prompt = XAI_PROMPT.format(
            claim=_norm_text(sample['statement']),
            evidence=_norm_text(sample['evidence']),
            verdict=sample['expected_relationship']
        )

        async with sem:
            response = await client.generate_async(
                model="deepseek/deepseek-chat-v3-0324",
                prompt=prompt,
                api_key=api_key,
                base_url=base_url,
                max_tokens=500,
                temperature=0.3
            )
        
        # Parse JSON
        cleaned = re.sub(r'```(?:json)?', '', response).strip()
        cleaned = re.sub(r'```', '', cleaned).strip()
        cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', cleaned)
        
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end != -1:
            json_str = cleaned[start:end+1]
            return json.loads(json_str)
        
        return None
    
    async def run_tests():
        client = LLMClient()
        results = []
        sem = asyncio.Semaphore(max_concurrency)
        
        try:
            tasks = [generate_xai(client, s, sem) for s in samples]
            xai_list = await asyncio.gather(*tasks, return_exceptions=True)

            for sample, xai in zip(samples, xai_list):
                if isinstance(xai, Exception) or not xai:
                    results.append({
                        "sample_id": sample['id'],
                        "xai": {},
                        "source": "JUDGE_LLM",
                        "error": str(xai) if isinstance(xai, Exception) else "parse_failed",
                    })
                    continue

                results.append({
                    "sample_id": sample['id'],
                    "xai": xai,
                    "source": "JUDGE_LLM"
                })
                    
        finally:
            await client.close()
        
        return results
    
    try:
        return asyncio.run(run_tests())
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return []


def compare_xai_structure(phobert_results, judge_results):
    """Compare XAI structure between PhoBERT and Judge."""
    print("\n" + "=" * 60)
    print("XAI STRUCTURE COMPARISON")
    print("=" * 60)
    
    # Expected fields
    expected_fields = [
        "relationship",
        "natural_explanation",
        "conflict_claim",
        "conflict_evidence"
    ]
    
    print("\n📊 Field presence check:")
    print("-" * 40)
    
    for p_res, j_res in zip(phobert_results, judge_results):
        sample_id = p_res['sample_id']
        print(f"\n{sample_id}:")

        p_xai = p_res['xai']
        j_xai = j_res['xai']

        # Check natural explanation
        p_nat = p_xai.get('natural_explanation', '')
        j_nat = j_xai.get('natural_explanation_vi', '')
        print(f"  natural_explanation:")
        print(f"    PhoBERT: {p_nat[:60]}...")
        print(f"    Judge:   {j_nat[:60]}...")

        # Check conflicts for REFUTES sample
        if sample_id == 'test_refute':
            p_conflict_claim = p_xai.get('claim_conflict_word', '')
            j_conflict_claim = j_xai.get('conflict_claim', '')
            print(f"  conflict_claim:")
            print(f"    PhoBERT: {p_conflict_claim}")
            print(f"    Judge:   {j_conflict_claim}")

            p_conflict_evidence = p_xai.get('evidence_conflict_word', '')
            j_conflict_evidence = j_xai.get('conflict_evidence', '')
            print(f"  conflict_evidence:")
            print(f"    PhoBERT: {p_conflict_evidence}")
            print(f"    Judge:   {j_conflict_evidence}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument(
        "--data",
        type=str,
        default=str(project_root / "dataset/processed/vifactcheck_pyvi/vifactcheck_test.jsonl"),
    )
    parser.add_argument("--max_concurrency", type=int, default=5)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--max_print", type=int, default=10)
    parser.add_argument("--only_fast", action="store_true", default=False)
    parser.add_argument("--full_text", action="store_true", default=False)
    parser.add_argument("--quote_max_len", type=int, default=0)
    args = parser.parse_args()

    print(" XAI CONSISTENCY TEST")
    print("Testing PhoBERT (fast path) vs Judge LLM (slow path)\n")

    data_path = Path(args.data)
    samples = load_samples_from_jsonl(data_path, args.n)
    if not samples:
        samples = TEST_SAMPLES
        print(f"⚠️ Could not load labeled samples from: {data_path}. Using built-in samples ({len(samples)}).")
    else:
        print(f"Loaded labeled samples: {len(samples)} from {data_path}")

    phobert_results = test_phobert_xai(
        samples,
        verbose=args.verbose,
        max_print=args.max_print,
        full_text=args.full_text,
        quote_max_len=args.quote_max_len,
    )

    judge_results = []
    if not args.only_fast:
        judge_results = test_judge_xai(samples, max_concurrency=args.max_concurrency)

    issue_counts: Dict[str, int] = {}
    for sample, p_res in zip(samples, phobert_results):
        for it in _check_fast_xai(sample, p_res.get("xai", {})):
            issue_counts[it] = issue_counts.get(it, 0) + 1

    if judge_results:
        for sample, j_res in zip(samples, judge_results):
            if j_res.get("error"):
                issue_counts["SLOW_LLM_ERROR"] = issue_counts.get("SLOW_LLM_ERROR", 0) + 1
                continue
            for it in _check_judge_xai(sample, j_res.get("xai", {})):
                issue_counts[it] = issue_counts.get(it, 0) + 1

    print("\n" + "=" * 60)
    print("✅ TEST COMPLETE")
    print("=" * 60)

    print("\n📝 SUMMARY:")
    print(f"  Samples: {len(samples)}")
    print(f"  PhoBERT XAI samples: {len(phobert_results)}")
    if judge_results:
        print(f"  Judge LLM samples: {len(judge_results)}")
    else:
        print("  Judge LLM samples: (skipped)")

    if phobert_results and judge_results and len(samples) <= 10:
        compare_xai_structure(phobert_results, judge_results)

    print("\nIssue counts:")
    if not issue_counts:
        print("  (none)")
    else:
        for k, v in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
