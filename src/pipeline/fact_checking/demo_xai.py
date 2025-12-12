"""
Demo script to test PhoBERT XAI
Run: python demo_xai.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.fact_checking.xai_phobert import load_xai_model
import json


def print_xai_result(claim: str, evidence: str, xai_result: dict, expected: str = None):
    """Pretty print XAI result"""
    predicted = xai_result['relationship']
    match = "✅" if expected and predicted == expected else ("❌" if expected else "")
    
    print("\n" + "="*80)
    print("CLAIM:")
    print(f"  {claim}")
    print("\nEVIDENCE:")
    print(f"  {evidence[:200]}...")  # Truncate for readability
    print("\n" + "-"*80)
    
    # Show prediction match
    if expected:
        print(f"PREDICTION: {predicted} (Expected: {expected}) {match}")
        print("-"*80)
    
    print("XAI OUTPUT:")
    print(f"  1. Relationship:        {xai_result['relationship']}")
    
    # Show conflict words if available
    claim_conflict = xai_result.get('claim_conflict_word', '')
    evidence_conflict = xai_result.get('evidence_conflict_word', '')
    if claim_conflict or evidence_conflict:
        print(f"  ⚡ Key Conflict:        '{claim_conflict}' ⚔️ '{evidence_conflict}'")
    
    print(f"  2. Natural Explanation: {xai_result['natural_explanation']}")
    print(f"\n  [Debug] Similarity Score: {xai_result['similarity_score']:.3f}")
    print("="*80 + "\n")


def main():
    """Main demo function"""
    
    # Path to trained model
    model_path = "results/fact_checking/pyvi/checkpoints/best_model_pyvi.pt"
    
    print("Loading PhoBERT XAI model...")
    xai = load_xai_model(model_path, device="cpu")
    print("✓ Model loaded successfully!\n")
    
    # Test cases - diverse examples for SUPPORTS, REFUTES, NEI
    test_cases = [
        # ============ REFUTES cases ============
        {
            "claim": "Việt Nam là quốc gia thứ hai đón ngọn đuốc SEA Games 31.",
            "evidence": "Việt Nam là quốc gia đầu tiên đón ngọn đuốc SEA Games 31 vào ngày 15/3/2022.",
            "expected": "REFUTES"
        },
        {
            "claim": "Chính phủ đã ban hành nghị định mới về thuế.",
            "evidence": "Bộ Tài chính đang xem xét dự thảo nghị định về cải cách thuế thu nhập cá nhân.",
            "expected": "REFUTES"
        },
        {
            "claim": "Dân số Việt Nam giảm 5% trong năm 2023.",
            "evidence": "Theo Tổng cục Thống kê, dân số Việt Nam tăng 0.95% trong năm 2023, đạt khoảng 100 triệu người.",
            "expected": "REFUTES"
        },
        
        # ============ SUPPORTS cases ============
        {
            "claim": "Giá vé phổ thông là 260.000 - 690.000 đồng mỗi lượt.",
            "evidence": "Giá vé phổ thông dao động từ 260.000 đến 690.000 đồng cho mỗi lượt tham quan.",
            "expected": "SUPPORTS"
        },
        {
            "claim": "Hà Nội là thủ đô của Việt Nam.",
            "evidence": "Hà Nội, thủ đô nước Cộng hòa Xã hội Chủ nghĩa Việt Nam, là trung tâm chính trị, văn hóa của cả nước.",
            "expected": "SUPPORTS"
        },
        {
            "claim": "SEA Games 31 được tổ chức tại Việt Nam vào năm 2022.",
            "evidence": "Đại hội Thể thao Đông Nam Á lần thứ 31 (SEA Games 31) diễn ra tại Việt Nam từ ngày 12 đến 23 tháng 5 năm 2022.",
            "expected": "SUPPORTS"
        },
        {
            "claim": "Việt Nam có đường bờ biển dài hơn 3000 km.",
            "evidence": "Việt Nam có đường bờ biển dài khoảng 3.260 km, trải dài từ Móng Cái đến Hà Tiên.",
            "expected": "SUPPORTS"
        },
        
        # ============ NEI cases ============
        {
            "claim": "iPhone 15 bán chạy nhất tại Việt Nam năm 2024.",
            "evidence": "Apple đã ra mắt iPhone 15 series vào tháng 9 năm 2023 với nhiều cải tiến về camera và chip A17.",
            "expected": "NEI"
        },
        {
            "claim": "Việt Nam sẽ đăng cai World Cup 2030.",
            "evidence": "FIFA đang xem xét các ứng cử viên đăng cai World Cup 2030, bao gồm các nước Nam Mỹ và châu Âu.",
            "expected": "NEI"
        },
        {
            "claim": "Đội tuyển bóng đá Việt Nam vô địch AFF Cup 2024.",
            "evidence": "Đội tuyển Việt Nam đã tham gia AFF Cup nhiều lần và từng vô địch vào năm 2018.",
            "expected": "NEI"
        }
    ]
    
    print(f"Running XAI on {len(test_cases)} test cases...\n")
    
    # Track results
    results = {"correct": 0, "total": 0, "by_label": {"SUPPORTS": [], "REFUTES": [], "NEI": []}}
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'#'*80}")
        print(f"# TEST CASE {i} (Expected: {case.get('expected', 'N/A')})")
        print(f"{'#'*80}")
        
        xai_result = xai.generate_xai(
            claim=case["claim"],
            evidence=case["evidence"]
        )
        
        print_xai_result(
            claim=case["claim"],
            evidence=case["evidence"],
            xai_result=xai_result,
            expected=case.get("expected")
        )
        
        # Track accuracy
        expected = case.get("expected")
        predicted = xai_result["relationship"]
        if expected:
            results["total"] += 1
            is_correct = predicted == expected
            if is_correct:
                results["correct"] += 1
            results["by_label"][expected].append({
                "case": i,
                "predicted": predicted,
                "correct": is_correct
            })
    
    # Print summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    print(f"Overall Accuracy: {results['correct']}/{results['total']} ({100*results['correct']/results['total']:.1f}%)")
    print()
    for label in ["SUPPORTS", "REFUTES", "NEI"]:
        cases = results["by_label"][label]
        if cases:
            correct = sum(1 for c in cases if c["correct"])
            print(f"  {label}: {correct}/{len(cases)} correct")
            for c in cases:
                status = "✅" if c["correct"] else f"❌ (predicted {c['predicted']})"
                print(f"    - Case {c['case']}: {status}")
    
    print("\n✓ Demo completed!")
    print("\nNext steps:")
    print("  1. Check if claim/evidence highlights make sense")
    print("  2. For visual highlighting, use Jupyter notebook")
    print("  3. Integrate into main pipeline for Fast Path XAI")


if __name__ == "__main__":
    main()
