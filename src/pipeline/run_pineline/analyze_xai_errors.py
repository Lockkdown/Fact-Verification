"""
XAI Error Analysis Script
Analyzes XAI explanations on 300 samples from dataset + 100 custom samples
to identify systematic errors and patterns.

Run: python scripts/analyze_xai_errors.py
Output: results/xai_error_analysis.json
"""

import sys
from pathlib import Path
import json
import random
from collections import defaultdict
from datetime import datetime

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.fact_checking.xai_phobert import load_xai_model
from datasets import load_dataset


# ============ Custom test samples (outside dataset) ============
CUSTOM_SAMPLES = [
    # REFUTES cases - clear contradiction
    {
        "claim": "Việt Nam thua Myanmar ở lượt cuối, qua đó không thể đi tiếp bảng B SEA Games 33.",
        "evidence": "Việt Nam thắng Myanmar 2-0 ở lượt cuối, qua đó đi tiếp với tư cách đội nhất bảng B SEA Games 33.",
        "expected": "REFUTES",
        "note": "thua vs thắng, không thể vs đi tiếp"
    },
    {
        "claim": "Tổng thống Biden từ chức vào tháng 1 năm 2024.",
        "evidence": "Tổng thống Biden tiếp tục nhiệm kỳ và không có thông báo từ chức nào trong năm 2024.",
        "expected": "REFUTES",
        "note": "từ chức vs tiếp tục nhiệm kỳ"
    },
    {
        "claim": "Apple ngừng sản xuất iPhone từ năm 2023.",
        "evidence": "Apple tiếp tục ra mắt iPhone 15 vào tháng 9 năm 2023 với doanh số kỷ lục.",
        "expected": "REFUTES",
        "note": "ngừng sản xuất vs tiếp tục ra mắt"
    },
    {
        "claim": "Việt Nam không tham gia ASEAN.",
        "evidence": "Việt Nam là thành viên chính thức của ASEAN từ năm 1995.",
        "expected": "REFUTES",
        "note": "không tham gia vs thành viên chính thức"
    },
    {
        "claim": "Hà Nội là thành phố lớn nhất Việt Nam về diện tích.",
        "evidence": "Thành phố Hồ Chí Minh có diện tích 2.095 km², trong khi Hà Nội có diện tích 3.358,6 km².",
        "expected": "SUPPORTS",
        "note": "Hà Nội lớn hơn về diện tích"
    },
    # SUPPORTS cases
    {
        "claim": "Sông Mekong chảy qua Việt Nam.",
        "evidence": "Sông Mekong bắt nguồn từ Tây Tạng, chảy qua Trung Quốc, Myanmar, Lào, Thái Lan, Campuchia và đổ ra biển tại Việt Nam.",
        "expected": "SUPPORTS",
        "note": "clear support"
    },
    {
        "claim": "Phở là món ăn truyền thống của Việt Nam.",
        "evidence": "Phở là một trong những món ăn đặc trưng nhất của ẩm thực Việt Nam, được UNESCO công nhận là di sản văn hóa phi vật thể.",
        "expected": "SUPPORTS",
        "note": "clear support"
    },
    # NEI cases
    {
        "claim": "Năm 2025 sẽ có bão lớn đổ bộ vào Việt Nam.",
        "evidence": "Theo thống kê, trung bình mỗi năm có 5-6 cơn bão ảnh hưởng đến Việt Nam.",
        "expected": "NEI",
        "note": "prediction vs historical data"
    },
    {
        "claim": "Giá vàng sẽ tăng trong năm tới.",
        "evidence": "Giá vàng năm nay biến động mạnh với nhiều phiên tăng giảm bất thường.",
        "expected": "NEI",
        "note": "future prediction"
    },
    {
        "claim": "Đội tuyển Việt Nam vô địch AFF Cup 2024.",
        "evidence": "AFF Cup 2024 sẽ diễn ra vào cuối năm với sự tham gia của các đội tuyển Đông Nam Á.",
        "expected": "NEI",
        "note": "result not mentioned"
    },
    # Edge cases - numbers
    {
        "claim": "GDP Việt Nam năm 2023 là 500 tỷ USD.",
        "evidence": "GDP Việt Nam năm 2023 đạt khoảng 430 tỷ USD.",
        "expected": "REFUTES",
        "note": "number mismatch: 500 vs 430"
    },
    {
        "claim": "Việt Nam có 63 tỉnh thành.",
        "evidence": "Việt Nam có 63 đơn vị hành chính cấp tỉnh, bao gồm 5 thành phố trực thuộc trung ương và 58 tỉnh.",
        "expected": "SUPPORTS",
        "note": "number match"
    },
    # Edge cases - negation
    {
        "claim": "Việt Nam không có biên giới với Trung Quốc.",
        "evidence": "Việt Nam có đường biên giới dài 1.281 km với Trung Quốc ở phía Bắc.",
        "expected": "REFUTES",
        "note": "negation contradiction"
    },
    # Edge cases - time
    {
        "claim": "World Cup 2022 diễn ra tại Qatar vào mùa hè.",
        "evidence": "World Cup 2022 được tổ chức tại Qatar từ 21/11 đến 18/12/2022, lần đầu tiên diễn ra vào mùa đông.",
        "expected": "REFUTES",
        "note": "mùa hè vs mùa đông"
    },
    # Semantic similarity
    {
        "claim": "Bác Hồ sinh năm 1890.",
        "evidence": "Chủ tịch Hồ Chí Minh sinh ngày 19 tháng 5 năm 1890 tại làng Kim Liên, huyện Nam Đàn, tỉnh Nghệ An.",
        "expected": "SUPPORTS",
        "note": "Bác Hồ = Chủ tịch Hồ Chí Minh"
    },
]


def load_vifactcheck_samples(n_samples: int = 300) -> list:
    """Load random samples from ViFactCheck dataset."""
    print(f"📥 Loading {n_samples} samples from ViFactCheck...")
    
    try:
        dataset = load_dataset("tranthaihoa/vifactcheck", split="test")
        
        # Random sample
        indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
        
        # Label mapping: 0=SUPPORTS, 1=REFUTES, 2=NEI
        label_map = {0: "SUPPORTS", 1: "REFUTES", 2: "NEI"}
        
        samples = []
        for idx in indices:
            item = dataset[idx]
            label_id = item["labels"]
            samples.append({
                "claim": item["Statement"],
                "evidence": item["Evidence"],
                "expected": label_map.get(label_id, "UNKNOWN"),
                "source": "vifactcheck",
                "idx": idx
            })
        
        print(f"   ✅ Loaded {len(samples)} samples")
        return samples
        
    except Exception as e:
        print(f"   ❌ Error loading dataset: {e}")
        return []


def analyze_xai_result(claim: str, evidence: str, xai_result: dict, expected: str) -> dict:
    """Analyze a single XAI result for errors."""
    errors = []
    warnings = []
    
    predicted = xai_result.get("relationship", "UNKNOWN")
    natural_explanation = xai_result.get("natural_explanation", "")
    claim_conflict = xai_result.get("claim_conflict_word", "")
    evidence_conflict = xai_result.get("evidence_conflict_word", "")
    
    # 1. Check prediction correctness
    prediction_correct = predicted == expected
    if not prediction_correct:
        errors.append(f"WRONG_PREDICTION: expected {expected}, got {predicted}")
    
    # 2. Check for underscore in display text (PyVi artifacts)
    if "_" in natural_explanation:
        errors.append(f"UNDERSCORE_IN_EXPLANATION: '{natural_explanation}'")
    
    # 4. For REFUTES - check conflict detection
    if predicted == "REFUTES":
        if not claim_conflict or not evidence_conflict:
            warnings.append("REFUTES_NO_CONFLICT_DETECTED")
        else:
            # Check if conflicts are meaningful
            if claim_conflict == evidence_conflict:
                errors.append(f"SAME_CONFLICT_WORDS: '{claim_conflict}'")
    
    # 5. Check natural explanation quality
    if not natural_explanation or len(natural_explanation) < 10:
        errors.append("EMPTY_OR_SHORT_EXPLANATION")
    
    # 7. Check for irrelevant conflict detection
    if predicted == "REFUTES" and claim_conflict and evidence_conflict:
        # Check if conflict words make semantic sense
        irrelevant_pairs = [
            # Words that shouldn't be marked as conflicts
            ("việt_nam", "việt_nam"),
            ("của", "của"),
            ("là", "là"),
        ]
        if (claim_conflict.lower(), evidence_conflict.lower()) in irrelevant_pairs:
            errors.append(f"IRRELEVANT_CONFLICT: '{claim_conflict}' vs '{evidence_conflict}'")
    
    return {
        "prediction_correct": prediction_correct,
        "predicted": predicted,
        "expected": expected,
        "errors": errors,
        "warnings": warnings,
        "claim_conflict": claim_conflict,
        "evidence_conflict": evidence_conflict,
        "natural_explanation": natural_explanation
    }


def run_analysis():
    """Main analysis function."""
    print("=" * 60)
    print("🔍 XAI Error Analysis Script")
    print("=" * 60)
    
    # Load XAI model
    print("\n📦 Loading PhoBERT XAI model...")
    model_path = project_root / "results/fact_checking/pyvi/checkpoints/best_model_pyvi.pt"
    
    try:
        xai = load_xai_model(str(model_path), device="cpu")
        print("   ✅ Model loaded")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Collect samples
    all_samples = []
    
    # 1. Dataset samples
    dataset_samples = load_vifactcheck_samples(300)
    all_samples.extend(dataset_samples)
    
    # 2. Custom samples
    print(f"\n📝 Adding {len(CUSTOM_SAMPLES)} custom samples...")
    for sample in CUSTOM_SAMPLES:
        sample["source"] = "custom"
    all_samples.extend(CUSTOM_SAMPLES)
    
    print(f"\n📊 Total samples: {len(all_samples)}")
    
    # Run analysis
    print("\n🔄 Analyzing XAI outputs...")
    results = []
    error_counts = defaultdict(int)
    warning_counts = defaultdict(int)
    
    for i, sample in enumerate(all_samples):
        if (i + 1) % 50 == 0:
            print(f"   Processing {i + 1}/{len(all_samples)}...")
        
        try:
            xai_result = xai.generate_xai(
                claim=sample["claim"],
                evidence=sample["evidence"]
            )
            
            analysis = analyze_xai_result(
                claim=sample["claim"],
                evidence=sample["evidence"],
                xai_result=xai_result,
                expected=sample["expected"]
            )
            
            analysis["sample"] = {
                "claim": sample["claim"][:100] + "..." if len(sample["claim"]) > 100 else sample["claim"],
                "evidence": sample["evidence"][:100] + "..." if len(sample["evidence"]) > 100 else sample["evidence"],
                "source": sample.get("source", "unknown"),
                "note": sample.get("note", "")
            }
            
            results.append(analysis)
            
            # Count errors and warnings
            for error in analysis["errors"]:
                error_type = error.split(":")[0]
                error_counts[error_type] += 1
            
            for warning in analysis["warnings"]:
                warning_type = warning.split(":")[0]
                warning_counts[warning_type] += 1
                
        except Exception as e:
            print(f"   ❌ Error on sample {i}: {e}")
            results.append({
                "sample": sample,
                "error": str(e),
                "errors": ["PROCESSING_ERROR"],
                "warnings": []
            })
            error_counts["PROCESSING_ERROR"] += 1
    
    # Summary statistics
    total = len(results)
    correct = sum(1 for r in results if r.get("prediction_correct", False))
    with_errors = sum(1 for r in results if len(r.get("errors", [])) > 0)
    with_warnings = sum(1 for r in results if len(r.get("warnings", [])) > 0)
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_samples": total,
        "dataset_samples": len(dataset_samples),
        "custom_samples": len(CUSTOM_SAMPLES),
        "prediction_accuracy": f"{correct / total * 100:.2f}%",
        "samples_with_errors": with_errors,
        "samples_with_warnings": with_warnings,
        "error_counts": dict(error_counts),
        "warning_counts": dict(warning_counts),
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("📈 ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total samples:        {total}")
    print(f"Prediction accuracy:  {summary['prediction_accuracy']}")
    print(f"Samples with errors:  {with_errors} ({with_errors/total*100:.1f}%)")
    print(f"Samples with warnings: {with_warnings} ({with_warnings/total*100:.1f}%)")
    
    print("\n🚨 ERROR BREAKDOWN:")
    for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
        print(f"   {error_type}: {count} ({count/total*100:.1f}%)")
    
    print("\n⚠️ WARNING BREAKDOWN:")
    for warning_type, count in sorted(warning_counts.items(), key=lambda x: -x[1]):
        print(f"   {warning_type}: {count} ({count/total*100:.1f}%)")
    
    # Save results
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "xai_error_analysis.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "detailed_results": results[:50],  # Save first 50 for inspection
            "error_samples": [r for r in results if len(r.get("errors", [])) > 0][:30]
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Print sample errors for inspection
    print("\n" + "=" * 60)
    print("📋 SAMPLE ERRORS (first 5)")
    print("=" * 60)
    
    error_samples = [r for r in results if len(r.get("errors", [])) > 0][:5]
    for i, sample in enumerate(error_samples, 1):
        print(f"\n--- Sample {i} ---")
        print(f"Claim: {sample['sample']['claim']}")
        print(f"Expected: {sample.get('expected', 'N/A')} | Got: {sample.get('predicted', 'N/A')}")
        print(f"Errors: {sample['errors']}")
        if sample.get('claim_conflict') or sample.get('evidence_conflict'):
            print(f"Conflicts: '{sample.get('claim_conflict', '')}' vs '{sample.get('evidence_conflict', '')}'")
        print(f"Explanation: {sample.get('natural_explanation', 'N/A')[:100]}")


if __name__ == "__main__":
    random.seed(42)  # Reproducibility
    run_analysis()
