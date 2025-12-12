"""
Generate XAI Report - Narrative-style Debate Explanation
=========================================================
Trích xuất và format lại kết quả debate thành báo cáo XAI dễ đọc,
tập trung vào NỘI DUNG hội thoại thay vì số liệu kỹ thuật.

Usage:
    python -m src.pipeline.run_pineline.generate_xai_report \
        --results-file results/vifactcheck/test/full_debate/vifactcheck_test_results.json \
        --output-dir results/vifactcheck/test/full_debate/xai_report \
        --num-examples 10
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


# === VERDICT MAPPING ===
VERDICT_EMOJI = {
    "SUPPORTED": "✅",
    "REFUTED": "❌", 
    "NEI": "⚠️",
    "NOT_ENOUGH_INFO": "⚠️",
    "Support": "✅",
    "Refute": "❌",
}

VERDICT_VI = {
    "SUPPORTED": "ĐÚNG",
    "REFUTED": "SAI",
    "NEI": "THIẾU THÔNG TIN",
    "NOT_ENOUGH_INFO": "THIẾU THÔNG TIN",
    "Support": "ĐÚNG",
    "Refute": "SAI",
}

AGENT_EMOJI = {
    "x-ai/grok-4-fast": "🎯",
    "google/gemini-2.5-flash": "💎",
    "openai/gpt-4o-mini": "🤖",
    "grok": "🎯",
    "gemini": "💎",
    "gpt": "🤖",
}

AGENT_NAME = {
    "x-ai/grok-4-fast": "Grok",
    "google/gemini-2.5-flash": "Gemini", 
    "openai/gpt-4o-mini": "GPT",
}


def get_verdict_emoji(verdict: str) -> str:
    """Get emoji for verdict."""
    v = verdict.upper() if verdict else ""
    if "SUPPORT" in v:
        return "✅"
    elif "REFUT" in v:
        return "❌"
    else:
        return "⚠️"


def get_verdict_vi(verdict: str) -> str:
    """Get Vietnamese translation for verdict."""
    v = verdict.upper() if verdict else ""
    if "SUPPORT" in v:
        return "ĐÚNG"
    elif "REFUT" in v:
        return "SAI"
    else:
        return "THIẾU THÔNG TIN"


def format_single_case(sample: Dict[str, Any], case_num: int) -> str:
    """Format a single case into narrative style."""
    lines = []
    
    # Header
    lines.append(f"### 📋 Case {case_num}")
    lines.append("")
    
    # Claim (truncate if too long)
    statement = sample.get("statement", "N/A")
    if len(statement) > 200:
        statement = statement[:200] + "..."
    lines.append(f"**Tuyên bố:** \"{statement}\"")
    lines.append("")
    
    # Evidence (truncate if too long)
    evidence = sample.get("evidence", "N/A")
    if len(evidence) > 300:
        evidence = evidence[:300] + "..."
    lines.append(f"**Bằng chứng:** \"{evidence}\"")
    lines.append("")
    
    # Gold label
    gold_label = sample.get("gold_label", "N/A")
    lines.append(f"**Nhãn thực tế:** {get_verdict_emoji(gold_label)} {get_verdict_vi(gold_label)}")
    lines.append("")
    
    # Model prediction
    model_verdict = sample.get("model_verdict", "N/A")
    model_correct = sample.get("model_correct", False)
    model_icon = "✅" if model_correct else "❌"
    lines.append(f"**PhoBERT dự đoán:** {get_verdict_emoji(model_verdict)} {get_verdict_vi(model_verdict)} {model_icon}")
    lines.append("")
    
    # Final verdict
    final_verdict = sample.get("final_verdict", "N/A")
    final_correct = sample.get("final_correct", False)
    final_icon = "✅" if final_correct else "❌"
    lines.append(f"**Kết luận cuối:** {get_verdict_emoji(final_verdict)} **{get_verdict_vi(final_verdict)}** {final_icon}")
    lines.append("")
    
    # Debate transcript
    debate_result = sample.get("debate_result", {})
    if debate_result:
        lines.append("---")
        lines.append("#### 🗣️ Hội đồng tranh luận")
        lines.append("")
        
        # Round 1
        r1_verdicts = debate_result.get("round_1_verdicts", {})
        if r1_verdicts:
            lines.append("**Vòng 1 - Phân tích độc lập:**")
            lines.append("")
            for agent_id, data in r1_verdicts.items():
                agent_name = AGENT_NAME.get(agent_id, agent_id.split("/")[-1])
                emoji = AGENT_EMOJI.get(agent_id, "🔹")
                verdict = data.get("verdict", "N/A")
                reasoning = data.get("reasoning", "N/A")
                lines.append(f"> {emoji} **{agent_name}** ({get_verdict_vi(verdict)}): \"{reasoning}\"")
                lines.append(">")
            lines.append("")
        
        # Round 2
        all_rounds = debate_result.get("all_rounds_verdicts", [])
        if len(all_rounds) >= 2:
            r2_data = all_rounds[1]  # Round 2 is index 1
            lines.append("**Vòng 2 - Tranh luận & Chốt kèo:**")
            lines.append("")
            for agent_id, data in r2_data.items():
                agent_name = AGENT_NAME.get(agent_id, agent_id.split("/")[-1])
                emoji = AGENT_EMOJI.get(agent_id, "🔹")
                verdict = data.get("verdict", "N/A")
                reasoning = data.get("reasoning", "N/A")
                # Check if changed
                r1_verdict = r1_verdicts.get(agent_id, {}).get("verdict", "")
                changed = r1_verdict != verdict
                change_note = " *(đổi ý)*" if changed else ""
                lines.append(f"> {emoji} **{agent_name}** ({get_verdict_vi(verdict)}{change_note}): \"{reasoning}\"")
                lines.append(">")
            lines.append("")
        
        # Judge conclusion
        judge_reasoning = debate_result.get("reasoning", "")
        if judge_reasoning:
            lines.append("**⚖️ Judge kết luận:**")
            lines.append(f"> \"{judge_reasoning}\"")
            lines.append("")
    
    lines.append("---")
    lines.append("")
    
    return "\n".join(lines)


def categorize_samples(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Categorize samples into Fixed, Broke, and Interesting cases."""
    categories = {
        "fixed": [],      # Model sai -> Debate đúng
        "broke": [],      # Model đúng -> Debate sai
        "consensus": [],  # Cả 3 agents đồng thuận từ đầu
        "dramatic": [],   # Có agent đổi ý
    }
    
    for sample in results:
        model_correct = sample.get("model_correct", False)
        final_correct = sample.get("final_correct", False)
        
        # Fixed: Model wrong, Debate correct
        if not model_correct and final_correct:
            categories["fixed"].append(sample)
        # Broke: Model correct, Debate wrong
        elif model_correct and not final_correct:
            categories["broke"].append(sample)
        
        # Check for consensus/dramatic
        debate_result = sample.get("debate_result", {})
        all_rounds = debate_result.get("all_rounds_verdicts", [])
        if len(all_rounds) >= 2:
            r1 = all_rounds[0]
            r2 = all_rounds[1]
            
            # Check if any agent changed verdict
            changed = False
            for agent_id in r1:
                if agent_id in r2:
                    if r1[agent_id].get("verdict") != r2[agent_id].get("verdict"):
                        changed = True
                        break
            
            if changed:
                categories["dramatic"].append(sample)
            else:
                # Check if all same in R1
                r1_verdicts = [v.get("verdict") for v in r1.values()]
                if len(set(r1_verdicts)) == 1:
                    categories["consensus"].append(sample)
    
    return categories


def generate_report(results_file: str, output_dir: str, num_examples: int = 10):
    """Generate the XAI narrative report."""
    
    # Load results
    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = data.get("results", [])
    total = len(results)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Categorize samples
    categories = categorize_samples(results)
    
    # Build report
    report_lines = []
    
    # Header
    report_lines.append("# 📊 BÁO CÁO XAI - GIẢI THÍCH KẾT QUẢ DEBATE")
    report_lines.append("")
    report_lines.append(f"**Ngày tạo:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report_lines.append(f"**Tổng số mẫu:** {total}")
    report_lines.append(f"**Độ chính xác Model:** {data.get('model_accuracy', 0)*100:.2f}%")
    report_lines.append(f"**Độ chính xác Debate:** {data.get('final_accuracy', 0)*100:.2f}%")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Summary
    report_lines.append("## 📈 Tổng quan")
    report_lines.append("")
    report_lines.append(f"- **Debate sửa đúng (Fixed):** {len(categories['fixed'])} cases")
    report_lines.append(f"- **Debate làm sai (Broke):** {len(categories['broke'])} cases")
    report_lines.append(f"- **Đồng thuận ngay (Consensus):** {len(categories['consensus'])} cases")
    report_lines.append(f"- **Có tranh luận gay gắt (Dramatic):** {len(categories['dramatic'])} cases")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Section 1: Fixed cases
    report_lines.append("## ✅ DEBATE SỬA ĐÚNG (Model sai → Debate đúng)")
    report_lines.append("")
    report_lines.append("*Những trường hợp PhoBERT dự đoán sai, nhưng sau khi tranh luận, hệ thống đã sửa lại đúng.*")
    report_lines.append("")
    
    fixed_samples = categories["fixed"][:num_examples]
    for i, sample in enumerate(fixed_samples, 1):
        report_lines.append(format_single_case(sample, i))
    
    if not fixed_samples:
        report_lines.append("*Không có case nào trong danh mục này.*")
        report_lines.append("")
    
    # Section 2: Broke cases
    report_lines.append("## ❌ DEBATE LÀM SAI (Model đúng → Debate sai)")
    report_lines.append("")
    report_lines.append("*Những trường hợp PhoBERT dự đoán đúng, nhưng sau tranh luận lại bị đổi thành sai (over-reasoning).*")
    report_lines.append("")
    
    broke_samples = categories["broke"][:num_examples]
    for i, sample in enumerate(broke_samples, 1):
        report_lines.append(format_single_case(sample, i))
    
    if not broke_samples:
        report_lines.append("*Không có case nào trong danh mục này.*")
        report_lines.append("")
    
    # Section 3: Dramatic cases
    report_lines.append("## 🔥 TRANH LUẬN GAY GẮT (Có agent đổi ý)")
    report_lines.append("")
    report_lines.append("*Những trường hợp có ít nhất 1 agent thay đổi quan điểm sau vòng tranh luận.*")
    report_lines.append("")
    
    dramatic_samples = categories["dramatic"][:num_examples]
    for i, sample in enumerate(dramatic_samples, 1):
        report_lines.append(format_single_case(sample, i))
    
    if not dramatic_samples:
        report_lines.append("*Không có case nào trong danh mục này.*")
        report_lines.append("")
    
    # Write report
    report_content = "\n".join(report_lines)
    report_file = output_path / "xai_narrative_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"✅ Đã tạo báo cáo XAI: {report_file}")
    print(f"   - Fixed cases: {len(categories['fixed'])}")
    print(f"   - Broke cases: {len(categories['broke'])}")
    print(f"   - Dramatic cases: {len(categories['dramatic'])}")
    
    return str(report_file)


def main():
    parser = argparse.ArgumentParser(description="Generate XAI Narrative Report")
    parser.add_argument(
        "--results-file",
        type=str,
        default="results/vifactcheck/test/full_debate/vifactcheck_test_results.json",
        help="Path to results JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/vifactcheck/test/full_debate/xai_report",
        help="Output directory for report"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Number of examples per category"
    )
    
    args = parser.parse_args()
    generate_report(args.results_file, args.output_dir, args.num_examples)


if __name__ == "__main__":
    main()
