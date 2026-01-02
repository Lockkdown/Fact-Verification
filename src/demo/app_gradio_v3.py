"""
Gradio Demo UI for Vietnamese Fact Checking (v3) - Professional Clean Interface

Features for Business Demo:
- Clean white theme (professional for corporate judges)
- Large, clear typography (easy to read)
- Simple 2-input layout (claim + evidence)
- Clear progress flow visualization
- Beautiful debate visualization with icons
- No complex confidence scores
- Majority vote system integration

Author: Lockdown
Date: Jan 2, 2026
"""

import gradio as gr
import asyncio
import logging
import os
import socket
import textwrap
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

# Suppress verbose logs
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.ERROR, format='%(message)s')
for logger_name in ["transformers", "torch", "httpx", "asyncio", "aiohttp"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

def _normalize_display_text(value: Any) -> str:
    if value is None:
        return ""
    s = str(value)
    s = textwrap.dedent(s)
    s = s.strip("\n ")
    return s

def _pick_free_port(preferred: int = 7860) -> int:
    """Pick a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Global pipeline
PIPELINE = None
INIT_STATUS = "Not initialized"

def init_pipeline():
    """Initialize pipeline at startup."""
    global PIPELINE, INIT_STATUS
    
    print("=" * 50)
    print("🔄 Initializing ViFactCheck Pipeline...")
    print("=" * 50)
    
    try:
        from src.pipeline.run_pineline.article_pipeline import ViFactCheckPipeline, PipelineConfig
        
        config = PipelineConfig(
            use_debate=True,
            use_async_debate=True,
            hybrid_enabled=True
        )
        PIPELINE = ViFactCheckPipeline(config)
        
        INIT_STATUS = "✅ Ready"
        print("✅ Pipeline ready!")
        return True
        
    except Exception as e:
        INIT_STATUS = f"❌ Error: {str(e)}"
        print(f"❌ Init failed: {e}")
        return False

# Sample data for demo
SAMPLES = [
    {
        "claim": "Việt Nam là quốc gia thứ hai đón ngọn đuốc Olympic Paris 2024.",
        "evidence": "Việt Nam là quốc gia đầu tiên đón ngọn đuốc Olympic Paris 2024 trong hành trình rước đuốc vòng quanh thế giới."
    },
    {
        "claim": "Tỷ lệ thất nghiệp của Việt Nam năm 2023 là 2.28%.",
        "evidence": "Theo Tổng cục Thống kê, tỷ lệ thất nghiệp của Việt Nam năm 2023 là 2.28%, giảm so với năm 2022."
    },
    {
        "claim": "Chính phủ Việt Nam đã ban hành luật về trí tuệ nhân tạo vào năm 2024.",
        "evidence": "Việt Nam đang xây dựng chiến lược quốc gia về trí tuệ nhân tạo đến năm 2030, với mục tiêu trở thành trung tâm đổi mới sáng tạo."
    }
]

def load_sample(idx: int) -> Tuple[str, str]:
    """Load sample by index."""
    if 0 <= idx < len(SAMPLES):
        s = SAMPLES[idx]
        return s["claim"], s["evidence"]
    return "", ""

def _compute_judge_from_rounds(all_rounds: List[Dict[str, Any]]) -> Dict[str, Any]:
    round_evaluations: List[Dict[str, Any]] = []
    previous_majority: Optional[str] = None

    for round_num, round_data in enumerate(all_rounds, 1):
        verdicts = [agent_data.get("verdict", "NEI") for agent_data in round_data.values()]
        counts: Dict[str, int] = {}
        for v in verdicts:
            counts[v] = counts.get(v, 0) + 1

        unique = len(set(verdicts))
        if unique == 1:
            consensus_status = "UNANIMOUS"
        elif len(counts) == 2:
            consensus_status = "MAJORITY"
        else:
            consensus_status = "SPLIT"

        majority_verdict = max(counts.items(), key=lambda x: x[1])[0] if counts else "NEI"
        stability_check = previous_majority is not None and majority_verdict == previous_majority

        # Early-stop rule (align with JudgeV2): unanimous + stable + round>=2, or last round.
        early_stop = (consensus_status == "UNANIMOUS" and stability_check and round_num >= 2) or (round_num == len(all_rounds))

        if consensus_status == "UNANIMOUS":
            judge_reasoning = "Các chuyên gia đồng thuận rõ ràng về kết luận ở vòng này."
        elif consensus_status == "MAJORITY":
            judge_reasoning = "Đa số nghiêng về một kết luận, nhưng vẫn còn ý kiến khác cần cân nhắc."
        else:
            judge_reasoning = "Ý kiến chia rẽ, cần thêm phản biện hoặc chốt NEI để an toàn."

        round_evaluations.append(
            {
                "round_num": round_num,
                "consensus_status": consensus_status,
                "majority_verdict": majority_verdict,
                "verdict_distribution": counts,
                "judge_reasoning": judge_reasoning,
                "early_stop_decision": early_stop,
                "stability_check": stability_check,
            }
        )
        previous_majority = majority_verdict

    # Final verdict from last round votes (align with majority-vote rule: 1-1-1 => NEI)
    final_counts = round_evaluations[-1]["verdict_distribution"] if round_evaluations else {}
    if final_counts:
        total_votes = sum(final_counts.values())
        majority_verdict, majority_count = max(final_counts.items(), key=lambda x: x[1])
        if majority_count == 1 and len(final_counts) == total_votes:
            final_verdict = "NEI"
        else:
            final_verdict = majority_verdict
    else:
        final_verdict = "NEI"

    if len(set(final_counts.keys())) == 1 and final_counts:
        final_reasoning = "Kết luận cuối: cả 3 chuyên gia đồng thuận, có thể chốt kết quả."
    elif final_verdict == "NEI":
        final_reasoning = "Kết luận cuối: ý kiến không thống nhất, nên trả về ‘Chưa đủ thông tin’."
    else:
        final_reasoning = "Kết luận cuối: đa số chuyên gia nghiêng về một nhãn, chốt theo đa số."

    return {
        "round_evaluations": round_evaluations,
        "final_verdict": final_verdict,
        "final_reasoning": final_reasoning,
    }

def format_debate_simple(result: Dict[str, Any]) -> str:
    """Simple, clean debate visualization for business presentation."""
    all_rounds = result.get("debate_all_rounds_verdicts", [])
    judge_round_evaluations = result.get("judge_round_evaluations", [])
    judge_final_reasoning = _normalize_display_text(result.get("judge_final_reasoning", ""))
    
    if not all_rounds:
        return """
        <div style="text-align: center; padding: 40px; background: #f8f9fa; border-radius: 12px; border: 2px dashed #dee2e6;">
            <div style="font-size: 18px; color: #6c757d; margin-bottom: 10px;">🚀 Phân tích nhanh</div>
            <div style="font-size: 16px; color: #868e96;">AI đã đưa ra kết luận dựa trên độ tin cậy cao</div>
        </div>
        """
    
    html = '<div style="background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">'
    
    # Header
    html += """
    <div style="background: rgba(15, 23, 42, 0.04); color: #111827; padding: 12px 20px; text-align: center; border-bottom: 1px solid #e5e7eb;">
        <h3 style="margin: 0; font-size: 22px; font-weight: 700; color: #111827;">🗣️ Quá trình phân tích</h3>
    </div>
    """
    
    for round_num, round_data in enumerate(all_rounds, 1):
        # Round header
        round_bg = "#f8f9fa" if round_num == 1 else "#fff"
        html += f"""
        <div style="background: {round_bg}; border-bottom: 1px solid #dee2e6; padding: 16px;">
            <div style="display: flex; align-items: center; margin-bottom: 12px;">
                <div style="background: #e2e8f0; color: #0f172a; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; margin-right: 12px;">{round_num}</div>
                <h4 style="margin: 0; color: #202124; font-size: 18px;">Vòng {round_num}</h4>
            </div>
        """

        
        # Agents in this round
        html += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px;">'
        
        for agent_name, agent_data in round_data.items():
            verdict = agent_data.get("verdict", "NEI")
            reasoning = _normalize_display_text(agent_data.get("reasoning", ""))
            
            # Clean agent name
            clean_name = agent_name.split('/')[-1].replace('-', ' ').title()
            
            # Verdict styling
            if verdict == "SUPPORTED":
                verdict_text = "HỖ TRỢ"
                verdict_color = "#34a853"
                verdict_icon = "✅"
            elif verdict == "REFUTED": 
                verdict_text = "BÁC BỎ"
                verdict_color = "#ea4335"
                verdict_icon = "❌"
            else:
                verdict_text = "CHƯA ĐỦ INFO"
                verdict_color = "#fbbc04"
                verdict_icon = "❓"
            
            html += f"""
            <div style="background: white; border: 2px solid {verdict_color}; border-radius: 8px; padding: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div style="font-weight: 600; color: #202124; font-size: 16px;">{clean_name}</div>
                    <div style="background: {verdict_color}; color: white; padding: 6px 12px; border-radius: 20px; font-size: 14px; font-weight: 600;">
                        {verdict_icon} {verdict_text}
                    </div>
                </div>
                <div style="color: #374151; font-size: 15px; line-height: 1.6; white-space: pre-wrap;">{reasoning}</div>
            </div>
            """
        
        html += '</div>'
        
        # Judge note AFTER the round (moved below debators)
        if isinstance(judge_round_evaluations, list) and len(judge_round_evaluations) >= round_num:
            ev = judge_round_evaluations[round_num - 1] or {}
            judge_reasoning = _normalize_display_text(ev.get("judge_reasoning", ""))
            if judge_reasoning:
                html += f"""
                <div style="margin: 16px 0 0 0; padding: 14px 16px; background: linear-gradient(135deg, #fef3c7, #fde68a); border: 2px solid #f59e0b; border-radius: 12px; box-shadow: 0 2px 8px rgba(245, 158, 11, 0.2);">
                    <div style="font-weight: 800; color: #92400e; margin-bottom: 8px; font-size: 16px;">👨‍⚖️ Thẩm phán DeepSeek</div>
                    <div style="color: #78350f; font-size: 15px; line-height: 1.6; font-weight: 500; white-space: pre-wrap;">{judge_reasoning}</div>
                </div>
                """
        html += '</div>'
    
    # Final result summary
    metrics = result.get("debate_metrics", {})
    if metrics:
        final_round = all_rounds[-1] if all_rounds else {}
        votes = {}
        for agent_data in final_round.values():
            verdict = agent_data.get("verdict", "NEI")
            votes[verdict] = votes.get(verdict, 0) + 1
        
        rounds_used = metrics.get("rounds_used", "?")
        
        html += f"""
        <div style="background: rgba(15, 23, 42, 0.04); padding: 20px; text-align: center; border-top: 1px solid #e5e7eb;">
            <h4 style="color: #111827; margin: 0 0 12px 0; font-size: 20px; font-weight: 800;">⚖️ Kết luận cuối ({rounds_used} vòng) - DeepSeek</h4>
            {f'<div style="max-width: 900px; margin: 0 auto 18px auto; color: #334155; font-size: 14px; line-height: 1.6; white-space: pre-wrap;">{judge_final_reasoning}</div>' if judge_final_reasoning else ''}
            <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
        """
        
        for verdict, count in votes.items():
            if verdict == "SUPPORTED":
                vote_text = "HỖ TRỢ"
                vote_color = "#34a853"
                vote_icon = "✅"
            elif verdict == "REFUTED":
                vote_text = "BÁC BỎ" 
                vote_color = "#ea4335"
                vote_icon = "❌"
            else:
                vote_text = "CHƯA ĐỦ"
                vote_color = "#fbbc04"
                vote_icon = "❓"
            
            html += f"""
            <div style="background: white; border: 2px solid {vote_color}; border-radius: 8px; padding: 12px 20px; text-align: center; min-width: 100px;">
                <div style="font-size: 24px; margin-bottom: 4px;">{vote_icon}</div>
                <div style="color: {vote_color}; font-weight: 600; font-size: 14px;">{vote_text}</div>
                <div style="color: #202124; font-weight: bold; font-size: 18px;">{count} phiếu</div>
            </div>
            """
        
        html += '</div></div>'
    
    html += '</div>'
    return html

async def verify_claim_v3(claim: str, evidence: str):
    """Main verification with clean progress flow."""
    if not claim.strip() or not evidence.strip():
        warn = """
        <div style="padding: 14px 16px; background: #fef2f2; border: 1px solid #fecaca; border-left: 6px solid #ef4444; border-radius: 10px; color: #991b1b; font-size: 15px; font-weight: 700; line-height: 1.4;">
            ❌ Vui lòng nhập đầy đủ <span style="font-weight: 900; color: #7f1d1d;">Tuyên bố</span> và <span style="font-weight: 900; color: #7f1d1d;">Bằng chứng</span> trước khi kiểm chứng.
        </div>
        """
        yield "", warn, "", ""
        return

    if PIPELINE is None:
        yield "", f"❌ Hệ thống chưa sẵn sàng: {INIT_STATUS}", "", ""
        return

    try:
        # Step 1: Analyzing with PhoBERT
        yield ("", 
               create_progress_step("🤖 PhoBERT đang phân tích...", 1, "active"),
               "", "")
        
        model_verdict, confidence, probs = PIPELINE._predict_verdict(claim, evidence)
        
        # Step 2: Routing decision
        yield ("",
               create_progress_step("⚖️ Quyết định luồng xử lý...", 2, "active"), 
               "", "")
        
        skipped = False
        hybrid_info = {"skipped": False}
        
        if (PIPELINE.debate_orchestrator and 
            getattr(PIPELINE.config, "hybrid_enabled", True) and 
            PIPELINE.debate_orchestrator.hybrid_enabled):
            
            hybrid_decision = PIPELINE.debate_orchestrator.hybrid_strategy.decide(
                model_verdict=model_verdict,
                model_probs=probs,
                debate_verdict="NEI",
                debate_confidence=0.0,
            )
            
            if hybrid_decision.decision_source == "MODEL_HIGH_CONF":
                skipped = True
                hybrid_info = {"skipped": True, "source": "MODEL_HIGH_CONF"}
            else:
                hybrid_info = {"skipped": False, "source": "DEBATE"}
        else:
            skipped = True
            hybrid_info = {"skipped": True, "source": "MODEL_ONLY"}

        if skipped:
            # Fast path - PhoBERT only
            yield ("",
                   create_progress_complete("Hoàn thành phân tích nhanh", "PhoBERT"),
                   "", "")
            
            final_verdict = model_verdict
            result = {
                "final_verdict": final_verdict,
                "debate_all_rounds_verdicts": None,
                "debate_metrics": None
            }
        else:
            # Slow path - Multi-agent debate  
            configured_max_rounds = 7
            try:
                configured_max_rounds = int(
                    getattr(getattr(PIPELINE, "debate_orchestrator", None), "debate_config", {})
                    .get("max_rounds", 7)
                    or 7
                )
            except Exception:
                configured_max_rounds = 7

            yield ("",
                   create_progress_step(f"🗣️ Debate vòng 1/{configured_max_rounds} - Chuyên gia phân tích...", 3, "active"),
                   "", "")
            
            from src.pipeline.debate.debator import Evidence as DebateEvidence
            
            debate_evidence = DebateEvidence(
                text=evidence,
                source="vifactcheck", 
                rank=1,
                nli_score={"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0, "other": 0.0},
                relevance_score=1.0,
            )

            orchestrator = PIPELINE.debate_orchestrator
            prev_max_rounds = None
            try:
                if getattr(orchestrator, "debate_config", None) is not None:
                    prev_max_rounds = orchestrator.debate_config.get("max_rounds")
                    orchestrator.debate_config["max_rounds"] = configured_max_rounds

                progress_queue: asyncio.Queue = asyncio.Queue()
                loop = asyncio.get_running_loop()

                def progress_cb(event: str, payload: Dict[str, Any]):
                    try:
                        loop.call_soon_threadsafe(progress_queue.put_nowait, (event, payload))
                    except Exception:
                        pass

                debate_task = asyncio.create_task(
                    orchestrator.debate_async(
                        claim=claim,
                        evidences=[debate_evidence],
                        model_verdict=model_verdict,
                        model_confidence=float(confidence),
                        progress_cb=progress_cb,
                    )
                )

                current_max = configured_max_rounds
                while True:
                    if debate_task.done() and progress_queue.empty():
                        break

                    try:
                        event, payload = await asyncio.wait_for(progress_queue.get(), timeout=0.25)
                    except asyncio.TimeoutError:
                        continue

                    if event == "ROUND_START":
                        r = int((payload or {}).get("round", 1))
                        mr = (payload or {}).get("max_rounds", current_max)
                        try:
                            current_max = int(mr) if mr != "unlimited" else current_max
                        except Exception:
                            pass

                        yield (
                            "",
                            create_progress_step(
                                f"🗣️ Debate vòng {r}/{current_max} - Chuyên gia tranh luận...",
                                3,
                                "active",
                            ),
                            "",
                            "",
                        )
                    elif event == "ROUND_DONE":
                        r = int((payload or {}).get("round", 1))
                        yield (
                            "",
                            create_progress_step(
                                f"🗣️ Hoàn tất vòng {r}/{current_max}",
                                3,
                                "active",
                            ),
                            "",
                            "",
                        )
                    elif event == "DEBATE_DONE":
                        yield (
                            "",
                            create_progress_step("👨‍⚖️ Thẩm phán đang biểu quyết...", 3, "active"),
                            "",
                            "",
                        )

                debate_result = await debate_task
            finally:
                if getattr(orchestrator, "debate_config", None) is not None:
                    if prev_max_rounds is not None or "max_rounds" in orchestrator.debate_config:
                        orchestrator.debate_config["max_rounds"] = prev_max_rounds

            all_rounds_verdicts = getattr(debate_result, "all_rounds_verdicts", None) or []
            max_rounds = len(all_rounds_verdicts) if all_rounds_verdicts else configured_max_rounds

            judge_round_evaluations: List[Dict[str, Any]] = []
            judge_final_reasoning: str = ""

            try:
                from src.pipeline.debate.judge_v2 import JudgeV2
                from src.pipeline.debate.debator import DebateArgument

                models_config = getattr(PIPELINE.debate_orchestrator, "models_config", {}) or {}
                judge_cfg = models_config.get("judge", {}) if isinstance(models_config, dict) else {}

                if isinstance(judge_cfg, dict) and judge_cfg.get("api_key") and judge_cfg.get("base_url"):
                    judge = JudgeV2(
                        model_config=judge_cfg,
                        llm_client=PIPELINE.debate_orchestrator.llm_client,
                        generate_xai=False,
                    )
                    judge.round_evaluations = []

                    debate_history = []
                    previous_majority_verdict: Optional[str] = None

                    def _infer_role(name: str) -> str:
                        n = (name or "").lower()
                        if "grok" in n:
                            return "Truth Seeker A"
                        if "gemini" in n:
                            return "Truth Seeker B"
                        if "gpt" in n:
                            return "Truth Seeker C"
                        return name

                    for r, round_data in enumerate(all_rounds_verdicts, 1):
                        round_arguments = []
                        for agent_name, agent_data in (round_data or {}).items():
                            role = agent_data.get("role") if isinstance(agent_data, dict) else None
                            role = role or _infer_role(agent_name)
                            verdict = agent_data.get("verdict", "NEI") if isinstance(agent_data, dict) else "NEI"
                            conf = float(agent_data.get("confidence", 0.5)) if isinstance(agent_data, dict) else 0.5
                            reasoning = agent_data.get("reasoning", "") if isinstance(agent_data, dict) else ""
                            key_points = agent_data.get("key_points", []) if isinstance(agent_data, dict) else []

                            round_arguments.append(
                                DebateArgument(
                                    debator_name=agent_name,
                                    role=role,
                                    round_num=r,
                                    verdict=verdict,
                                    confidence=conf,
                                    reasoning=reasoning,
                                    key_points=key_points or [],
                                    evidence_citations=[],
                                )
                            )

                        debate_history.append(round_arguments)

                        ev = judge.evaluate_round(
                            claim=claim,
                            evidence_list=[debate_evidence],
                            round_num=r,
                            round_arguments=round_arguments,
                            previous_majority_verdict=previous_majority_verdict,
                            max_rounds=len(all_rounds_verdicts),
                        )
                        previous_majority_verdict = ev.majority_verdict

                    final = judge.make_final_decision(
                        claim=claim,
                        evidence_list=[debate_evidence],
                        debate_history=debate_history,
                        stop_reason=getattr(debate_result, "stop_reason", ""),
                    )

                    final_verdict = final.verdict
                    judge_final_reasoning = getattr(final, "reasoning", "") or ""
                    judge_round_evaluations = [
                        {
                            "round_num": getattr(ev, "round_num", None),
                            "consensus_status": getattr(ev, "consensus_status", ""),
                            "majority_verdict": getattr(ev, "majority_verdict", ""),
                            "verdict_distribution": getattr(ev, "verdict_distribution", {}) or {},
                            "judge_reasoning": getattr(ev, "judge_reasoning", "") or "",
                            "early_stop_decision": getattr(ev, "early_stop_decision", False),
                            "stability_check": getattr(ev, "stability_check", False),
                        }
                        for ev in (getattr(final, "round_evaluations", None) or [])
                    ]
                else:
                    raise RuntimeError("Missing judge config")
            except Exception:
                judge_info = _compute_judge_from_rounds(all_rounds_verdicts)
                final_verdict = judge_info.get("final_verdict", getattr(debate_result, "verdict", "NEI"))
                judge_final_reasoning = judge_info.get("final_reasoning", "")
                judge_round_evaluations = judge_info.get("round_evaluations", [])

            yield ("",
                   create_progress_complete("Hoàn thành tranh luận", "Judge"),
                   "", "")
            result = {
                "final_verdict": final_verdict,
                "debate_all_rounds_verdicts": all_rounds_verdicts,
                "judge_round_evaluations": judge_round_evaluations,
                "judge_final_reasoning": judge_final_reasoning,
                "debate_metrics": {
                    "rounds_used": debate_result.rounds_used,
                    "stop_reason": debate_result.stop_reason,
                }
            }
        
        # Final result
        verdict_html = create_final_verdict(final_verdict)
        path_html = "PhoBERT (Nhanh)" if skipped else "Multi-Agent Debate + Judge"
        debate_html = format_debate_simple(result)
        
        yield (verdict_html, create_progress_complete("✅ Hoàn thành", ""), path_html, debate_html)
        
    except Exception as e:
        yield ("", f"❌ Lỗi hệ thống: {str(e)}", "", "")

def create_progress_step(message: str, step: int, status: str = "active") -> str:
    """Create progress step visualization."""
    if status == "active":
        return f"""
        <div style="display: flex; align-items: center; padding: 16px; background: #f8fafc; border: 1px solid #e5e7eb; border-left: 4px solid #2563eb; border-radius: 10px; max-width: 760px; margin: 12px auto;">
            <div style="width: 24px; height: 24px; border: 3px solid #2563eb; border-top: 3px solid transparent; border-radius: 50%; animation: spin 1s linear infinite; margin-right: 14px;"></div>
            <span style="font-size: 18px; color: #111827; font-weight: 700;">{message}</span>
        </div>
        """
    return ""

def create_progress_complete(message: str, method: str = "") -> str:
    """Create completion message."""
    return f"""
    <div style="display: flex; align-items: center; padding: 16px; background: #f0fdf4; border: 1px solid #d1fae5; border-left: 4px solid #10b981; border-radius: 10px; max-width: 760px; margin: 12px auto;">
        <div style="width: 24px; height: 24px; background: #10b981; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 14px;">
            <span style="color: white; font-weight: bold; font-size: 14px;">✓</span>
        </div>
        <span style="font-size: 18px; color: #065f46; font-weight: 700;">{message}</span>
        {f'<span style="margin-left: 12px; background: #10b981; color: white; padding: 6px 12px; border-radius: 8px; font-size: 14px; font-weight: 600;">{method}</span>' if method else ''}
    </div>
    """

def create_final_verdict(verdict: str) -> str:
    """Create clean verdict display."""
    verdict_map = {
        "Support": "HỖ TRỢ", 
        "Refute": "BÁC BỎ",
        "NEI": "CHƯA ĐỦ THÔNG TIN",
        "SUPPORTED": "HỖ TRỢ",
        "REFUTED": "BÁC BỎ", 
        "NOT_ENOUGH_INFO": "CHƯA ĐỦ THÔNG TIN"
    }
    
    vn_verdict = verdict_map.get(verdict, verdict)
    
    if vn_verdict == "HỖ TRỢ":
        color = "#34a853"
        icon = "✅"
        bg = "#ecfdf5"
    elif vn_verdict == "BÁC BỎ":
        color = "#ea4335" 
        icon = "❌"
        bg = "#fef2f2"
    else:
        color = "#fbbc04"
        icon = "❓"
        bg = "#fffbeb"
    
    return f"""
    <div style="text-align: center; padding: 24px; background: {bg}; border-radius: 12px; border: 2px solid {color}; max-width: 760px; margin: 16px auto;">
        <div style="font-size: 36px; margin-bottom: 12px;">{icon}</div>
        <div style="color: {color}; font-size: 22px; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; line-height: 1.2;">
            {vn_verdict}
        </div>
    </div>
    """

# Clean white theme CSS
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.gradio-container {
    background: #ffffff !important;
    max-width: 100% !important;
    margin: 0 !important;
    padding: 20px !important;
    --block-border-width: 0px !important;
    --block-border-color: transparent !important;
}

body {
    background: #ffffff !important;
    min-height: 100vh;
}

/* Labels (block-info) must be readable on white background */
.gradio-container span[data-testid="block-info"],
.gradio-container label span[data-testid="block-info"] {
    color: #111827 !important;
}

.gr-input, .gr-text-input, textarea {
    background: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    color: #374151 !important;
    font-size: 16px !important;
    padding: 12px !important;
    transition: all 0.2s ease !important;
}

.gr-input:focus, .gr-text-input:focus, textarea:focus {
    border-color: transparent !important;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15) !important;
    outline: none !important;
}

.gr-button {
    background: #2563eb !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    padding: 12px 20px !important;
    transition: all 0.2s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

.gr-button:hover {
    background: #1d4ed8 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
}

/* Primary button in Gradio v4+/v5+ is often plain <button class="lg primary ..."> */
.gradio-container button.primary,
.gradio-container button.lg.primary,
.gradio-container button.primary.lg {
    background: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
}

/* Prevent the main CTA from stretching full width */
#verify_row {
    justify-content: center !important;
}

#verify_btn button {
    width: fit-content !important;
    min-width: 320px !important;
}

.gradio-container button.primary:hover,
.gradio-container button.lg.primary:hover,
.gradio-container button.primary.lg:hover {
    background: #1d4ed8 !important;
    color: #ffffff !important;
}

/* Sample buttons - nuclear override approach */
.gradio-container .gr-button,
.gradio-container button,
.gr-button,
button {
    /* Only target small buttons (samples) not large primary button */
}

.gradio-container button.sm,
.gradio-container .gr-button.sm,
button.sm {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #374151 !important;
    border: 2px solid #d1d5db !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    text-transform: none !important;
    letter-spacing: normal !important;
}

.gradio-container .gr-button[variant="secondary"]:hover,
.gradio-container button[variant="secondary"]:hover,
.gradio-container .gr-button.secondary:hover,
.gradio-container button.secondary:hover,
.gradio-container .sm.secondary:hover,
.gradio-container button.sm.secondary:hover,
.gradio-container [data-testid] button[variant="secondary"]:hover,
.gradio-container [data-testid] .gr-button[variant="secondary"]:hover,
button.secondary:hover,
button[variant="secondary"]:hover,
.secondary:hover {
    background: #f9fafb !important;
    border-color: #9ca3af !important;
    color: #111827 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
    background-color: #f9fafb !important;
}

/* Even more aggressive override */
* [class*="secondary"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
}

* [class*="secondary"]:hover {
    background: #f9fafb !important; 
    background-color: #f9fafb !important;
}

label {
    color: #374151 !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    margin-bottom: 6px !important;
}

h1, h2, h3, h4 {
    color: #1f2937 !important;
    font-weight: 700 !important;
}

/* Make the whole form a single rounded card; avoid "scalloped" corners between stacked blocks */
.gradio-container .form {
    background: #ffffff !important;
    border-radius: 16px !important;
    overflow: hidden !important;
    box-shadow: none !important;
}

.block, .gr-block {
    background: white !important;
    border-radius: 0px !important;
    border: none !important;
    box-shadow: none !important;
    border-width: 0 !important;
    border-style: none !important;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

footer {
    display: none !important;
}

/* Card styling */
.gr-form, .gr-panel {
    background: white !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    border: 1px solid #e5e7eb !important;
}
"""

def create_demo_v3():
    """Create professional clean demo interface."""
    
    with gr.Blocks(
        title="Kiểm chứng tiếng Việt"
    ) as demo:
        
        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 6px 0 14px 0; background: transparent;">
            <h1 style="font-size: 40px; font-weight: 800; margin: 0; color: #111827 !important;">🔍 Kiểm chứng sự thật tiếng Việt</h1>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.HTML("<div style='height: 6px;'></div>")
                
                claim_input = gr.Textbox(
                    label="🎯 Tuyên bố (Claim)",
                    placeholder="Nhập tuyên bố cần kiểm chứng...",
                    lines=3,
                    max_lines=5
                )
                
                evidence_input = gr.Textbox(
                    label="📋 Bằng chứng (Evidence)", 
                    placeholder="Nhập bằng chứng hoặc ngữ cảnh liên quan...",
                    lines=4,
                    max_lines=8
                )
                
                with gr.Row(elem_id="verify_row"):
                    verify_btn = gr.Button("🚀 KIỂM CHỨNG NGAY", variant="primary", scale=0, size="lg", elem_id="verify_btn")
                
                # Sample buttons
                gr.HTML("<h4 style='color: #6b7280; margin: 14px 0 10px 0; text-align: center;'>📋 Mẫu thử nghiệm:</h4>")
                with gr.Row():
                    sample_btn1 = gr.Button("Mẫu 1: Olympic", size="sm", variant="secondary")
                    sample_btn2 = gr.Button("Mẫu 2: Thất nghiệp", size="sm", variant="secondary")  
                    sample_btn3 = gr.Button("Mẫu 3: AI Law", size="sm", variant="secondary")
                
                # Progress section
                gr.HTML("<h3 style='color: #374151; font-size: 18px; margin: 18px 0 10px 0;'>⏳ Tiến trình xử lý</h3>")
                progress_output = gr.HTML(value="", show_progress="hidden")
                
                # Result section  
                gr.HTML("<h3 style='color: #374151; font-size: 18px; margin: 18px 0 10px 0;'>🎯 Kết quả</h3>")
                verdict_output = gr.HTML(show_progress="hidden")
                path_output = gr.HTML(show_progress="hidden")
        
        # Debate visualization (full width)
        gr.HTML("<div style='height: 8px;'></div>")
        debate_output = gr.HTML(show_progress="hidden")
        
        # Event handlers
        verify_btn.click(
            fn=verify_claim_v3,
            inputs=[claim_input, evidence_input],
            outputs=[verdict_output, progress_output, path_output, debate_output],
            show_progress="hidden"
        )
        
        # Sample data handlers
        sample_btn1.click(lambda: load_sample(0), outputs=[claim_input, evidence_input])
        sample_btn2.click(lambda: load_sample(1), outputs=[claim_input, evidence_input]) 
        sample_btn3.click(lambda: load_sample(2), outputs=[claim_input, evidence_input])
    
    return demo

if __name__ == "__main__":
    print("🚀 Starting ViFactCheck Demo V3...")
    port = _pick_free_port(7861)  # Different port from V2
    print(f"📍 URL: http://localhost:{port}\n")
    
    if init_pipeline():
        print("\n🌐 Launching V3 web UI...")
        demo = create_demo_v3()
        
        try:
            import asyncio.runners as _asyncio_runners
            asyncio.run = _asyncio_runners.run
        except Exception:
            pass
            
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            theme=gr.themes.Monochrome(
                primary_hue="slate",
                secondary_hue="gray",
                neutral_hue="slate",
                text_size="lg",
                font=("Inter", "system-ui", "sans-serif")
            ),
            css=CUSTOM_CSS,
            share=False,
            show_error=True
        )
    else:
        print("❌ Cannot start demo without pipeline.")
