"""
Gradio Demo UI for Vietnamese Fact Checking (v2)

Features:
- Clean UI: Dark gray theme, white text, no purple
- Pre-loaded pipeline at startup (no lazy loading)
- Shows Fast Path (PhoBERT) vs Slow Path (Debate) results
- Displays round-by-round debate process

Author: Lockdown
Date: Dec 25, 2025
"""

import gradio as gr
import asyncio
import logging
import os
import socket
from pathlib import Path
from typing import Dict, Any, Tuple

# Suppress ALL verbose warnings and logs BEFORE imports
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup minimal logging
logging.basicConfig(level=logging.ERROR, format='%(message)s')
for logger_name in ["transformers", "torch", "httpx", "asyncio", "aiohttp"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

def _pick_free_port(preferred: int = 7860) -> int:
    """Pick a free TCP port. Try preferred first, otherwise let OS pick."""
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

# Global pipeline instance
PIPELINE = None
INIT_STATUS = "Not initialized"


def init_pipeline():
    """Initialize pipeline at startup."""
    global PIPELINE, INIT_STATUS
    
    print("=" * 50)
    print("🔄 Initializing ViFactCheck Pipeline...")
    print("   (This takes 30-60 seconds on first run)")
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
        print("=" * 50)
        print("✅ Pipeline ready!")
        print("=" * 50)
        return True
        
    except Exception as e:
        INIT_STATUS = f"❌ Error: {str(e)}"
        print(f"❌ Init failed: {e}")
        return False

# Sample data
SAMPLES = [
    {
        "claim": "Việt Nam là quốc gia thứ hai đón ngọn đuốc Olympic Paris 2024.",
        "evidence": "Việt Nam là quốc gia đầu tiên đón ngọn đuốc Olympic Paris 2024 trong hành trình rước đuốc vòng quanh thế giới.",
        "label": "Refute"
    },
    {
        "claim": "Tỷ lệ thất nghiệp của Việt Nam năm 2023 là 2.28%.",
        "evidence": "Theo Tổng cục Thống kê, tỷ lệ thất nghiệp của Việt Nam năm 2023 là 2.28%, giảm so với năm 2022.",
        "label": "Support"
    },
    {
        "claim": "Chính phủ Việt Nam đã ban hành luật về trí tuệ nhân tạo vào năm 2024.",
        "evidence": "Việt Nam đang xây dựng chiến lược quốc gia về trí tuệ nhân tạo đến năm 2030, với mục tiêu trở thành trung tâm đổi mới sáng tạo.",
        "label": "NEI"
    }
]


def load_sample(idx: int) -> Tuple[str, str, str]:
    """Load sample by index."""
    if 0 <= idx < len(SAMPLES):
        s = SAMPLES[idx]
        return s["claim"], s["evidence"], f"Label: {s['label']}"
    return "", "", ""


def format_debate_html(result: Dict[str, Any]) -> str:
    """Format debate transcript with full structured output."""
    all_rounds = result.get("debate_all_rounds_verdicts", [])
    
    if not all_rounds:
        return "<p style='color: #888; font-style: italic;'>Không có tranh luận (Độ tin cậy cao)</p>"
    
    html = '<div style="font-size: 16px; line-height: 1.6;">'
    
    for round_num, round_data in enumerate(all_rounds, 1):
        html += f'<details open style="margin: 12px 0; border: 1px solid #444; border-radius: 8px;"><summary style="cursor: pointer; font-weight: bold; color: #fff; background: #1a1a1a; padding: 10px; border-radius: 8px 8px 0 0; font-size: 17px;">🔍 Round {round_num}</summary>'
        html += '<div style="padding: 15px; background: #2a2a2a; border-radius: 0 0 8px 8px;">'
        
        for agent_name, agent_data in round_data.items():
            verdict = agent_data.get("verdict", "NEI")
            reasoning = agent_data.get("reasoning", "")
            
            verdict_color = "#4a7c59" if verdict == "SUPPORTED" else "#7c4a4a" if verdict == "REFUTED" else "#5a5a5a"
            
            html += f"""
            <div style="margin: 12px 0; padding: 12px; background: #333; border-left: 4px solid {verdict_color}; border-radius: 4px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px; align-items: center;">
                    <strong style="color: #fff; font-size: 18px;">{agent_name.split('/')[-1]}</strong>
                    <span style="background: {verdict_color}; padding: 4px 10px; border-radius: 4px; font-size: 13px; font-weight: bold; color: #fff;">{verdict}</span>
                </div>
            """
            
            # Round 1: Show parts-based analysis
            if round_num == 1:
                parts = agent_data.get("parts", [])
                if parts:
                    html += '<div style="margin: 8px 0;"><strong style="color: #88c999; font-size: 15px;">📋 Key Parts Analysis:</strong></div>'
                    html += '<div style="margin-left: 15px;">'
                    for i, part in enumerate(parts, 1):
                        part_text = part.get("part", "")
                        status = part.get("status", "")
                        quote = part.get("quote", "")
                        
                        status_color = "#4a7c59" if status == "COVERED" else "#7c4a4a" if status == "CONFLICT" else "#666"
                        status_icon = "✅" if status == "COVERED" else "❌" if status == "CONFLICT" else "❓"
                        
                        html += f"""
                        <div style="margin: 6px 0; padding: 10px; background: #404040; border-radius: 4px; border-left: 3px solid {status_color};">
                            <div style="font-size: 16px; color: #ddd; font-weight: 500;"><strong>{i}.</strong> {part_text}</div>
                            <div style="font-size: 15px; margin: 4px 0;"><span style="color: {status_color}; font-weight: bold;">{status_icon} {status}</span></div>
                            <div style="font-size: 15px; color: #bbb; font-style: italic;">Quote: "{quote if quote and quote != 'NULL' else 'N/A'}"</div>
                        </div>
                        """
                    html += '</div>'
            
            # Round 2+: Show rebuttal analysis (using same green color scheme as R1)
            else:
                key_parts = agent_data.get("key_parts_checked", [])
                decision_change = agent_data.get("decision_change", "MAINTAIN")
                rebuttals = agent_data.get("rebuttals", [])
                
                if key_parts:
                    html += f'<div style="margin: 8px 0; font-size: 16px; color: #88c999;"><strong>🔍 Checked:</strong> {", ".join(key_parts)}</div>'
                
                # Use green color scheme consistent with R1
                change_color = "#7c4a4a" if decision_change == "CHANGE" else "#4a7c59" if decision_change == "MAINTAIN" else "#88c999"
                html += f'<div style="margin: 8px 0; font-size: 16px;"><strong style="color: {change_color}; font-size: 17px;">📝 Decision:</strong> <span style="color: {change_color}; font-weight: bold;">{decision_change}</span></div>'
                
                if rebuttals:
                    html += '<div style="margin: 8px 0;"><strong style="color: #88c999; font-size: 17px;">💬 Rebuttals:</strong></div>'
                    html += '<div style="margin-left: 15px;">'
                    for reb in rebuttals:
                        agent_name_reb = reb.get("agent", "")
                        issue = reb.get("issue", "")
                        counter_quote = reb.get("counter_quote", "")
                        
                        html += f"""
                        <div style="margin: 6px 0; padding: 10px; background: #404040; border-radius: 4px; border-left: 3px solid #88c999;">
                            <div style="font-size: 16px; color: #88c999; font-weight: bold;">vs {agent_name_reb}</div>
                            <div style="font-size: 15px; color: #ddd;">Issue: {issue}</div>
                            <div style="font-size: 15px; color: #bbb; font-style: italic;">Counter: "{counter_quote if counter_quote and counter_quote != 'NULL' else 'N/A'}"</div>
                        </div>
                        """
                    html += '</div>'
            
            # Always show reasoning
            if reasoning:
                html += f'<div style="margin: 10px 0; padding: 12px; background: #1a1a1a; border-radius: 4px; font-size: 16px; color: #ddd; line-height: 1.5;">{reasoning}</div>'
            
            html += '</div>'
        
        html += '</div></details>'
    
    # Majority Vote - Visual representation with 3 debators
    metrics = result.get("debate_metrics", {})
    if metrics:
        # Get final round verdicts for vote visualization
        final_round = all_rounds[-1] if all_rounds else {}
        debator_votes = []
        for agent_name, agent_data in final_round.items():
            verdict = agent_data.get("verdict", "NEI")
            short_name = agent_name.split('/')[-1]
            verdict_color = "#4a7c59" if verdict == "SUPPORTED" else "#7c4a4a" if verdict == "REFUTED" else "#5a5a5a"
            debator_votes.append((short_name, verdict, verdict_color))
        
        html += f"""
        <div style="margin-top: 20px; padding: 15px; background: linear-gradient(135deg, #2a3a4a, #1a2a3a); border-radius: 8px; border-left: 4px solid #4a7c59;">
            <div style="text-align: center; margin-bottom: 15px;">
                <h4 style="color: #fff; margin: 0; font-size: 18px;">⚖️ Majority Vote</h4>
                <div style="color: #bcd; font-size: 14px; margin-top: 5px;">{metrics.get('rounds_used', '?')} rounds • {metrics.get('stop_reason', 'unknown')}</div>
            </div>
            <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
        """
        
        for short_name, verdict, verdict_color in debator_votes:
            html += f"""
                <div style="text-align: center; padding: 10px; background: #1a1a1a; border-radius: 8px; border: 2px solid {verdict_color}; min-width: 120px;">
                    <div style="color: #fff; font-weight: bold; font-size: 14px; margin-bottom: 5px;">{short_name}</div>
                    <div style="background: {verdict_color}; color: #fff; padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: bold;">{verdict}</div>
                </div>
            """
        
        html += """
            </div>
        </div>
        """
    
    html += '</div>'
    return html


async def verify_claim(claim: str, evidence: str):
    """
    Main verification function (no progress bar to avoid UI clutter).
    
    Returns: verdict_html, path_html, debate_html, status, dev_trace_html
    """
    if not claim.strip() or not evidence.strip():
        yield "⚠️ Nhập claim và evidence", "", "", "<div style='color:#ffb74d;'>Thiếu input</div>", ""
        return

    if PIPELINE is None:
        yield f"❌ Pipeline chưa sẵn sàng: {INIT_STATUS}", "", "", "<div style='color:#ef5350;'>Lỗi</div>", ""
        return

    try:
        def _status(msg: str) -> str:
            return (
                "<div class='status-row'>"
                "  <span class='spinner'></span>"
                f"  <span>{msg}</span>"
                "</div>"
            )

        def _status_simple() -> str:
            return _status("Đang phân tích...")

        # Step 0: preparing
        yield "", "", "", _status("Đang chuẩn bị dữ liệu..."), ""

        # Step 1: PhoBERT prediction (sync)
        yield "", "", "", _status("Đang chạy PhoBERT..."), ""
        model_verdict, confidence, probs = PIPELINE._predict_verdict(claim, evidence)

        # Step 2: routing decision
        skipped = False
        hybrid_info: Dict[str, Any] = {"skipped": False}
        if PIPELINE.debate_orchestrator and getattr(PIPELINE.config, "hybrid_enabled", True) and PIPELINE.debate_orchestrator.hybrid_enabled:
            hybrid_decision = PIPELINE.debate_orchestrator.hybrid_strategy.decide(
                model_verdict=model_verdict,
                model_probs=probs,
                debate_verdict="NEI",
                debate_confidence=0.0,
            )

            if hybrid_decision.decision_source == "MODEL_HIGH_CONF":
                skipped = True
                hybrid_info = {
                    "source": "MODEL_HIGH_CONF",
                    "threshold": PIPELINE.debate_orchestrator.hybrid_threshold,
                    "skipped": True,
                }
            else:
                hybrid_info = {
                    "source": "DEBATE",
                    "threshold": PIPELINE.debate_orchestrator.hybrid_threshold,
                    "skipped": False,
                }
        else:
            skipped = True
            hybrid_info = {"source": "MODEL_ONLY", "skipped": True}

        yield "", "", "", _status(
            f"Routing: {'PhoBERT (model-only)' if skipped else 'Multi-Agent Debate'}..."
        ), ""

        result: Dict[str, Any] = {
            "statement": claim,
            "evidence": evidence[:200] + "..." if len(evidence) > 200 else evidence,
            "model_verdict": model_verdict,
            "model_confidence": round(float(confidence), 4),
            "model_probs": {k: round(float(v), 6) for k, v in (probs or {}).items()},
            "hybrid_info": hybrid_info,
        }

        # Step 3: Fast vs Slow execution
        if skipped:
            # Fast Path - no debate needed
            result["final_verdict"] = model_verdict
            result["debate_metrics"] = None
            result["debate_all_rounds_verdicts"] = None
        else:
            # Slow Path debate
            yield "", "", "", _status("Multi-Agent Debate: đang tranh luận..."), ""

            from src.pipeline.debate.debator import Evidence as DebateEvidence

            debate_evidence = DebateEvidence(
                text=evidence,
                source="vifactcheck",
                rank=1,
                nli_score={"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0, "other": 0.0},
                relevance_score=1.0,
            )

            # Always show progress
            q: asyncio.Queue = asyncio.Queue()
            last_status_html = _status("Đang chuẩn bị dữ liệu...")
            last_event_ts = asyncio.get_event_loop().time()
            started_ts = last_event_ts
            current_round: int = 0
            phase: str = "debate"  # debate | judge
            max_rounds_ui: str = "?"  # display only (e.g., "7" or "unlimited")

            def _progress_cb(event: str, payload: Dict[str, Any]):
                q.put_nowait({"event": event, "payload": payload})

            debate_task = asyncio.create_task(
                PIPELINE.debate_orchestrator.debate_async(
                    claim=claim,
                    evidences=[debate_evidence],
                    model_verdict=model_verdict,
                    model_confidence=float(confidence),
                    progress_cb=_progress_cb,
                )
            )

            # Stream progress while debate is running
            while not debate_task.done():
                try:
                    item = await asyncio.wait_for(q.get(), timeout=0.4)
                    ev = item.get("event")
                    pl = item.get("payload") or {}
                    last_event_ts = asyncio.get_event_loop().time()
                    if ev == "ROUND_START":
                        try:
                            current_round = int(pl.get("round") or 0)
                        except Exception:
                            current_round = current_round or 0
                        try:
                            mr = pl.get("max_rounds")
                            max_rounds_ui = str(mr) if mr is not None else max_rounds_ui
                        except Exception:
                            pass
                        phase = "debate"
                        last_status_html = _status(f"Multi-Agent Debate: Round {current_round}/{max_rounds_ui} đang chạy...")
                        yield "", "", "", last_status_html, ""
                    elif ev == "ROUND_DONE":
                        vc = pl.get("vote_counts") or {}
                        vc_txt = ", ".join([f"{k}:{v}" for k, v in vc.items()])
                        try:
                            current_round = int(pl.get("round") or current_round or 0)
                        except Exception:
                            pass
                        try:
                            mr = pl.get("max_rounds")
                            max_rounds_ui = str(mr) if mr is not None else max_rounds_ui
                        except Exception:
                            pass
                        phase = "debate"
                        last_status_html = _status(f"Multi-Agent Debate: Round {current_round}/{max_rounds_ui} xong ({vc_txt})")
                        yield "", "", "", last_status_html, ""
                    elif ev == "JUDGE_START":
                        last_status_html = _status("Tổng hợp kết quả debate...")
                        phase = "judge"
                        yield "", "", "", last_status_html, ""
                    elif ev == "DEBATE_START":
                        last_status_html = _status("Multi-Agent Debate: bắt đầu tranh luận...")
                        phase = "debate"
                        yield "", "", "", last_status_html, ""
                    elif ev == "JUDGE_DONE":
                        v = pl.get("verdict")
                        c = pl.get("confidence")
                        last_status_html = _status(f"Đã tổng hợp: {v} (conf={c})")
                        phase = "judge"
                        yield "", "", "", last_status_html, ""
                    else:
                        # ignore other events
                        pass
                except asyncio.TimeoutError:
                    # keep UI alive, but don't overwrite meaningful status
                    now = asyncio.get_event_loop().time()
                    if (now - last_event_ts) > 2.0 and (now - started_ts) > 2.0:
                        # fallback if events are slow / missing
                        if phase == "judge":
                            last_status_html = _status("Đang tổng hợp kết quả...")
                        else:
                            fallback_round = current_round if current_round > 0 else 1
                            last_status_html = _status(f"Multi-Agent Debate: Round {fallback_round}/{max_rounds_ui} đang chạy...")
                    yield "", "", "", last_status_html, ""

            debate_result = await debate_task

            result["debate_verdict"] = debate_result.verdict
            result["debate_confidence"] = round(float(debate_result.confidence), 4)
            result["debate_reasoning"] = debate_result.reasoning
            result["debate_round_1_verdicts"] = debate_result.round_1_verdicts
            result["debate_all_rounds_verdicts"] = getattr(debate_result, "all_rounds_verdicts", None)
            result["debate_metrics"] = {
                "rounds_used": debate_result.rounds_used,
                "early_stopped": debate_result.early_stopped,
                "stop_reason": debate_result.stop_reason,
                "mvp_agent": debate_result.mvp_agent,
                "debator_agreements": debate_result.debator_agreements,
                "decision_path": getattr(debate_result, "decision_path", None),
                "best_quote_from": getattr(debate_result, "best_quote_from", None),
            }
            result["final_verdict"] = debate_result.verdict
        
        # Extract results
        final_verdict = result.get("final_verdict", "NEI")
        model_verdict = result.get("model_verdict", "NEI")
        model_conf = result.get("model_confidence", 0.0)
        debate_verdict = result.get("debate_verdict")
        debate_conf = result.get("debate_confidence", 0.0)

        # Determine path
        hybrid_info = result.get("hybrid_info", {}) or {}
        skipped = hybrid_info.get("skipped", False)
        path_name = "PhoBERT (model-only)" if skipped else "Multi-Agent Debate"
        
        # Verdict display
        # Translate to Vietnamese
        verdict_map = {
            "Support": "HỖ TRỢ", 
            "Refute": "BÁC BỎ", 
            "NEI": "CHƯA ĐỦ THÔNG TIN",
            "SUPPORTED": "HỖ TRỢ",
            "REFUTED": "BÁC BỎ",
            "NOT_ENOUGH_INFO": "CHƯA ĐỦ THÔNG TIN"
        }
        vn_verdict = verdict_map.get(final_verdict, final_verdict)
        
        # Simple colors (desaturated)
        verdict_colors = {
            "HỖ TRỢ": "#2e7d32", # Muted Green
            "BÁC BỎ": "#c62828", # Muted Red
            "CHƯA ĐỦ THÔNG TIN": "#424242" # Dark Grey
        }
        verdict_color = verdict_colors.get(vn_verdict, "#424242")
        
        confidence = model_conf if skipped else debate_conf
        
        verdict_html = f"""
        <div style="text-align: center; padding: 20px;">
            <span style="background: {verdict_color}; color: #fff; padding: 15px 30px; 
                         border-radius: 8px; font-size: 24px; font-weight: bold; letter-spacing: 1px;">
                {vn_verdict}
            </span>
        </div>
        """
        
        # Path display
        path_color = "#2e7d32" if skipped else "#1565c0"
        path_name_vn = "PhoBERT (Model-only)" if skipped else "Multi-Agent Debate"
        
        path_html = f"""
        <div style="text-align: center; margin-top: 10px;">
            <span style="border: 1px solid {path_color}; color: {path_color}; padding: 5px 12px; border-radius: 15px; font-size: 14px;">
                {path_name_vn}
            </span>
        </div>
        """
        
        # Always show dev trace
        metrics = result.get("debate_metrics") or {}
        decision_path = metrics.get("decision_path")
        best_quote_from = metrics.get("best_quote_from")
        rounds_used = metrics.get("rounds_used")
        stop_reason = metrics.get("stop_reason")
        hybrid_source = hybrid_info.get("source")
        threshold = hybrid_info.get("threshold")

        dev_trace_html = "<div style='font-size:13px; line-height:1.5; background:#1a1a1a; border:1px solid #333; padding:12px; border-radius:8px;'>"
        dev_trace_html += "<div style='font-weight:700; margin-bottom:8px;'>Dev Trace</div>"
        dev_trace_html += f"<div><strong>PhoBERT:</strong> verdict={model_verdict}, conf={model_conf}</div>"
        dev_trace_html += f"<div><strong>Routing:</strong> {'FAST_PATH' if skipped else 'SLOW_PATH'}</div>"
        if threshold is not None:
            dev_trace_html += f"<div><strong>Hybrid:</strong> threshold={threshold}, source={hybrid_source}</div>"
        else:
            dev_trace_html += f"<div><strong>Hybrid:</strong> source={hybrid_source}</div>"
        if not skipped:
            dev_trace_html += f"<div><strong>Debate:</strong> rounds_used={rounds_used}, stop_reason={stop_reason}</div>"
            dev_trace_html += f"<div><strong>Majority Vote:</strong> decision_path={decision_path}, best_quote_from={best_quote_from}</div>"
        dev_trace_html += "</div>"
        
        # Debate display
        debate_html = format_debate_html(result)
        
        yield verdict_html, path_html, debate_html, "<div style='color:#66bb6a; font-weight:600;'>✅ Hoàn thành</div>", dev_trace_html
        return
        
    except Exception as e:
        yield f"❌ Lỗi: {str(e)}", "", "", f"<div style='color:#ef5350;'>Lỗi: {str(e)}</div>", ""
        return


# Custom CSS for dark theme - High Contrast & Large Text
CUSTOM_CSS = """
.gradio-container {
    background: #121212 !important;
}
.status-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: linear-gradient(135deg, #1a2332, #2a3342);
    border-left: 4px solid #4a6fa5;
    border-radius: 6px;
    margin: 8px 0;
}
.spinner {
    width: 14px;
    height: 14px;
    border: 2px solid #333;
    border-top-color: #4a6fa5;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}
.debate-container {
    max-height: 800px;
    overflow-y: auto;
    border: 1px solid #444;
    border-radius: 8px;
    padding: 16px;
    background: #1a1a1a;
    font-size: 15px;
    line-height: 1.6;
}
@keyframes spin {
    to { transform: rotate(360deg); }
}
.gr-box, .gr-form, .gr-panel, .gr-input, .gr-text-input, textarea, .gr-button {
    font-size: 16px !important;
}
.gr-input, .gr-text-input, textarea {
    background: transparent !important;
    color: #ffffff !important;
    border: 1px solid #444 !important;
}
.gr-input:focus, .gr-text-input:focus, textarea:focus {
    border-color: #666 !important;
    outline: none !important;
}
label {
    color: #ffffff !important;
    font-size: 16px !important;
    font-weight: bold !important;
}
p, span, div {
    color: #eeeeee;
}
h1, h2, h3, h4 {
    color: #ffffff !important;
}
.prose {
    font-size: 16px !important;
    color: #eeeeee !important;
}
footer {
    display: none !important;
}
/* Make input container transparent */
.block, .form, .panel, .gr-group, .gr-block, .gr-form {
    background: transparent !important;
    border: none !important;
}
/* Hide multiple progress indicators - show only one */
.progress-bar, .progress-text {
    display: none !important;
}
.wrap.default.full.svelte-1kptzi7 {
    display: block !important;
}
"""


def create_demo():
    """Create Gradio demo interface."""
    
    with gr.Blocks(
        title="Verification Demo",
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue="slate",
            secondary_hue="slate",
            neutral_hue="slate",
            text_size="lg"
        )
    ) as demo:
        
        gr.HTML("""
        <div style="text-align: center; padding: 25px 0;">
            <h1 style="color: #fff; margin: 0; font-size: 32px; font-weight: 700;">🔍 Verification News</h1>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                claim_input = gr.Textbox(
                    label="Claim (Tuyên bố cần kiểm tra)",
                    placeholder="Nhập tuyên bố...",
                    lines=3
                )
                
                evidence_input = gr.Textbox(
                    label="Evidence (Bằng chứng)",
                    placeholder="Nhập bằng chứng...",
                    lines=4
                )
                
                with gr.Row():
                    verify_btn = gr.Button("🚀 KIỂM TRA", variant="primary", scale=2, size="lg")

                status = gr.HTML(label="Status", value="", visible=True)
        
        gr.HTML("<hr style='border-color: #333; margin: 20px 0;'>")
        
        # Results + Debate Layout (side by side)
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h3 style='color: #fff; text-align: center; font-size: 20px; margin-bottom: 15px;'>📊 KẾT QUẢ</h3>")
                verdict_output = gr.HTML(show_progress="hidden")
                path_output = gr.HTML(show_progress="hidden")
            
            with gr.Column(scale=3):  # Even bigger space for debate
                gr.HTML("<h3 style='color: #fff; font-size: 20px; margin-bottom: 15px;'>🗣️ TRANH LUẬN CHI TIẾT</h3>")
                debate_output = gr.HTML(show_progress="hidden", elem_classes=["debate-container"])

        gr.HTML("<hr style='border-color: #333; margin: 20px 0;'>")
        
        # Dev trace (collapsed by default)
        with gr.Row():
            with gr.Column():
                with gr.Accordion("🛠️ DEV TRACE", open=False):
                    dev_trace_output = gr.HTML(show_progress="hidden")
        
        # Event handlers
        verify_btn.click(
            fn=verify_claim,
            inputs=[claim_input, evidence_input],
            outputs=[verdict_output, path_output, debate_output, status, dev_trace_output],
            show_progress="minimal"  # Only show one progress indicator
        )
    
    return demo


if __name__ == "__main__":
    print("🚀 Starting ViFactCheck Demo...")
    port = _pick_free_port(7860)
    print(f"📍 URL: http://localhost:{port}\n")
    
    # Pre-initialize pipeline BEFORE launching UI
    if init_pipeline():
        print("\n🌐 Launching web UI...")
        demo = create_demo()

        # nest_asyncio may monkeypatch asyncio.run() and break uvicorn in threads on Windows.
        # Restore stdlib implementation to avoid: "There is no current event loop in thread".
        try:
            import asyncio.runners as _asyncio_runners

            asyncio.run = _asyncio_runners.run  # type: ignore[attr-defined]
        except Exception:
            pass

        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
            show_error=True
        )
    else:
        print("❌ Cannot start demo without pipeline. Check errors above.")
