# Fake News Detection (Fact Verification) - ViFactCheck + Multi-Agent Debate

Dự án này xây dựng một pipeline **Fact Verification (kiểm chứng thông tin)** cho tiếng Việt dựa trên bộ dữ liệu **ViFactCheck**.

Mục tiêu:
- Dự đoán quan hệ giữa **Tuyên bố** và **Bằng chứng** theo 3 nhãn:
  - **SUPPORTS**: Bằng chứng ủng hộ tuyên bố
  - **REFUTES**: Bằng chứng bác bỏ tuyên bố
  - **NOT_ENOUGH_INFO (NEI)**: Chưa đủ thông tin để kết luận
- Cung cấp **XAI (Explainable AI)**: giải thích ngắn gọn, dễ hiểu, và có thông tin đối chiếu khi bác bỏ.
- Khi mô hình nền không chắc chắn, kích hoạt **Multi-Agent Debate** (nhiều tác nhân tranh luận) để tăng độ tin cậy.

---

## Kiến trúc tổng quan

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  📥 INPUT (ViFactCheck Dataset)                                             │
│  ───────────────────────────────────────────────────────────────────────    │
│  • Statement (Claim): Tuyên bố cần kiểm chứng                               │
│  • Evidence: Bằng chứng từ bài báo                                          │
│  • Label: SUPPORTED / REFUTED / NOT_ENOUGH_INFO                             │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  🤖 MODULE 1: CLASSIFICATION & ROUTING (PhoBERT-base)                       │
│  ───────────────────────────────────────────────────────────────────────    │
│  • Fine-tuned on ViFactCheck (PyVi preprocessing)                           │
│  • OUTPUT: verdict + confidence (0.0 - 1.0)                                 │
│                                                                             │
│                ┌────────────────────────────────────────┐                   │
│                │ 🚦 CONFIDENCE ROUTER                   │                   │
│                │ IF confidence >= Threshold             │                   │
│                │    → HIGH CONFIDENCE PATH              │                   │
│                │ ELSE:                                  │                   │
│                │    → LOW CONFIDENCE PATH (Debate)      │                   │
│                └───────────────────┬────────────────────┘                   │
└─────────────────────────────────┬──┴────────────────────────────────────────┘
                                  │
        ┌─────────────────────────REJECT──────────────────────────┐
        │ (High Confidence)                                       │ (Low Confidence)
        ▼                                                         ▼
┌───────────────────────────────────────┐  ┌────────────────────────────────────────────────────────────────────────┐
│ 🚀 FAST PATH (Model)                  │  │ ⚔️ SLOW PATH: MULTI-AGENT DEBATE COUNCIL                              │
│ ───────────────────────────────────── │  │ ────────────────────────────────────────────────────────────────────── │
│ • Use Model Verdict directly          │  │ 🎯 Purpose: Deep reasoning for hard/ambiguous cases                   │
│ • No Debate Cost, Fast Response       │  │                                                                        │
│                                       │  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│                                       │  │  │ 🎯 GROK 4 FAST  │  │ 💎 GEMINI 2.5  │  │ 🤖 GPT-4o MINI  │        │
│                                       │  │  │ Debator A       │  │ Debator B       │  │ Debator C       │         │
│                                       │  │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │
│                                       │  │           │                    │                    │                  │
│                                       │  │           ▼                    ▼                    ▼                  │
│                                       │  │  ┌─────────────────────────────────────────────────────────────────┐   │
│                                       │  │  │  📖 ROUND 1: Independent Analysis (JSON verdict + quote)        │  │
│                                       │  │  └─────────────────────────────┬───────────────────────────────────┘   │
│                                       │  │                                ▼                                       │
│                                       │  │  ┌─────────────────────────────────────────────────────────────────┐   │
│                                       │  │  │  � ROUND 2-K: Cross-Examination / Dispute Resolution           │   │ 
│                                       │  │  └─────────────────────────────┬───────────────────────────────────┘   │
│                                       │  │                                ▼                                       │
│                                       │  │  ┌─────────────────────────────────────────────────────────────────┐   │
│                                       │  │  │  ✅ ADJUDICATION (Rule-base):                                   │   │
│                                       │  │  │   - Fixed K: run full K rounds, then majority vote last round   │   │
│                                       │  │  │   - EarlyStop (max K): stop when unanimous_r AND stable vs r-1  │   │
│                                       │  │  │     then majority vote last round                               │   │
│                                       │  │  └─────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────┴──────────────────────────────────────┬────────────────────────────────────┘
                                                                               ▼ 
                                        ┌─────────────────────────────────────────────────────────────────────────────┐
                                        │  📤 FINAL OUTPUT (Hybrid Verdict)                                           │
                                        │  ────────────────────────────────────────────────────────────────────────── │
                                        │                                                                             │
                                        │  {                                                                          │
                                        │    "final_verdict": "REFUTED",                                              │
                                        │    "method": "DEBATE" or "MODEL",                                           │
                                        │                                                                             │
                                        │    "source": "SLOW_PATH" or "FAST_PATH"                                     │
                                        │  }                                                                          │
                                        └─────────────────────────────────────────────────────────────────────────────┘
```

</details>

### Mô tả 2 nhánh chính

- **Fast Path (PhoBERT + LIME XAI)**
  - Kích hoạt khi model confidence **≥ 85%** (threshold có thể tune).
  - Chạy nhanh, tiết kiệm API calls (không gọi LLM).
  - XAI sinh bằng **LIME** (rule-based) → giải thích tự nhiên + highlight conflict words.

- **Slow Path (Multi-Agent Debate + Judge + XAI)**
  - Kích hoạt khi case khó / model confidence **< 85%**.
  - 3 debators đại diện cho 3 stance: **Support**, **Refute**, **NEI**.
  - Tranh luận qua 2-4 rounds với arguments & rebuttals.
  - **Judge** tổng hợp và đưa ra kết luận cuối.
  - XAI được chuẩn hoá theo cùng schema với Fast Path.

### Hybrid Strategy Logic

```python
if model_confidence >= threshold:  # Default: 0.85
    final_verdict = model_verdict   # Trust Model → FAST PATH
else:
    final_verdict = debate_verdict  # Trust Debate → SLOW PATH
```

> **Research basis**: DOWN Framework (Debate Only When Necessary, 2025)

---

## Thành viên
- Trương Hoàng Phúc - 2280602486
- Ngô Xuân Hạo - 2280600867
