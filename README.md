# Fact Verification

Dự án này xây dựng một pipeline **Fact Verification (xác minh tuyên bố)** cho tiếng Việt dựa trên bộ dữ liệu **ViFactCheck**.

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
│                           HYBRID DEBATE PIPELINE                            │
│                        (DOWN Framework - 2025)                              │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌───────────────┐
                              │   INPUT       │
                              │ Claim + Evid  │
                              └───────┬───────┘
                                      │
                                      ▼
                        ┌─────────────────────────┐
                        │     PhoBERT Model       │
                        │   (3-label classifier)  │
                        │  SUPPORT│REFUTE│NEI     │
                        └────────────┬────────────┘
                                     │
                            ┌────────┴────────┐
                            │  Verdict + Conf │
                            └────────┬────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │     HYBRID DECISION            │
                    │  Confidence >= 0.85 threshold? │
                    └───────────────┬────────────────┘
                                    │
                   ┌────────────────┴────────────────┐
                   │                                 │
        YES (≥85%) │                                 │ NO (<85%)
                   │                                 │
                   ▼                                 ▼
    ┌──────────────────────────┐    ┌──────────────────────────────────────┐
    │      FAST PATH           │    │          SLOW PATH                   │
    │    Trust Model           │    │      Multi-Agent Debate              │
    │    (Skip Debate)         │    │                                      │
    └────────────┬─────────────┘    │  ┌─────────────────────────────────┐ │
                 │                  │  │       DEBATORS (3 agents)       │ │
                 │                  │  │  ┌─────┐  ┌─────┐  ┌─────┐      │ │
                 │                  │  │  │ D1  │  │ D2  │  │ D3  │      │ │
                 │                  │  │  │Supp │  │Refut│  │NEI  │      │ │
                 │                  │  │  └──┬──┘  └──┬──┘  └──┬──┘      │ │
                 │                  │  │     │        │        │         │ │
                 │                  │  │     └────────┼────────┘         │ │
                 │                  │  │              ▼                  │ │
                 │                  │  │    ┌─────────────────┐          │ │
                 │                  │  │    │   ROUNDS        │          │ │
                 │                  │  │    │  (2 rounds)     │          │ │
                 │                  │  │    │   Arguments &   │          │ │
                 │                  │  │    │   Rebuttals     │          │ │
                 │                  │  │    └────────┬────────┘          │ │
                 │                  │  │             ▼                   │ │
                 │                  │  │    ┌─────────────────┐          │ │
                 │                  │  │    │   JUDGE         │          │ │
                 │                  │  │    │  Final Verdict  │          │ │
                 │                  │  │    │  + Confidence   │          │ │
                 │                  │  │    └────────┬────────┘          │ │
                 │                  │  └─────────────┼─────────────────┘ │
                 │                  │                │                   │
                 │                  └────────────────┼───────────────────┘
                 │                                   │
                 └────────────────┬──────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │       FINAL VERDICT         │
                    │   SUPPORTED│REFUTED│NEI     │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │       XAI GENERATOR         │
                    │   Natural Language Explain  │
                    │   - Summary                 │
                    │   - Key Evidence            │
                    │   - Reasoning Chain         │
                    └─────────────────────────────┘
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
