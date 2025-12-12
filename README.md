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
Pipeline có 2 nhánh chính:

- **Fast Path (PhoBERT + XAI)**
  - Chạy nhanh, phù hợp khi mô hình có độ tin cậy cao.
  - XAI được sinh theo hướng “giải thích tự nhiên” (natural explanation).

- **Slow Path (Multi-Agent Debate + Judge + XAI)**
  - Kích hoạt khi case khó / mô hình không chắc chắn.
  - Nhiều debators tranh luận theo vòng (round), sau đó **Judge** tổng hợp và đưa ra kết luận cuối.
  - XAI được chuẩn hoá theo cùng schema với Fast Path.

---

## Thành viên
- Trương Hoàng Phúc - 2280602486
- Ngô Xuân Hạo - 2280600867
