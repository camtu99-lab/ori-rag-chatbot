"""
Bộ công cụ đánh giá độc lập tầng truy xuất dữ liệu (Retrieval-Only Evaluation):
1. Mục tiêu & Triết lý thiết kế:
   - Đánh giá phân lập (Isolated Testing): Đo lường thuần túy chất lượng retrieval mà không phụ thuộc vào tầng Generation của LLM; giải quyết bài toán chi phí token và độ trễ của `evaluate.py`.
   - Đối chuẩn thực nghiệm (A/B Benchmarking): Cung cấp số liệu định lượng chuẩn xác để đối soát hiệu quả trước và sau khi nâng cấp từ Dense-only lên Hybrid Search.
2. Tính tương thích ngược (Interface Decoupling):
   - Chỉ tương tác qua public API `retrieve(query) -> dict` từ `rag.py`.
   - Giữ nguyên script mà không cần sửa đổi khi backend retriever nâng cấp thuật toán nội bộ.
3. Đầu ra (Reporting):
   - Terminal: Tóm tắt các chỉ số thống kê tổng hợp (Hit Rate, Precision, Recall...).
   - File xuất: `retrieval_eval_report.csv` lưu chi tiết context trả về và kết quả đánh giá của từng câu hỏi.
"""
from __future__ import annotations

import csv
import json
import time
from typing import Any

from rag import retrieve

TEST_CASES_PATH = "test_cases.json"
REPORT_PATH = "retrieval_eval_report.csv"

# Giá trị "expected_source" đặc biệt: câu hỏi KHÔNG kỳ vọng tìm thấy chunk
# liên quan nào trong knowledge base (câu hỏi ngoài phạm vi quán).
NO_SOURCE = "none"

# 1. LOAD DỮ LIỆU TEST (dùng lại đúng test_cases.json của evaluate.py, để
# 2 file luôn đánh giá trên cùng 1 bộ câu hỏi, không lệch nhau).
def load_test_cases(path: str = TEST_CASES_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# 2. CÁC HÀM ĐO / TÍNH METRIC CHO TỪNG CÂU HỎI
def get_retrieved_sources(results: list[Any]) -> list[str]:
    """Lấy field 'type' (menu_item / menu_option / faq / promotion) từ
    metadata của từng chunk trả về."""
    sources = []
    for r in results:
        meta = getattr(r, "metadata", None) or {}
        sources.append(meta.get("type", "unknown"))
    return sources

def check_retriever_correct(
    expected_source: str, retrieved_sources: list[str], context: str
) -> bool:
    """Retriever Accuracy (1 câu hỏi) - giống hệt logic trong evaluate.py,
    tách riêng lại đây để file này chạy độc lập, không import chatbot.py
    (chatbot.py kéo theo llm.py -> cần API key, khởi tạo model... không
    cần thiết khi chỉ đo retrieval)."""
    if expected_source == NO_SOURCE:
        return context.strip() == ""
    return expected_source in retrieved_sources


def check_context_precision(
    expected_source: str, retrieved_sources: list[str]
) -> float | None:
    """Context Precision = số chunk đúng source / tổng số chunk trả về.
    None cho câu hỏi "none" (không có source đúng để so khớp)."""
    if expected_source == NO_SOURCE:
        return None
    if not retrieved_sources:
        return 0.0
    relevant = sum(1 for s in retrieved_sources if s == expected_source)
    return relevant / len(retrieved_sources)

def check_reciprocal_rank(expected_source: str, retrieved_sources: list[str]) -> float | None:
    """Reciprocal Rank = 1 / (vị trí xuất hiện đầu tiên của expected_source
    trong danh sách kết quả, tính từ 1). 0 nếu không xuất hiện.
    None cho câu hỏi "none" (không áp dụng khái niệm rank).

    Dùng để tính MRR (Mean Reciprocal Rank) ở phần tổng hợp - đo hybrid
    search có đẩy chunk ĐÚNG lên vị trí CAO HƠN hay không, chứ không chỉ
    "có xuất hiện hay không" như Retriever Accuracy."""
    if expected_source == NO_SOURCE:
        return None
    for rank, source in enumerate(retrieved_sources, start=1):
        if source == expected_source:
            return 1.0 / rank
    return 0.0

def check_false_positive(expected_source: str, context: str) -> bool | None:
    """Riêng cho câu hỏi expected_source == "none": True nếu retriever vẫn
    trả về context (đáng lẽ phải rỗng) - đây là dấu hiệu retriever "ép"
    trả kết quả không liên quan, rất quan trọng để không làm bot bịa
    thông tin. None cho các câu hỏi khác (không áp dụng)."""
    if expected_source != NO_SOURCE:
        return None
    return context.strip() != ""

# 3. VÒNG LẶP ĐÁNH GIÁ CHÍNH
def evaluate_single_case(case: dict) -> dict:
    question = case["question"]
    expected_source = case["expected_source"]

    t0 = time.perf_counter()
    result = retrieve(question)
    latency = time.perf_counter() - t0

    retrieved_sources = get_retrieved_sources(result["results"])
    retriever_correct = check_retriever_correct(
        expected_source, retrieved_sources, result["context"]
    )
    precision = check_context_precision(expected_source, retrieved_sources)
    reciprocal_rank = check_reciprocal_rank(expected_source, retrieved_sources)
    false_positive = check_false_positive(expected_source, result["context"])

    return {
        "question": question,
        "expected_source": expected_source,
        "retrieved_sources": retrieved_sources,
        "retriever_correct": retriever_correct,
        "context_precision": precision,
        "reciprocal_rank": reciprocal_rank,
        "false_positive": false_positive,
        "latency": latency,
    }

def run_evaluation() -> None:
    test_cases = load_test_cases()
    eval_results = [evaluate_single_case(case) for case in test_cases]

    write_report(eval_results)
    print_summary(eval_results)

# 4. OUTPUT: CSV + TERMINAL SUMMARY
def write_report(eval_results: list[dict], path: str = REPORT_PATH) -> None:
    fieldnames = [
        "Question",
        "Expected Source",
        "Retrieved Sources",
        "Retriever Correct",
        "Context Precision",
        "Reciprocal Rank",
        "False Positive (none case)",
        "Latency",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in eval_results:
            writer.writerow({
                "Question": r["question"],
                "Expected Source": r["expected_source"],
                "Retrieved Sources": ", ".join(r["retrieved_sources"]) or "(none)",
                "Retriever Correct": r["retriever_correct"],
                "Context Precision": (
                    "" if r["context_precision"] is None else round(r["context_precision"], 3)
                ),
                "Reciprocal Rank": (
                    "" if r["reciprocal_rank"] is None else round(r["reciprocal_rank"], 3)
                ),
                "False Positive (none case)": (
                    "" if r["false_positive"] is None else r["false_positive"]
                ),
                "Latency": round(r["latency"], 3),
            })
    print(f"Đã ghi báo cáo chi tiết vào {path}\n")

def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def print_summary(eval_results: list[dict]) -> None:
    total = len(eval_results)

    retriever_accuracy = _avg([1.0 if r["retriever_correct"] else 0.0 for r in eval_results])

    precisions = [r["context_precision"] for r in eval_results if r["context_precision"] is not None]
    context_precision = _avg(precisions)

    ranks = [r["reciprocal_rank"] for r in eval_results if r["reciprocal_rank"] is not None]
    mrr = _avg(ranks)

    fp_flags = [r["false_positive"] for r in eval_results if r["false_positive"] is not None]
    false_positive_rate = _avg([1.0 if fp else 0.0 for fp in fp_flags]) if fp_flags else None

    avg_latency = _avg([r["latency"] for r in eval_results])

    print("=" * 48)
    print("Retrieval-Only Evaluation Report (no LLM call)")
    print("=" * 48)
    print(f"Total Questions          : {total}")
    print(f"Retriever Accuracy       : {retriever_accuracy:.1%}")
    print(f"Context Precision        : {context_precision:.1%}")
    print(f"Mean Reciprocal Rank     : {mrr:.3f}")
    if false_positive_rate is not None:
        print(f"False Positive Rate      : {false_positive_rate:.1%}  "
              f"(trên {len(fp_flags)} câu 'none' - retriever lỡ trả context dù không nên)")
    print(f"Average Retrieval Latency: {avg_latency:.3f}s")
    print("=" * 48)
    print(
        "\nGợi ý: lưu lại các số trên (vd. đổi tên retrieval_eval_report.csv\n"
        "thành retrieval_eval_dense_baseline.csv) TRƯỚC khi đổi sang hybrid\n"
        "search, để có căn cứ so sánh khách quan sau khi implement xong."
    )

if __name__ == "__main__":
    run_evaluation()
