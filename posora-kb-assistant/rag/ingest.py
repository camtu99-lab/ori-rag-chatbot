"""
Đọc dữ liệu (menu.json, menu.md, faq.md, promotions.md), chunk theo cấu trúc
ngữ nghĩa và nạp vào Vector Database thông qua LangChain VectorStore.
Vector Database cụ thể (Redis Cloud, Qdrant, ...) được cấu hình hoàn toàn
trong vector_store.py. File này không phụ thuộc vào bất kỳ Vector DB cụ thể nào.
"""

import os
import re
import json
from langchain_core.documents import Document
from .vector_store import get_vector_store, reset_collection

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def _path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)

# 1. CHUNKING - mỗi loại file có cấu trúc riêng nên tách riêng logic
def chunk_menu_items(filename: str = "menu.json") -> list[dict]:
    """Mỗi món trong menu.json -> 1 chunk, kèm metadata giá/danh mục
    để sau này có thể lọc (filter) theo category hoặc khoảng giá."""
    items = json.load(open(_path(filename), encoding="utf-8"))
    chunks = []
    for it in items:
        if it["price_m"] == it["price_l"]:
            text = (
                f"Món: {it['name']} (Danh mục: {it['category']}). "
                f"Giá: {it['price_m']:,} VNĐ. "
                f"Mô tả: {it['description']}")
        else:
            text = (
                f"Món: {it['name']} (Danh mục: {it['category']}). "
                f"Giá Size M: {it['price_m']:,} VNĐ, "
                f"Size L: {it['price_l']:,} VNĐ. "
                f"Mô tả: {it['description']}")
        chunks.append({
            "text": text,
            "source": "menu.json",
            "type": "menu_item",
            "name": it["name"],
            "category": it["category"],
            "price_m": it["price_m"],
            "price_l": it["price_l"],
        })
    return chunks

def chunk_menu_customization(filename: str = "menu.md") -> list[dict]:
    """Tách chunk các tùy chọn chung (đường, đá, topping...) từ menu.md, loại trừ các mục món đã có trong menu.json.
    - Phân loại bằng cấu trúc Markdown: Bỏ qua danh sách đánh số ("N. **Tên món**") và chỉ lấy danh sách bullet ("* **Nhãn:**").
    - Chunking mịn theo từng bullet cấp cha (kèm heading ngữ cảnh) để tránh loãng vector; giữ nguyên các dòng con thụt lề.
    """
    text = open(_path(filename), encoding="utf-8").read()
    sections = re.split(r"\n(?=## )", text)

    chunks = []
    for sec in sections:
        sec = sec.strip()
        heading_match = re.match(r"## (.+)", sec)
        # Bỏ qua phần mở đầu trước heading '## ' đầu tiên (tiêu đề H1/lời giới thiệu) do không chứa cấu trúc tùy chọn.
        if not sec or heading_match is None:
            continue
        # Bỏ section liệt kê món (xem giải thích ở docstring) - phát hiện thuần theo cấu trúc list đánh số, không hardcode số La Mã.
        if re.search(r"\n\d+\.\s+\*\*", sec):
            continue

        heading = heading_match.group(1)
        items = re.split(r"\n(?=\*\s+\*\*)", sec)
        for item in items:
            item = item.strip()
            # Bỏ mẩu chỉ gồm đúng 1 dòng heading Markdown, không có nội dung
            # theo sau (cùng logic lọc heading-only đã dùng ở chunk_promotions).
            if not item or re.fullmatch(r"#{1,6}\s+.+", item):
                continue
            prefixed = item if item.startswith("##") else (
                f"[{heading}]\n{item}" if heading else item
            )
            chunks.append({
                "text": prefixed,
                "source": "menu.md",
                "type": "menu_option",
            })
    return chunks

def chunk_faq(filename: str = "faq.md") -> list[dict]:
    """Tách theo từng cặp Q:/A: - mỗi cặp là 1 chunk độc lập, đủ ngữ cảnh."""
    text = open(_path(filename), encoding="utf-8").read()
    blocks = re.split(r"\n(?=Q:)", text)
    chunks = []
    for b in blocks:
        b = b.strip()
        if not b.startswith("Q:"):
            continue
        chunks.append({"text": b, "source": "faq.md", "type": "faq"})
    return chunks

def chunk_promotions(filename: str = "promotions.md") -> list[dict]:
    """Chia nhỏ promotions.md theo từng section (##) và từng mục con (CTKM/hạng thành viên), kèm heading để giữ ngữ cảnh.
    - Hỗ trợ cả 2 định dạng mục: Đánh số (`1. **...**`) và gạch đầu dòng (`- **...**`) để tránh gộp nguyên section làm loãng vector.
    - Giữ nguyên các dòng con thụt lề (sub-bullets) trong chunk cha tương ứng.
    """
    text = open(_path(filename), encoding="utf-8").read()
    sections = re.split(r"\n(?=## )", text)
    chunks = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        heading_match = re.match(r"## (.+)", sec)
        heading = heading_match.group(1) if heading_match else ""

        items = re.split(r"\n(?=\d+\.\s+\*\*|-\s+\*\*)", sec)
        for item in items:
            item = item.strip()
            # Loại bỏ chunk rác chỉ chứa DUY NHẤT 1 dòng heading (H1-H6) không có nội dung kèm theo.
            # Dùng regex (không cờ re.S) để tránh lọt heading cấp H2/H3 vào vector store mà vẫn giữ nguyên chunk đa dòng hợp lệ.
            if not item or re.fullmatch(r"#{1,6}\s+.+", item):
                continue
            # Tránh lặp heading 2 lần khi item chính là đoạn mở đầu section
            prefixed = item if item.startswith("##") else (
                f"[{heading}]\n{item}" if heading else item
            )
            chunks.append({
                "text": prefixed,
                "source": "promotions.md",
                "type": "promotion",
            })
    return chunks

# 2. CHUYỂN CHUNK -> LANGCHAIN DOCUMENT
def _chunks_to_documents(chunks: list[dict]) -> list[Document]:
    """Chuyển đổi dict chunk sang đối tượng chuẩn `langchain_core.documents.Document`.
    Giữ nguyên toàn bộ schema metadata (source, type, name, category, price_m, price_l) 
    và gán trực tiếp chuỗi văn bản vào `page_content`, loại bỏ sự trùng lặp text trong metadata cũ.
    """
    documents = []
    for chunk in chunks:
        metadata = {key: value for key, value in chunk.items() if key != "text"}
        documents.append(Document(page_content=chunk["text"], metadata=metadata))
    return documents

# 3. NẠP VÀO VECTOR STORE (qua LangChain VectorStore, không gọi QdrantClient trực tiếp)
def build_index() -> None:
    all_chunks = (
        chunk_menu_items()
        + chunk_menu_customization()
        + chunk_faq()
        + chunk_promotions()
    )
    documents = _chunks_to_documents(all_chunks)

    # Xoá sạch collection cũ trước khi nạp lại - tương đương hành vi cũ
    # (client.delete_collection() rồi add() lại), tránh dữ liệu cũ sót lại.
    reset_collection()
    vector_store = get_vector_store()

    # add_documents() của LangChain tự động: embed từng Document bằng
    # embedding model đã cấu hình trong vector_store.py, rồi upsert vào
    # Vector DB - tương đương những gì client.add() làm trước đây, chỉ khác
    # là đi qua interface chuẩn của LangChain thay vì API riêng của Qdrant.
    vector_store.add_documents(documents, ids=list(range(len(documents))))

    print(f"Đã nạp {len(all_chunks)} chunks vào Vector Store:")
    by_type: dict[str, int] = {}
    for c in all_chunks:
        by_type[c["type"]] = by_type.get(c["type"], 0) + 1
    for t, n in by_type.items():
        print(f"  - {t}: {n} chunks")

if __name__ == "__main__":
    build_index()
