# Ori – Trợ lý hội thoại bán hàng (RAG + Tool Calling)

Chạy lại project `posora-kb-assistant` trên Google Colab, không cần Redis Cloud
(vector store cục bộ bằng NumPy). Gồm 3 notebook và một ứng dụng Streamlit.

## Cấu trúc

| Đường dẫn | Nội dung |
|---|---|
| `posora-kb-assistant/` | Mã nguồn và dữ liệu gốc của chatbot (đã bỏ `pass.txt` và model giọng nói `.onnx`) |
| `notebooks/Ori_RAG_Chatbot_Base.ipynb` | Notebook gốc, chat bằng vòng lặp `input()` |
| `notebooks/Ori_RAG_Chatbot_Gradio.ipynb` | Thêm giao diện Gradio |
| `notebooks/Ori_RAG_Chatbot_Streamlit.ipynb` | Thêm giao diện Streamlit (mở link qua cloudflared) |
| `streamlit_app/` | `app.py` và `ori_bootstrap.py`, cùng mã mà notebook Streamlit tự ghi ra |

## Chạy trên Colab

1. Nén thư mục mã nguồn thành đúng tên mà notebook tìm:
   ```bash
   zip -r Ma__nguo__n_va__Du___lie__u.zip posora-kb-assistant
   ```
2. Mở một notebook trong `notebooks/` trên Colab, chọn Runtime **T4 GPU**.
3. Chạy theo hướng dẫn trong notebook. Tải file zip ở bước 1 lên khi được hỏi,
   và nhập `GROQ_API_KEY` (hoặc đổi `BACKEND` sang `gemini` / `ollama`).

## Lưu ý

- Zip không chứa trọng số PhoBERT nên Bước 4 huấn luyện lại mô hình NER bằng dữ liệu sinh từ menu.
  F1 trên tập kiểm thử sinh cùng nguồn sẽ rất cao, không dùng để kết luận khả năng khái quát.
- **Không đưa khoá API hay file `.env` lên repo.** Khoá được nhập khi chạy notebook.
- Link `gradio.live` / `trycloudflare.com` là công khai và tạm thời; ai có link đều dùng được
  chatbot bằng khoá API của bạn.

## Chạy trên Streamlit Cloud

1. Deploy với *Main file path* = `streamlit_app/app.py`. `requirements.txt` ở thư mục gốc được cài tự động.
2. Vào **Manage app → Settings → Secrets** và thêm (không commit khoá này lên repo):
   ```toml
   GROQ_API_KEY = "gsk_..."
   # tuỳ chọn:
   # GROQ_MODEL = "llama-3.3-70b-versatile"
   ```
3. Chế độ này gọi thẳng Groq (không có LiteLLM Gateway), tắt reranker và **không có NER PhoBERT**
   (repo không chứa trọng số), nên chatbot trả lời bằng đường RAG/LLM; việc đặt món có thể kém chính xác hơn bản Colab.
4. Mô hình embedding `multilingual-e5-large` (~2,2 GB) có thể vượt giới hạn bộ nhớ của Streamlit Cloud miễn phí.
   Nếu app bị khởi động lại hoặc báo hết tài nguyên, hãy chạy trên Colab hoặc Hugging Face Spaces.
