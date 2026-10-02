import os, sys, uuid
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
import ori_bootstrap

st.set_page_config(page_title="Ori - Trợ lý bán hàng", page_icon="☕")

PROJ = os.environ["ORI_PROJ"]
USE_RERANKER = os.environ.get("ORI_USE_RERANKER", "1") == "1"


@st.cache_resource(show_spinner="Đang nạp mô hình và tri thức (lần đầu mất vài phút)...")
def load():
    return ori_bootstrap.setup(PROJ, USE_RERANKER)


ask, reset_history, ner_status = load()

if "sid" not in st.session_state:
    st.session_state.sid = "st-" + uuid.uuid4().hex[:8]
    st.session_state.msgs = []
    reset_history(st.session_state.sid)


def _reset():
    reset_history(st.session_state.sid)
    st.session_state.msgs = []


EXAMPLES = [
    "Quán có món cà phê muối không, giá bao nhiêu?",
    "Cho mình 2 ly Cà Phê Đen Đá size L",
    "Xem giỏ hàng giúp mình",
    "Thanh toán",
]

with st.sidebar:
    st.header("Ori")
    st.button("🗑️ Xoá hội thoại & giỏ hàng", on_click=_reset, use_container_width=True)
    st.caption("Câu hỏi mẫu")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex
    st.caption(ner_status)

st.title("☕ Ori – Trợ lý hội thoại bán hàng")

for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Nhập tin nhắn...") or st.session_state.pop("pending", None)
if prompt:
    st.session_state.msgs.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Ori đang trả lời..."):
            try:
                reply = ask(prompt, session_id=st.session_state.sid)
            except Exception as e:
                reply = f"⚠️ Lỗi: {e}"
        st.markdown(reply)
    st.session_state.msgs.append({"role": "assistant", "content": reply})
