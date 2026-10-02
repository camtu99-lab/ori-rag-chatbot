import os, sys, types
from pathlib import Path
import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


class LocalVectorStore(VectorStore):
    """Vector store trong bo nho, cosine similarity (cang cao cang giong)."""
    def __init__(self, embedding):
        self._emb = embedding
        self._docs = []
        self._mat = None

    @property
    def embeddings(self):
        return self._emb

    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
        s = cls(embedding)
        s.add_texts(texts, metadatas)
        return s

    def add_texts(self, texts, metadatas=None, **kwargs):
        metadatas = metadatas or [{} for _ in texts]
        return self.add_documents([Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)])

    def add_documents(self, documents, **kwargs):
        vecs = np.asarray(self._emb.embed_documents([d.page_content for d in documents]), dtype="float32")
        vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
        self._docs += list(documents)
        self._mat = vecs if self._mat is None else np.vstack([self._mat, vecs])
        return [str(i) for i in range(len(self._docs))]

    def similarity_search_with_score(self, query, k=4, **kwargs):
        if self._mat is None:
            return []
        q = np.asarray(self._emb.embed_query(query), dtype="float32")
        q /= (np.linalg.norm(q) + 1e-12)
        sims = self._mat @ q
        top = np.argsort(-sims)[:k]
        return [(self._docs[i], float(sims[i])) for i in top]

    def similarity_search(self, query, k=4, **kwargs):
        return [d for d, _ in self.similarity_search_with_score(query, k)]


def _register_vector_store():
    store = {"obj": None}

    def get_vector_store():
        if store["obj"] is None:
            from rag.embeddings import get_embeddings
            store["obj"] = LocalVectorStore(get_embeddings())
        return store["obj"]

    def reset_collection():
        store["obj"] = None

    mod = types.ModuleType("rag.vector_store")
    mod.get_vector_store = get_vector_store
    mod.reset_collection = reset_collection
    mod.LocalVectorStore = LocalVectorStore
    sys.modules["rag.vector_store"] = mod   # phai chay truoc khi import rag


_DEFAULTS = {
    "LLM_GATEWAY_TIMEOUT_SECONDS": "120",
    "LLM_CHAT_TEMPERATURE": "0.7",
    "LLM_CHAT_MAX_TOKENS": "300",
    "LLM_INTENT_TEMPERATURE": "0",
    "LLM_INTENT_MAX_TOKENS": "200",
    "CONTEXT_STORE_BACKEND": "memory",
    "CONTEXT_MAX_INPUT_TOKENS": "8192",
    "CONTEXT_RESERVED_OUTPUT_TOKENS": "768",
    "CONTEXT_MAX_HISTORY_TOKENS": "3200",
    "CONTEXT_MAX_EXTERNAL_TOKENS": "3200",
    "PYTHONUTF8": "1",
}


def configure_env(secrets=None):
    """Chuan bi bien moi truong khi chay ngoai notebook (vd Streamlit Cloud: khong co Gateway / .env).
    Tra ve chuoi loi (tieng Viet) neu thieu khoa API, nguoc lai tra ve None."""
    for k, v in (secrets or {}).items():
        if isinstance(v, (str, int, float, bool)):
            os.environ.setdefault(str(k), str(v))
    for k, v in _DEFAULTS.items():
        os.environ.setdefault(k, v)
    if not os.environ.get("LLM_GATEWAY_URL"):
        # Khong co LiteLLM Gateway -> goi thang Groq (API tuong thich OpenAI)
        key = os.environ.get("GROQ_API_KEY", "").strip()
        if not key or key == "placeholder":
            return ("Thiếu GROQ_API_KEY. Vào Manage app → Settings → Secrets và thêm dòng: "
                    'GROQ_API_KEY = "gsk_..."')
        model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        os.environ["LLM_GATEWAY_URL"] = "https://api.groq.com/openai/v1"
        os.environ["LITELLM_MASTER_KEY"] = key
        os.environ["LLM_GATEWAY_MODEL"] = model
        os.environ["LLM_GATEWAY_INTENT_MODEL"] = model
    return None


def _patch_ner():
    try:
        return _patch_ner_impl()
    except ImportError as e:
        return "NER: tat (thieu thu vien: " + str(e)[:60] + ")"
    except Exception as e:
        return "NER: tat (" + repr(e)[:80] + ")"


def _patch_ner_impl():
    import intent.phobert_runtime as pr
    test = "cho minh 2 ly ca phe den da size lon it duong nha"
    try:
        pr.get_phobert_runtime().extract(test)
        return "NER: pipeline chuan OK"
    except FileNotFoundError as e:
        return "NER: chua co mo hinh (" + str(e)[:80] + ")"
    except Exception:
        pass

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    class ManualNERPipeline:
        def __init__(self, model, tok):
            self.model, self.tok = model.eval(), tok
            self.dev = next(model.parameters()).device
        def __call__(self, text):
            words = text.split()
            ids, first = [self.tok.cls_token_id], []
            for w in words:
                pid = self.tok.convert_tokens_to_ids(self.tok.tokenize(w) or [self.tok.unk_token])
                first.append(len(ids)); ids += pid
            ids.append(self.tok.sep_token_id)
            with torch.no_grad():
                logits = self.model(input_ids=torch.tensor([ids], device=self.dev)).logits[0]
            tags = [self.model.config.id2label[int(logits[i].argmax())] for i in first]
            out, cur = [], None
            for w, t in zip(words, tags):
                if t == "O":
                    cur = None; continue
                typ = t[2:]
                if t.startswith("B-") or cur is None or cur["entity_group"] != typ:
                    cur = {"entity_group": typ, "word": w}; out.append(cur)
                else:
                    cur["word"] += " " + w
            return out

    def _build(model_dir):
        tok = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
        pr._configure_labels(model, model_dir)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        return pr.PhoBERTRuntime(model_dir, tok, model, ManualNERPipeline(model, tok))

    pr._get_runtime_for_dir.cache_clear()
    pr._build_runtime = _build
    try:
        pr.get_phobert_runtime().extract(test)
        return "NER: dung bo du doan thu cong"
    except Exception as e:
        return "NER: loi " + repr(e)[:120]


def setup(proj, use_reranker=True):
    proj = str(proj)
    os.chdir(proj)
    if proj not in sys.path:
        sys.path.insert(0, proj)
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(proj) / ".env")
    except Exception:
        pass

    _register_vector_store()

    import rag.hybrid_retriever as hr
    if not use_reranker:
        class _PassThroughReranker:
            def rerank_with_scores(self, query, documents, top_k):
                return [(d, 0.0) for d in list(documents)[:top_k]]
        hr.get_reranker = lambda: _PassThroughReranker()

    from rag.ingest import build_index
    build_index()

    ner_status = _patch_ner()

    import logging
    logging.basicConfig(level=logging.WARNING)
    from chatbot import ask, reset_history
    return ask, reset_history, ner_status
