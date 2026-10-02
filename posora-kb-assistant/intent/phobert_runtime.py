"""Production PhoBERT token-classification runtime.
Label names are discovered from model metadata. BIO/BILOU prefixes are
normalized generically, so newly trained labels flow through without editing
this runtime.
"""
from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MODEL_DIRS = (
    _PROJECT_ROOT / "ner_model" / "best_phobert_ner",
    _PROJECT_ROOT / "ner_model",
)

def _candidate_model_dirs(extra_dirs: Iterable[str | Path] | None = None) -> tuple[Path, ...]:
    candidates: list[Path] = []
    env_dir = os.getenv("PHOBERT_NER_MODEL_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    if extra_dirs:
        candidates.extend(Path(path).expanduser() for path in extra_dirs)
    candidates.extend(_DEFAULT_MODEL_DIRS)

    result: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        resolved = path.resolve()
        key = os.path.normcase(str(resolved))
        if key not in seen:
            seen.add(key)
            result.append(resolved)
    return tuple(result)

def _resolve_model_dir(extra_dirs: Iterable[str | Path] | None = None) -> Path:
    candidates = _candidate_model_dirs(extra_dirs)
    for path in candidates:
        if path.is_dir() and (path / "config.json").is_file():
            return path
    searched = "\\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Không tìm thấy PhoBERT NER model. Đã kiểm tra:\\n  - {searched}")

def _mapping_from_labels_data(data: object) -> dict[int, str] | None:
    if isinstance(data, list):
        return {i: str(label) for i, label in enumerate(data)}
    if not isinstance(data, dict):
        return None

    for key in ("labels", "id2label", "label2id"):
        nested = data.get(key)
        if nested is None:
            continue
        if key == "labels" and isinstance(nested, list):
            return {i: str(label) for i, label in enumerate(nested)}
        if isinstance(nested, dict):
            parsed = _mapping_from_labels_data(nested)
            if parsed:
                return parsed

    if data and all(str(key).isdigit() for key in data):
        return {int(key): str(value) for key, value in data.items()}
    if data and all(isinstance(value, int) for value in data.values()):
        return {int(value): str(key) for key, value in data.items()}
    return None

def _load_external_labels(model_dir: Path) -> dict[int, str] | None:
    path = model_dir / "labels.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Không đọc được labels.json (%s): %s", path, exc)
        return None
    return _mapping_from_labels_data(data)

def _has_meaningful_config_labels(model: Any) -> bool:
    id2label = getattr(model.config, "id2label", None)
    if not isinstance(id2label, dict) or not id2label:
        return False
    labels = [str(value) for value in id2label.values()]
    return not all(label.upper().startswith("LABEL_") for label in labels)

def _configure_labels(model: Any, model_dir: Path) -> None:
    external = _load_external_labels(model_dir)
    if external:
        model.config.id2label = dict(sorted(external.items()))
        model.config.label2id = {label: idx for idx, label in model.config.id2label.items()}
        return
    if not _has_meaningful_config_labels(model):
        raise RuntimeError("Model không có label mapping hợp lệ trong config.json/labels.json")

def _normalize_entity_name(label: str) -> str | None:
    label = str(label).strip()
    if not label or label.upper() == "O":
        return None
    upper = label.upper()
    for prefix in ("B-", "I-", "L-", "U-", "S-", "E-"):
        if upper.startswith(prefix):
            label = label[len(prefix):]
            break
    normalized = label.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized or None

def _collapse(values: list[str]) -> str | list[str]:
    # Preserve multiple spans instead of silently dropping a second topping/etc.
    return values[0] if len(values) == 1 else values

@dataclass(slots=True)
class PhoBERTRuntime:
    model_dir: Path
    tokenizer: object
    model: object
    pipeline: object

    def extract(self, text: str) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            return {}

        grouped: dict[str, list[str]] = {}
        for item in self.pipeline(text):
            key = _normalize_entity_name(item.get("entity_group") or item.get("entity"))
            if key is None:
                continue
            word = str(item.get("word", "")).strip()
            if word:
                grouped.setdefault(key, []).append(word)
        return {key: _collapse(values) for key, values in grouped.items() if values}

def _build_runtime(model_dir: Path) -> PhoBERTRuntime:
    try:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
    except ImportError as exc:
        raise RuntimeError("Thiếu transformers/torch cho PhoBERT NER") from exc

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    _configure_labels(model, model_dir)

    ner_pipeline = pipeline(
        task="token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1,
    )
    return PhoBERTRuntime(model_dir, tokenizer, model, ner_pipeline)

@lru_cache(maxsize=4)
def _get_runtime_for_dir(model_dir: str) -> PhoBERTRuntime:
    return _build_runtime(Path(model_dir))

def get_phobert_runtime(candidate_dirs: Iterable[str | Path] | None = None) -> PhoBERTRuntime:
    model_dir = _resolve_model_dir(candidate_dirs)
    return _get_runtime_for_dir(str(model_dir))
