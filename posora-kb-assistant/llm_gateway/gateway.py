from __future__ import annotations
import os
import shutil
import subprocess
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().with_name("config.yaml")
ENV_PATH = ROOT_DIR / ".env"

def _load_environment() -> None:
    if not ENV_PATH.is_file():
        raise FileNotFoundError(f"Không tìm thấy file .env: {ENV_PATH}")

    load_dotenv(ENV_PATH, override=False)

    required = (
        "LITELLM_MASTER_KEY",
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
        "GATEWAY_GROQ_MODEL",
        "GATEWAY_GEMINI_MODEL",
        "GATEWAY_LOCAL_MODEL",
        "GATEWAY_LOCAL_API_BASE",
    )

    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError("Thiếu biến môi trường: " + ", ".join(missing))

def _find_litellm() -> str:
    executable = shutil.which("litellm")
    if not executable:
        raise RuntimeError("Không tìm thấy lệnh 'litellm'. " "Hãy cài: pip install 'litellm[proxy]'")
    return executable

def main() -> None:
    _load_environment()

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Không tìm thấy config Gateway: {CONFIG_PATH}")

    command = [
        _find_litellm(),
        "--config",
        str(CONFIG_PATH),
        "--port",
        os.getenv("LLM_GATEWAY_PORT", "4000"),
    ]

    print(f"Starting LLM Gateway with: {CONFIG_PATH}")
    subprocess.run(command, cwd=ROOT_DIR, check=True)

if __name__ == "__main__":
    main()
