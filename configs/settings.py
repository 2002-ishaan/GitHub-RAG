"""
configs/settings.py
────────────────────────────────────────────────────────────────
Central configuration for the GitHub Documentation Assistant.

ENDPOINT DETAILS (per professor's final project instructions):
    - LLM / chat    : https://rsm-8430-finalproject.bjlkeng.io/v1
    - Embeddings    : https://rsm-8430-a2.bjlkeng.io/v1
    - API key       : your student ID (read from ID.txt line 3)
    - LLM model     : IGNORED (server ignores model name)
    - Embedding     : BAAI/bge-base-en-v1.5
────────────────────────────────────────────────────────────────
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator
from loguru import logger

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
CONFIGS_DIR  = PROJECT_ROOT / "configs"
DATA_DIR     = PROJECT_ROOT / "data"
LOGS_DIR     = PROJECT_ROOT / "logs"


def _read_student_id() -> str:
    """
    Read student ID from ID.txt (line 3).
    Falls back to QWEN_API_KEY env var if ID.txt not found.
    """
    id_path = PROJECT_ROOT / "ID.txt"
    if id_path.exists():
        lines = id_path.read_text().strip().splitlines()
        if len(lines) >= 3 and lines[2].strip():
            return lines[2].strip()
    # Fallback
    return os.getenv("QWEN_API_KEY", "")


class Settings(BaseModel):
    """All runtime settings in one validated object."""

    # ── LLM — Final project chat endpoint ────────────────────────────────
    qwen_api_key:    str
    qwen_base_url:   str   = "https://rsm-8430-finalproject.bjlkeng.io/v1"
    llm_model:       str   = "IGNORED"
    llm_temperature: float = 0.0

    # ── Embeddings — A2 endpoint (per professor's instructions) ───────────
    embedding_base_url: str = "https://rsm-8430-a2.bjlkeng.io/v1"
    embedding_model:    str = "BAAI/bge-base-en-v1.5"

    # ── ChromaDB ──────────────────────────────────────────────────────────
    chroma_persist_dir:     str = str(DATA_DIR / "chroma_db")
    chroma_collection_name: str = "github_docs"

    # ── Retrieval ─────────────────────────────────────────────────────────
    top_k_retrieval: int = 10
    top_k_rerank:    int = 5

    # ── Agent persistence ─────────────────────────────────────────────────
    sqlite_db_path: str = str(DATA_DIR / "agent_state.db")

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_dir:   str = str(LOGS_DIR)

    @field_validator("qwen_api_key")
    @classmethod
    def api_key_must_exist(cls, v: str) -> str:
        if not v:
            raise ValueError(
                "\n\n❌  Student ID not found!\n"
                "    Make sure ID.txt exists in your project root with:\n"
                "      Line 1: Your Name\n"
                "      Line 2: your@email.com\n"
                "      Line 3: your_student_id\n"
            )
        return v


def load_settings() -> Settings:
    """Load and validate all settings."""
    student_id = _read_student_id()

    return Settings(
        qwen_api_key=student_id,
        qwen_base_url=os.getenv(
            "QWEN_BASE_URL",
            "https://rsm-8430-finalproject.bjlkeng.io/v1",
        ),
        llm_model=os.getenv("LLM_MODEL", "IGNORED"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        embedding_base_url=os.getenv(
            "EMBEDDING_BASE_URL",
            "https://rsm-8430-a2.bjlkeng.io/v1",
        ),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"),
        chroma_persist_dir=os.getenv(
            "CHROMA_PERSIST_DIR",
            str(DATA_DIR / "chroma_db"),
        ),
        chroma_collection_name=os.getenv(
            "CHROMA_COLLECTION_NAME",
            "github_docs",
        ),
        top_k_retrieval=int(os.getenv("TOP_K_RETRIEVAL", "10")),
        top_k_rerank=int(os.getenv("TOP_K_RERANK", "5")),
        sqlite_db_path=os.getenv(
            "SQLITE_DB_PATH",
            str(DATA_DIR / "agent_state.db"),
        ),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_dir=os.getenv("LOG_DIR", str(LOGS_DIR)),
    )


def load_prompts() -> dict:
    """Load prompt templates from the YAML file."""
    prompts_path = CONFIGS_DIR / "prompts.yaml"
    with open(prompts_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(settings: Settings) -> None:
    """Configure logging with loguru."""
    LOGS_DIR.mkdir(exist_ok=True)
    logger.remove()

    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=settings.log_level,
        format="<green>{time:HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan> - {message}",
        colorize=True,
    )

    logger.add(
        sink=f"{settings.log_dir}/agent.log",
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
        rotation="1 day",
        retention="7 days",
        compression="zip",
    )

    logger.info(
        f"GitHub Docs Agent | "
        f"endpoint={settings.qwen_base_url} | "
        f"embedding={settings.embedding_model}"
    )
