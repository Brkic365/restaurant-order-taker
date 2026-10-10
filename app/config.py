"""Runtime configuration, read from environment variables (and a local .env file)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MENU_PATH = Path(__file__).parent / "jelovnik.json"


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None
    gemini_model: str
    llm_timeout_s: float
    llm_max_attempts: int
    menu_path: Path
    fallback_enabled: bool
    session_ttl_s: float


def get_settings() -> Settings:
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
        llm_timeout_s=float(os.getenv("LLM_TIMEOUT_S", "20")),
        llm_max_attempts=int(os.getenv("LLM_MAX_ATTEMPTS", "2")),
        menu_path=Path(os.getenv("MENU_PATH", str(DEFAULT_MENU_PATH))),
        fallback_enabled=os.getenv("FALLBACK_ENABLED", "true").lower() in ("1", "true", "yes"),
        session_ttl_s=float(os.getenv("SESSION_TTL_S", "1800")),
    )
