"""FastAPI entry point.

POST /order         one message in, structured order out (the task's endpoint)
POST /conversation  the same, but keeps the order across the messages of one call
GET  /              a small chat page for trying /conversation in the browser
"""

import logging
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.responses import FileResponse

from app.config import get_settings
from app.fallback import FallbackOrderParser
from app.llm import GeminiOrderParser, LLMBadOutput, LLMError, LLMQuotaExceeded, OrderParser, ParsedOrder
from app.menu import Menu, MenuItem
from app.schemas import (
    ConversationRequest,
    ConversationResponse,
    OrderItem,
    OrderRequest,
    OrderResponse,
)
from app.service import build_response, order_items
from app.sessions import SessionStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(
    title="Restaurant AI Order Taker",
    description="Extracts menu items and quantities from what a guest says, using Gemini.",
    version="1.1.0",
)


@lru_cache
def get_menu() -> Menu:
    return Menu.from_file(get_settings().menu_path)


@lru_cache
def _gemini_parser() -> GeminiOrderParser:
    settings = get_settings()
    return GeminiOrderParser(
        api_key=settings.gemini_api_key or "",
        model=settings.gemini_model,
        menu=get_menu(),
        timeout_s=settings.llm_timeout_s,
        max_attempts=settings.llm_max_attempts,
    )


def get_parser() -> OrderParser:
    if not get_settings().gemini_api_key:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not set on the server.")
    return _gemini_parser()


def get_fallback() -> OrderParser | None:
    return FallbackOrderParser() if get_settings().fallback_enabled else None


@lru_cache
def get_sessions() -> SessionStore:
    return SessionStore(ttl_s=get_settings().session_ttl_s)


def parse_with_fallback(
    text: str,
    parser: OrderParser,
    fallback: OrderParser | None,
    current_order: dict[str, int] | None = None,
) -> tuple[ParsedOrder, bool]:
    """Returns the parse and whether it came from the fallback (degraded mode)."""
    try:
        return parser.parse(text, current_order), False
    except (LLMError, LLMBadOutput) as exc:
        if fallback is not None:
            log.warning("Gemini failed (%s), answering with the rule-based fallback", type(exc).__name__)
            return fallback.parse(text, current_order), True
        if isinstance(exc, LLMQuotaExceeded):
            raise HTTPException(status_code=503, detail="The language model quota is exhausted, please try again later.")
        if isinstance(exc, LLMError):
            raise HTTPException(status_code=503, detail="The language model is unavailable, please try again.")
        raise HTTPException(status_code=502, detail="The language model returned an unusable answer.")


CHAT_PAGE = Path(__file__).parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def chat_page() -> FileResponse:
    return FileResponse(CHAT_PAGE)


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "llm_configured": bool(settings.gemini_api_key),
        "model": settings.gemini_model,
        "fallback_enabled": settings.fallback_enabled,
    }


@app.get("/menu", response_model=list[MenuItem])
def menu(menu: Menu = Depends(get_menu)) -> list[MenuItem]:
    return menu.items


@app.post("/order", response_model=OrderResponse)
def order(
    request: OrderRequest,
    parser: OrderParser = Depends(get_parser),
    fallback: OrderParser | None = Depends(get_fallback),
    menu: Menu = Depends(get_menu),
) -> OrderResponse:
    parsed, degraded = parse_with_fallback(request.text, parser, fallback)
    response = build_response(parsed, menu)
    response.degraded = degraded
    return response


@app.post("/conversation", response_model=ConversationResponse)
def conversation(
    request: ConversationRequest,
    parser: OrderParser = Depends(get_parser),
    fallback: OrderParser | None = Depends(get_fallback),
    menu: Menu = Depends(get_menu),
    sessions: SessionStore = Depends(get_sessions),
) -> ConversationResponse:
    if request.session_id:
        session = sessions.get(request.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Unknown or expired session_id; omit it to start a new one.")
    else:
        session = sessions.create()

    parsed, degraded = parse_with_fallback(request.text, parser, fallback, session.order)
    result = build_response(parsed, menu, previous_order=session.order)

    session.order = {item.id: item.quantity for item in result.items}
    session.turns += 1
    sessions.save(session)
    return ConversationResponse(
        **result.model_dump(exclude={"degraded"}), degraded=degraded, session_id=session.id, turn=session.turns
    )


@app.get("/conversation/{session_id}", response_model=list[OrderItem])
def conversation_order(
    session_id: str, menu: Menu = Depends(get_menu), sessions: SessionStore = Depends(get_sessions)
) -> list[OrderItem]:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown or expired session_id.")
    return order_items(session.order, menu)


@app.delete("/conversation/{session_id}", status_code=204)
def end_conversation(session_id: str, sessions: SessionStore = Depends(get_sessions)) -> Response:
    if not sessions.delete(session_id):
        raise HTTPException(status_code=404, detail="Unknown or expired session_id.")
    return Response(status_code=204)
