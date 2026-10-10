import pytest
from fastapi.testclient import TestClient

from app.llm import ParsedLine, ParsedOrder
from app.fallback import FallbackOrderParser
from app.main import app, get_fallback, get_parser, get_sessions
from app.sessions import SessionStore
from app.schemas import Intent


def line(heard, quantity, menu_id=None, ambiguous=False):
    return ParsedLine(heard=heard, quantity=quantity, menu_id=menu_id, ambiguous=ambiguous)


def parsed(lines=(), intent=Intent.order, asks_meat_free=False, suggestion_ids=(), clarification=None):
    return ParsedOrder(
        intent=intent,
        lines=list(lines),
        asks_meat_free=asks_meat_free,
        suggestion_ids=list(suggestion_ids),
        clarification=clarification,
    )


class FakeParser:
    """Stands in for Gemini: returns a canned parse, or raises a canned error."""

    def __init__(self):
        self.result = parsed()
        self.error = None
        self.calls = []
        self.orders_seen = []

    def parse(self, text, current_order=None):
        self.calls.append(text)
        self.orders_seen.append(dict(current_order) if current_order else None)
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def fake_parser():
    """Gemini replaced by FakeParser; fallback off unless a test enables it."""
    fake = FakeParser()
    store = SessionStore()
    app.dependency_overrides[get_parser] = lambda: fake
    app.dependency_overrides[get_fallback] = lambda: None
    app.dependency_overrides[get_sessions] = lambda: store
    yield fake
    app.dependency_overrides.clear()


@pytest.fixture
def with_fallback(fake_parser):
    app.dependency_overrides[get_fallback] = lambda: FallbackOrderParser()
    return fake_parser


@pytest.fixture
def client():
    return TestClient(app)
