"""Public API models (request/response)."""

from enum import Enum

from pydantic import BaseModel, Field, field_validator

MAX_TEXT_LENGTH = 1000


class OrderRequest(BaseModel):
    text: str = Field(..., description="What the guest said, e.g. 'dvije margarite i jednu colu'")

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be empty")
        if len(value) > MAX_TEXT_LENGTH:
            raise ValueError(f"text must be at most {MAX_TEXT_LENGTH} characters")
        return value


class Intent(str, Enum):
    order = "order"  # guest is ordering
    question = "question"  # guest asks something (e.g. "imate li nešto bez mesa?")
    other = "other"  # greeting, unrelated or unintelligible text


class OrderItem(BaseModel):
    id: str
    quantity: int = Field(ge=1)
    name: str
    unit_price_eur: float
    line_total_eur: float


class UnavailableItem(BaseModel):
    text: str
    quantity: int = Field(ge=1)


class Suggestion(BaseModel):
    id: str
    name: str
    price_eur: float


class OrderResponse(BaseModel):
    items: list[OrderItem]
    unavailable: list[UnavailableItem]
    unclear: list[UnavailableItem] = []
    suggestions: list[Suggestion] = []
    intent: Intent
    total_eur: float
    needs_clarification: bool = False
    clarification: str | None = None
    degraded: bool = False  # True when Gemini was unavailable and the rule-based fallback answered


class ConversationRequest(OrderRequest):
    session_id: str | None = Field(
        default=None, max_length=64, description="Omit on the first message of a call; then send back the returned id."
    )


class ConversationResponse(OrderResponse):
    """`items` is the whole order so far; `unavailable`/`unclear`/`suggestions` refer to this message."""

    session_id: str
    turn: int
