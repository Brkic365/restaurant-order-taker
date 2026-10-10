"""Menu loading. The menu is the single source of truth for valid item ids."""

import json
from pathlib import Path

from pydantic import BaseModel, Field

# Categories that count as food when a guest asks for a recommendation.
FOOD_CATEGORIES = {"pizza", "salata"}


class MenuItem(BaseModel):
    id: str
    naziv: str
    kategorija: str
    cijena_eur: float = Field(ge=0)
    bez_mesa: bool


class Menu:
    def __init__(self, items: list[MenuItem]):
        if not items:
            raise ValueError("Menu is empty")
        self.items = items
        self.by_id = {item.id: item for item in items}
        if len(self.by_id) != len(items):
            raise ValueError("Menu contains duplicate ids")

    @classmethod
    def from_file(cls, path: Path) -> "Menu":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([MenuItem.model_validate(entry) for entry in raw])

    def ids(self) -> list[str]:
        return list(self.by_id)

    def get(self, item_id: str) -> MenuItem | None:
        return self.by_id.get(item_id)

    def meat_free_food(self) -> list[MenuItem]:
        return [i for i in self.items if i.bez_mesa and i.kategorija in FOOD_CATEGORIES]

    def as_prompt_table(self) -> str:
        lines = ["id | naziv | kategorija | bez_mesa"]
        for i in self.items:
            lines.append(f"{i.id} | {i.naziv} | {i.kategorija} | {'da' if i.bez_mesa else 'ne'}")
        return "\n".join(lines)
