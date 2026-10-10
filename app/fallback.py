"""Rule-based fallback parser, used only when Gemini is unreachable.

It is deliberately simple: it recognises menu items by word stems and Croatian
number words, and reports unknown items that come with a quantity. It does NOT
understand corrections, so every response it produces is marked ``degraded``
and, when the sentence looks like a correction, asks the guest to repeat.
A simple, honest answer is better than a 503 in the middle of a phone call.
"""

import re
import unicodedata

from app.llm import ParsedLine, ParsedOrder
from app.schemas import Intent

# Stems are matched against text without diacritics, so "kapričozu" -> "kapricozu".
ITEM_PATTERNS: dict[str, list[str]] = {
    "margarita": [r"margh?[ae]rit\w*"],
    "capricciosa": [r"[ck]apric\w*"],
    "diavola": [r"dj?iavol\w*"],
    "vegetariana": [r"vegetar\w*"],
    "quattro_formaggi": [r"quattro\w*", r"formagg\w*", r"(?:cetiri|4) sira"],
    "cezar_salata": [r"cezar\w*", r"caesar\w*"],
    "mijesana_salata": [r"mi?j?esan\w*"],
    "coca_cola": [r"coca\w*", r"koka\w*", r"coke", r"\b[ck]ol(?:a|u|e|om)\b"],
    "mineralna_voda": [r"mineraln\w*", r"\bvod(?:a|u|e|om)\b", r"water"],
    "pivo": [r"\bpiv(?:o|a|u|om)?\b", r"tocen\w*", r"\bbeers?\b"],
}

NUMBER_WORDS = {
    "jedan": 1, "jedna": 1, "jednu": 1, "jedno": 1, "jednog": 1, "jednom": 1,
    "dva": 2, "dvije": 2, "dvoje": 2, "par": 2,
    "tri": 3, "troje": 3, "cetiri": 4, "pet": 5, "sest": 6, "sedam": 7,
    "osam": 8, "devet": 9, "deset": 10,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
}
FILLER_WORDS = {"molim", "jos", "please", "pls", "hvala", "mi", "nam", "za", "sa", "s", "x"}
GENERIC_NAME = re.compile(r"(?:pizz|pic|salat)\w*")
SEPARATORS = re.compile(r",|;|\.|\bi\b|\bte\b|\bplus\b|\band\b|\ba\b")
CORRECTION_MARKERS = re.compile(r"\bma ne\b|\bipak\b|\bzapravo\b|\bumjesto\b|\bbez\b|\bne\b")
MEAT_FREE = re.compile(r"bez mesa|vegetarijansk\w*|vegan\w*|posn\w*")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _quantity(words: list[str]) -> int | None:
    for word in words:
        match = re.fullmatch(r"x?(\d+)x?", word)
        if match:
            return int(match.group(1))
        if word in NUMBER_WORDS:
            return NUMBER_WORDS[word]
    return None


class FallbackOrderParser:
    def __init__(self):
        self._patterns = {
            item_id: [re.compile(p) for p in patterns] for item_id, patterns in ITEM_PATTERNS.items()
        }

    def _find_items(self, segment: str) -> list[tuple[int, int, str]]:
        """All menu-item mentions in a segment as (start, end, menu_id), left to right."""
        found: list[tuple[int, int, str]] = []
        for item_id, patterns in self._patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(segment):
                    if not any(s < match.end() and match.start() < e for s, e, _ in found):
                        found.append((match.start(), match.end(), item_id))
        merged: list[tuple[int, int, str]] = []
        for start, end, item_id in sorted(found):
            # "quattro formaggi" matches two stems of the same item: count it once.
            if merged and merged[-1][2] == item_id and not segment[merged[-1][1]:start].strip():
                merged[-1] = (merged[-1][0], end, item_id)
            else:
                merged.append((start, end, item_id))
        return merged

    def parse(self, text: str, current_order: dict[str, int] | None = None) -> ParsedOrder:
        normalized = normalize(text)
        current_lines = [
            ParsedLine(heard=item_id, quantity=qty, menu_id=item_id, ambiguous=False)
            for item_id, qty in (current_order or {}).items()
        ]
        if MEAT_FREE.search(normalized) and "?" in text:
            return ParsedOrder(intent=Intent.question, lines=current_lines, asks_meat_free=True,
                               suggestion_ids=[], clarification=None)

        quantities = dict(current_order or {})
        unknown: list[ParsedLine] = []
        for segment in SEPARATORS.split(normalized):
            segment = segment.strip()
            if not segment:
                continue
            found = self._find_items(segment)
            if found:
                previous_end = 0
                for i, (start, end, item_id) in enumerate(found):
                    # The quantity is the number word between the previous item and this one
                    # ("dvije margarite"), or right after the last item ("margarita x2").
                    after = segment[end:] if i == len(found) - 1 else ""
                    words = re.findall(r"\w+", segment[previous_end:start] + " " + after)
                    quantities[item_id] = quantities.get(item_id, 0) + (_quantity(words) or 1)
                    previous_end = end
                continue
            words = re.findall(r"\w+", segment)
            qty = _quantity(words)
            if qty:  # "dva hamburgera": a quantity plus something we don't sell
                name = " ".join(w for w in words
                                if w not in NUMBER_WORDS and w not in FILLER_WORDS and not w.isdigit())
                if name:
                    unknown.append(ParsedLine(heard=name, quantity=qty, menu_id=None,
                                              ambiguous=bool(GENERIC_NAME.fullmatch(name))))

        lines = [ParsedLine(heard=i, quantity=q, menu_id=i, ambiguous=False) for i, q in quantities.items()]
        clarification = None
        if CORRECTION_MARKERS.search(normalized) or not (lines or unknown):
            clarification = "Nisam siguran da sam dobro razumio, možete li ponoviti narudžbu?"
        return ParsedOrder(
            intent=Intent.order if lines or unknown else Intent.other,
            lines=lines + unknown,
            asks_meat_free=False,
            suggestion_ids=[],
            clarification=clarification,
        )
