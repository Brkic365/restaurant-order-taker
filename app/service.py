"""Turns the model's raw parse into a validated API response.

This is the safety net: whatever the model says, the response only ever
contains ids that exist on the menu, positive integer quantities, and every
item the guest mentioned ends up somewhere (ordered, unavailable or unclear).
"""

from app.llm import ParsedOrder
from app.menu import Menu
from app.schemas import Intent, OrderItem, OrderResponse, Suggestion, UnavailableItem

MAX_QUANTITY = 99


def build_response(
    parsed: ParsedOrder, menu: Menu, previous_order: dict[str, int] | None = None
) -> OrderResponse:
    """``previous_order`` is the order so far in a conversation (None for a single message)."""
    ordered: dict[str, int] = {}
    unavailable: dict[str, UnavailableItem] = {}
    unclear: dict[str, UnavailableItem] = {}
    clarification = parsed.clarification
    flagged_quantity = False

    for line in parsed.lines:
        if line.quantity <= 0:  # e.g. an item the guest cancelled
            continue
        quantity = line.quantity
        if quantity > MAX_QUANTITY:
            quantity = MAX_QUANTITY
            flagged_quantity = True

        heard = (line.heard or line.menu_id or "").strip()
        item = menu.get(line.menu_id) if line.menu_id else None

        if item is not None:
            ordered[item.id] = ordered.get(item.id, 0) + quantity
        else:
            # Not on the menu (or the model invented an id): keep what the guest said.
            target = unclear if line.ambiguous else unavailable
            key = heard.lower()
            if key in target:
                target[key].quantity += quantity
            else:
                target[key] = UnavailableItem(text=heard or "?", quantity=quantity)

    items = order_items(ordered, menu)

    if parsed.asks_meat_free:
        # Deterministic: derived from the menu's bez_mesa flag, not from the model.
        suggested = menu.meat_free_food()
    else:
        suggested = [menu.by_id[i] for i in dict.fromkeys(parsed.suggestion_ids) if i in menu.by_id]
    suggestions = [Suggestion(id=i.id, name=i.naziv, price_eur=i.cijena_eur) for i in suggested]

    if flagged_quantity and not clarification:
        clarification = f"Količina je ograničena na {MAX_QUANTITY}, molim potvrdite narudžbu."
    if unclear and not clarification:
        clarification = "Možete li pojasniti: " + ", ".join(u.text for u in unclear.values()) + "?"

    intent = parsed.intent
    if items and intent == Intent.other:
        intent = Intent.order
    if previous_order is not None and intent != Intent.order:
        # A question or small talk mid-call must never change the order, even if the
        # model forgot to repeat it.
        items = order_items(previous_order, menu)

    return OrderResponse(
        items=items,
        unavailable=list(unavailable.values()),
        unclear=list(unclear.values()),
        suggestions=suggestions,
        intent=intent,
        total_eur=round(sum(i.line_total_eur for i in items), 2),
        needs_clarification=bool(unclear) or flagged_quantity or bool(parsed.clarification),
        clarification=clarification,
    )


def order_items(quantities: dict[str, int], menu: Menu) -> list[OrderItem]:
    return [
        OrderItem(
            id=item_id,
            quantity=qty,
            name=menu.by_id[item_id].naziv,
            unit_price_eur=menu.by_id[item_id].cijena_eur,
            line_total_eur=round(menu.by_id[item_id].cijena_eur * qty, 2),
        )
        for item_id, qty in quantities.items()
        if item_id in menu.by_id
    ]
