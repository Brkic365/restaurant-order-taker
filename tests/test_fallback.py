"""The rule-based fallback used when Gemini is down."""

import pytest

from app.config import DEFAULT_MENU_PATH
from app.fallback import FallbackOrderParser
from app.llm import LLMBadOutput, LLMError
from app.menu import Menu
from app.service import build_response


@pytest.fixture(scope="module")
def run():
    menu = Menu.from_file(DEFAULT_MENU_PATH)
    parser = FallbackOrderParser()
    return lambda text, current=None: build_response(parser.parse(text, current), menu)


def ids(response):
    return {i.id: i.quantity for i in response.items}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("dvije margarite i jednu colu", {"margarita": 2, "coca_cola": 1}),
        ("jednu kapričozu i dva piva molim", {"capricciosa": 1, "pivo": 2}),
        ("četiri kole i dvije cezar salate", {"coca_cola": 4, "cezar_salata": 2}),
        ("2 quattro formaggi, 3 piva", {"quattro_formaggi": 2, "pivo": 3}),
        ("pizzu sa četiri sira i vodu", {"quattro_formaggi": 1, "mineralna_voda": 1}),
        ("Kapricoza x2 pls", {"capricciosa": 2}),
        ("dvije margarite tri piva", {"margarita": 2, "pivo": 3}),
        ("two margheritas and a coke please", {"margarita": 2, "coca_cola": 1}),
    ],
)
def test_simple_orders(run, text, expected):
    assert ids(run(text)) == expected


def test_unknown_item_with_quantity_is_unavailable(run):
    result = run("dva hamburgera i jednu margaritu")
    assert ids(result) == {"margarita": 1}
    assert [(u.text, u.quantity) for u in result.unavailable] == [("hamburgera", 2)]


def test_generic_pizza_is_unclear(run):
    result = run("jednu pizzu i colu")
    assert ids(result) == {"coca_cola": 1}
    assert result.unclear[0].text == "pizzu"


def test_meat_free_question(run):
    result = run("imate li nešto bez mesa?")
    assert result.items == []
    assert [s.id for s in result.suggestions] == ["margarita", "vegetariana", "quattro_formaggi", "mijesana_salata"]


def test_corrections_ask_to_repeat(run):
    result = run("tri margarite, ma ne, ipak dvije")
    assert result.needs_clarification


def test_nonsense_asks_to_repeat(run):
    result = run("dobar dan")
    assert result.items == [] and result.needs_clarification


def test_adds_to_current_order(run):
    assert ids(run("i jednu colu", {"margarita": 2})) == {"margarita": 2, "coca_cola": 1}


@pytest.mark.parametrize("error", [LLMError("timeout"), LLMBadOutput("garbage")])
def test_order_endpoint_degrades_instead_of_failing(client, with_fallback, error):
    with_fallback.error = error
    response = client.post("/order", json={"text": "dvije margarite i jednu colu"})
    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is True
    assert {i["id"]: i["quantity"] for i in body["items"]} == {"margarita": 2, "coca_cola": 1}
