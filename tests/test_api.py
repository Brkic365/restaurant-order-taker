"""API behaviour with the model mocked out: validation, errors and post-processing."""

import pytest

from app.llm import LLMBadOutput, LLMError
from app.schemas import Intent
from tests.conftest import line, parsed


def ids(body):
    return {i["id"]: i["quantity"] for i in body["items"]}


# --- The five examples from the task, as the model would parse them ---------


def test_two_margaritas_and_a_cola(client, fake_parser):
    fake_parser.result = parsed([line("margarita", 2, "margarita"), line("cola", 1, "coca_cola")])
    body = client.post("/order", json={"text": "dvije margarite i jednu colu"}).json()
    assert ids(body) == {"margarita": 2, "coca_cola": 1}
    assert body["unavailable"] == []
    assert body["total_eur"] == 21.0


def test_capricciosa_and_beers(client, fake_parser):
    fake_parser.result = parsed([line("kapričoza", 1, "capricciosa"), line("pivo", 2, "pivo")])
    body = client.post("/order", json={"text": "jednu kapričozu i dva piva molim"}).json()
    assert ids(body) == {"capricciosa": 1, "pivo": 2}


def test_hamburger_is_unavailable_not_substituted(client, fake_parser):
    fake_parser.result = parsed([line("hamburger", 2), line("margarita", 1, "margarita")])
    body = client.post("/order", json={"text": "dva hamburgera i jednu margaritu"}).json()
    assert ids(body) == {"margarita": 1}
    assert body["unavailable"] == [{"text": "hamburger", "quantity": 2}]


def test_meat_free_question_suggests_from_menu_without_ordering(client, fake_parser):
    # Even if the model suggests wrong ids, suggestions come from the menu's bez_mesa flag.
    fake_parser.result = parsed(
        intent=Intent.question, asks_meat_free=True, suggestion_ids=["diavola", "pivo"]
    )
    body = client.post("/order", json={"text": "imate li nešto bez mesa za nas dvoje?"}).json()
    assert body["items"] == []
    assert body["intent"] == "question"
    assert [s["id"] for s in body["suggestions"]] == [
        "margarita", "vegetariana", "quattro_formaggi", "mijesana_salata"
    ]


def test_correction_keeps_final_quantity(client, fake_parser):
    fake_parser.result = parsed([line("margarita", 2, "margarita"), line("cola", 1, "coca_cola")])
    body = client.post("/order", json={"text": "tri margarite i colu, ma ne, ipak dvije margarite"}).json()
    assert ids(body) == {"margarita": 2, "coca_cola": 1}


# --- Guarding against model mistakes -----------------------------------------


def test_invented_menu_id_goes_to_unavailable(client, fake_parser):
    fake_parser.result = parsed([line("lazanje", 1, "lasagne")])
    body = client.post("/order", json={"text": "jedne lazanje"}).json()
    assert body["items"] == []
    assert body["unavailable"] == [{"text": "lazanje", "quantity": 1}]


def test_duplicate_lines_are_merged(client, fake_parser):
    fake_parser.result = parsed([line("margarita", 1, "margarita"), line("margaritu", 2, "margarita")])
    body = client.post("/order", json={"text": "margaritu i još dvije margarite"}).json()
    assert ids(body) == {"margarita": 3}


def test_zero_or_negative_quantities_are_dropped(client, fake_parser):
    fake_parser.result = parsed([line("cola", 0, "coca_cola"), line("pivo", -1, "pivo")])
    body = client.post("/order", json={"text": "colu, ma ne, bez cole"}).json()
    assert body["items"] == []


def test_huge_quantity_is_capped_and_flagged(client, fake_parser):
    fake_parser.result = parsed([line("pivo", 5000, "pivo")])
    body = client.post("/order", json={"text": "5000 piva"}).json()
    assert ids(body) == {"pivo": 99}
    assert body["needs_clarification"] is True


def test_ambiguous_item_goes_to_unclear(client, fake_parser):
    fake_parser.result = parsed([line("pizza", 1, None, ambiguous=True)], clarification="Koju pizzu želite?")
    body = client.post("/order", json={"text": "jednu pizzu"}).json()
    assert body["unclear"] == [{"text": "pizza", "quantity": 1}]
    assert body["unavailable"] == []
    assert body["needs_clarification"] is True
    assert body["clarification"] == "Koju pizzu želite?"


def test_suggestion_ids_not_on_menu_are_dropped(client, fake_parser):
    fake_parser.result = parsed(intent=Intent.question, suggestion_ids=["pivo", "fanta"])
    body = client.post("/order", json={"text": "što imate za piti?"}).json()
    assert [s["id"] for s in body["suggestions"]] == ["pivo"]


def test_line_prices(client, fake_parser):
    fake_parser.result = parsed([line("diavola", 3, "diavola")])
    item = client.post("/order", json={"text": "tri diavole"}).json()["items"][0]
    assert item["unit_price_eur"] == 11.5
    assert item["line_total_eur"] == 34.5
    assert item["name"] == "Pizza Diavola (ljuta kobasica)"


# --- Input validation ----------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [{"text": ""}, {"text": "   "}, {"text": "x" * 1001}, {}, {"text": None}, {"text": 123}],
)
def test_bad_input_is_rejected_before_calling_model(client, fake_parser, payload):
    response = client.post("/order", json=payload)
    assert response.status_code == 422
    assert fake_parser.calls == []


def test_non_json_body(client, fake_parser):
    response = client.post("/order", content="dvije margarite", headers={"content-type": "text/plain"})
    assert response.status_code == 422


def test_text_is_trimmed(client, fake_parser):
    client.post("/order", json={"text": "  jednu colu  "})
    assert fake_parser.calls == ["jednu colu"]


# --- Model / API failures -------------------------------------------------------


def test_model_unreachable_returns_503(client, fake_parser):
    fake_parser.error = LLMError("timeout")
    response = client.post("/order", json={"text": "jednu colu"})
    assert response.status_code == 503


def test_model_garbage_returns_502(client, fake_parser):
    fake_parser.error = LLMBadOutput("not json")
    response = client.post("/order", json={"text": "jednu colu"})
    assert response.status_code == 502


def test_missing_api_key_returns_503(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = client.post("/order", json={"text": "jednu colu"})
    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]


def test_health_and_menu(client):
    assert client.get("/health").json()["status"] == "ok"
    assert len(client.get("/menu").json()) == 10


def test_quota_exhausted_returns_503(client, fake_parser):
    from app.llm import LLMQuotaExceeded

    fake_parser.error = LLMQuotaExceeded("429")
    response = client.post("/order", json={"text": "jednu colu"})
    assert response.status_code == 503
    assert "quota" in response.json()["detail"]


def test_chat_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "/conversation" in response.text
