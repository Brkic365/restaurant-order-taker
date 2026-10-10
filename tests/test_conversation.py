"""Multi-turn /conversation endpoint, with the model mocked."""

from app.schemas import Intent
from tests.conftest import line, parsed


def ids(body):
    return {i["id"]: i["quantity"] for i in body["items"]}


def test_order_builds_up_over_several_messages(client, fake_parser):
    fake_parser.result = parsed([line("margarita", 2, "margarita")])
    first = client.post("/conversation", json={"text": "dvije margarite"}).json()
    sid = first["session_id"]
    assert first["turn"] == 1 and ids(first) == {"margarita": 2}

    # The model returns the full updated order; the server passes it the order so far.
    fake_parser.result = parsed([line("margarita", 2, "margarita"), line("cola", 1, "coca_cola")])
    second = client.post("/conversation", json={"text": "i jednu colu", "session_id": sid}).json()
    assert fake_parser.orders_seen[-1] == {"margarita": 2}
    assert second["turn"] == 2 and ids(second) == {"margarita": 2, "coca_cola": 1}
    assert second["total_eur"] == 21.0

    fake_parser.result = parsed([line("margarita", 2, "margarita")])
    third = client.post("/conversation", json={"text": "ipak bez cole", "session_id": sid}).json()
    assert ids(third) == {"margarita": 2}

    assert {i["id"] for i in client.get(f"/conversation/{sid}").json()} == {"margarita"}


def test_question_mid_call_keeps_the_order(client, fake_parser):
    fake_parser.result = parsed([line("diavola", 1, "diavola")])
    sid = client.post("/conversation", json={"text": "jednu diavolu"}).json()["session_id"]

    # Model forgets to repeat the order while answering a question: the order must survive.
    fake_parser.result = parsed(intent=Intent.question, asks_meat_free=True)
    body = client.post("/conversation", json={"text": "imate li nešto bez mesa?", "session_id": sid}).json()
    assert ids(body) == {"diavola": 1}
    assert len(body["suggestions"]) == 4


def test_unavailable_items_are_reported_per_message(client, fake_parser):
    fake_parser.result = parsed([line("hamburger", 1)])
    sid = client.post("/conversation", json={"text": "jedan hamburger"}).json()["session_id"]
    fake_parser.result = parsed([line("pivo", 1, "pivo")])
    body = client.post("/conversation", json={"text": "onda jedno pivo", "session_id": sid}).json()
    assert body["unavailable"] == []
    assert ids(body) == {"pivo": 1}


def test_sessions_are_independent(client, fake_parser):
    fake_parser.result = parsed([line("pivo", 1, "pivo")])
    a = client.post("/conversation", json={"text": "pivo"}).json()["session_id"]
    fake_parser.result = parsed([line("cola", 1, "coca_cola")])
    b = client.post("/conversation", json={"text": "colu"}).json()["session_id"]
    assert a != b
    assert [i["id"] for i in client.get(f"/conversation/{a}").json()] == ["pivo"]


def test_unknown_session_is_404(client, fake_parser):
    response = client.post("/conversation", json={"text": "colu", "session_id": "nope"})
    assert response.status_code == 404


def test_end_conversation(client, fake_parser):
    sid = client.post("/conversation", json={"text": "bok"}).json()["session_id"]
    assert client.delete(f"/conversation/{sid}").status_code == 204
    assert client.get(f"/conversation/{sid}").status_code == 404


def test_conversation_falls_back_when_model_is_down(client, with_fallback):
    from app.llm import LLMError

    with_fallback.result = parsed([line("margarita", 1, "margarita")])
    sid = client.post("/conversation", json={"text": "jednu margaritu"}).json()["session_id"]
    with_fallback.error = LLMError("down")
    body = client.post("/conversation", json={"text": "i dva piva", "session_id": sid}).json()
    assert body["degraded"] is True
    assert ids(body) == {"margarita": 1, "pivo": 2}
