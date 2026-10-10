"""GeminiOrderParser retry and error mapping, with the Google client mocked."""

from types import SimpleNamespace

import pytest

from app.config import DEFAULT_MENU_PATH
from app.llm import SYSTEM_PROMPT, GeminiOrderParser, LLMBadOutput, LLMError
from app.menu import Menu

GOOD = '{"intent":"order","lines":[{"heard":"cola","quantity":1,"menu_id":"coca_cola","ambiguous":false}],"asks_meat_free":false,"suggestion_ids":[],"clarification":null}'


@pytest.fixture
def menu():
    return Menu.from_file(DEFAULT_MENU_PATH)


def make_parser(menu, responses, max_attempts=2):
    parser = GeminiOrderParser(api_key="test", model="test-model", menu=menu, max_attempts=max_attempts)
    queue = list(responses)

    def generate_content(**_):
        nxt = queue.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return SimpleNamespace(text=nxt)

    parser._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    return parser


def test_valid_output(menu):
    result = make_parser(menu, [GOOD]).parse("colu")
    assert result.lines[0].menu_id == "coca_cola"


def test_retries_after_garbage(menu):
    result = make_parser(menu, ["not json at all", GOOD]).parse("colu")
    assert result.lines[0].quantity == 1


def test_retries_after_network_error(menu):
    result = make_parser(menu, [TimeoutError("slow"), GOOD]).parse("colu")
    assert result.intent.value == "order"


def test_gives_up_on_repeated_garbage(menu):
    with pytest.raises(LLMBadOutput):
        make_parser(menu, ['{"intent": "pizza"}', "[]"]).parse("colu")


def test_gives_up_on_repeated_network_errors(menu):
    with pytest.raises(LLMError):
        make_parser(menu, [ConnectionError(), ConnectionError()]).parse("colu")


def test_prompt_contains_every_menu_id(menu):
    prompt = SYSTEM_PROMPT.format(menu=menu.as_prompt_table())
    for item_id in menu.ids():
        assert item_id in prompt


def _client_error(code):
    from google.genai import errors

    return errors.ClientError(code, {"error": {"code": code, "message": "nope", "status": "X"}})


def test_quota_error_is_not_retried(menu):
    from app.llm import LLMQuotaExceeded

    parser = make_parser(menu, [_client_error(429), GOOD])
    with pytest.raises(LLMQuotaExceeded):
        parser.parse("colu")


def test_unknown_model_is_not_retried(menu):
    parser = make_parser(menu, [_client_error(404), GOOD])
    with pytest.raises(LLMError):
        parser.parse("colu")
