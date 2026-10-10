"""Runs the labelled sentences in eval/cases.json against the REAL model and scores them.

Usage:
    python -m eval.run_eval                 # in-process, uses GEMINI_API_KEY
    python -m eval.run_eval --url http://localhost:8000   # against a running server
    python -m eval.run_eval --conversations   # multi-turn conversations (eval/conversations.json)

A case passes when ordered items and quantities match exactly, every expected
unavailable item is reported (with its quantity) and nothing extra is reported
as unavailable, and, if given, suggestions match.
"""

import argparse
import json
import sys
import time
from pathlib import Path

CASES_PATH = Path(__file__).parent / "cases.json"
CONVERSATIONS_PATH = Path(__file__).parent / "conversations.json"


def call_in_process(text: str) -> dict:
    from app.main import get_menu, get_parser
    from app.service import build_response

    return build_response(get_parser().parse(text), get_menu()).model_dump(mode="json")


def call_conversation(url: str | None, text: str, session_id: str | None) -> dict:
    payload = {"text": text, "session_id": session_id}
    if url:
        import httpx

        response = httpx.post(f"{url.rstrip('/')}/conversation", json=payload, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
        return response.json()
    from fastapi.testclient import TestClient

    from app.main import app

    response = TestClient(app).post("/conversation", json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    return response.json()


def run_conversations(url: str | None, delay: float) -> tuple[int, int]:
    """Each turn is scored on the whole order so far. Returns (passed turns, total turns)."""
    conversations = json.loads(CONVERSATIONS_PATH.read_text(encoding="utf-8"))
    passed = total = 0
    for conv in conversations:
        session_id = None
        print(f"\n-- {conv['name']}")
        for turn in conv["turns"]:
            total += 1
            try:
                body = call_conversation(url, turn["text"], session_id)
                session_id = body["session_id"]
                problems = check(turn, body)
            except Exception as exc:
                problems = [f"error: {exc!r}"]
            passed += not problems
            print(f"[{'PASS' if not problems else 'FAIL'}] {turn['text']}")
            for p in problems:
                print(f"        {p}")
            if delay:
                time.sleep(delay)
    return passed, total


def call_http(url: str, text: str) -> dict:
    import httpx

    response = httpx.post(f"{url.rstrip('/')}/order", json={"text": text}, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    return response.json()


def _matches(expected_text: str, actual_text: str) -> bool:
    # Croatian inflection: "hamburger" vs "hamburgera", "tun" vs "pizza s tunom".
    e, a = expected_text.lower(), actual_text.lower()
    return e in a or a in e


def check(case: dict, body: dict) -> list[str]:
    problems = []
    got_items = {i["id"]: i["quantity"] for i in body["items"]}
    if got_items != case.get("items", {}):
        problems.append(f"items {got_items} != {case.get('items', {})}")

    expected_un = {**case.get("unavailable", {}), **case.get("unclear", {})}
    got_un = body["unavailable"] + body.get("unclear", [])
    for text, qty in expected_un.items():
        hit = [u for u in got_un if _matches(text, u["text"])]
        if not hit:
            problems.append(f"missing unavailable '{text}'")
        elif hit[0]["quantity"] != qty:
            problems.append(f"unavailable '{text}' qty {hit[0]['quantity']} != {qty}")
    extra = [u["text"] for u in got_un if not any(_matches(t, u["text"]) for t in expected_un)]
    if extra:
        problems.append(f"unexpected unavailable {extra}")

    if "suggestions" in case:
        got = [s["id"] for s in body["suggestions"]]
        if case.get("suggestions_mode") == "subset":
            if not got or not set(got) <= set(case["suggestions"]) | {"mijesana_salata"}:
                problems.append(f"suggestions {got} not within {case['suggestions']}")
        elif set(got) != set(case["suggestions"]):
            problems.append(f"suggestions {got} != {case['suggestions']}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="Base URL of a running server; default runs in-process")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds between calls (free-tier rate limits)")
    parser.add_argument("--conversations", action="store_true", help="Run the multi-turn conversations instead")
    args = parser.parse_args()

    if args.conversations:
        passed, total = run_conversations(args.url, args.delay)
        print(f"\n{passed}/{total} conversation turns passed ({passed / total:.0%})")
        return 0 if passed == total else 1

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    passed, latencies = 0, []
    for case in cases:
        start = time.perf_counter()
        try:
            body = call_http(args.url, case["text"]) if args.url else call_in_process(case["text"])
            problems = check(case, body)
        except Exception as exc:  # report and keep going
            problems = [f"error: {exc!r}"]
        latencies.append(time.perf_counter() - start)
        status = "PASS" if not problems else "FAIL"
        passed += not problems
        print(f"[{status}] {case['text']}")
        for p in problems:
            print(f"        {p}")
        if any("quota" in p.lower() or "RESOURCE_EXHAUSTED" in p for p in problems):
            print("\nAPI quota exhausted, stopping early.")
            break
        if args.delay:
            time.sleep(args.delay)

    ran = len(latencies)
    latencies.sort()
    print(f"\n{passed}/{ran} passed ({passed / ran:.0%}) of {len(cases)} cases, "
          f"median latency {latencies[ran // 2]:.2f}s, max {latencies[-1]:.2f}s")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
