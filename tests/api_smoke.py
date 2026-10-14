"""End-to-end test through the HTTP API (exactly what a client / benchmark sees).

Requires the server to run (docker compose up, or uv run aegra dev). Plays a scripted candidate:

    uv run python tests/api_smoke.py                       # http://localhost:2026, German
    uv run python tests/api_smoke.py --url http://host:2026 --lang fr
"""

from __future__ import annotations

import argparse
import sys
import time

from langgraph_sdk import get_sync_client

SCRIPTS = {
    "de": [
        "Hallo",
        "Ich möchte Koch werden.",
        "Ja, danke, mit dem Bus.",
        "Ich bin Lena, 15, und koche gern mit meiner Grossmutter.",
    ],
    "fr": [
        "Bonjour",
        "Je voudrais devenir employée de commerce.",
        "Oui, merci, en train.",
        "Je m'appelle Léa, j'ai 15 ans et j'aime organiser.",
    ],
    "it": [
        "Ciao",
        "Vorrei diventare cuoco.",
        "Sì, grazie, in autobus.",
        "Mi chiamo Luca, ho 15 anni e cucino volentieri.",
    ],
}
FILLER = {
    "de": "In der Schnupperlehre habe ich zum Beispiel Gemüse gerüstet und beim Anrichten geholfen.",
    "fr": "Pendant mon stage, j'ai par exemple préparé des brochures et répondu au téléphone.",
    "it": "Durante lo stage, per esempio, ho preparato le verdure e aiutato a impiattare.",
}


def last_ai(values: dict) -> str:
    msgs = values.get("messages", [])
    for m in reversed(msgs):
        if m.get("type") == "ai":
            c = m.get("content")
            return c if isinstance(c, str) else str(c)
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:2026")
    ap.add_argument("--lang", default="de", choices=list(SCRIPTS))
    ap.add_argument("--max-turns", type=int, default=40)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    client = get_sync_client(url=args.url)
    thread = client.threads.create()
    msgs = SCRIPTS[args.lang] + [FILLER[args.lang]] * args.max_turns
    t_all = time.time()
    prev_reply = None
    for i, msg in enumerate(msgs[: args.max_turns]):
        t0 = time.time()
        values = client.runs.wait(thread["thread_id"], "agent", input={"messages": [{"role": "user", "content": msg}]})
        reply = last_ai(values)
        if not isinstance(values, dict) or "messages" not in values or (i > 0 and reply == prev_reply):
            print(f"FAIL at turn {i + 1}: the run returned no new coach message. Server response:\n{str(values)[:800]}")
            print("Check the server log: docker compose logs --tail 60 hackapertustemplate26")
            return 1
        prev_reply = reply
        print(
            f"[{i + 1:>2}] YOU: {msg[:70]}\n     COACH ({time.time() - t0:.1f}s, {values.get('action')}, {values.get('stage')}): {reply[:160]}"
        )
        if values.get("finished"):
            print(
                f"\nOK: interview finished after {i + 1} turns in {time.time() - t_all:.0f}s, "
                f"llm_calls={values.get('llm_calls')} answers={values.get('answers')}"
            )
            print("\n--- FEEDBACK (start) ---\n" + reply[:1200])
            return 0
    print("FAIL: interview did not finish")
    return 1


if __name__ == "__main__":
    sys.exit(main())
