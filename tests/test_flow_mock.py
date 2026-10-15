"""Run a full interview against the mock LLM: python tests/test_flow_mock.py"""

import os
import sys
from pathlib import Path

os.environ["COACH_MOCK_LLM"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from coach.graph import create_graph  # noqa: E402

g = create_graph(checkpointer=MemorySaver())

cfg = {"configurable": {"thread_id": "t1"}}
answers = [
    "/start posting=P-01 lang=de",
    "Ja, gut, danke.",
    "Ich bin Luca, 15, und gehe in die 3. Sek in Chur. In der Freizeit baue ich Lego Technic Modelle mit Motoren.",
    "Weil es cool ist.",
    "In der Schnupperlehre habe ich im CAD eine Halterung gezeichnet und gesehen, wie sie gefräst wurde.",
    "Ich arbeite genau und frage nach, wenn ich etwas nicht verstehe, zum Beispiel im Werken.",
    "Mich interessiert, wie aus einer Zeichnung ein echtes Bauteil wird.",
    "Ich bin geduldig und gebe nicht schnell auf, auch wenn ein Modell nicht sofort funktioniert.",
    "Manchmal will ich zu viel auf einmal, deshalb mache ich mir jetzt eine Liste.",
] * 2 + [
    "Wie viele Lernende habt ihr?",
    "Nein, danke, ich habe keine weiteren Fragen.",
    "Nein.",
    "Danke.",
]
out: dict = {}
for i, a in enumerate(answers):
    g.invoke({"messages": [("user", a)]}, cfg)
    out = g.get_state(cfg).values
    print(
        f"\n[{i}] CAND: {a}\n    COACH ({out['action']}, {out['stage']}, q{out['q_index']}): {out['messages'][-1].content[:160]}"
    )
    if out.get("finished"):
        break
calls_per_answer = out["llm_calls"] / max(out["answers"], 1)
print("\nplan:", out["plan"])
print("llm_calls", out["llm_calls"], "answers", out["answers"], "calls/answer", round(calls_per_answer, 2))
assert out.get("finished"), "the interview did not reach the feedback"
assert calls_per_answer < 5, "more than 5 LLM calls per answer"
print("OK")
