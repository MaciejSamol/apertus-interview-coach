"""Interactive interview in the terminal against the real model (no Aegra / UI needed).

    uv run python tests/chat_cli.py                      # free start, e.g. "Hallo, ich möchte Koch werden"
    uv run python tests/chat_cli.py "/start scenario=S-04"
    uv run python tests/chat_cli.py "/start posting=P-01 lang=it address=formal mode=spoken"

Type 'q' to quit. After each turn it prints action, stage, LLM calls and latency.
"""

import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")

print("Loading coach ...", flush=True)
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from coach.config import get_settings  # noqa: E402
from coach.graph import create_graph  # noqa: E402

s = get_settings()
print(f"Model: {s.llm_model} @ {s.llm_base_url} (mock={s.mock_llm})", flush=True)
g = create_graph(checkpointer=MemorySaver())
cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}

first = sys.argv[1] if len(sys.argv) > 1 else input("YOU> ")
msg = first
while msg.strip().lower() not in {"q", "quit", "exit"}:
    t0 = time.time()
    print("   ... (coach is thinking)", flush=True)
    try:
        g.invoke({"messages": [("user", msg)]}, cfg)
    except Exception as e:  # show model or connection errors instead of hanging silently
        print(f"ERROR: {type(e).__name__}: {e}")
        break
    st = g.get_state(cfg).values
    print(f"\nCOACH> {st['messages'][-1].content}")
    a = st.get("last_assessment") or {}
    print(
        f"   [{st.get('action')} | stage={st.get('stage')} | q={st.get('q_index')}/{len(st.get('plan', []))} | "
        f"scores={a.get('scores')} | flags={a.get('problem_flags')} | "
        f"calls={st.get('llm_calls')} answers={st.get('answers')} | {time.time() - t0:.1f}s]\n"
    )
    if st.get("finished"):
        break
    msg = input("YOU> ")
