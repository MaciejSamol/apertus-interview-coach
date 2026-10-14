"""End-to-end simulation: an LLM role-plays the candidate of a dataset scenario through a full interview,
then deterministic checks (and optionally an LLM judge) score the run.

    uv run python tests/simulate.py                         # scenario S-01
    uv run python tests/simulate.py S-01 S-04 S-05          # several scenarios (in parallel)
    uv run python tests/simulate.py --all --workers 4       # all 28 scenarios
    uv run python tests/simulate.py S-03 --quality 1        # force a weak candidate (1-4)

    uv run python tests/simulate.py S-01 --judge            # add the LLM judge

Reports are written to runs/<timestamp>/ (one .md per scenario, summary.md, results.json).
The judge uses the same endpoint unless JUDGE_BASE_URL / JUDGE_MODEL / JUDGE_API_KEY are set.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from openai import OpenAI  # noqa: E402

from coach import llm  # noqa: E402
from coach.config import get_settings  # noqa: E402
from coach.graph import create_graph  # noqa: E402
from coach.knowledge import base_lang, kb  # noqa: E402

LANG_NAMES = {"de": "German", "fr": "French", "it": "Italian"}
MAX_TURNS = 45


def candidate_system(cand: dict, scenario: dict, quality: int | None) -> str:
    lang = base_lang(scenario.get("lang") or cand.get("interview_language"))
    persona = dict(cand.get("simulation_persona") or {})
    if quality:
        persona["typical_quality_level"] = quality
    profile = {
        k: cand.get(k)
        for k in (
            "first_name",
            "age",
            "place",
            "school",
            "hobbies",
            "experiences",
            "career_choice_story",
            "strengths",
            "development_areas",
            "future_plans",
            "interview_experience",
        )
    }
    q = persona.get("typical_quality_level", 3)
    style = {
        1: "Your answers are very short, vague, sometimes just a few words. You rarely give examples.",
        2: "Your answers are short and a bit generic; you give an example only when asked.",
        3: "Your answers are okay, with some concrete details.",
        4: "Your answers are clear and concrete, with real examples from your life.",
    }.get(q, "")
    return f"""You role-play a Swiss school student in a PRACTICE apprenticeship interview. Stay fully in character.
Speak {LANG_NAMES[lang]} like a real {cand.get("age", 15)}-year-old (simple words, not perfect, no bullet points).
Write standard {LANG_NAMES[lang]} (a few typical Swiss words are fine); do NOT write in dialect, even if the persona mentions it.
Answer ONLY the interviewer's last question in 1-4 sentences. Never say you are an AI. Never write the interviewer's lines.
If asked whether you have questions, ask 1-2 realistic questions about the apprenticeship or company; when asked again, say you have no more questions.
Quality of your answers: {style}

YOUR PROFILE (use only these facts, do not invent big new facts):
{json.dumps(profile, ensure_ascii=False)}

YOUR PERSONA / WAY OF SPEAKING:
{json.dumps(persona, ensure_ascii=False)}

Output only what you say."""


def candidate_reply(system: str, history: list[tuple[str, str]]) -> str:
    convo = "\n".join(f"{who}: {txt}" for who, txt in history[-10:])
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"CONVERSATION:\n{convo}\n\nYour answer as the candidate:"},
    ]
    out = llm.chat(msgs, purpose="candidate", max_tokens=200, temperature=0.8)
    out = re.sub(r"^\s*(Candidate|Kandidat\w*|Candidat\w*|Candidat[oa]|\w+)\s*:\s*", "", out.strip())
    return out.strip().strip('"') or "Ich weiss nicht."


BANNED = re.compile(r"\b(perfekt|genial|perfect|parfait|génial|perfetto|geniale)\w*", re.I)


def auto_checks(st: dict, turns: list[dict]) -> dict:
    k = kb()
    fb = st["messages"][-1].content if st.get("finished") else ""
    stages_seen = {t["stage"] for t in turns}
    criteria_rows = sum(1 for c in k.criteria.values() if c["label"][base_lang(st["lang"])].lower() in fb.lower())
    answers = max(st.get("answers", 0), 1)
    return {
        "finished": bool(st.get("finished")),
        "turns": len(turns),
        "main_questions_asked": st.get("q_index", 0) + 1,
        "plan_length": len(st.get("plan", [])),
        "followups": sum(1 for t in turns if t["action"] == "follow_up"),
        "redirects": sum(1 for t in turns if t["action"] == "redirect"),
        "support": sum(1 for t in turns if t["action"] == "support"),
        "company_answers": sum(1 for t in turns if t["action"] == "answer_company"),
        "stages_covered": sorted(stages_seen),
        "llm_calls": st.get("llm_calls", 0),
        "calls_per_answer": round(st.get("llm_calls", 0) / answers, 2),
        "rule_ok_calls": st.get("llm_calls", 0) / answers < 5,
        "avg_turn_s": round(sum(t["seconds"] for t in turns) / max(len(turns), 1), 1),
        "feedback_criteria_found": criteria_rows,
        "feedback_banned_words": BANNED.findall(re.sub(r"«[^»]*»|\"[^\"]*\"|„[^“]*“", "", fb)),
        "feedback_chars": len(fb),
        "assessments_with_scores": sum(1 for t in st.get("turns", []) if t["assessment"].get("scores")),
    }


def judge(scenario: dict, transcript: str, feedback: str) -> dict:
    s = get_settings()
    client = OpenAI(
        base_url=os.getenv("JUDGE_BASE_URL", s.llm_base_url),
        api_key=os.getenv("JUDGE_API_KEY", s.llm_api_key),
        timeout=s.llm_timeout,
        max_retries=1,
    )
    model = os.getenv("JUDGE_MODEL", s.llm_model)
    rules = kb().guideline_text()
    sc_list = scenario.get("success_criteria_for_coach", [])
    crit = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(sc_list))
    prompt = f"""You are a strict evaluator of an AI interview coach for Swiss teenagers (apprenticeship interviews).

SCENARIO SUCCESS CRITERIA FOR THE COACH:
{crit}

FEEDBACK RULES (ids):
{rules}

INTERVIEW TRANSCRIPT:
{transcript[:12000]}

FINAL FEEDBACK GIVEN BY THE COACH:
{feedback[:6000]}

Be critical: a pass needs clear evidence in the transcript. Check in particular whether the interviewer asked the
questions clearly, praised or graded during the interview (not allowed), repeated itself, or invented company facts.

Return ONLY JSON with exactly {len(sc_list)} entries in "criteria" (one per numbered criterion above):
{{"criteria": [{{"n": <number>, "pass": <true|false>, "reason": "<one sentence with evidence>"}}],
 "feedback_rule_violations": [{{"id": "<rule id>", "evidence": "<quote from the feedback>"}}],
 "interview_quality_1_10": <int>, "feedback_quality_1_10": <int>,
 "summary": "<3-5 sentences: what the coach did well and what went wrong>"}}"""
    if s.mock_llm:
        return {"criteria": [], "feedback_rule_violations": [], "summary": "(mock)"}
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0,
        response_format={"type": "json_object"},
    )
    return llm.parse_json(resp.choices[0].message.content or "")


def safe_run(sid: str, quality, out_dir: Path, use_judge: bool):
    """One failing scenario (e.g. the model server is briefly unreachable) must not stop the whole run."""
    try:
        return run_scenario(sid, quality, out_dir, use_judge)
    except Exception as e:
        print(f"[{sid}] ERROR: {type(e).__name__}: {str(e)[:200]}", flush=True)
        (out_dir / f"{sid}.error.txt").write_text(f"{type(e).__name__}: {e}", encoding="utf-8")
        return None


def run_scenario(sid: str, quality: int | None, out_dir: Path, use_judge: bool = False) -> dict:
    k = kb()
    sc = k.scenarios[sid]
    cand = k.candidates[sc["candidate_id"]]
    g = create_graph(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    cand_sys = candidate_system(cand, sc, quality)

    history: list[tuple[str, str]] = []
    turns: list[dict] = []
    msg = f"/start scenario={sid}"
    t_start = time.time()
    st: dict = {}
    for _ in range(MAX_TURNS):
        t0 = time.time()
        g.invoke({"messages": [("user", msg)]}, cfg)
        st = g.get_state(cfg).values
        coach = st["messages"][-1].content
        if not msg.startswith("/start"):
            history.append(("Candidate", msg))
        history.append(("Interviewer", coach))
        a = st.get("last_assessment") or {}
        turns.append(
            {
                "candidate": None if msg.startswith("/start") else msg,
                "coach": coach,
                "action": st.get("action"),
                "stage": st.get("stage"),
                "q": f"{st.get('q_index')}/{len(st.get('plan', []))}",
                "scores": a.get("scores"),
                "flags": a.get("problem_flags"),
                "seconds": round(time.time() - t0, 1),
            }
        )
        print(
            f"[{sid}] turn {len(turns):>2} {st.get('action'):<14} {st.get('stage'):<20} {turns[-1]['seconds']}s",
            flush=True,
        )
        if st.get("finished"):
            break
        msg = candidate_reply(cand_sys, history)

    checks = auto_checks(st, turns)
    feedback = st["messages"][-1].content if st.get("finished") else ""
    transcript = "\n".join(f"{w}: {t}" for w, t in history[:-1])
    try:
        verdict = judge(sc, transcript, feedback) if use_judge else {"summary": "(judge disabled, use --judge)"}
    except Exception as e:  # a judge failure must not lose the run
        verdict = {"error": str(e)}
    passed = [c for c in verdict.get("criteria", []) if c.get("pass")]
    result = {
        "scenario": sid,
        "title": sc["title"],
        "lang": sc["lang"],
        "candidate": cand["first_name"],
        "quality_forced": quality,
        "seconds_total": round(time.time() - t_start),
        "checks": checks,
        "judge": verdict,
        "criteria_passed": f"{len(passed)}/{len(verdict.get('criteria', []))}",
    }
    write_report(out_dir / f"{sid}.md", result, turns, feedback)
    (out_dir / f"{sid}.json").write_text(
        json.dumps({**result, "turns": turns, "feedback": feedback}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def write_report(path: Path, r: dict, turns: list[dict], feedback: str) -> None:
    c, j = r["checks"], r["judge"]
    lines = [
        f"# {r['scenario']}: {r['title']}",
        f"Candidate: {r['candidate']} · language: {r['lang']} · time: {r['seconds_total']} s",
        "",
        "## Judge summary",
        j.get("summary", j.get("error", "-")),
        "",
        f"Interview quality: {j.get('interview_quality_1_10', '-')}/10 · "
        f"feedback quality: {j.get('feedback_quality_1_10', '-')}/10 · "
        f"scenario criteria: {r['criteria_passed']}",
        "",
        "## Scenario criteria",
    ]
    sc_crit = kb().scenarios[r["scenario"]].get("success_criteria_for_coach", [])
    for item in j.get("criteria", []):
        n = item.get("n")
        text = sc_crit[n - 1] if isinstance(n, int) and 0 < n <= len(sc_crit) else ""
        lines.append(f"- {'✅' if item.get('pass') else '❌'} {text} — {item.get('reason', '')}")
    if j.get("feedback_rule_violations"):
        lines += ["", "## Feedback rule violations"]
        lines += [f"- {v.get('id')}: {v.get('evidence', '')}" for v in j["feedback_rule_violations"]]
    lines += ["", "## Automatic checks", "```", json.dumps(c, ensure_ascii=False, indent=1), "```", "", "## Transcript"]
    for t in turns:
        if t["candidate"]:
            lines.append(f"**Candidate:** {t['candidate']}")
        lines.append(
            f"**Coach** `[{t['action']} · {t['stage']} · q{t['q']} · {t['seconds']}s · {t['scores']} {t['flags'] or ''}]`:\n{t['coach']}\n"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="*", default=["S-01"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--quality", type=int, choices=[1, 2, 3, 4])
    ap.add_argument(
        "--judge", action="store_true", help="also run the LLM judge (slow; Apertus 8B judge is unreliable)"
    )
    args = ap.parse_args()
    ids = sorted(kb().scenarios) if args.all else args.scenarios
    out_dir = ROOT / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    s = get_settings()
    print(f"Model: {s.llm_model} @ {s.llm_base_url} · scenarios: {', '.join(ids)} · output: {out_dir}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = [r for r in ex.map(lambda sid: safe_run(sid, args.quality, out_dir, args.judge), ids) if r]
    failed = [sid for sid in ids if not (out_dir / f"{sid}.json").exists()]
    if failed:
        print(f"\nFailed scenarios (connection or model error): {' '.join(failed)}")
        print(f"Rerun only these: uv run python tests/simulate.py {' '.join(failed)}")

    rows = [
        "| Scenario | Language | Turns | Follow-ups | Calls/answer | Avg. turn | Criteria | Interview | Feedback |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        c, j = r["checks"], r["judge"]
        rows.append(
            f"| {r['scenario']} | {r['lang']} | {c['turns']} | {c['followups']} | {c['calls_per_answer']} | "
            f"{c['avg_turn_s']} s | {r['criteria_passed']} | {j.get('interview_quality_1_10', '-')} | {j.get('feedback_quality_1_10', '-')} |"
        )
    (out_dir / "summary.md").write_text("\n".join(rows), encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "\n".join(rows))
    print(f"\nReports: {out_dir}")


if __name__ == "__main__":
    main()
