"""Offline evaluation of the assess step against the 698 expert-annotated answers.

For every annotated answer we build the same prompt the coach uses (few-shot examples from the
same question, but never the answer itself = leave-one-out), call the model once and compare with
the expert annotation: coach_action, safety (distress / redirect), criterion scores, flags.

    uv run python tests/eval_assess.py                    # all answers, 8 parallel requests
    uv run python tests/eval_assess.py --limit 150        # quick stratified sample
    uv run python tests/eval_assess.py --workers 12

Report: runs/eval-assess-<timestamp>/report.md (+ predictions.jsonl)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from coach import llm, prompts  # noqa: E402
from coach.config import get_settings  # noqa: E402
from coach.graph import DISCOURAGED, DISTRESS, CoachState, clean_assessment  # noqa: E402
from coach.knowledge import _jsonl, base_lang, kb  # noqa: E402
from coach.planner import stage_of  # noqa: E402

ACTIONS = ["continue", "probe", "redirect", "support_and_refer"]
SAFETY_FLAGS = {"distress_signal"}
REDIRECT_FLAGS = {"manipulation_attempt", "discriminatory", "off_topic", "inappropriate_tone"}


def load_items() -> list[dict]:
    d = get_settings().data_dir
    answers = {a["id"]: a for a in _jsonl(d / "answers/answers.jsonl")}
    items = []
    for ann in _jsonl(d / "annotations/annotations.jsonl"):
        a = answers.get(ann["answer_id"])
        if not a or not a.get("question_id") or a["question_id"] not in kb().questions:
            continue
        items.append({"answer": a, "ann": ann})
    return items


def sample(items: list[dict], limit: int | None) -> list[dict]:
    if not limit or limit >= len(items):
        return items
    rnd = random.Random(42)
    # keep ALL safety-relevant cases, sample the rest stratified by language
    must = [i for i in items if i["ann"]["coach_action"] in ("support_and_refer", "redirect")]
    rest = [i for i in items if i not in must]
    rnd.shuffle(rest)
    by_lang = defaultdict(list)
    for i in rest:
        by_lang[base_lang(i["answer"]["lang"])].append(i)
    out, n = list(must), max(limit - len(must), 0)
    while n > 0 and any(by_lang.values()):
        for lang in list(by_lang):
            if by_lang[lang] and n > 0:
                out.append(by_lang[lang].pop())
                n -= 1
    return out


def predict(item: dict) -> dict:
    a, ann = item["answer"], item["ann"]
    k = kb()
    lang = a["lang"]
    qid = a["question_id"]
    qtext = (a.get("question_text") or {}).get(base_lang(lang)) or k.question_text(qid, lang)
    state = CoachState(lang=lang, current_qid=qid, current_question_text=qtext, stage=stage_of(qid))
    msgs = prompts.assess_messages(state, a["text"], exclude_id=a["id"])
    t0 = time.time()
    try:
        raw = llm.chat(msgs, purpose="assess", json_mode=True, max_tokens=350)
        pred = clean_assessment(llm.parse_json(raw), qid, a["text"])
        err = None
    except Exception as e:  # a single failure must not stop the evaluation
        pred, err = {"coach_action": "continue", "scores": {}, "problem_flags": []}, str(e)
    text = a["text"].replace("’", "'")
    rule_distress = bool(DISTRESS.search(text))
    rule_discouraged = bool(DISCOURAGED.search(text))
    return {
        "answer_id": a["id"],
        "lang": lang,
        "qid": qid,
        "text": a["text"],
        "expert_action": ann["coach_action"],
        "expert_flags": ann.get("problem_flags", []),
        "expert_scores": ann.get("criteria_scores", {}),
        "expert_level": ann["quality_level"],
        "raw_action": pred.get("coach_action"),
        "pred_action": effective_action(pred),
        "pred_flags": pred.get("problem_flags", []),
        "pred_scores": pred.get("scores", {}),
        "follow_up": pred.get("follow_up"),
        "rule_distress": rule_distress,
        "rule_discouraged": rule_discouraged,
        "seconds": round(time.time() - t0, 1),
        "error": err,
    }


def effective_action(pred: dict) -> str:
    """What the coach actually does: same probe policy as graph.decide (mean score <= 2.5)."""
    a = pred.get("coach_action")
    sc = list((pred.get("scores") or {}).values())
    if a == "probe" and sc and sum(sc) / len(sc) > 2.5:
        return "continue"
    return a


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.0f}% ({a}/{b})" if b else "–"


def report(preds: list[dict]) -> str:
    out = []
    n = len(preds)
    acc = sum(p["pred_action"] == p["expert_action"] for p in preds)
    out += ["# Assess-step evaluation", "", f"Answers evaluated: {n} · model: {get_settings().llm_model}", ""]
    out += ["## coach_action", "", f"Exact agreement with the expert: **{pct(acc, n)}**", ""]
    # confusion matrix
    cm = Counter((p["expert_action"], p["pred_action"]) for p in preds)
    out += ["| expert ↓ / model → | " + " | ".join(ACTIONS) + " |", "|---|" + "---:|" * len(ACTIONS)]
    for e in ACTIONS:
        out.append(f"| {e} | " + " | ".join(str(cm[(e, m)]) for m in ACTIONS) + " |")
    # per language
    out += ["", "| language | agreement | probe recall | probe precision |", "|---|---:|---:|---:|"]
    by = defaultdict(list)
    for p in preds:
        by[base_lang(p["lang"]) if not p["lang"].startswith("gsw") else "gsw"].append(p)
    for lang, ps in sorted(by.items()):
        a = sum(p["pred_action"] == p["expert_action"] for p in ps)
        tp = sum(p["expert_action"] == "probe" and p["pred_action"] == "probe" for p in ps)
        ep = sum(p["expert_action"] == "probe" for p in ps)
        pp = sum(p["pred_action"] == "probe" for p in ps)
        out.append(f"| {lang} | {pct(a, len(ps))} | {pct(tp, ep)} | {pct(tp, pp)} |")

    # safety
    out += ["", "## Safety", ""]
    dist = [p for p in preds if p["expert_action"] == "support_and_refer" or SAFETY_FLAGS & set(p["expert_flags"])]
    caught_model = sum(p["pred_action"] == "support_and_refer" or "distress_signal" in p["pred_flags"] for p in dist)
    caught_sys = sum(
        p["pred_action"] == "support_and_refer"
        or "distress_signal" in p["pred_flags"]
        or p["rule_distress"]
        or p["rule_discouraged"]
        for p in dist
    )
    false_alarm = sum((p["pred_action"] == "support_and_refer" or p["rule_distress"]) for p in preds if p not in dist)
    out += [
        f"- Distress cases (expert): {len(dist)}",
        f"- caught by the model alone: **{pct(caught_model, len(dist))}**",
        f"- caught by the full system (model + rules): **{pct(caught_sys, len(dist))}**",
        f"- false alarms on other answers (support or strong-distress rule): {false_alarm}",
    ]
    red = [p for p in preds if p["expert_action"] == "redirect"]
    red_ok = sum(p["pred_action"] == "redirect" for p in red)
    red_soft = sum(p["pred_action"] in ("redirect", "probe", "support_and_refer") for p in red)
    false_red = sum(p["pred_action"] == "redirect" for p in preds if p["expert_action"] != "redirect")
    out += [
        f"- Redirect cases (manipulation, off-topic, disrespect): {len(red)}; model redirect: **{pct(red_ok, len(red))}**, "
        f"at least not 'continue': {pct(red_soft, len(red))}; false redirects: {false_red}"
    ]
    out += ["", "Missed distress / redirect cases:", ""]
    for p in dist + red:
        ok = (
            p in dist
            and (
                p["pred_action"] == "support_and_refer"
                or "distress_signal" in p["pred_flags"]
                or p["rule_distress"]
                or p["rule_discouraged"]
            )
        ) or (p in red and p["pred_action"] == "redirect")
        if not ok:
            out.append(
                f"- `{p['answer_id']}` [{p['lang']}] expert={p['expert_action']} {p['expert_flags']} → model={p['pred_action']} "
                f"{p['pred_flags']}: «{p['text'][:160]}»"
            )

    # scores
    out += [
        "",
        "## Criterion scores (1–4)",
        "",
        "| criterion | n | mean abs. error | bias (model − expert) | exact |",
        "|---|---:|---:|---:|---:|",
    ]
    per = defaultdict(list)
    for p in preds:
        for c, e in p["expert_scores"].items():
            m = p["pred_scores"].get(c)
            if isinstance(e, int) and isinstance(m, int):
                per[c].append((m, e))
    allpairs = []
    for c, pairs in sorted(per.items()):
        allpairs += pairs
        mae = sum(abs(m - e) for m, e in pairs) / len(pairs)
        bias = sum(m - e for m, e in pairs) / len(pairs)
        ex = sum(m == e for m, e in pairs)
        out.append(f"| {c} | {len(pairs)} | {mae:.2f} | {bias:+.2f} | {pct(ex, len(pairs))} |")
    if allpairs:
        mae = sum(abs(m - e) for m, e in allpairs) / len(allpairs)
        bias = sum(m - e for m, e in allpairs) / len(allpairs)
        out.append(
            f"| **all** | {len(allpairs)} | **{mae:.2f}** | **{bias:+.2f}** | {pct(sum(m == e for m, e in allpairs), len(allpairs))} |"
        )
    # by expert level: does the model separate weak from strong answers?
    out += ["", "| expert quality level | n | mean model score |", "|---:|---:|---:|"]
    for lvl in (1, 2, 3, 4):
        vals = [
            sum(p["pred_scores"].values()) / len(p["pred_scores"])
            for p in preds
            if p["expert_level"] == lvl and p["pred_scores"]
        ]
        if vals:
            out.append(f"| {lvl} | {len(vals)} | {sum(vals) / len(vals):.2f} |")

    # flags
    out += [
        "",
        "## Problem flags",
        "",
        "| flag | expert | model | both | recall | precision |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    flags = sorted({f for p in preds for f in p["expert_flags"] + p["pred_flags"]})
    for f in flags:
        e = sum(f in p["expert_flags"] for p in preds)
        m = sum(f in p["pred_flags"] for p in preds)
        b = sum(f in p["expert_flags"] and f in p["pred_flags"] for p in preds)
        out.append(f"| {f} | {e} | {m} | {b} | {pct(b, e)} | {pct(b, m)} |")
    errs = sum(1 for p in preds if p["error"])
    out += ["", f"Errors: {errs} · mean latency: {sum(p['seconds'] for p in preds) / max(n, 1):.1f} s"]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    items = sample(load_items(), args.limit)
    out = ROOT / "runs" / f"eval-assess-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    s = get_settings()
    print(f"Model: {s.llm_model} @ {s.llm_base_url} · answers: {len(items)} · output: {out}", flush=True)

    done = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, p in enumerate(pool.map(predict, items), 1):
            done.append(p)
            if i % 20 == 0 or i == len(items):
                print(f"  {i}/{len(items)}", flush=True)

    with (out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for p in done:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    rep = report(done)
    (out / "report.md").write_text(rep, encoding="utf-8")
    print("\n" + rep)
    print(f"\nReport: {out / 'report.md'}")


if __name__ == "__main__":
    main()
