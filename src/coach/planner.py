"""Deterministic interview plan: which questions from the FHGR bank are asked, in which order.

Keeping the plan outside the LLM makes interviews consistent across scenarios and languages.
"""

from __future__ import annotations

from .knowledge import kb

# Covers the six interview stages with 14 main questions.
DEFAULT_PLAN = [
    "Q-01-01",  # small talk
    "Q-02-01",  # tell us about yourself
    "Q-03-01",  # why this occupation
    "Q-03-04",  # why our company
    "Q-04-04",  # knowledge of the occupation
    "Q-05-01",  # strengths
    "Q-05-03",  # weakness
    "Q-06-07",  # trial apprenticeship (Schnupperlehre)
    "@BF",  # occupation-specific question for the Berufsfeld
    "Q-SIT-02",  # situational
    "Q-DIF-01",  # difficult question
    "Q-08-02",  # future goals
    "Q-09-01",  # candidate questions, answered as the company
    "Q-10-01",  # closing
]

# Used when the candidate has not chosen an occupation yet: no company- or occupation-specific questions.
GENERAL_PLAN = [
    "Q-01-01",
    "Q-02-01",
    "Q-02-08",
    "Q-08-06",
    "Q-06-01",
    "Q-05-01",
    "Q-05-03",
    "Q-06-07",
    "Q-07-01",
    "Q-SIT-02",
    "Q-DIF-05",
    "Q-08-01",
    "Q-09-02",
    "Q-10-01",
]


def build_plan(
    posting: dict | None, scenario: dict | None, bf_key: str | None = None, general: bool = False
) -> list[str]:
    k = kb()
    if general and not scenario and not posting:
        return [q for q in GENERAL_PLAN if q in k.questions]
    if scenario and scenario.get("planned_question_ids"):
        plan = [q for q in scenario["planned_question_ids"] if q in k.questions]
    else:
        plan = []
        bf_key = (posting or {}).get("occupation", {}).get("berufsfeld_key") or bf_key
        for qid in DEFAULT_PLAN:
            if qid == "@BF":
                plan += k.berufsfeld_questions.get(bf_key, [])[:1]
            else:
                plan.append(qid)
    # The interview always ends with the candidate's questions and a closing question.
    if not any(k.questions[q]["flow_stage"] == "candidate_questions" for q in plan):
        plan.append("Q-09-01")
    if plan[-1] not in {q for q in k.questions if k.questions[q]["flow_stage"] == "closing"}:
        plan.append("Q-10-01")
    return plan


def stage_of(qid: str | None) -> str:
    if not qid:
        return "introduction"
    return kb().questions[qid]["flow_stage"]
