"""LangGraph definition of the interview coach.

Per candidate answer the graph makes one LLM call (assess). The interviewer node calls the model only for
the greeting, answers about the company, redirects and support messages; questions and reactions are fixed
texts. The final feedback adds two parallel calls. On average this is about 1.2 calls per answer.

    START ─► entry ─┬─► setup ───────────────────────► interviewer ─► END
                    ├─► assess ─► decide ─┬──────────► interviewer ─► END
                    │                     └──────────► feedback ────► END
                    └─► finished ──────────────────────────────────► END
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from . import llm, prompts
from .config import get_settings
from .i18n import STRINGS, t
from .knowledge import base_lang, kb
from .planner import build_plan, stage_of

log = logging.getLogger("coach.graph")


class CoachState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    started: bool = False
    finished: bool = False
    paused: bool = False
    lang: str = "de"
    address: str = "informal"
    feedback_mode: str = "written"
    scenario_id: str | None = None
    posting: dict | None = None
    occupation_info: dict | None = None
    occupation_label: str | None = None
    plan: list[str] = Field(default_factory=list)
    q_index: int = -1
    current_qid: str | None = None
    current_question_text: str | None = None
    stage: str = "introduction"
    followups: int = 0
    total_followups: int = 0
    last_followup_q: int = -99
    prev_stage: str | None = None
    announced: list[str] = Field(default_factory=list)
    intake_pending: bool = False
    general: bool = False
    situation: str | None = None
    distress_seen: bool = False
    support_given: bool = False
    encouraged: bool = False
    announced_encouragement: bool = False
    company_questions: int = 0
    turns: list[dict] = Field(default_factory=list)
    last_assessment: dict = Field(default_factory=dict)
    action: str | None = None
    follow_up_hint: str | None = None
    llm_calls: int = 0
    answers: int = 0
    feedback_issues: list[str] = Field(default_factory=list)


_STOP = {
    "de": {"ich", "und", "der", "die", "das", "nicht", "bin", "habe", "mit", "ist", "hallo", "guten", "tag"},
    "fr": {"je", "et", "le", "la", "les", "pas", "suis", "avec", "est", "bonjour", "salut", "merci"},
    "it": {"io", "e", "il", "la", "non", "sono", "con", "che", "buongiorno", "ciao", "grazie"},
    "gsw": {"ich", "bi", "han", "isch", "nöd", "nid", "grüezi", "grüessech", "hoi", "mer", "öppis", "gsi"},
}


def detect_lang(text: str) -> str | None:
    words = set(re.findall(r"[a-zàâäçéèêëîïôöùûüß]+", text.lower()))
    scores = {k: len(words & v) for k, v in _STOP.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return None
    if best == "gsw":
        return "gsw"
    return best


def _parse_start_command(text: str) -> dict[str, str]:
    """'/start lang=fr posting=P-04 scenario=S-04 address=formal mode=spoken'"""
    if not text.strip().startswith("/start"):
        return {}
    return dict(re.findall(r"(\w+)=([\w\-]+)", text))


def _text(m: BaseMessage) -> str:
    c = m.content
    if isinstance(c, list):
        return " ".join(b.get("text", "") for b in c if isinstance(b, dict))
    return str(c)


def _history(state: CoachState, n: int = 8) -> str:
    lines = []
    for m in state.messages[-n:]:
        who = "Candidate" if isinstance(m, HumanMessage) else "Interviewer"
        lines.append(f"{who}: {_text(m)}")
    return "\n".join(lines)


def _clean_utterance(text: str) -> str:
    text = re.sub(r"^\s*(Interviewer|Intervieweur|Intervistatore|\w+ \w+)\s*:\s*", "", text.strip())
    return text.replace("**", "").strip().strip('"')


def entry(state: CoachState) -> str:
    if not state.started or state.intake_pending:
        return "setup"
    if state.finished:
        return "finished"
    return "assess"


def setup(state: CoachState, config: RunnableConfig) -> dict[str, Any]:
    """First turn: read config / dev command, detect language, find occupation and company, build the plan.
    If nothing tells us which apprenticeship to practise (e.g. the user just says "Hallo"), ask for it first."""
    s = get_settings()
    k = kb()
    text = _text(state.messages[-1]) if state.messages else ""
    cfg = {**(config.get("configurable") or {}), **_parse_start_command(text)}

    scenario = k.scenarios.get(cfg.get("scenario") or cfg.get("scenario_id") or "")
    posting = k.postings.get(cfg.get("posting") or cfg.get("posting_id") or "")
    if scenario and not posting:
        posting = k.postings.get(scenario["posting_id"])
    occupation_info, occupation_label = None, cfg.get("occupation")
    if not posting and text and not text.startswith("/start"):
        occupation_info = k.find_apprenticeship(text)
        posting = k.posting_for(occupation_info) or (None if occupation_info else k.find_posting(text))
        if occupation_info:
            occupation_label = occupation_info["Beruf"]
        elif not posting:
            occupation_label = occupation_label or k.occupation_phrase(text)
    if occupation_label and not occupation_info and not posting:
        occupation_info = k.find_apprenticeship(occupation_label)

    detected = detect_lang(text) if not text.startswith("/start") else None
    lang = (
        cfg.get("lang")
        or (scenario or {}).get("lang")
        or (detected if detected else None)
        or (state.lang if state.intake_pending else None)
        or (posting or {}).get("lang")
        or s.default_lang
    )
    base = {
        "started": True,
        "lang": lang,
        "address": cfg.get("address", state.address if state.intake_pending else s.default_address),
        "feedback_mode": cfg.get("mode", state.feedback_mode if state.intake_pending else s.feedback_mode),
    }

    # "Stopp" before the interview has started: there is nothing to give feedback on.
    if state.intake_pending and _is_end_request(text):
        return {
            **base,
            "intake_pending": False,
            "finished": True,
            "action": "intake",
            "messages": [AIMessage(content=STRINGS["finished"][base_lang(lang)])],
        }

    general = False
    if not (scenario or posting or occupation_info or occupation_label):
        if not state.intake_pending and not UNDECIDED.search(text):
            return {
                **base,
                "intake_pending": True,
                "action": "intake",
                "messages": [AIMessage(content=STRINGS["intake"][base_lang(lang)])],
            }
        # Answer to the intake question: a short occupation name is kept as label; "don't know yet",
        # small talk or anything else starts a general interview, so the conversation never gets stuck.
        words = re.findall(r"[\wÀ-ÿ'/-]+", text)
        if (
            0 < len(words) <= 3
            and not UNDECIDED.search(text)
            and not SMALLTALK.fullmatch(" ".join(words).lower())
            and "?" not in text
        ):
            occupation_label = " ".join(words)
        else:
            general = True

    plan = build_plan(posting, scenario, k.berufsfeld_key(occupation_info), general=general)
    qid = plan[0]
    return {
        **base,
        "intake_pending": False,
        "general": general,
        "scenario_id": (scenario or {}).get("id"),
        "situation": cfg.get("situation") or (scenario or {}).get("situation") or "apprenticeship_interview",
        "posting": posting,
        "occupation_info": occupation_info,
        "occupation_label": occupation_label,
        "plan": plan,
        "q_index": 0,
        "current_qid": qid,
        "current_question_text": k.question_text(qid, lang),
        "stage": stage_of(qid),
        "action": "start",
    }


UNDECIDED = re.compile(
    r"(weiss (es )?(noch )?nicht|weiß (es )?(noch )?nicht|weiss nöd|weiss no nöd|keine ahnung|noch (nicht )?(sicher|entschieden)|"
    r"bin mir (noch )?nicht sicher|je ne sais pas|sais pas encore|pas encore (décidé|choisi)|aucune idée|"
    r"non (lo )?so|non ho ancora deciso|non ne ho idea|non sono ancora sicur)",
    re.I,
)
SMALLTALK = re.compile(
    r"(ja|nein|gut|danke|ja danke|gut danke|okay|ok|hallo|hoi|grüezi|merci|oui|non|bien|bonjour|salut|"
    r"ciao|grazie|sì|si|bene|buongiorno)( \w+)?"
)


def route_after_setup(state: CoachState) -> str:
    return END if state.action == "intake" else "interviewer"


BUZZWORDS = re.compile(
    r"(motiviert|dynamisch|teamfähig|flexibel|belastbar|zuverlässig und|kommunikativ|motivé|dynamique|polyvalent|"
    r"esprit d'équipe|motivat[oa]|dinamic[oa]|flessibile|spirito di squadra)",
    re.I,
)


def clean_assessment(a: dict, qid: str | None, answer: str) -> dict:
    """Keep the assess JSON inside the schema; small models sometimes drift."""
    a = dict(a or {})
    allowed = set(kb().question_criteria(qid))
    scores = a.get("scores") if isinstance(a.get("scores"), dict) else {}
    a["scores"] = {k: v for k, v in scores.items() if k in allowed and isinstance(v, int) and 1 <= v <= 4}
    flags = a.get("problem_flags") if isinstance(a.get("problem_flags"), list) else []
    if len(answer.split()) > 4:
        flags = [f for f in flags if f != "one_word"]
    known = {f.strip() for f in prompts.PROBLEM_FLAGS.split(",")}
    flags = [f for f in flags if f in known]
    if "memorized_phrases" in flags and not BUZZWORDS.search(answer):
        flags.remove("memorized_phrases")  # over-used by the model: 1 % precision in tests/eval_assess.py
    a["problem_flags"] = flags
    if a.get("coach_action") not in {"continue", "probe", "redirect", "support_and_refer"}:
        a["coach_action"] = "continue"
    a["evidence_quote"] = _grounded_quote(a.get("evidence_quote"), answer)
    return a


def _words(text: str | None) -> list[str]:
    return re.findall(r"\w+", (text or "").lower())


def _grounded_quote(quote: str | None, answer: str) -> str:
    """Evidence must be the candidate's own words. If the model's quote is not (mostly) in the answer,
    fall back to the beginning of the answer."""
    q, a = _words(quote), set(_words(answer))
    if q and sum(w in a for w in q) / len(q) >= 0.8:
        return quote.strip().strip('«»"')
    words = answer.split()
    return " ".join(words[:15]) + (" …" if len(words) > 15 else "")


def assess(state: CoachState) -> dict[str, Any]:
    answer = _text(state.messages[-1])
    if state.paused:  # the reply to a support message is not graded
        return {"last_assessment": {"coach_action": "continue", "scores": {}}, "paused": False}
    if _is_end_request(answer):  # "Stopp" is not an answer: no LLM call, no score
        return {"last_assessment": {"coach_action": "continue", "scores": {}}}
    raw = llm.chat(prompts.assess_messages(state, answer), purpose="assess", json_mode=True, max_tokens=350)
    a = clean_assessment(llm.parse_json(raw), state.current_qid, answer)
    return {"last_assessment": a, "llm_calls": state.llm_calls + 1, "answers": state.answers + 1}


MAX_FOLLOWUPS_TOTAL = 4
DISCOURAGED = re.compile(
    r"(absag(e|a|en)|abgesagt|abgelehnt|keine (lehrstelle|zusage)|nöd leicht|macht mir (den kopf|angst|stress)|"
    r"refus|pas été pris|personne ne (me|veut)|rifiut|non mi hanno pres|nessuno mi)",
    re.I,
)
DISTRESS = re.compile(
    r"((ich|i) (kann|ka|cha) (eh |sowieso |halt )?(nüt|nüüt|nichts|nix)|(nöd|nid|nicht) so schlau|"
    r"(bin|bi) (halt |eh |eifach )?(zu |z'? ?)?(dumm|blöd|nüt wert)|nimmt mi(ch)? (eh |sowieso )?(niemert|niemand|keiner)|"
    r"nehmen (eh|sowieso) (öpper|jemand|en) (anders|andere)|bringt (eh|doch) (nüt|nichts)|alles (ist |isch )?sinnlos|"
    r"will nicht mehr|mag nicht mehr|keinen sinn mehr|je suis nul(le)?(?! en)|je n'y arriverai jamais|ça sert à rien|j'en peux plus|"
    r"sono (un |una )?(stupid|incapace|negat)|non ce la faccio|non serve a niente|non valgo niente|"
    r"non sono fatt[oa] per (niente|nulla)|nessuno mi (prende|vuole)|je ne suis bon(ne)? à rien)",
    re.I,
)
NO_MORE_QUESTIONS = re.compile(
    r"(keine (weiteren |anderen )?fragen|keine fragen mehr|nichts mehr|plus (vraiment )?de questions|pas (d'autres|de) questions|"
    r"nessun'altra domanda|non ho (più|altre) domande|nessuna domanda|no more questions)",
    re.I,
)


WANTS_INFO = re.compile(
    r"(möchte (gerne |gern )?wissen|würde (gerne |gern )?wissen|frage mich|j'aimerais savoir|je voudrais savoir|je me demande|"
    r"vorrei sapere|mi chiedo|mi piacerebbe sapere|i would like to know|i wonder)",
    re.I,
)


END_REQUEST = re.compile(
    r"^\W*(stopp?|halt|ende|fin|basta|feedback)\W*$|"
    r"(gespräch (beenden|abbrechen)|ich (will|möchte|würde gern) (jetzt )?(aufhören|abbrechen|beenden)|"
    r"(will|möchte|hätte gern) (jetzt )?(mein |das )?feedback|ich wott (uf(h|ö)|abbräche)|"
    r"(je veux|j'aimerais|je voudrais) (arrêter|terminer|mon feedback|le feedback|mon retour)|"
    r"(arrêter|terminer) l'entretien|fin de l'entretien|"
    r"(voglio|vorrei) (smettere|finire|terminare|il feedback)|(finire|terminare) il colloquio)",
    re.I,
)


def _is_end_request(answer: str) -> bool:
    norm = answer.replace("’", "'").strip()
    return len(norm.split()) <= 12 and bool(END_REQUEST.search(norm)) and not DISTRESS.search(norm)


def _same_answer(a: str, b: str) -> bool:
    """The candidate gives (almost) the same answer again, e.g. a memorised text for every question.
    Short repeated replies ("je n'ai plus de questions") are non-answers, not memorised texts."""
    wa, wb = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return len(wa) >= 8 and len(wa & wb) / max(len(wa | wb), 1) >= 0.85


def decide(state: CoachState) -> dict[str, Any]:
    s = get_settings()
    k = kb()
    a = state.last_assessment
    answer = _text(state.messages[-1])
    asked = _text(state.messages[-2]) if len(state.messages) >= 2 else state.current_question_text
    # The candidate can end the interview at any time ("Stopp", "je veux arrêter", "voglio il feedback").
    if _is_end_request(answer):
        return {"action": "feedback"}
    upd: dict[str, Any] = {}
    turns = list(state.turns)
    if a.get("scores") or a.get("evidence_quote"):
        turns.append({"qid": state.current_qid, "question": asked, "answer": answer, "assessment": a})
    upd["turns"] = turns

    act = a.get("coach_action", "continue")
    norm = answer.replace("’", "'").strip()
    prev_answers = [_text(m) for m in state.messages[:-1] if isinstance(m, HumanMessage)]
    if (
        prev_answers
        and _same_answer(answer, prev_answers[-1])
        and state.followups < 1
        and state.stage not in {"candidate_questions", "closing"}
    ):
        return {**upd, "action": "repeat_question", "followups": state.followups + 1}
    qtype = k.questions.get(state.current_qid or "", {}).get("type")
    light = qtype in {"smalltalk", "offer", "closing", "situation_specific"}  # short answers are fine
    too_short = len(answer.split()) <= 3 and not light

    # Candidate questions: answer as the company until the candidate has no more questions.
    wants_info = "?" in answer or WANTS_INFO.search(norm)
    says_no_more = bool(NO_MORE_QUESTIONS.search(norm)) and not wants_info
    asks = bool(a.get("is_question_to_company")) and not says_no_more and bool(wants_info)
    if state.stage in {"candidate_questions", "closing"} and asks and state.company_questions < 3:
        return {**upd, "action": "answer_company", "company_questions": state.company_questions + 1}
    flags = a.get("problem_flags") or []
    # "support_and_refer" alone is not enough (false alarms pathologise normal answers): it must come with
    # the distress flag, or the explicit-phrase rule must fire.
    distress = (act == "support_and_refer" and "distress_signal" in flags) or bool(DISTRESS.search(norm))
    if distress and not state.support_given:
        # Pause, respond supportively, refer to a real person and drop the remaining difficult questions.
        rest = [q for q in state.plan[state.q_index + 1 :] if not q.startswith("Q-DIF")]
        return {
            **upd,
            "action": "support",
            "paused": True,
            "support_given": True,
            "distress_seen": True,
            "plan": state.plan[: state.q_index + 1] + rest,
        }
    if distress:
        upd["distress_seen"] = True
    if not state.encouraged and not distress and DISCOURAGED.search(norm):
        upd["encouraged"] = True
        upd["distress_seen"] = True  # the feedback adds the referral sentence
        # No "what if you don't get the job" questions for a discouraged candidate.
        upd["plan"] = state.plan[: state.q_index + 1] + [
            q for q in state.plan[state.q_index + 1 :] if not q.startswith("Q-DIF")
        ]
    if act == "redirect" and state.followups < s.max_followups + 1:
        return {**upd, "action": "redirect", "followups": state.followups + 1}

    # Deterministic probing policy, so interviews stay comparable across languages: probe only really weak
    # answers, at most MAX_FOLLOWUPS_TOTAL per interview, never twice on the same question and never on two
    # consecutive questions. The threshold (mean score <= 2.5) was tuned with tests/eval_assess.py.
    scores = [v for v in (a.get("scores") or {}).values() if isinstance(v, int)]
    weak = too_short or (act == "probe" and (not scores or sum(scores) / len(scores) <= 2.5))
    can_probe = (
        not light
        and state.followups < s.max_followups
        and state.total_followups < MAX_FOLLOWUPS_TOTAL
        and state.q_index - state.last_followup_q > 1
    )
    if weak and can_probe:
        return {
            **upd,
            "action": "follow_up",
            "followups": state.followups + 1,
            "total_followups": state.total_followups + 1,
            "last_followup_q": state.q_index,
            "follow_up_hint": a.get("follow_up") or None,
        }

    plan = upd.get("plan", state.plan)
    nxt = state.q_index + 1
    if nxt >= len(plan):
        return {**upd, "action": "feedback"}
    qid = plan[nxt]
    return {
        **upd,
        "action": "next",
        "q_index": nxt,
        "current_qid": qid,
        "current_question_text": k.question_text(qid, state.lang),
        "stage": stage_of(qid),
        "prev_stage": state.stage,
        "followups": 0,
    }


def route_after_decide(state: CoachState) -> str:
    return "feedback" if state.action == "feedback" else "interviewer"


PRAISE = re.compile(
    r"\b(sehr gut|toll|super|wertvoll|perfekt|genial|hervorragend|ausgezeichnet|très bien|parfait|génial|excellent|"
    r"molto bene|perfetto|ottimo|eccellente|gut durchdacht|klingt nach|werden genannt|wird genannt|"
    r"information|sehr schön|bravo|gut|bien|bene|bon|buono|klar|clair|chiaro|utile|nützlich|wichtig|important|importante|"
    r"strukturiert|organisiert|beobachtet|noté|notato|interessant\w*|spannend|sehr|très|molto|vous|Sie)\b",
    re.I,
)


def _ack(state: CoachState) -> str:
    """Short neutral reaction: a fixed phrase, or the one produced by the assess call in "llm" ack mode."""
    lang = base_lang(state.lang)
    ack = (state.last_assessment.get("ack") or "").strip().strip('"') if get_settings().ack_mode == "llm" else ""
    prev = [m for m in state.messages if isinstance(m, AIMessage)][-3:]
    prev_txt = [_text(m) for m in prev]
    repeated = any(p.lower().startswith(" ".join(ack.lower().split()[:3])) for p in prev_txt if ack)
    if not ack or "?" in ack or len(ack.split()) > 16 or PRAISE.search(ack) or repeated:
        options = STRINGS["ack"][lang]
        last = _text(prev[-1]) if prev else ""
        ack = next(
            (
                o
                for k in range(len(options))
                for o in [options[(state.answers + k) % len(options)]]
                if not last.startswith(o)
            ),
            options[0],
        )
    return ack


TRANSITIONS = {
    "motivation": {
        "de": "Jetzt möchte ich mehr über deine Motivation erfahren.",
        "fr": "Maintenant, j'aimerais en savoir plus sur ta motivation.",
        "it": "Adesso vorrei sapere di più sulla tua motivazione.",
    },
    "strengths_weaknesses": {
        "de": "Reden wir jetzt über deine Stärken und Schwächen.",
        "fr": "Parlons maintenant de tes forces et de tes points faibles.",
        "it": "Parliamo adesso dei tuoi punti forti e deboli.",
    },
    "situational": {
        "de": "Ich beschreibe dir jetzt ein paar Situationen.",
        "fr": "Je vais maintenant te décrire quelques situations.",
        "it": "Adesso ti descrivo alcune situazioni.",
    },
    "candidate_questions": {
        "de": "Wir sind fast am Ende.",
        "fr": "Nous arrivons presque à la fin.",
        "it": "Siamo quasi alla fine.",
    },
}


def interviewer(state: CoachState) -> dict[str, Any]:
    action = state.action or "next"
    if action == "next" and state.encouraged and not state.announced_encouragement:
        lang = base_lang(state.lang)
        text = f"{STRINGS['encourage'][lang]} {state.current_question_text}"
        return {"messages": [AIMessage(content=text)], "follow_up_hint": None, "announced_encouragement": True}
    qtype = kb().questions.get(state.current_qid or "", {}).get("type")
    announce = (
        action == "next"
        and state.stage in TRANSITIONS
        and state.stage not in state.announced
        and state.prev_stage != state.stage
        and (state.stage != "situational" or qtype == "situational")
    )
    if announce:
        ack = f"{_ack(state)} {TRANSITIONS[state.stage][base_lang(state.lang)]}"
        return {
            "messages": [AIMessage(content=f"{ack} {state.current_question_text}")],
            "follow_up_hint": None,
            "announced": [*state.announced, state.stage],
        }
    # No LLM call: a neutral reaction plus the planned question verbatim, consistent across languages.
    if action == "next":
        return {"messages": [AIMessage(content=f"{_ack(state)} {state.current_question_text}")], "follow_up_hint": None}
    if action == "follow_up" and state.follow_up_hint and state.follow_up_hint.strip().endswith("?"):
        return {"messages": [AIMessage(content=state.follow_up_hint.strip())], "follow_up_hint": None}

    if action == "repeat_question":
        text = STRINGS["repeat_question"][base_lang(state.lang)].format(q=state.current_question_text)
        return {"messages": [AIMessage(content=text)], "follow_up_hint": None}
    if action == "start" and state.situation == "phone_call_trial":
        p = state.posting or {}
        company = p.get("company", {}).get("name", "")
        name = (p.get("interviewer") or {}).get("name", "")
        text = STRINGS["phone_pickup"][base_lang(state.lang)].format(company=company, name=name)
        if state.current_qid != "Q-OTH-01":
            text = f"{text} {state.current_question_text}"
        return {"messages": [AIMessage(content=text)], "follow_up_hint": None}
    task = prompts.interviewer_task(action, state, state.current_question_text, state.follow_up_hint)
    user = f"CONVERSATION SO FAR:\n{_history(state)}\n\nYOUR TASK:\n{task}\n\nReply with your spoken words only."
    msgs = [{"role": "system", "content": prompts.interviewer_system(state)}, {"role": "user", "content": user}]
    out = _clean_utterance(llm.chat(msgs, purpose="interviewer", max_tokens=220))
    if action == "support" and not re.search(
        r"BIZ|Berufsberat|Lehrperson|Lehrer|Eltern|orientation|enseignant|parents|orientament|docente|genitori",
        out,
        re.I,
    ):
        out = f"{out} {STRINGS['referral'][base_lang(state.lang)]}"
    if action == "start":
        if any(isinstance(m, AIMessage) for m in state.messages):
            # The intake message already greeted the candidate: drop a second greeting.
            out = re.sub(
                r"^\s*((und|et|e)\s+)?((hallo|hoi|grüezi|guten tag|herzlich willkommen|willkommen|bonjour|salut|"
                r"bienvenue|ciao|buongiorno|benvenut[oa])\b[\s!,.]*((und|et|e)\s+)?)+",
                "",
                out,
                flags=re.I,
            )
            out = out[:1].upper() + out[1:]
        if state.general:
            out = f"{STRINGS['general_ok'][base_lang(state.lang)]} {out}"
        out = f"{out} {state.current_question_text}"
    return {
        "messages": [AIMessage(content=_swiss(out, state.lang))],
        "llm_calls": state.llm_calls + 1,
        "follow_up_hint": None,
    }


BANNED = re.compile(
    r"\b(perfekt|genial|hervorragend|ausgezeichnet|perfect|parfait|génial|excellent|perfetto|geniale|ottimo|eccellente|"
    r"nie eine lehrstelle)\w*",
    re.I,
)
# No exaggerated praise in the feedback; applied outside candidate quotes.
SOFTEN = {
    "de": {
        "perfekte": "sehr gute",
        "perfekt": "sehr gut",
        "hervorragende": "sehr gute",
        "hervorragend": "sehr gut",
        "ausgezeichnete": "sehr gute",
        "ausgezeichnet": "sehr gut",
        "genial": "stark",
        "bestens": "gut",
        "perfekten": "sehr guten",
        "perfektes": "sehr gutes",
    },
    "fr": {
        "parfaitement": "bien",
        "excellemment": "bien",
        "excellent": "bon",
        "excellente": "bonne",
        "parfait": "bien",
        "parfaite": "bien",
        "génial": "bien",
    },
    "it": {
        "perfettamente": "bene",
        "ottimamente": "bene",
        "ottimo": "buon",
        "ottima": "buona",
        "eccellente": "buono",
        "perfetto": "buono",
        "perfetta": "buona",
    },
}


def _swiss(text: str, lang: str) -> str:
    return text.replace("ß", "ss") if base_lang(lang) == "de" else text


def _enforce_table(text: str, state: CoachState, lang: str) -> str:
    """Rebuild the feedback table: all 11 criteria in fixed order with the levels computed in code."""
    k = kb()
    lines = text.splitlines()
    idx = [i for i, line in enumerate(lines) if line.strip().startswith("|")]
    found: dict[str, list[str]] = {}
    for i in idx:
        cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        if len(cells) >= 5:
            for key, c in k.criteria.items():
                if cells[0].lower().startswith(c["label"][lang].lower()[:8]):
                    found.setdefault(key, cells[:5])
    levels = prompts.criterion_levels(state)
    quotes_map = prompts.assign_quotes(state)
    no_ev = prompts.NO_EVIDENCE[lang]
    rows = [f"| {prompts.COLS[lang]} |", "|---|---|---|---|---|"]
    for key, c in k.criteria.items():
        level, quotes = levels[key]
        word = prompts.level_word(level, lang)
        cells = found.get(key)
        if level is None:
            cells = [c["label"][lang], "–", no_ev, "–", "–"]
        elif not cells or not any(x and x != "–" for x in cells[2:5]):
            cells = [c["label"][lang], word, f"«{quotes[0]}»" if quotes else "–", "–", "–"]
        else:
            cells = [c["label"][lang], word, *cells[2:5]]
        if level is not None and key in quotes_map:
            cells[2] = f"«{quotes_map[key]}»"  # a real quote, never a paraphrase
        cells = [x if x else "–" for x in cells]
        rows.append("| " + " | ".join(cells) + " |")
    table = set(idx)
    rest = [line for i, line in enumerate(lines) if i not in table]
    start = idx[0] if idx else 0
    before = [line for line in lines[:start] if line not in rows]
    after = rest[len(before) :]
    return "\n".join(before + rows + after).strip()


def _soften(text: str, lang: str) -> str:
    parts = re.split(r"(«[^»]*»)", text)  # candidate quotes stay untouched
    for i in range(0, len(parts), 2):
        for bad, good in SOFTEN.get(lang, {}).items():
            parts[i] = re.sub(rf"\b{bad}\b", good, parts[i], flags=re.I)
    return "".join(parts)


def _short_quotes(text: str, max_words: int = 18) -> str:
    def cut(m: re.Match) -> str:
        words = m.group(1).split()
        return m.group(0) if len(words) <= max_words else "«" + " ".join(words[:max_words]) + " …»"

    return re.sub(r"«([^»]*)»", cut, text)


def _level_words(text: str, lang: str) -> str:
    """Replace a bare level number in the level column by its word ('Gut' instead of '3')."""
    labels = {str(sc["value"]): sc[lang] for sc in kb().scale}
    lines = []
    for line in text.splitlines():
        cells = line.split("|")
        if line.strip().startswith("|") and len(cells) > 3 and cells[2].strip() in labels:
            cells[2] = f" {labels[cells[2].strip()]} "
            line = "|".join(cells)
        lines.append(line)
    return "\n".join(lines)


def feedback(state: CoachState) -> dict[str, Any]:
    lang = base_lang(state.lang)
    if state.feedback_mode == "spoken":
        out, calls = llm.chat(prompts.feedback_messages(state, "spoken"), purpose="feedback", max_tokens=500), 1
    else:
        # Two parallel calls (criteria 1-6 | criteria 7-11 and summary): vLLM batches them, halving the latency.
        with ThreadPoolExecutor(max_workers=2) as ex:
            fa = ex.submit(
                llm.chat, prompts.feedback_messages(state, "written", "a"), purpose="feedback", max_tokens=700
            )
            fb = ex.submit(
                llm.chat, prompts.feedback_messages(state, "written", "b"), purpose="feedback", max_tokens=900
            )
            out, calls = f"{fa.result()}\n\n{fb.result()}", 2
        out = _enforce_table(out, state, lang)
    out = _swiss(_short_quotes(_soften(_level_words(out, lang), lang)), lang)
    if state.general:
        paras = out.rstrip().split("\n\n")
        paras.insert(max(len(paras) - 1, 0), STRINGS["undecided_tip"][lang])
        out = "\n\n".join(paras)
    if state.distress_seen:
        paras = out.rstrip().split("\n\n")
        paras.insert(max(len(paras) - 1, 0), STRINGS["referral"][lang])  # the encouraging sentence stays last
        out = "\n\n".join(paras)
    own_words = re.sub(r"«[^»]*»|\"[^\"]*\"|„[^“]*“", "", out)  # candidate quotes are not checked
    issues = [f"banned word: {m.group(0)}" for m in BANNED.finditer(own_words)]
    if issues:
        log.warning("feedback rule issues: %s", issues)
    intro = t("feedback_intro", lang)
    if state.q_index + 1 < len(state.plan) and state.q_index < 5:
        # Stopped early: say that the assessment rests on very few answers.
        intro = f"{intro}\n\n{STRINGS['short_interview'][lang].format(n=len(state.turns))}"
    text = f"{intro}\n\n{out}"
    return {
        "messages": [AIMessage(content=text)],
        "finished": True,
        "llm_calls": state.llm_calls + calls,
        "feedback_issues": issues,
    }


def finished(state: CoachState) -> dict[str, Any]:
    return {"messages": [AIMessage(content=t("finished", base_lang(state.lang)))]}


def create_graph(checkpointer=None):
    g = StateGraph(CoachState)
    g.add_node("setup", setup)
    g.add_node("assess", assess)
    g.add_node("decide", decide)
    g.add_node("interviewer", interviewer)
    g.add_node("feedback", feedback)
    g.add_node("finished", finished)
    g.add_conditional_edges(START, entry, ["setup", "assess", "finished"])
    g.add_conditional_edges("setup", route_after_setup, ["interviewer", END])
    g.add_edge("assess", "decide")
    g.add_conditional_edges("decide", route_after_decide, ["interviewer", "feedback"])
    g.add_edge("interviewer", END)
    g.add_edge("feedback", END)
    g.add_edge("finished", END)
    return g.compile(checkpointer=checkpointer)
