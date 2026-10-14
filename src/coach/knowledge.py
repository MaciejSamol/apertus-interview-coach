"""Loads the FHGR datasets once and offers lookups used by the planner and the prompts."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import get_settings

LANGS = ("de", "fr", "it")


def base_lang(lang: str | None) -> str:
    """Output language of the coach: 'gsw-be' -> 'de', unknown -> 'de'."""
    if not lang:
        return "de"
    lang = lang.lower()
    if lang.startswith("gsw"):
        return "de"
    return lang[:2] if lang[:2] in LANGS else "de"


def _jsonl(p: Path) -> list[dict[str, Any]]:
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class Knowledge:
    questions: dict[str, dict] = field(default_factory=dict)
    postings: dict[str, dict] = field(default_factory=dict)
    candidates: dict[str, dict] = field(default_factory=dict)
    scenarios: dict[str, dict] = field(default_factory=dict)
    criteria: dict[str, dict] = field(default_factory=dict)
    scale: list[dict] = field(default_factory=list)
    guidelines: list[dict] = field(default_factory=list)
    examples_by_question: dict[str, list[dict]] = field(default_factory=lambda: defaultdict(list))
    apprenticeships: list[dict] = field(default_factory=list)
    berufsfeld_questions: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def question_text(self, qid: str, lang: str) -> str:
        q = self.questions[qid]
        return q["text"].get(base_lang(lang)) or q["text"]["de"]

    def question_criteria(self, qid: str | None) -> list[str]:
        if not qid or qid not in self.questions:
            return ["clarity", "relevance", "communication"]
        return self.questions[qid].get("criteria") or ["clarity", "relevance"]

    def criterion_label(self, key: str, lang: str) -> str:
        return self.criteria[key]["label"].get(base_lang(lang), key)

    def rubric_text(self, keys: list[str]) -> str:
        lines = []
        for k in keys:
            c = self.criteria.get(k)
            if not c:
                continue
            lv = "; ".join(f"{lvl['value']}={lvl['description']['en']}" for lvl in c["levels"])
            lines.append(f"- {k} ({c['what_is_assessed']['en']}): {lv}")
        return "\n".join(lines)

    def guideline_text(self) -> str:
        return "\n".join(f"{r['id']}: {r['en']}" for r in self.guidelines)

    def examples_for(self, qid: str | None, lang: str, exclude_id: str | None = None) -> list[dict]:
        """Two expert-annotated answers to the same question (weakest and strongest), same language if possible.
        `exclude_id` keeps the offline evaluation leave-one-out."""
        pool = [e for e in self.examples_by_question.get(qid or "", []) if e["answer_id"] != exclude_id]
        if not pool:
            return []
        bl = base_lang(lang)
        same = [e for e in pool if base_lang(e["lang"]) == bl] or pool
        same = sorted(same, key=lambda e: e["quality_level"])
        return [same[0]] if len(same) == 1 else [same[0], same[-1]]

    GENERAL_EXAMPLE_IDS = ("A-DE2-0062", "A-DE2-0027", "A-DE2-0032", "A-DE1-0017", "A-OCC1-0067")

    def general_examples(self, exclude_text: str = "") -> list[dict]:
        """Fixed examples for special answers (distress, manipulation, off-topic, unmotivated, badmouthing).
        The answer under assessment is excluded, so the offline evaluation stays leave-one-out."""
        out = []
        for exs in self.examples_by_question.values():
            for e in exs:
                if e["answer_id"] in self.GENERAL_EXAMPLE_IDS and e["answer"].strip() != exclude_text.strip():
                    out.append(e)
        return sorted(out, key=lambda e: self.GENERAL_EXAMPLE_IDS.index(e["answer_id"]))

    def find_posting(self, text: str) -> dict | None:
        """Match free text (e.g. "ich möchte Koch werden") to a posting by occupation name."""
        t = text.lower()
        best, best_score = None, 0.0
        for p in self.postings.values():
            for key in ("name_de", "name_fr", "name_it"):
                name = p["occupation"].get(key, "").lower()
                stem = name.split("/")[0].split(" ")[0][:6]
                if stem and stem in t:
                    score = len(stem)
                    if score > best_score:
                        best, best_score = p, score
        return best

    _GENERIC = frozenset(
        {
            "fachmann",
            "fachfrau",
            "frau",
            "mann",
            "efz",
            "eba",
            "assistent",
            "assistentin",
            "und",
            "in",
            "-in",
            "fachrichtung",
            "schwerpunkt",
            "lehre",
            "werden",
            "möchte",
            "ich",
            "eine",
            "einen",
            "als",
        }
    )

    def find_apprenticeship(self, text: str) -> dict | None:
        """Match free text ("ich möchte Koch werden", "Fachfrau Gesundheit") to an apprenticeship from occupations.csv."""
        words = {w.strip(".,!?;:«»\"'()").lower().replace("-", "") for w in text.split()}
        words = {w for w in words if len(w) >= 4 and w not in self._GENERIC}
        if not words:
            return None
        best, best_key = None, (0, 0, 0)
        for o in self.apprenticeships:
            tokens = set()
            for part in re.split(r"[ /(),]+", o["Beruf"].lower()):
                part = part.replace("-", "")
                if len(part) >= 4 and part not in self._GENERIC:
                    tokens.add(part)
            exact = {t for t in tokens if t in words}
            hits = [
                t
                for t in tokens
                for w in words
                if w == t or (len(t) >= 5 and len(w) >= 5 and (w.startswith(t) or t.startswith(w)))
            ]
            if hits:
                key = (len(exact), len(set(hits)), -len(o["Beruf"]))
                if key > best_key:
                    best, best_key = o, key
        return best

    _PHRASE = re.compile(
        r"(?:lehre (?:als|zum|zur)|als|zum|zur|devenir|apprentissage (?:de|d'|comme|en)|comme|diventare|"
        r"apprendistato (?:da|di|come|in)|come)\s+([\wÀ-ÿ'/-]+(?:\s+[\wÀ-ÿ'/-]+)?)|([\wÀ-ÿ/-]+)\s+werden",
        re.I,
    )

    def occupation_phrase(self, text: str) -> str | None:
        """Extract the desired occupation from free text; used as a label when nothing in the data matches."""
        m = self._PHRASE.search(text)
        if not m:
            return None
        phrase = (m.group(1) or m.group(2) or "").strip(" .,!?")
        return phrase or None

    def posting_for(self, occupation: dict | None) -> dict | None:
        """A posting (fictional company with facts) for the same occupation, if the dataset has one."""
        if not occupation:
            return None
        return next(
            (p for p in self.postings.values() if p["occupation"].get("occupation_id") == occupation.get("id")), None
        )

    def berufsfeld_key(self, occupation: dict | None) -> str | None:
        if not occupation:
            return None
        fields = (occupation.get("Berufsfelder") or "").lower()
        for q in self.questions.values():
            bf = q.get("berufsfeld")
            if bf and bf["name_de"].lower() in fields:
                return bf["key"]
        return None


def _load() -> Knowledge:
    d = get_settings().data_dir
    kb = Knowledge()
    kb.questions = {q["id"]: q for q in _jsonl(d / "questions/questions.jsonl")}
    kb.postings = {p["id"]: p for p in _jsonl(d / "profiles/postings.jsonl")}
    kb.candidates = {c["id"]: c for c in _jsonl(d / "profiles/candidates.jsonl")}
    kb.scenarios = {s["id"]: s for s in _jsonl(d / "scenarios/scenarios.jsonl")}
    crit = json.loads((d / "rubric/criteria.json").read_text(encoding="utf-8"))
    kb.criteria = {c["key"]: c for c in crit["criteria"]}
    kb.scale = crit["scale"]
    gl = json.loads((d / "rubric/feedback_guidelines.json").read_text(encoding="utf-8"))
    kb.guidelines = [r for g in gl["groups"] for r in g["rules"]]

    answers = {a["id"]: a for a in _jsonl(d / "answers/answers.jsonl")}
    for ann in _jsonl(d / "annotations/annotations.jsonl"):
        a = answers.get(ann["answer_id"])
        if not a or not a.get("question_id"):
            continue
        kb.examples_by_question[a["question_id"]].append(
            {
                "answer_id": a["id"],
                "lang": a["lang"],
                "answer": a.get("normalized_de") or a["text"],
                "quality_level": ann["quality_level"],
                "scores": ann["criteria_scores"],
                "coach_action": ann["coach_action"],
                "follow_up": ann.get("follow_up_question"),
                "flags": ann.get("problem_flags", []),
            }
        )

    for q in kb.questions.values():
        bf = q.get("berufsfeld")
        if bf:
            kb.berufsfeld_questions[bf["key"]].append(q["id"])

    occ = d / "occupations.csv"
    if occ.exists():
        with occ.open(encoding="utf-8") as f:
            kb.apprenticeships = [o for o in csv.DictReader(f) if "Grundbildung" in (o.get("Bildungstypen") or "")]
    return kb


@lru_cache(maxsize=1)
def kb() -> Knowledge:
    return _load()
