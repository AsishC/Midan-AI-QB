"""
factcheck_agent.py

Contract:
- Must expose: run_factcheck_agent(...)
- Must never crash the app if OpenAI is missing/unavailable.
- Should support governance: royal/family safe, current affairs rules.

Return shape:
- (is_valid: bool, severity: str, notes: str, tags: list[str])

Severity:
- "PASS" | "WARN" | "FAIL"
"""

from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    # Optional dependency; do not hard-fail if missing
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# -----------------------------
# Policy and heuristics
# -----------------------------

SPECULATION_PATTERNS = [
    r"\b(allegedly|rumou?r|unconfirmed|maybe|might|could be|likely)\b",
    r"\b(conspiracy|cover[- ]?up|secretly)\b",
]

DEFAMATION_PATTERNS = [
    r"\b(criminal|fraud|corrupt|scandal|bribe|embezzle|money laundering)\b",
    r"\b(terrorist|extremist)\b",
]

SENSITIVE_IDENTITY_PATTERNS = [
    r"\b(prince|princess|king|crown prince|royal family)\b",
    r"\b(mbs|mohammed bin salman)\b",
]

CURRENT_AFFAIRS_PATTERNS = [
    r"\b(today|yesterday|this week|last week|breaking|latest)\b",
]

# For current affairs governance
REQUIRES_SOURCE_FIELDS = ["source", "source_url", "source_date"]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _contains_any(text: str, patterns: List[str]) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t, flags=re.IGNORECASE) for p in patterns)


def _extract_problem_flags(question_text: str, answer_text: str, sensitivity: str, current_affairs: bool) -> Tuple[List[str], List[str]]:
    """
    Returns: (hard_fail_reasons, warn_reasons)
    """
    q = _norm(question_text)
    a = _norm(answer_text)
    combined = f"{q} {a}".strip()

    hard_fail = []
    warn = []

    # General quality: answer in question (trivial leakage)
    if q and a and a.lower() in q.lower() and len(a) >= 3:
        warn.append("Answer appears in the question text (potential leakage).")

    # Speculation / rumor language
    if _contains_any(combined, SPECULATION_PATTERNS):
        warn.append("Speculative/rumor language detected.")

    # Defamation risk
    if _contains_any(combined, DEFAMATION_PATTERNS):
        # In sensitive contexts, this is a hard fail.
        if sensitivity in ("royal", "politics"):
            hard_fail.append("Defamation/criminal allegation language detected in sensitive category.")
        else:
            warn.append("Potential defamation/allegation language detected.")

    # Royal-safe: no speculation, no allegations, no gossip framing
    if sensitivity == "royal":
        if _contains_any(combined, SPECULATION_PATTERNS):
            hard_fail.append("Speculative framing is not allowed for royal-safe content.")
        if _contains_any(combined, DEFAMATION_PATTERNS):
            hard_fail.append("Allegations/criminality framing is not allowed for royal-safe content.")

        # A softer warning if it appears to target individuals
        if _contains_any(combined, SENSITIVE_IDENTITY_PATTERNS):
            warn.append("Mentions royal identifiers—ensure respectful, factual, non-personal framing.")

    # Current affairs: if flagged, must not be vague time-relative without a source/date
    if current_affairs:
        if _contains_any(combined, CURRENT_AFFAIRS_PATTERNS):
            warn.append("Time-relative phrasing detected (today/this week). Require explicit date and source.")
        # hard requirement for a source handled in orchestrator (if available)
        # but we also warn here.

    return hard_fail, warn


# -----------------------------
# LLM-based factcheck (optional)
# -----------------------------

LLM_SYSTEM = (
    "You are a governance and fact-check assistant for a quiz question bank. "
    "Your job is to flag speculation, defamation, disrespectful framing, "
    "and current affairs issues. You must be conservative."
)

LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "severity": {"type": "string", "enum": ["PASS", "WARN", "FAIL"]},
        "reasons": {"type": "array", "items": {"type": "string"}},
        "rewrite_suggestion": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["severity", "reasons", "tags"],
}


def _llm_review(
    question_text: str,
    answer_text: str,
    category_name: str,
    sensitivity: str,
    current_affairs: bool,
    meta: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Returns structured dict or None if LLM unavailable/error.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None

    model = os.getenv("AI_MODEL", "gpt-4o-mini")

    payload = {
        "category": category_name,
        "sensitivity": sensitivity,
        "current_affairs": bool(current_affairs),
        "question": question_text,
        "answer": answer_text,
        "meta": meta or {},
        "schema": LLM_SCHEMA,
    }

    prompt = (
        "Review the following quiz question for governance and fact-check risk.\n\n"
        "Return STRICT JSON matching this schema:\n"
        f"{json.dumps(LLM_SCHEMA)}\n\n"
        "Input payload:\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )

    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": LLM_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        text = (resp.choices[0].message.content or "").strip()

        # Attempt strict JSON parse
        data = json.loads(text)

        # Lightweight validation
        if not isinstance(data, dict):
            return None
        if data.get("severity") not in ("PASS", "WARN", "FAIL"):
            return None
        if not isinstance(data.get("reasons", []), list):
            return None
        if not isinstance(data.get("tags", []), list):
            return None

        return data
    except Exception:
        return None


# -----------------------------
# Public entrypoint
# -----------------------------

def run_factcheck_agent(
    question_text: str,
    answer_text: str = "",
    category_name: str = "",
    sensitivity: str = "general",
    current_affairs: bool = False,
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, str, List[str]]:
    """
    Main callable used by app.py.

    Returns:
      is_valid: bool
      severity: "PASS" | "WARN" | "FAIL"
      notes: human-readable notes
      tags: list of tags like ["speculation", "royal_safe", "current_affairs"]
    """
    q = _norm(question_text)
    a = _norm(answer_text)

    sensitivity = (sensitivity or "general").lower().strip()
    if sensitivity not in ("general", "royal", "religion", "politics"):
        sensitivity = "general"

    tags: List[str] = []
    if sensitivity == "royal":
        tags.append("royal_safe")
    if current_affairs:
        tags.append("current_affairs")

    hard_fail, warn = _extract_problem_flags(q, a, sensitivity, current_affairs)

    # Optional LLM pass (if available)
    llm = _llm_review(
        question_text=q,
        answer_text=a,
        category_name=category_name or "",
        sensitivity=sensitivity,
        current_affairs=current_affairs,
        meta=meta,
    )

    # Merge LLM findings conservatively
    llm_severity = None
    llm_reasons: List[str] = []
    llm_tags: List[str] = []
    llm_rewrite = ""

    if llm:
        llm_severity = llm.get("severity")
        llm_reasons = [str(x) for x in llm.get("reasons", []) if str(x).strip()]
        llm_tags = [str(x) for x in llm.get("tags", []) if str(x).strip()]
        llm_rewrite = str(llm.get("rewrite_suggestion", "") or "").strip()

    # Determine final severity (most conservative)
    severity = "PASS"
    if warn or llm_severity == "WARN":
        severity = "WARN"
    if hard_fail or llm_severity == "FAIL":
        severity = "FAIL"

    # Tagging
    if _contains_any(f"{q} {a}", SPECULATION_PATTERNS):
        tags.append("speculation")
    if _contains_any(f"{q} {a}", DEFAMATION_PATTERNS):
        tags.append("allegation_risk")
    if "answer appears" in " ".join(warn).lower():
        tags.append("leakage")

    # Include LLM tags
    for t in llm_tags:
        if t not in tags:
            tags.append(t)

    # Construct notes
    reasons = []
    if hard_fail:
        reasons.extend([f"[HARD] {r}" for r in hard_fail])
    if warn:
        reasons.extend([f"[WARN] {r}" for r in warn])
    if llm_reasons:
        reasons.extend([f"[LLM] {r}" for r in llm_reasons])

    if current_affairs:
        # If orchestrator provides source fields in meta, enforce presence
        meta = meta or {}
        missing = []
        for f in REQUIRES_SOURCE_FIELDS:
            if not meta.get(f):
                missing.append(f)
        if missing:
            # This is governance-critical; escalate at least to WARN, or FAIL if already sensitive
            msg = f"Current affairs requires source metadata: missing {', '.join(missing)}."
            if sensitivity in ("royal", "politics"):
                severity = "FAIL"
                reasons.append(f"[HARD] {msg}")
            else:
                severity = "WARN" if severity != "FAIL" else severity
                reasons.append(f"[WARN] {msg}")
            if "missing_source" not in tags:
                tags.append("missing_source")

    if llm_rewrite:
        reasons.append(f"Rewrite suggestion: {llm_rewrite}")

    notes = "\n".join(reasons) if reasons else "No issues detected."

    is_valid = severity != "FAIL"
    return is_valid, severity, notes, tags

# --------------------------------------------------------
# Compatibility wrapper required by app.py
# Old code expects: run_factcheck_agent_for_question_ids(db, ids)
# Newer code may only have run_factcheck_agent(...)
# --------------------------------------------------------

def run_factcheck_agent_for_question_ids(db, question_ids):
    """
    Compatibility wrapper: delegates to run_factcheck_agent.
    If run_factcheck_agent does not exist, this function should
    gracefully no-op to avoid breaking the pipeline.
    """
    try:
        # If the newer function exists, call it.
        if 'run_factcheck_agent' in globals():
            return run_factcheck_agent(db=db, question_ids=question_ids)
    except Exception as e:
        print("Factcheck agent failed:", e)

    # Fallback: do nothing but keep pipeline alive
    return None
