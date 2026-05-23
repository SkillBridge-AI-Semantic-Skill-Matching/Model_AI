import re
from collections import defaultdict
from typing import Dict, Any, List


def infer_group_label_bahasa_panjang(cv_text: str, min_chars_short: int = 600) -> str:
    """Infer group_label dari proxy yang tidak sensitif.

    Proxy:
    - Bahasa: heuristik jumlah token bahasa Inggris teknis
    - Panjang CV: short/long berdasarkan panjang karakter
    """
    if cv_text is None:
        cv_text = ""

    text = cv_text.lower()

    # Heuristik kata/istilah Inggris yang umum di CV tech
    en_markers = [
        "experience",
        "projects",
        "responsibilities",
        "skills",
        "education",
        "professional",
        "react",
        "node",
        "javascript",
        "typescript",
        "python",
        "tensorflow",
        "pytorch",
        "docker",
        "kubernetes",
        "aws",
        "gcp",
        "azure",
        "reactjs",
        "front-end",
        "back-end",
        "ml",
        "ai",
        "engineer",
        "software",
    ]

    en_count = sum(1 for m in en_markers if m in text)

    bahasa = "en" if en_count >= 3 else "id"

    # Panjang CV proxy
    cv_len = len(text)
    panjang = "long" if cv_len >= min_chars_short else "short"

    return f"{bahasa}_{panjang}"


def init_fairness_accumulator() -> Dict[str, Any]:
    return {
        "groups": {},
        "total_events": 0,
    }


def add_event(acc: Dict[str, Any], event: Dict[str, Any]) -> None:
    """Accumulates a fairness event.

    event schema expected:
      {
        group_label: str,
        match_score: float,
        prob_cocok: float|None,
        has_gap: bool
      }
    """
    group_label = event.get("group_label", "unknown")
    match_score = event.get("match_score")
    prob_cocok = event.get("prob_cocok")
    has_gap = bool(event.get("has_gap", False))

    if group_label not in acc["groups"]:
        acc["groups"][group_label] = {
            "count": 0,
            "sum_match_score": 0.0,
            "sum_prob_cocok": 0.0,
            "count_prob_cocok": 0,
            "gap_count": 0,
        }

    g = acc["groups"][group_label]
    g["count"] += 1
    if match_score is not None:
        g["sum_match_score"] += float(match_score)

    if prob_cocok is not None:
        g["sum_prob_cocok"] += float(prob_cocok)
        g["count_prob_cocok"] += 1

    if has_gap:
        g["gap_count"] += 1

    acc["total_events"] += 1


def compute_fairness_report(acc: Dict[str, Any]) -> Dict[str, Any]:
    groups_report: Dict[str, Any] = {}

    for gl, g in acc.get("groups", {}).items():
        count = g.get("count", 0)
        mean_match_score = (g.get("sum_match_score", 0.0) /
                            count) if count else None
        mean_prob_cocok = (
            (g.get("sum_prob_cocok", 0.0) / g.get("count_prob_cocok", 0))
            if g.get("count_prob_cocok", 0)
            else None
        )
        gap_rate = (g.get("gap_count", 0) / count) if count else None

        groups_report[gl] = {
            "count": count,
            "mean_match_score": mean_match_score,
            "mean_prob_cocok": mean_prob_cocok,
            "gap_rate": gap_rate,
            "gap_count": g.get("gap_count", 0),
        }

    return {
        "total_events": acc.get("total_events", 0),
        "groups": groups_report,
    }
