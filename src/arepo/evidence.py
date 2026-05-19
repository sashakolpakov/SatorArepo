"""Evidence aggregation for public Human/AI scoring."""

from dataclasses import dataclass

import numpy as np


AMBIGUOUS_MARGIN = 0.10
AMBIGUOUS_GEOMETRIC_CONFIDENCE = 0.10
HARD_MARGIN = 0.25
HARD_GEOMETRIC_CONFIDENCE = 0.25


@dataclass(frozen=True)
class EvidenceThresholds:
    """Thresholds for converting scores into evidence classes."""

    ambiguous_margin: float = AMBIGUOUS_MARGIN
    ambiguous_geometric_confidence: float = AMBIGUOUS_GEOMETRIC_CONFIDENCE
    hard_margin: float = HARD_MARGIN
    hard_geometric_confidence: float = HARD_GEOMETRIC_CONFIDENCE


DEFAULT_THRESHOLDS = EvidenceThresholds()


def posterior_margin(p_ai):
    """Distance from the 50/50 boundary, scaled to [0, 1]."""
    return round(abs(float(p_ai) - 0.5) * 2.0, 4)


def evidence_class(
    p_ai,
    geometric_confidence,
    thresholds=DEFAULT_THRESHOLDS,
):
    """Classify one score as hard, soft, or ambiguous evidence."""
    p_ai = float(p_ai)
    margin = posterior_margin(p_ai)
    if geometric_confidence is None:
        raise ValueError("MoE geometric confidence is required for evidence classification")
    geom = float(geometric_confidence)
    side = "ai" if p_ai >= 0.5 else "human"

    if (
        margin <= thresholds.ambiguous_margin
        or geom <= thresholds.ambiguous_geometric_confidence
    ):
        return "ambiguous"
    if margin >= thresholds.hard_margin and geom >= thresholds.hard_geometric_confidence:
        return f"hard_{side}"
    return f"soft_{side}"


def evidence_score_fields(
    p_ai,
    geometric_confidence,
    thresholds=DEFAULT_THRESHOLDS,
):
    """Return public evidence fields for a single score."""
    p_ai = float(p_ai)
    margin = posterior_margin(p_ai)
    if geometric_confidence is None:
        raise ValueError("MoE geometric confidence is required for evidence classification")
    geom = round(float(geometric_confidence), 4)
    klass = evidence_class(p_ai, geom, thresholds=thresholds)
    return {
        "posterior_margin": margin,
        "geometric_confidence": geom,
        "evidence_class": klass,
        "hard_gate_accepted": klass in {"hard_ai", "hard_human"},
    }


def _weights(rows, key):
    return np.array([max(1, float(row.get(key, row.get("chars", 1)))) for row in rows], dtype=float)


def _longest_run(rows, klass):
    longest = 0
    current = 0
    for row in rows:
        if row.get("evidence_class") == klass:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _fraction(rows, predicate, weights):
    flags = np.array([1.0 if predicate(row) else 0.0 for row in rows], dtype=float)
    return float(np.average(flags, weights=weights))


def _weighted_side(rows, side, weight_key, weight_mode):
    rows = [row for row in rows if row.get("evidence_class") != "ambiguous"]
    if not rows:
        return 0.5
    weights = _weights(rows, weight_key)
    if weight_mode == "margin":
        weights = weights * np.array([float(row.get("posterior_margin", row.get("margin", 0.0))) for row in rows])
    elif weight_mode == "geometric":
        weights = weights * np.array([float(row.get("geometric_confidence", 0.0)) for row in rows])
    if float(weights.sum()) <= 0.0:
        return 0.5
    if side == "ai":
        values = np.array([1.0 if row.get("p_ai", 0.5) >= 0.5 else 0.0 for row in rows], dtype=float)
    else:
        values = np.array([1.0 if row.get("p_ai", 0.5) < 0.5 else 0.0 for row in rows], dtype=float)
    return float(np.average(values, weights=weights))


def _compact_window(row):
    if row is None:
        return None
    return {
        "index": row.get("index"),
        "title": row.get("title", ""),
        "scale_words": row.get("scale_words"),
        "start_word": row.get("start_word"),
        "end_word": row.get("end_word"),
        "p_ai": row.get("p_ai"),
        "p_human": row.get("p_human"),
        "posterior_margin": row.get("posterior_margin", row.get("margin")),
        "geometric_confidence": row.get("geometric_confidence"),
        "evidence_class": row.get("evidence_class"),
    }


def aggregate_evidence(rows, weight_key="word_count"):
    """Aggregate window/passage evidence into document-level metrics."""
    rows = list(rows)
    if not rows:
        return {
            "n_windows": 0,
            "mean_p_ai": 0.5,
            "mean_p_human": 0.5,
            "median_p_ai": 0.5,
            "max_p_ai": 0.5,
            "min_p_ai": 0.5,
            "variance_p_ai": 0.0,
            "hard_ai_fraction": 0.0,
            "hard_human_fraction": 0.0,
            "ambiguous_fraction": 1.0,
            "soft_evidence_balance": 0.0,
            "longest_hard_ai_run": 0,
            "longest_hard_human_run": 0,
            "strongest_hard_ai_window": None,
            "strongest_hard_human_window": None,
            "margin_weighted_ai_evidence": 0.5,
            "margin_weighted_human_evidence": 0.5,
            "geometric_confidence_weighted_ai_evidence": 0.5,
            "geometric_confidence_weighted_human_evidence": 0.5,
        }

    weights = _weights(rows, weight_key)
    p_ai = np.array([float(row["p_ai"]) for row in rows], dtype=float)
    hard_ai = [row for row in rows if row.get("evidence_class") == "hard_ai"]
    hard_human = [row for row in rows if row.get("evidence_class") == "hard_human"]
    soft_ai_fraction = _fraction(rows, lambda r: r.get("evidence_class") == "soft_ai", weights)
    soft_human_fraction = _fraction(rows, lambda r: r.get("evidence_class") == "soft_human", weights)
    strongest_hard_ai = max(hard_ai, key=lambda row: (row.get("geometric_confidence", 0.0), row["p_ai"]), default=None)
    strongest_hard_human = max(hard_human, key=lambda row: (row.get("geometric_confidence", 0.0), row["p_human"]), default=None)

    margin_ai = _weighted_side(rows, "ai", weight_key, "margin")
    geom_ai = _weighted_side(rows, "ai", weight_key, "geometric")
    return {
        "n_windows": len(rows),
        "mean_p_ai": float(np.average(p_ai, weights=weights)),
        "mean_p_human": float(1.0 - np.average(p_ai, weights=weights)),
        "median_p_ai": float(np.median(p_ai)),
        "max_p_ai": float(np.max(p_ai)),
        "min_p_ai": float(np.min(p_ai)),
        "variance_p_ai": float(np.average((p_ai - np.average(p_ai, weights=weights)) ** 2, weights=weights)),
        "hard_ai_fraction": _fraction(rows, lambda r: r.get("evidence_class") == "hard_ai", weights),
        "hard_human_fraction": _fraction(rows, lambda r: r.get("evidence_class") == "hard_human", weights),
        "ambiguous_fraction": _fraction(rows, lambda r: r.get("evidence_class") == "ambiguous", weights),
        "soft_ai_fraction": soft_ai_fraction,
        "soft_human_fraction": soft_human_fraction,
        "soft_evidence_balance": soft_ai_fraction - soft_human_fraction,
        "longest_hard_ai_run": _longest_run(rows, "hard_ai"),
        "longest_hard_human_run": _longest_run(rows, "hard_human"),
        "strongest_hard_ai_window": _compact_window(strongest_hard_ai),
        "strongest_hard_human_window": _compact_window(strongest_hard_human),
        "margin_weighted_ai_evidence": margin_ai,
        "margin_weighted_human_evidence": 1.0 - margin_ai,
        "geometric_confidence_weighted_ai_evidence": geom_ai,
        "geometric_confidence_weighted_human_evidence": 1.0 - geom_ai,
    }


def document_evidence_fields(document_score, summary):
    """Return document-level evidence labels from accumulated local evidence."""
    hard_ai = float(summary.get("hard_ai_fraction", 0.0))
    hard_human = float(summary.get("hard_human_fraction", 0.0))
    ambiguous = float(summary.get("ambiguous_fraction", 1.0))
    soft_balance = float(summary.get("soft_evidence_balance", 0.0))
    p_ai = float(document_score.get("p_ai", 0.5))
    posterior_lean = "ai" if p_ai >= 0.5 else "human"

    if int(summary.get("n_windows", 0)) <= 0:
        klass = "ambiguous"
        label = "No local evidence"
    elif hard_ai > 0.0 and hard_human > 0.0:
        klass = "ambiguous"
        label = "Mixed hard evidence"
    elif hard_ai > 0.0:
        klass = "hard_ai"
        label = "Hard AI evidence"
    elif hard_human > 0.0:
        klass = "hard_human"
        label = "Hard Human evidence"
    elif ambiguous >= 0.80:
        klass = "ambiguous"
        label = "Ambiguous evidence"
    elif soft_balance >= 0.20:
        klass = "soft_ai"
        label = "Soft AI evidence"
    elif soft_balance <= -0.20:
        klass = "soft_human"
        label = "Soft Human evidence"
    else:
        klass = "ambiguous"
        label = "Mixed soft evidence"

    return {
        "document_evidence_class": klass,
        "document_evidence_label": label,
        "posterior_lean": posterior_lean,
        "posterior_label": document_score.get("label"),
        "posterior_verdict": document_score.get("verdict"),
    }


def ranked_evidence(rows, limit=5):
    """Return compact ranked evidence lists for public display."""
    rows = list(rows)
    hard_ai = [row for row in rows if row.get("evidence_class") == "hard_ai"]
    hard_human = [row for row in rows if row.get("evidence_class") == "hard_human"]
    ambiguous = [row for row in rows if row.get("evidence_class") == "ambiguous"]
    return {
        "strongest_ai": [
            _compact_window(row)
            for row in sorted(hard_ai, key=lambda r: (r.get("geometric_confidence", 0.0), r["p_ai"]), reverse=True)[:limit]
        ],
        "strongest_human": [
            _compact_window(row)
            for row in sorted(hard_human, key=lambda r: (r.get("geometric_confidence", 0.0), r["p_human"]), reverse=True)[:limit]
        ],
        "ambiguous": [
            _compact_window(row)
            for row in sorted(ambiguous, key=lambda r: r.get("posterior_margin", r.get("margin", 0.0)))[:limit]
        ],
    }


def scale_consistency(rows):
    """Summarize how evidence changes across window scales."""
    by_scale = {}
    for row in rows:
        scale = row.get("scale_words")
        if scale is None:
            continue
        by_scale.setdefault(scale, []).append(row)
    scales = {}
    hard_ai_scales = 0
    hard_human_scales = 0
    for scale, scale_rows in sorted(by_scale.items()):
        summary = aggregate_evidence(scale_rows)
        scales[str(scale)] = {
            "n_windows": summary["n_windows"],
            "mean_p_ai": summary["mean_p_ai"],
            "hard_ai_fraction": summary["hard_ai_fraction"],
            "hard_human_fraction": summary["hard_human_fraction"],
            "ambiguous_fraction": summary["ambiguous_fraction"],
        }
        hard_ai_scales += 1 if summary["hard_ai_fraction"] > 0 else 0
        hard_human_scales += 1 if summary["hard_human_fraction"] > 0 else 0
    if not scales:
        mode = "no_window_evidence"
    elif hard_ai_scales and hard_human_scales:
        mode = "mixed_across_scales"
    elif hard_ai_scales:
        mode = "ai_evidence_across_scales" if hard_ai_scales > 1 else "ai_evidence_single_scale"
    elif hard_human_scales:
        mode = "human_evidence_across_scales" if hard_human_scales > 1 else "human_evidence_single_scale"
    else:
        mode = "no_hard_evidence"
    return {"mode": mode, "scales": scales}
