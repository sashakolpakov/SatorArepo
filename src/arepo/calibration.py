"""Public display policy for Arepo scores."""

from .evidence import evidence_score_fields, posterior_margin

LOW_CONFIDENCE_RADIUS = 0.05


def confidence_from_probability(p_ai):
    """Return score margin from the 50/50 boundary."""
    return posterior_margin(p_ai)


def verdict_from_probability(p_ai):
    """Use a strict 50/50 Human/AI split with a low-confidence band."""
    p_ai = float(p_ai)
    if p_ai >= 0.5:
        verdict = "likely_ai"
        side = "AI"
    else:
        verdict = "likely_human"
        side = "Human"

    low_confidence = abs(p_ai - 0.5) <= LOW_CONFIDENCE_RADIUS
    if low_confidence:
        return verdict, f"Low confidence: more {side}-like", "low"
    return verdict, f"More {side}-like", "normal"


def verdict_from_prediction(prediction, p_ai):
    """Use the MoE prediction for the public verdict."""
    if prediction == "ai":
        verdict, side = "likely_ai", "AI"
    else:
        verdict, side = "likely_human", "Human"

    low_confidence = abs(float(p_ai) - 0.5) <= LOW_CONFIDENCE_RADIUS
    if low_confidence:
        return verdict, f"Low confidence: {side}", "low"
    return verdict, f"Likely {side}", "normal"


def public_score(
    p_human,
    prediction=None,
    geometric_confidence=None,
):
    """Build the public-facing MoE score object."""
    if geometric_confidence is None:
        raise ValueError("MoE geometric confidence is required for public scoring")

    p_human = round(float(p_human), 4)
    p_ai = round(1.0 - p_human, 4)
    if prediction in {"human", "ai"}:
        verdict, label, confidence_band = verdict_from_prediction(prediction, p_ai)
        basis = "vote_majority"
    else:
        verdict, label, confidence_band = verdict_from_probability(p_ai)
        basis = "posterior_50_50"

    score = {
        "verdict": verdict,
        "label": label,
        "p_human": p_human,
        "p_ai": p_ai,
        "confidence": confidence_from_probability(p_ai),
        "margin": confidence_from_probability(p_ai),
        "confidence_band": confidence_band,
        "calibration_basis": basis,
    }
    score.update(evidence_score_fields(p_ai, geometric_confidence=geometric_confidence))
    return score
