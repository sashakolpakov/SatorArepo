"""Text-regime stub.

The public path no longer uses keyword or style heuristics for abstention.
"""


def infer_text_regime(text):
    """Return a no-op regime record for API compatibility."""
    return {
        "regime": "unclassified",
        "abstain": False,
        "reason": "",
    }
