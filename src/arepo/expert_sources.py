"""Local named expert sources used for mixture-expert selection experiments."""

import json
from pathlib import Path

from .training_data import add_dataset, load_historic_civic


DATA_DIR = Path(__file__).parent / "data"


LOCAL_EXPERT_FILES = {
    "historic_civic_expanded": DATA_DIR / "historic_civic_expanded.jsonl",
    "public_domain_narrative": DATA_DIR / "public_domain_narrative.jsonl",
    "educational_explanatory": DATA_DIR / "educational_explanatory.jsonl",
}


def load_jsonl_source(path):
    """Load a two-class local JSONL expert source."""
    human_texts = []
    ai_texts = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["label"] == "human":
                human_texts.append(row["text"])
            elif row["label"] == "ai":
                ai_texts.append(row["text"])
    return human_texts, ai_texts


def local_expert_records(names):
    """Return labeled records for named local expert sources."""
    records = []
    for name in names:
        if name == "historic_civic":
            human_texts, ai_texts = load_historic_civic()
        else:
            if name not in LOCAL_EXPERT_FILES:
                raise ValueError(f"Unknown local expert source: {name}")
            human_texts, ai_texts = load_jsonl_source(LOCAL_EXPERT_FILES[name])
        add_dataset(records, name, human_texts, ai_texts)
    return records
