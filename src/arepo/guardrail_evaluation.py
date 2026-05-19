#!/usr/bin/env python3
"""Build and evaluate hard guardrails for the MoE engine."""

import argparse
from pathlib import Path

import numpy as np

from .core import DEFAULT_MODELS
from .corpus import get_demo_corpus
from .download import (
    load_arepo_essays,
    load_hc3,
    load_mage,
    load_raid,
    load_sample_texts,
    load_wiki,
)
from .document_report import split_word_windows
from .web import DEFAULT_MOE_MODEL, _classify, _extract_features_single, _get_device, _load_moe_model, _preload_models


EXTRA_HUMAN_CONTROLS = [
    {
        "id": "classic-dickens",
        "title": "A Tale of Two Cities",
        "source": "Charles Dickens, 1859",
        "kind": "public-domain classic",
        "text": (
            "It was the best of times, it was the worst of times, it was the age of wisdom, "
            "it was the age of foolishness, it was the epoch of belief, it was the epoch of "
            "incredulity, it was the season of Light, it was the season of Darkness, it was "
            "the spring of hope, it was the winter of despair."
        ),
    },
    {
        "id": "classic-austen",
        "title": "Pride and Prejudice",
        "source": "Jane Austen, 1813",
        "kind": "public-domain classic",
        "text": (
            "It is a truth universally acknowledged, that a single man in possession of a "
            "good fortune, must be in want of a wife. However little known the feelings or "
            "views of such a man may be on his first entering a neighbourhood, this truth is "
            "so well fixed in the minds of the surrounding families, that he is considered "
            "as the rightful property of some one or other of their daughters."
        ),
    },
    {
        "id": "historic-civic-federalist",
        "title": "Federalist Style Human Control",
        "source": "Public-domain civic prose",
        "kind": "historic civic",
        "text": (
            "Among the numerous advantages promised by a well constructed Union, none "
            "deserves to be more accurately developed than its tendency to break and control "
            "the violence of faction. The friend of popular governments never finds himself "
            "so much alarmed for their character and fate as when he contemplates their "
            "propensity to this dangerous vice."
        ),
    },
    {
        "id": "formal-human-email",
        "title": "Formal Human Email Control",
        "source": "Local human-written control",
        "kind": "formal short human",
        "text": (
            "Dear team, I reviewed the revised schedule and I am comfortable with the proposed "
            "dates. Please keep the Friday checkpoint on the calendar, since that is the only "
            "time this week when everyone can confirm the final dependency list. I will send "
            "my notes before noon tomorrow."
        ),
    },
]


EXTRA_AI_CONTROLS = [
    {
        "id": "ai-policy-template",
        "title": "Synthetic Policy Template",
        "source": "Local generated-style control",
        "kind": "synthetic policy",
        "text": (
            "A practical workplace communication policy should define expectations, document "
            "the approval process, and provide clear escalation paths. First, teams should "
            "identify the responsible owner for each recurring decision. Second, managers "
            "should review the policy quarterly and update examples when common questions arise."
        ),
    },
    {
        "id": "ai-formal-email",
        "title": "Synthetic Formal Email",
        "source": "Local generated-style control",
        "kind": "synthetic formal email",
        "text": (
            "Thank you for sharing the updated timeline. I have reviewed the proposed milestones "
            "and they appear reasonable based on the current scope. To keep the project moving, "
            "I recommend confirming ownership for each deliverable and scheduling a short review "
            "after the first implementation checkpoint."
        ),
    },
    {
        "id": "ai-educational-answer",
        "title": "Synthetic Educational Answer",
        "source": "Local generated-style control",
        "kind": "synthetic explanatory",
        "text": (
            "Photosynthesis is the process by which plants convert light energy into chemical "
            "energy. In simple terms, the plant uses sunlight, water, and carbon dioxide to "
            "produce glucose and oxygen. This allows the plant to store energy for growth while "
            "also releasing oxygen into the atmosphere."
        ),
    },
]


def expected_status(expected, verdict):
    """Return ok/false-positive/false-negative for a public verdict."""
    if expected == "human" and verdict == "likely_ai":
        return "FAIL false positive"
    if expected == "ai" and verdict == "likely_human":
        return "FAIL false negative"
    return "ok"


def append_labeled(records, prefix, source, kind, human_texts, ai_texts):
    """Append labeled rows from a source to the guardrail list."""
    for i, text in enumerate(human_texts, start=1):
        records.append({
            "id": f"{prefix}-human-{i}",
            "title": f"{source} Human {i}",
            "source": source,
            "kind": kind,
            "expected": "human",
            "text": text,
        })
    for i, text in enumerate(ai_texts, start=1):
        records.append({
            "id": f"{prefix}-ai-{i}",
            "title": f"{source} AI {i}",
            "source": source,
            "kind": kind,
            "expected": "ai",
            "text": text,
        })


def append_window_records(records, window_words, stride_words, min_window_words):
    """Append overlapping word-window guardrails for longer rows."""
    if window_words is None or window_words <= 0:
        return records

    expanded = list(records)
    for record in records:
        windows = split_word_windows(
            record["text"],
            window_words=window_words,
            stride_words=stride_words,
            min_words=min_window_words,
        )
        if len(windows) <= 1:
            continue
        for window_index, window in enumerate(windows, start=1):
            expanded.append({
                "id": f"{record['id']}-window-{window_index}",
                "title": f"{record['title']}: {window['title']}",
                "source": record["source"],
                "kind": f"{record.get('kind', 'unknown')} window",
                "expected": record["expected"],
                "text": window["text"],
                "parent_id": record["id"],
                "start_word": window["start_word"],
                "end_word": window["end_word"],
                "word_count": window["word_count"],
            })
    return expanded


def guardrail_records(
    max_wiki=2,
    suite="basic",
    holdout_start=500,
    holdout_per_class=3,
    window_words=None,
    stride_words=None,
    min_window_words=40,
):
    """Return labeled guardrail texts."""
    records = []

    for item in get_demo_corpus():
        records.append({
            "id": item["id"],
            "title": item["title"],
            "source": item["source"],
            "kind": item.get("kind", "demo"),
            "expected": item["expected"],
            "text": "\n\n".join(p["text"] for p in item["passages"]),
        })
        if suite == "expanded":
            for passage_index, passage in enumerate(item["passages"], start=1):
                records.append({
                    "id": f"{item['id']}-passage-{passage_index}",
                    "title": f"{item['title']}: {passage['title']}",
                    "source": item["source"],
                    "kind": f"{item.get('kind', 'demo')} passage",
                    "expected": item["expected"],
                    "text": passage["text"],
                })

    _, bundled_ai = load_sample_texts()
    for i, text in enumerate(bundled_ai[:2], start=1):
        records.append({
            "id": f"bundled-gpt-{i}",
            "title": f"Bundled GPT Sample {i}",
            "source": "src/arepo/data",
            "kind": "bundled generated",
            "expected": "ai",
            "text": text,
        })

    if suite == "expanded":
        for row in EXTRA_HUMAN_CONTROLS:
            records.append({**row, "expected": "human"})
        for row in EXTRA_AI_CONTROLS:
            records.append({**row, "expected": "ai"})

    try:
        wiki_human, wiki_ai = load_wiki(max_per_class=max_wiki)
    except Exception as exc:
        print(f"WARNING: Wiki guardrails unavailable: {exc}")
        wiki_human, wiki_ai = [], []

    for i, text in enumerate(wiki_human, start=1):
        records.append({
            "id": f"wiki-human-{i}",
            "title": f"Wiki Human Intro {i}",
            "source": "GPT-Wiki-Intro wiki_intro",
            "kind": "wiki",
            "expected": "human",
            "text": text,
        })
    for i, text in enumerate(wiki_ai, start=1):
        records.append({
            "id": f"wiki-generated-{i}",
            "title": f"Wiki Generated Intro {i}",
            "source": "GPT-Wiki-Intro generated_intro",
            "kind": "wiki generated",
            "expected": "ai",
            "text": text,
        })

    if suite == "expanded":
        holdout_loaders = [
            ("hc3-holdout", "HC3 holdout", "qa/chatbot holdout", load_hc3),
            ("wiki-holdout", "GPT-Wiki-Intro holdout", "wiki holdout", load_wiki),
            ("raid-holdout", "RAID holdout", "mixed-domain holdout", load_raid),
            ("mage-holdout", "MAGE holdout", "mixed-domain holdout", load_mage),
            (
                "arepo-holdout",
                "Arepo essay holdout",
                "essay holdout",
                load_arepo_essays,
            ),
        ]
        for prefix, source, kind, loader in holdout_loaders:
            try:
                human_texts, ai_texts = loader(
                    max_per_class=holdout_per_class,
                    skip_per_class=holdout_start,
                )
            except Exception as exc:
                print(f"WARNING: {source} guardrails unavailable: {exc}")
                human_texts, ai_texts = [], []
            append_labeled(records, prefix, source, kind, human_texts, ai_texts)

    if stride_words is None:
        stride_words = max(1, int(window_words / 2)) if window_words else None

    return append_window_records(records, window_words, stride_words, min_window_words)


def score_records(records, model_path):
    """Score guardrail records with the MoE engine."""
    device = _get_device()
    experts = _load_moe_model(model_path)
    loaded_models = _preload_models(DEFAULT_MODELS, device)

    scored = []
    for i, record in enumerate(records, start=1):
        print(f"  Guardrail {i}/{len(records)}: {record['title']}", flush=True)
        features = _extract_features_single(record["text"], loaded_models, device)
        if np.any(np.isnan(features)):
            scored.append({**record, "status": "FAIL feature extraction", "error": "NaN features"})
            continue
        score = _classify(features, experts, text=record["text"])
        status = expected_status(record["expected"], score["verdict"])
        scored.append({**record, **score, "status": status, "chars": len(record["text"])})
    return scored


def fmt_pct(value):
    """Format probability as a percent."""
    return f"{float(value) * 100:.1f}%"


def write_report(scored, path, model_path):
    """Write guardrail markdown report."""
    failures = [r for r in scored if r["status"] != "ok"]
    lines = [
        "# Guardrail Evaluation",
        "",
        f"- MoE model: `{model_path}`",
        f"- Result: {'FAIL' if failures else 'PASS'}",
        f"- Failures: {len(failures)} / {len(scored)}",
        "",
        "| ID | Kind | Expected | Verdict | Human | AI | Geom conf | Evidence | Status |",
        "|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in scored:
        if "p_human" in row:
            lines.append(
                f"| {row['id']} | {row.get('kind', '')} | {row['expected']} | {row['label']} | "
                f"{fmt_pct(row['p_human'])} | {fmt_pct(row['p_ai'])} | "
                f"{fmt_pct(row.get('geometric_confidence', 0.0))} | "
                f"{row.get('evidence_class', '')} | {row['status']} |"
            )
        else:
            lines.append(
                f"| {row['id']} | {row.get('kind', '')} | {row['expected']} | error | "
                f"n/a | n/a | n/a | {row['status']} |"
            )
    lines.extend([
        "",
        "## Rejection Rule",
        "",
        "A candidate MoE model is rejected if any mandatory guardrail fails. "
        "The current failure mode is especially bad when historic human text is false-positive AI "
        "and generated control text is false-negative human under the same model.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate hard MoE guardrails")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MOE_MODEL),
        help="Path to MoE .npz model.",
    )
    parser.add_argument("--output", default="reports/guardrail_evaluation.md")
    parser.add_argument("--max-wiki", type=int, default=2)
    parser.add_argument("--suite", choices=["basic", "expanded"], default="basic")
    parser.add_argument("--holdout-start", type=int, default=500)
    parser.add_argument("--holdout-per-class", type=int, default=3)
    parser.add_argument("--window-words", type=int, default=0)
    parser.add_argument("--stride-words", type=int, default=None)
    parser.add_argument("--min-window-words", type=int, default=40)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = guardrail_records(
        max_wiki=args.max_wiki,
        suite=args.suite,
        holdout_start=args.holdout_start,
        holdout_per_class=args.holdout_per_class,
        window_words=args.window_words,
        stride_words=args.stride_words,
        min_window_words=args.min_window_words,
    )
    scored = score_records(records, model_path)
    write_report(scored, output_path, model_path)

    failures = [r for r in scored if r["status"] != "ok"]
    print(f"Wrote {output_path}")
    print(f"Guardrail result: {'FAIL' if failures else 'PASS'} ({len(failures)} failures)")
    if args.fail_on_error and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
