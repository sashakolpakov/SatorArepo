#!/usr/bin/env python3
"""Run one-by-one expert addition experiments."""

import argparse
from collections import Counter
from pathlib import Path

from .expert_sources import local_expert_records
from .mixture_experts import (
    evaluate_guardrails,
    load_or_extract_cached_features,
    save_experts,
    train_dataset_and_cluster_experts,
)
from .training_data import load_training_records


DEFAULT_SEQUENCE = [
    ("00_baseline", []),
    ("01_historic_civic", ["historic_civic"]),
    ("02_historic_civic_expanded", ["historic_civic", "historic_civic_expanded"]),
    (
        "03_public_domain_narrative",
        ["historic_civic", "historic_civic_expanded", "public_domain_narrative"],
    ),
    (
        "04_educational_explanatory",
        [
            "historic_civic",
            "historic_civic_expanded",
            "public_domain_narrative",
            "educational_explanatory",
        ],
    ),
]


def summarize_rows(rows):
    """Summarize guardrail failures overall and by kind."""
    failures = [row for row in rows if row["status"] != "ok"]
    by_kind = Counter(row.get("kind", "unknown") for row in failures)
    return {
        "rows": len(rows),
        "failures": len(failures),
        "by_kind": dict(sorted(by_kind.items())),
    }


def write_sequence_report(results, output_path):
    """Write a markdown report for one-by-one expert additions."""
    lines = [
        "# One-by-One Expert Addition",
        "",
        "This run starts from the external source experts and adds named local experts one at a time.",
        "Each step trains a fresh expert bank and evaluates the same expanded guardrail suite.",
        "",
        "| Step | Local experts | Rows | Failures | Delta | Model | Report |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    previous = None
    for result in results:
        delta = ""
        if previous is not None:
            delta = result["summary"]["failures"] - previous["summary"]["failures"]
        experts = ", ".join(result["local_experts"]) if result["local_experts"] else "none"
        lines.append(
            f"| {result['step']} | {experts} | {result['summary']['rows']} | "
            f"{result['summary']['failures']} | {delta} | `{result['model']}` | "
            f"`{result['report']}` |"
        )
        previous = result

    lines.extend(["", "## Failures by Kind", ""])
    for result in results:
        lines.extend([
            f"### {result['step']}",
            "",
            "| Kind | Failures |",
            "|---|---:|",
        ])
        if result["summary"]["by_kind"]:
            for kind, count in result["summary"]["by_kind"].items():
                lines.append(f"| {kind} | {count} |")
        else:
            lines.append("| none | 0 |")
        lines.append("")

    lines.extend([
        "## Rule",
        "",
        "Keep an added expert only if it reduces failures in its target slice without adding worse failures elsewhere.",
        "A lower total failure count is not enough by itself; the failure kinds must also make sense.",
        "",
    ])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_sequence(args):
    """Train and evaluate cumulative expert additions."""
    base_records = load_training_records("extended", args.max_per_class)
    results = []
    previous_cache = args.base_cache

    for step, local_names in DEFAULT_SEQUENCE:
        print(f"\n=== {step}: {local_names or 'baseline'} ===", flush=True)
        records = list(base_records)
        records.extend(local_expert_records(local_names))

        cache_path = Path(args.cache_dir) / f"feature_cache_{step}.npz"
        model_path = Path(args.model_dir) / f"mixture_experts.{step}.npz"
        report_path = Path(args.report_dir) / f"expert_addition_{step}.md"

        features, labels, groups, _ = load_or_extract_cached_features(
            records,
            cache_path,
            base_cache_path=previous_cache,
        )
        experts = train_dataset_and_cluster_experts(
            features,
            labels,
            groups,
            random_state=args.random_state,
        )
        save_experts(
            model_path,
            experts,
            metadata={
                "step": step,
                "local_experts": local_names,
                "max_per_class": args.max_per_class,
            },
        )
        rows = evaluate_guardrails(
            experts,
            report_path,
            temperature=args.temperature,
            ai_veto_threshold=args.ai_veto_threshold,
            ai_veto_min_weight=args.ai_veto_min_weight,
            human_veto_threshold=args.human_veto_threshold,
            human_veto_min_weight=args.human_veto_min_weight,
            suite="expanded",
            max_wiki=args.max_wiki,
            holdout_start=args.max_per_class,
            holdout_per_class=args.holdout_per_class,
            window_words=args.window_words,
            stride_words=args.stride_words,
            min_window_words=args.min_window_words,
        )
        summary = summarize_rows(rows)
        results.append({
            "step": step,
            "local_experts": local_names,
            "model": str(model_path),
            "report": str(report_path),
            "summary": summary,
        })
        previous_cache = cache_path
        print(f"{step}: {summary['failures']} / {summary['rows']} failures", flush=True)

    write_sequence_report(results, args.output)
    print(f"Wrote {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Run one-by-one expert selection")
    parser.add_argument("--max-per-class", type=int, default=500)
    parser.add_argument("--base-cache", default="reports/feature_cache_extended_500_historic_civic.npz")
    parser.add_argument("--cache-dir", default="reports/expert_selection_cache")
    parser.add_argument("--model-dir", default="src/arepo/models/expert_selection")
    parser.add_argument("--report-dir", default="reports/expert_selection")
    parser.add_argument("--output", default="reports/expert_addition_sequence.md")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--ai-veto-threshold", type=float, default=0.89)
    parser.add_argument("--ai-veto-min-weight", type=float, default=0.20)
    parser.add_argument("--human-veto-threshold", type=float, default=0.95)
    parser.add_argument("--human-veto-min-weight", type=float, default=0.20)
    parser.add_argument("--max-wiki", type=int, default=5)
    parser.add_argument("--holdout-per-class", type=int, default=3)
    parser.add_argument("--window-words", type=int, default=120)
    parser.add_argument("--stride-words", type=int, default=60)
    parser.add_argument("--min-window-words", type=int, default=45)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    run_sequence(args)


if __name__ == "__main__":
    main()
