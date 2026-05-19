#!/usr/bin/env python3
"""Greedy expert selection from an empty accepted expert set."""

import argparse
from collections import Counter
import copy
from pathlib import Path

import numpy as np

from .core import DEFAULT_MODELS
from .document_report import split_word_windows
from .embedding_context import load_or_extract_cached_embedding_context
from .expert_sources import LOCAL_EXPERT_FILES, local_expert_records
from .guardrail_evaluation import expected_status, fmt_pct, guardrail_records
from .mixture_experts import (
    fit_reliability_from_labeled_rows,
    kmeans_labels,
    load_or_extract_cached_features,
    predict_mixture,
    save_experts,
    stable_text_hash,
    train_dataset_and_cluster_experts,
    train_expert_for_group,
)
from .training_data import load_training_records
from .web import _extract_features_single, _get_device, _preload_models


BASE_CANDIDATES = ["sample", "cgtd", "hc3", "wiki", "raid", "mage", "arepo"]
LOCAL_CANDIDATES = [
    "historic_civic",
    "historic_civic_expanded",
    "public_domain_narrative",
    "educational_explanatory",
]


def parse_int_list(value):
    """Parse a comma-separated integer list."""
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def add_scale_tagged_window_records(
    records,
    window_words_values,
    stride_fraction=0.5,
    min_words=45,
    max_per_group_label=300,
):
    """Append labeled training windows as separate scale-tagged expert groups."""
    if not window_words_values:
        return records

    expanded = list(records)
    counts = Counter()
    for record in records:
        for window_words in window_words_values:
            key = (record["group"], int(record["label"]), window_words)
            stride_words = max(1, int(round(window_words * stride_fraction)))
            windows = split_word_windows(
                record["text"],
                window_words=window_words,
                stride_words=stride_words,
                min_words=min_words,
            )
            if len(windows) <= 1:
                continue
            for window in windows:
                if counts[key] >= max_per_group_label:
                    break
                expanded.append({
                    "text": window["text"],
                    "label": record["label"],
                    "group": f"{record['group']}@w{window_words}",
                })
                counts[key] += 1
    return expanded


def load_candidate_records(args):
    """Load all candidate source records."""
    records = load_training_records("extended", args.max_per_class)
    records.extend(local_expert_records(LOCAL_CANDIDATES))
    return add_scale_tagged_window_records(
        records,
        parse_int_list(args.train_window_words),
        stride_fraction=args.train_window_stride_fraction,
        min_words=args.train_min_window_words,
        max_per_group_label=args.train_window_max_per_group_label,
    )


def score_guardrails(
    features_by_row,
    rows,
    experts,
    args,
    context_features_by_row=None,
):
    """Score guardrails with a selected expert subset."""
    scored = []
    if context_features_by_row is None:
        context_features_by_row = [None] * len(rows)
    for row, features, context_features in zip(rows, features_by_row, context_features_by_row):
        result = predict_mixture(
            features,
            experts,
            context_features=context_features,
            temperature=args.temperature,
            ai_veto_threshold=args.ai_veto_threshold,
            ai_veto_min_weight=args.ai_veto_min_weight,
            human_veto_threshold=args.human_veto_threshold,
            human_veto_min_weight=args.human_veto_min_weight,
            competence_metric=args.competence_metric,
            competence_max_distance=args.competence_max_distance,
            competence_strength=args.competence_strength,
            reliability_strength=args.reliability_strength,
            reliability_floor=args.reliability_floor,
            alignment_threshold=args.alignment_threshold,
        )
        if args.min_geometric_confidence is not None:
            confidence = result["geometric_confidence"]["confidence"]
            accepted = confidence >= args.min_geometric_confidence
            result["accepted"] = bool(accepted)
            result["acceptance_gate"] = {
                "rule": "geometric_confidence",
                "confidence": float(confidence),
                "min_confidence": float(args.min_geometric_confidence),
                "aligned_majority_weight": float(
                    result["geometric_confidence"]["aligned_majority_weight"]
                ),
                "aligned_minority_weight": float(
                    result["geometric_confidence"]["aligned_minority_weight"]
                ),
                "alignment_threshold": float(args.alignment_threshold),
                "active_expert": result["experts"][0]["group"] if result["experts"] else "",
                "context_space": args.context_space,
            }
        verdict = "likely_ai" if result["prediction"] == "ai" else "likely_human"
        status = expected_status(row["expected"], verdict)
        scored.append({**row, **result, "status": status})
    return scored


def summarize(scored):
    """Summarize failures by kind."""
    failures = [row for row in scored if row["status"] != "ok"]
    by_kind = Counter(row.get("kind", "unknown") for row in failures)
    return {
        "rows": len(scored),
        "failures": len(failures),
        "by_kind": dict(sorted(by_kind.items())),
    }


def acceptance_summary(scored):
    """Summarize accepted-vs-abstained rows for confidence-gated reports."""
    if not any("acceptance_gate" in row for row in scored):
        return None
    accepted = [row for row in scored if row.get("accepted")]
    accepted_failures = [row for row in accepted if row["status"] != "ok"]
    abstained = [row for row in scored if not row.get("accepted")]
    return {
        "accepted": len(accepted),
        "abstained": len(abstained),
        "accepted_failures": len(accepted_failures),
        "accepted_accuracy": (
            (len(accepted) - len(accepted_failures)) / len(accepted)
            if accepted else 0.0
        ),
        "coverage": len(accepted) / len(scored) if scored else 0.0,
    }


def failure_ids(scored):
    """Return the set of failed row IDs."""
    return {row["id"] for row in scored if row["status"] != "ok"}


def row_error_weight(row, args):
    """Return target coverage weight for a currently failed row."""
    target = getattr(args, "coverage_target", "all_failures")
    if target == "false_negatives" and row["status"] != "FAIL false negative":
        return 0.0
    if target == "false_positives" and row["status"] != "FAIL false positive":
        return 0.0
    if row["status"] == "FAIL false negative":
        return args.false_negative_weight
    if row["status"] == "FAIL false positive":
        return args.false_positive_weight
    return 1.0


def is_protected_break(row, args):
    """Return whether breaking a previously correct row is high-cost."""
    if row.get("expected") != "human":
        return False
    kind = row.get("kind", "").lower()
    tokens = [
        token.strip().lower()
        for token in args.protected_kind_tokens.split(",")
        if token.strip()
    ]
    return any(token in kind for token in tokens)


def set_coverage_delta(current_scored, trial_scored, args):
    """Compute set-coverage utility for adding one candidate expert."""
    if current_scored is None:
        failures = len([row for row in trial_scored if row["status"] != "ok"])
        fixed_weight = len(trial_scored) - failures
        return {
            "utility": float(fixed_weight),
            "fixed": int(fixed_weight),
            "fixed_weight": float(fixed_weight),
            "broken": 0,
            "protected_broken": 0,
            "break_penalty": 0.0,
        }

    fixed = 0
    fixed_weight = 0.0
    broken = 0
    protected_broken = 0
    break_penalty = 0.0
    for before, after in zip(current_scored, trial_scored):
        before_failed = before["status"] != "ok"
        after_failed = after["status"] != "ok"
        if before_failed and not after_failed:
            fixed += 1
            fixed_weight += row_error_weight(before, args)
        elif not before_failed and after_failed:
            broken += 1
            if is_protected_break(after, args):
                protected_broken += 1
                break_penalty += args.protected_break_penalty
            else:
                break_penalty += args.break_penalty
    utility = float(fixed_weight - break_penalty)
    if getattr(args, "max_protected_breaks", -1) >= 0 and protected_broken > args.max_protected_breaks:
        utility = float("-inf")
    if getattr(args, "max_breaks", -1) >= 0 and broken > args.max_breaks:
        utility = float("-inf")
    return {
        "utility": utility,
        "fixed": fixed,
        "fixed_weight": float(fixed_weight),
        "broken": broken,
        "protected_broken": protected_broken,
        "break_penalty": float(break_penalty),
    }


def target_residual_indices(scored, args):
    """Return residual row indices matching the requested target."""
    target = getattr(args, "residual_target", "false_negatives")
    indices = []
    for i, row in enumerate(scored):
        if row["status"] == "ok":
            continue
        if target == "false_negatives" and row["status"] != "FAIL false negative":
            continue
        if target == "false_positives" and row["status"] != "FAIL false positive":
            continue
        indices.append(i)
    return indices


def nearest_balanced_indices(center, training_context, labels, per_class):
    """Return nearest balanced training indices around a residual context center."""
    center = np.asarray(center, dtype=float).reshape(1, -1)
    distances = np.linalg.norm(training_context - center, axis=1)
    selected = []
    for label in (0, 1):
        label_indices = np.flatnonzero(labels == label)
        order = label_indices[np.argsort(distances[label_indices])]
        selected.extend(order[:per_class].tolist())
    return np.array(sorted(set(selected)), dtype=int)


def train_residual_context_experts(
    scored,
    guardrail_features,
    guardrail_context_features,
    training_features,
    training_labels,
    training_context_features,
    args,
):
    """Train local experts around current residual errors without using guardrails as training rows."""
    residual_indices = target_residual_indices(scored, args)
    if not residual_indices:
        return []

    guardrail_context = guardrail_features if guardrail_context_features is None else guardrail_context_features
    training_context = training_features if training_context_features is None else training_context_features
    residual_context = guardrail_context[residual_indices]
    residual_rows = [scored[i] for i in residual_indices]

    centers = []
    if args.residual_neighborhoods > 0:
        for i, row in enumerate(residual_rows[: args.residual_neighborhoods]):
            centers.append((f"residual_neighborhood:{row['id']}", residual_context[i]))

    if args.residual_clusters > 0:
        scale = np.maximum(residual_context.std(axis=0), 1e-6)
        centered = (residual_context - residual_context.mean(axis=0)) / scale
        cluster_count = min(args.residual_clusters, len(residual_context))
        cluster_labels = kmeans_labels(
            centered,
            cluster_count,
            random_state=args.random_state,
        )
        for cluster_id in range(cluster_count):
            mask = cluster_labels == cluster_id
            if not np.any(mask):
                continue
            centers.append((
                f"residual_cluster:c{cluster_id}",
                residual_context[mask].mean(axis=0),
            ))

    experts = []
    for name, center in centers:
        selected = nearest_balanced_indices(
            center,
            training_context,
            training_labels,
            args.residual_neighbors_per_class,
        )
        if (
            np.sum(training_labels[selected] == 0) < args.residual_min_class
            or np.sum(training_labels[selected] == 1) < args.residual_min_class
        ):
            continue
        experts.extend(train_expert_for_group(
            name,
            training_features[selected],
            training_labels[selected],
            context_features=None if training_context_features is None else training_context_features[selected],
            random_state=args.random_state,
            reliability=args.reliability_strength > 0,
            reliability_bins=args.reliability_bins,
            min_reliability_count=args.min_reliability_count,
        ))
        if args.residual_include_inverted and experts:
            inverted = copy.deepcopy(experts[-1])
            inverted["group"] = f"{name}:inverted"
            inverted["orientation"] = -1 * int(inverted["orientation"])
            experts.append(inverted)
    return experts


def load_or_extract_guardrail_features(rows, cache_path):
    """Load guardrail feature cache or extract features for the current rows."""
    cache_path = Path(cache_path)
    text_hashes = np.array([stable_text_hash(row["text"]) for row in rows])
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if (
            "text_hashes" in cached
            and len(cached["text_hashes"]) == len(text_hashes)
            and np.array_equal(cached["text_hashes"], text_hashes)
        ):
            return cached["features"]

    device = _get_device()
    loaded_models = _preload_models(DEFAULT_MODELS, device)
    guardrail_features = []
    for i, row in enumerate(rows, start=1):
        print(f"Extracting guardrail {i}/{len(rows)}: {row['title']}", flush=True)
        guardrail_features.append(_extract_features_single(row["text"], loaded_models, device))
    guardrail_features = np.asarray(guardrail_features)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, features=guardrail_features, text_hashes=text_hashes)
    return guardrail_features


def reliability_calibration_rows(args):
    """Return separate held-out rows for cross-regime reliability calibration."""
    rows = guardrail_records(
        max_wiki=args.max_wiki,
        suite="expanded",
        holdout_start=args.max_per_class + args.holdout_per_class,
        holdout_per_class=args.reliability_holdout_per_class,
        window_words=args.window_words,
        stride_words=args.stride_words,
        min_window_words=args.min_window_words,
    )
    return [row for row in rows if "holdout" in row.get("kind", "")]


def calibrate_expert_reliability(experts, rows, features, args):
    """Attach held-out cross-regime reliability tables to experts."""
    labels = np.array([1 if row["expected"] == "ai" else 0 for row in rows], dtype=int)
    for expert in experts:
        expert["reliability"] = fit_reliability_from_labeled_rows(
            expert,
            features,
            labels,
            n_bins=args.reliability_bins,
            min_reliability_count=args.min_reliability_count,
        )


def heldout_test_rows(args):
    """Return disjoint held-out rows for testing a selected ensemble."""
    test_holdout_start = args.test_holdout_start
    if test_holdout_start is None:
        test_holdout_start = args.max_per_class + args.holdout_per_class
    test_window_words = args.window_words if args.test_window_words is None else args.test_window_words
    test_stride_words = args.stride_words if args.test_stride_words is None else args.test_stride_words
    rows = guardrail_records(
        max_wiki=args.max_wiki,
        suite="expanded",
        holdout_start=test_holdout_start,
        holdout_per_class=args.test_holdout_per_class,
        window_words=test_window_words,
        stride_words=test_stride_words,
        min_window_words=args.min_window_words,
    )
    if args.test_only_holdout:
        rows = [row for row in rows if "holdout" in row.get("kind", "")]
    return rows


def write_step_report(path, step_name, experts, scored, args):
    """Write a detailed report for one accepted expert set."""
    summary = summarize(scored)
    acceptance = acceptance_summary(scored)
    lines = [
        f"# Greedy Expert Step: {step_name}",
        "",
        f"- Failures: {summary['failures']} / {summary['rows']}",
        f"- Experts: {', '.join(expert['group'] for expert in experts)}",
        f"- Expert temperature: {args.temperature}",
        f"- AI disagreement threshold: {args.ai_veto_threshold}",
        f"- AI disagreement minimum expert weight: {args.ai_veto_min_weight}",
        f"- Human disagreement threshold: {args.human_veto_threshold}",
        f"- Human disagreement minimum expert weight: {args.human_veto_min_weight}",
        f"- Context space: {args.context_space}",
        f"- Competence metric: {args.competence_metric}",
        f"- Competence maximum distance: {args.competence_max_distance}",
        f"- Competence strength: {args.competence_strength}",
        f"- Reliability strength: {args.reliability_strength}",
        f"- Reliability floor: {args.reliability_floor}",
        f"- Selection objective: {args.selection_objective}",
        f"- Window words: {args.window_words}",
        f"- Window stride words: {args.stride_words}",
    ]
    if acceptance is not None:
        lines.extend([
            "- Acceptance gate: 4D geometric confidence",
            f"- Alignment threshold: {args.alignment_threshold}",
            f"- Minimum geometric confidence: {args.min_geometric_confidence}",
            f"- Accepted rows: {acceptance['accepted']} / {summary['rows']}",
            f"- Abstained rows: {acceptance['abstained']} / {summary['rows']}",
            f"- Accepted failures: {acceptance['accepted_failures']} / {acceptance['accepted']}",
            f"- Accepted accuracy: {acceptance['accepted_accuracy'] * 100:.1f}%",
            f"- Coverage: {acceptance['coverage'] * 100:.1f}%",
        ])
    lines.extend([
        "",
        "## Failure Summary",
        "",
        "| Kind | Failures |",
        "|---|---:|",
    ])
    if summary["by_kind"]:
        for kind, count in summary["by_kind"].items():
            lines.append(f"| {kind} | {count} |")
    else:
        lines.append("| none | 0 |")

    lines.extend(["", "## Guardrails", ""])
    if acceptance is None:
        lines.extend([
            "| ID | Kind | Expected | Prediction | Human | AI | Geom conf | Status | Rule | Top experts |",
            "|---|---|---|---|---:|---:|---:|---|---|---|",
        ])
    else:
        lines.extend([
            "| ID | Kind | Expected | Prediction | Human | AI | Geom conf | Status | Accepted | Gate | Rule | Top experts |",
            "|---|---|---|---|---:|---:|---:|---|---|---|---|---|",
        ])
    for row in scored:
        top = ", ".join(
            f"{expert['group']} w={expert['weight'] * 100:.1f}% ai={expert['oriented_ai'] * 100:.1f}% d={expert['context_distance']:.2f} a={expert['plane_alignment']:.2f} r={expert['reliability']:.2f}"
            for expert in row["experts"][:3]
        )
        override = row.get("override")
        if override and override["rule"] == "ai_expert_disagreement":
            rule = f"{override['expert']} ai={override['expert_ai'] * 100:.1f}% w={override['expert_weight'] * 100:.1f}%"
        elif override:
            rule = f"{override['expert']} human={override['expert_human'] * 100:.1f}% w={override['expert_weight'] * 100:.1f}%"
        else:
            rule = row["base_prediction"]
        geom = row.get("geometric_confidence", {})
        geom_conf = fmt_pct(geom.get("confidence", 0.0))
        if acceptance is None:
            lines.append(
                f"| {row['id']} | {row.get('kind', '')} | {row['expected']} | {row['prediction']} | "
                f"{fmt_pct(row['p_human'])} | {fmt_pct(row['p_ai'])} | {geom_conf} | "
                f"{row['status']} | {rule} | {top} |"
            )
        else:
            gate = row.get("acceptance_gate", {})
            threshold = gate.get("min_confidence", args.min_geometric_confidence)
            gate_label = (
                f"{fmt_pct(gate.get('confidence', 0.0))} >= {fmt_pct(threshold)}; "
                f"aligned {fmt_pct(gate.get('aligned_majority_weight', 0.0))}/"
                f"{fmt_pct(gate.get('aligned_minority_weight', 0.0))}"
            )
            accepted = "yes" if row.get("accepted") else "no"
            lines.append(
                f"| {row['id']} | {row.get('kind', '')} | {row['expected']} | {row['prediction']} | "
                f"{fmt_pct(row['p_human'])} | {fmt_pct(row['p_ai'])} | {geom_conf} | "
                f"{row['status']} | {accepted} | {gate_label} | {rule} | {top} |"
            )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_selection_report(path, trials, accepted, final_summary, args):
    """Write the greedy selection summary."""
    lines = [
        "# Greedy Expert Selection",
        "",
        "Selection starts with the configured seed experts, or zero accepted experts when no seed is provided. Each round tests every remaining candidate added to the current accepted set.",
        "",
        f"- Guardrail rows: {final_summary['rows']}",
        f"- Final failures: {final_summary['failures']}",
        f"- Window words: {args.window_words}",
        f"- Window stride words: {args.stride_words}",
        f"- Context space: {args.context_space}",
        f"- Competence metric: {args.competence_metric}",
        f"- Competence maximum distance: {args.competence_max_distance}",
        f"- Competence strength: {args.competence_strength}",
        f"- Reliability strength: {args.reliability_strength}",
        f"- Reliability floor: {args.reliability_floor}",
        f"- Selection objective: {args.selection_objective}",
        f"- False-negative fix weight: {args.false_negative_weight}",
        f"- False-positive fix weight: {args.false_positive_weight}",
        f"- Standard break penalty: {args.break_penalty}",
        f"- Protected human break penalty: {args.protected_break_penalty}",
        f"- Coverage target: {args.coverage_target}",
        f"- Minimum total improvement: {args.min_total_improvement}",
        f"- Maximum standard/protected breaks: {args.max_breaks} / {args.max_protected_breaks}",
        f"- Seed experts: {args.seed_experts or 'none'}",
        f"- Train window words: {args.train_window_words or 'none'}",
        f"- Residual target: {args.residual_target}",
        f"- Residual neighborhoods/clusters: {args.residual_neighborhoods} / {args.residual_clusters}",
        f"- Residual neighbors per class: {args.residual_neighbors_per_class}",
        f"- Residual include inverted twins: {args.residual_include_inverted}",
        f"- Protected kind tokens: {args.protected_kind_tokens}",
        "",
        "## Accepted Experts",
        "",
        "| Round | Expert Added | Failures | Improvement | Utility | Fixed | Broken | Protected Broken | Report |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in accepted:
        lines.append(
            f"| {item['round']} | {item['expert']} | {item['failures']} | "
            f"{item['improvement']} | {item.get('utility', 0):.2f} | "
            f"{item.get('fixed', 0)} | {item.get('broken', 0)} | "
            f"{item.get('protected_broken', 0)} | `{item['report']}` |"
        )
    if not accepted:
        lines.append("| 0 | none | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")

    lines.extend([
        "",
        "## Trial Ledger",
        "",
        "| Round | Candidate | Failures | Improvement | Utility | Fixed | Broken | Protected Broken | Accepted |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for trial in trials:
        lines.append(
            f"| {trial['round']} | {trial['candidate']} | {trial['failures']} | "
            f"{trial['improvement']} | {trial.get('utility', 0):.2f} | "
            f"{trial.get('fixed', 0)} | {trial.get('broken', 0)} | "
            f"{trial.get('protected_broken', 0)} | {'yes' if trial['accepted'] else 'no'} |"
        )

    lines.extend([
        "",
        "## Final Failures by Kind",
        "",
        "| Kind | Failures |",
        "|---|---:|",
    ])
    if final_summary["by_kind"]:
        for kind, count in final_summary["by_kind"].items():
            lines.append(f"| {kind} | {count} |")
    else:
        lines.append("| none | 0 |")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args):
    """Run greedy expert selection."""
    records = load_candidate_records(args)
    features, labels, groups, valid_indices = load_or_extract_cached_features(
        records,
        args.cache,
        base_cache_path=args.base_cache,
    )
    context_features = None
    if args.context_space == "embedding":
        all_context_features, context_valid_indices = load_or_extract_cached_embedding_context(
            records,
            args.embedding_cache,
            max_length=args.embedding_max_length,
        )
        if not np.array_equal(context_valid_indices, valid_indices):
            by_index = {
                int(source_idx): vector
                for source_idx, vector in zip(context_valid_indices, all_context_features)
            }
            context_features = np.vstack([by_index[int(source_idx)] for source_idx in valid_indices])
        else:
            context_features = all_context_features
    all_experts = train_dataset_and_cluster_experts(
        features,
        labels,
        groups,
        context_features=context_features,
        random_state=args.random_state,
        clusters_per_group=args.clusters_per_group,
        min_cluster_class=args.min_cluster_class,
        dataset_union_pairs=args.dataset_union_pairs,
        min_dataset_union_class=args.min_dataset_union_class,
        union_cluster_pairs=args.union_cluster_pairs,
        min_union_class=args.min_union_class,
        reliability=args.reliability_strength > 0,
        reliability_bins=args.reliability_bins,
        min_reliability_count=args.min_reliability_count,
    )
    if args.reliability_calibration == "guardrail_holdout":
        calibration_rows = reliability_calibration_rows(args)
        calibration_features = load_or_extract_guardrail_features(
            calibration_rows,
            args.reliability_guardrail_cache,
        )
        calibrate_expert_reliability(all_experts, calibration_rows, calibration_features, args)
    experts_by_group = {expert["group"]: expert for expert in all_experts}
    candidates = [name for name in BASE_CANDIDATES + LOCAL_CANDIDATES if name in experts_by_group]
    candidates.extend(sorted(name for name in experts_by_group if "@w" in name and ":c" not in name))
    if args.clusters_per_group > 0:
        candidates.extend(sorted(name for name in experts_by_group if ":c" in name))
    if args.dataset_union_pairs > 0:
        candidates.extend(sorted(name for name in experts_by_group if name.startswith("dataset_union:")))
    if args.union_cluster_pairs > 0:
        candidates.extend(sorted(name for name in experts_by_group if name.startswith("union:")))
    candidates = list(dict.fromkeys(candidates))
    candidate_prefixes = [
        prefix.strip()
        for prefix in args.candidate_prefixes.split(",")
        if prefix.strip()
    ]
    if candidate_prefixes:
        candidates = [
            name for name in candidates
            if any(name.startswith(prefix) for prefix in candidate_prefixes)
        ]
    seed_names = [
        name.strip()
        for name in args.seed_experts.split(",")
        if name.strip()
    ]
    missing_seed_names = [name for name in seed_names if name not in experts_by_group]
    if missing_seed_names:
        raise ValueError(f"Unknown seed expert(s): {', '.join(missing_seed_names)}")

    rows = guardrail_records(
        max_wiki=args.max_wiki,
        suite="expanded",
        holdout_start=args.max_per_class,
        holdout_per_class=args.holdout_per_class,
        window_words=args.window_words,
        stride_words=args.stride_words,
        min_window_words=args.min_window_words,
    )
    guardrail_features = load_or_extract_guardrail_features(rows, args.guardrail_cache)
    guardrail_context_features = None
    if args.context_space == "embedding":
        guardrail_context_features, guardrail_context_valid = load_or_extract_cached_embedding_context(
            rows,
            args.guardrail_embedding_cache,
            max_length=args.embedding_max_length,
        )
        if len(guardrail_context_valid) != len(rows) or not np.array_equal(
            guardrail_context_valid,
            np.arange(len(rows)),
        ):
            raise RuntimeError("Embedding context extraction failed for one or more guardrail rows")

    selected = list(dict.fromkeys(seed_names))
    remaining = [name for name in candidates if name not in selected]
    current_failures = len(rows)
    current_scored = None
    accepted = []
    trials = []
    if selected:
        current_scored = score_guardrails(
            guardrail_features,
            rows,
            [experts_by_group[name] for name in selected],
            args,
            context_features_by_row=guardrail_context_features,
        )
        current_failures = summarize(current_scored)["failures"]
        step_report = Path(args.report_dir) / f"{args.report_prefix}_00_seed.md"
        write_step_report(
            step_report,
            f"seed: {', '.join(selected)}",
            [experts_by_group[name] for name in selected],
            current_scored,
            args,
        )
        accepted.append({
            "round": 0,
            "expert": ", ".join(selected),
            "failures": current_failures,
            "improvement": 0,
            "utility": 0.0,
            "fixed": 0,
            "broken": 0,
            "protected_broken": 0,
            "report": str(step_report),
        })
        residual_experts = train_residual_context_experts(
            current_scored,
            guardrail_features,
            guardrail_context_features,
            features,
            labels,
            context_features,
            args,
        )
        for expert in residual_experts:
            experts_by_group[expert["group"]] = expert
            if (
                not candidate_prefixes
                or any(expert["group"].startswith(prefix) for prefix in candidate_prefixes)
            ):
                remaining.append(expert["group"])
        if residual_experts:
            print(f"Added {len(residual_experts)} residual experts", flush=True)

    for round_index in range(1, args.max_rounds + 1):
        best = None
        for candidate in remaining:
            subset_names = selected + [candidate]
            subset_experts = [experts_by_group[name] for name in subset_names]
            scored = score_guardrails(
                guardrail_features,
                rows,
                subset_experts,
                args,
                context_features_by_row=guardrail_context_features,
            )
            failures = summarize(scored)["failures"]
            improvement = current_failures - failures
            coverage = set_coverage_delta(current_scored, scored, args)
            trial = {
                "round": round_index,
                "candidate": candidate,
                "failures": failures,
                "improvement": improvement,
                **coverage,
                "accepted": False,
                "scored": scored,
            }
            trials.append(trial)
            if args.selection_objective == "failure_count":
                better = best is None or failures < best["failures"]
            else:
                better = (
                    best is None
                    or trial["utility"] > best["utility"]
                    or (
                        trial["utility"] == best["utility"]
                        and failures < best["failures"]
                    )
                )
            if better:
                best = trial

        if best is None:
            break
        if args.selection_objective == "failure_count":
            should_accept = best["improvement"] > 0
        else:
            should_accept = (
                best["utility"] > args.min_coverage_utility
                and best["improvement"] >= args.min_total_improvement
            )
        if not should_accept:
            break

        best["accepted"] = True
        selected.append(best["candidate"])
        remaining.remove(best["candidate"])
        current_failures = best["failures"]
        current_scored = best["scored"]
        step_report = Path(args.report_dir) / f"{args.report_prefix}_{round_index:02d}_{best['candidate']}.md"
        write_step_report(
            step_report,
            f"round {round_index}: {best['candidate']}",
            [experts_by_group[name] for name in selected],
            current_scored,
            args,
        )
        accepted.append({
            "round": round_index,
            "expert": best["candidate"],
            "failures": current_failures,
            "improvement": best["improvement"],
            "utility": best.get("utility", 0.0),
            "fixed": best.get("fixed", 0),
            "broken": best.get("broken", 0),
            "protected_broken": best.get("protected_broken", 0),
            "report": str(step_report),
        })
        print(
            f"Round {round_index}: accepted {best['candidate']} "
            f"({current_failures} failures, improvement {best['improvement']}, "
            f"utility {best.get('utility', 0.0):.2f})",
            flush=True,
        )

    if current_scored is None:
        final_summary = {"rows": len(rows), "failures": len(rows), "by_kind": {"no expert": len(rows)}}
        final_experts = []
    else:
        final_summary = summarize(current_scored)
        final_experts = [experts_by_group[name] for name in selected]
    if final_experts:
        save_experts(
            args.output_model,
            final_experts,
            metadata={
                "selected": selected,
                "max_per_class": args.max_per_class,
                "strategy": "greedy_from_zero",
                "seed_experts": seed_names,
                "coverage_target": args.coverage_target,
            },
        )

    write_selection_report(args.output, trials, accepted, final_summary, args)
    if final_experts and args.test_output:
        test_rows = heldout_test_rows(args)
        test_features = load_or_extract_guardrail_features(
            test_rows,
            args.test_guardrail_cache,
        )
        test_context_features = None
        if args.context_space == "embedding":
            test_context_features, test_context_valid = load_or_extract_cached_embedding_context(
                test_rows,
                args.test_guardrail_embedding_cache,
                max_length=args.embedding_max_length,
            )
            if len(test_context_valid) != len(test_rows) or not np.array_equal(
                test_context_valid,
                np.arange(len(test_rows)),
            ):
                raise RuntimeError("Embedding context extraction failed for one or more held-out test rows")
        test_scored = score_guardrails(
            test_features,
            test_rows,
            final_experts,
            args,
            context_features_by_row=test_context_features,
        )
        write_step_report(
            args.test_output,
            "held-out test",
            final_experts,
            test_scored,
            args,
        )
        test_summary = summarize(test_scored)
        test_acceptance = acceptance_summary(test_scored)
        print(f"Wrote {args.test_output}")
        print(f"Held-out test failures: {test_summary['failures']} / {test_summary['rows']}")
        if test_acceptance is not None:
            print(
                "Accepted failures: "
                f"{test_acceptance['accepted_failures']} / {test_acceptance['accepted']} "
                f"({test_acceptance['accepted_accuracy'] * 100:.1f}% accuracy, "
                f"{test_acceptance['coverage'] * 100:.1f}% coverage)"
            )
    print(f"Wrote {args.output}")
    if final_experts:
        print(f"Wrote {args.output_model}")


def main():
    parser = argparse.ArgumentParser(description="Greedy expert selection from zero experts")
    parser.add_argument("--max-per-class", type=int, default=500)
    parser.add_argument("--train-window-words", default="")
    parser.add_argument("--train-window-stride-fraction", type=float, default=0.5)
    parser.add_argument("--train-min-window-words", type=int, default=45)
    parser.add_argument("--train-window-max-per-group-label", type=int, default=300)
    parser.add_argument("--base-cache", default="src/arepo/models/report_caches/feature_cache_extended_500_historic_civic.npz")
    parser.add_argument("--cache", default="src/arepo/models/report_caches/greedy_expert_selection_features.npz")
    parser.add_argument("--embedding-cache", default="src/arepo/models/report_caches/greedy_embedding_context_features.npz")
    parser.add_argument("--guardrail-cache", default="src/arepo/models/report_caches/greedy_guardrail_features.npz")
    parser.add_argument("--guardrail-embedding-cache", default="src/arepo/models/report_caches/greedy_guardrail_embedding_context_features.npz")
    parser.add_argument("--test-guardrail-cache", default="src/arepo/models/report_caches/greedy_test_guardrail_features.npz")
    parser.add_argument("--test-guardrail-embedding-cache", default="src/arepo/models/report_caches/greedy_test_guardrail_embedding_context_features.npz")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--report-prefix", default="greedy_round")
    parser.add_argument("--output", default="reports/greedy_expert_selection.md")
    parser.add_argument("--output-model", default="src/arepo/models/greedy_expert_selection.npz")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--ai-veto-threshold", type=float, default=0.89)
    parser.add_argument("--ai-veto-min-weight", type=float, default=0.20)
    parser.add_argument("--human-veto-threshold", type=float, default=0.95)
    parser.add_argument("--human-veto-min-weight", type=float, default=0.20)
    parser.add_argument("--context-space", choices=["feature", "embedding"], default="feature")
    parser.add_argument("--embedding-max-length", type=int, default=512)
    parser.add_argument("--competence-metric", choices=["z_distance", "cosine", "plane"], default="z_distance")
    parser.add_argument("--competence-max-distance", type=float, default=None)
    parser.add_argument("--competence-strength", type=float, default=0.0)
    parser.add_argument("--reliability-strength", type=float, default=0.0)
    parser.add_argument("--reliability-floor", type=float, default=0.05)
    parser.add_argument("--reliability-bins", type=int, default=3)
    parser.add_argument("--min-reliability-count", type=int, default=3)
    parser.add_argument("--alignment-threshold", type=float, default=0.8)
    parser.add_argument(
        "--min-geometric-confidence",
        type=float,
        default=None,
        help="Accept only rows whose 4D geometric confidence is at least this value.",
    )
    parser.add_argument(
        "--reliability-calibration",
        choices=["source_cv", "guardrail_holdout"],
        default="source_cv",
    )
    parser.add_argument("--reliability-holdout-per-class", type=int, default=8)
    parser.add_argument(
        "--reliability-guardrail-cache",
        default="src/arepo/models/report_caches/greedy_reliability_guardrail_features.npz",
    )
    parser.add_argument("--max-wiki", type=int, default=5)
    parser.add_argument("--holdout-per-class", type=int, default=3)
    parser.add_argument("--test-output", default="")
    parser.add_argument("--test-holdout-start", type=int, default=None)
    parser.add_argument("--test-holdout-per-class", type=int, default=4)
    parser.add_argument("--test-window-words", type=int, default=None)
    parser.add_argument("--test-stride-words", type=int, default=None)
    parser.add_argument("--test-only-holdout", action="store_true")
    parser.add_argument("--window-words", type=int, default=120)
    parser.add_argument("--stride-words", type=int, default=60)
    parser.add_argument("--min-window-words", type=int, default=45)
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--clusters-per-group", type=int, default=0)
    parser.add_argument("--min-cluster-class", type=int, default=8)
    parser.add_argument("--dataset-union-pairs", type=int, default=0)
    parser.add_argument("--min-dataset-union-class", type=int, default=None)
    parser.add_argument("--union-cluster-pairs", type=int, default=0)
    parser.add_argument("--min-union-class", type=int, default=None)
    parser.add_argument("--candidate-prefixes", default="")
    parser.add_argument(
        "--selection-objective",
        choices=["failure_count", "set_coverage"],
        default="failure_count",
    )
    parser.add_argument("--false-negative-weight", type=float, default=1.25)
    parser.add_argument("--false-positive-weight", type=float, default=1.0)
    parser.add_argument("--break-penalty", type=float, default=1.0)
    parser.add_argument("--protected-break-penalty", type=float, default=2.5)
    parser.add_argument(
        "--coverage-target",
        choices=["all_failures", "false_negatives", "false_positives"],
        default="all_failures",
    )
    parser.add_argument("--max-breaks", type=int, default=-1)
    parser.add_argument("--max-protected-breaks", type=int, default=-1)
    parser.add_argument("--seed-experts", default="")
    parser.add_argument("--min-coverage-utility", type=float, default=0.0)
    parser.add_argument("--min-total-improvement", type=int, default=-999999)
    parser.add_argument(
        "--residual-target",
        choices=["all_failures", "false_negatives", "false_positives"],
        default="false_negatives",
    )
    parser.add_argument("--residual-neighborhoods", type=int, default=0)
    parser.add_argument("--residual-clusters", type=int, default=0)
    parser.add_argument("--residual-neighbors-per-class", type=int, default=40)
    parser.add_argument("--residual-min-class", type=int, default=10)
    parser.add_argument("--residual-include-inverted", action="store_true")
    parser.add_argument(
        "--protected-kind-tokens",
        default="public-domain,wiki,essay,historic civic",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
