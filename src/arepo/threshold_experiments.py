#!/usr/bin/env python3
"""Threshold and abstention experiments over renormalization score CSV files."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load_rows(path, mode, window_size, statistic):
    """Load score rows for one renormalization configuration."""
    rows = []
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["mode"] != mode:
                continue
            if int(row["window_size"]) != window_size:
                continue
            if row["statistic"] != statistic:
                continue
            row["label"] = int(row["label"])
            row["score"] = float(row["asymmetry_score"])
            rows.append(row)
    return rows


def roc_auc(rows, orientation=1):
    """Compute pairwise ROC AUC for score rows."""
    positives = [orientation * r["score"] for r in rows if r["label"] == 1]
    negatives = [orientation * r["score"] for r in rows if r["label"] == 0]
    if not positives or not negatives:
        return None

    wins = 0
    ties = 0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1
            elif positive == negative:
                ties += 1
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def metrics(rows, decisions):
    """Compute metrics for decisions where None means abstain."""
    covered = [(row, decision) for row, decision in zip(rows, decisions) if decision is not None]
    tp = tn = fp = fn = 0
    for row, decision in covered:
        label = row["label"]
        if decision == 1 and label == 1:
            tp += 1
        elif decision == 1 and label == 0:
            fp += 1
        elif decision == 0 and label == 1:
            fn += 1
        else:
            tn += 1

    n_covered = len(covered)
    total = len(rows)
    return {
        "n": total,
        "covered": n_covered,
        "coverage": n_covered / total if total else 0.0,
        "abstained": total - n_covered,
        "accuracy": (tp + tn) / n_covered if n_covered else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def threshold_candidates(scores):
    """Return thresholds between observed score values."""
    values = sorted(set(scores))
    if not values:
        return []
    thresholds = [values[0] - 1e-9]
    thresholds.extend((left + right) / 2.0 for left, right in zip(values, values[1:]))
    thresholds.append(values[-1] + 1e-9)
    return thresholds


def best_threshold(rows, orientation=1, max_fpr=None):
    """Choose threshold maximizing Youden's J, optionally under an FPR cap."""
    oriented_scores = [orientation * row["score"] for row in rows]
    best = None
    for threshold in threshold_candidates(oriented_scores):
        decisions = [
            1 if orientation * row["score"] >= threshold else 0
            for row in rows
        ]
        values = metrics(rows, decisions)
        if max_fpr is not None and values["false_positive_rate"] > max_fpr:
            continue
        tpr = 1.0 - values["false_negative_rate"]
        youden = tpr - values["false_positive_rate"]
        candidate = (youden, values["accuracy"], -values["false_positive_rate"], threshold, values)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return None, None
    return best[3], best[4]


def group_by_dataset(rows):
    """Group score rows by dataset."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset"]].append(row)
    return dict(groups)


def per_dataset_calibration(rows):
    """Fit per-dataset orientation and threshold on the same rows.

    This is an oracle upper-bound diagnostic, not a deployable estimate.
    """
    calibration = {}
    for dataset, group in group_by_dataset(rows).items():
        auc = roc_auc(group)
        orientation = 1 if auc is not None and auc >= 0.5 else -1
        threshold, values = best_threshold(group, orientation)
        calibration[dataset] = {
            "auc": auc,
            "orientation": orientation,
            "threshold": threshold,
            "metrics": values,
        }
    return calibration


def apply_calibration(rows, calibration, abstain_datasets=None):
    """Apply per-dataset calibration with optional dataset abstention."""
    abstain_datasets = set(abstain_datasets or [])
    decisions = []
    for row in rows:
        if row["dataset"] in abstain_datasets:
            decisions.append(None)
            continue
        params = calibration[row["dataset"]]
        oriented = params["orientation"] * row["score"]
        decisions.append(1 if oriented >= params["threshold"] else 0)
    return metrics(rows, decisions)


def render_report(rows, output_path, mode, window_size, statistic, weak_auc_floor):
    """Write a markdown report."""
    zero = metrics(rows, [1 if row["score"] >= 0.0 else 0 for row in rows])
    global_threshold, global_metrics = best_threshold(rows, orientation=1)
    fpr_threshold, fpr_metrics = best_threshold(rows, orientation=1, max_fpr=0.05)
    calibration = per_dataset_calibration(rows)
    oracle_metrics = apply_calibration(rows, calibration)

    weak = {
        dataset
        for dataset, values in calibration.items()
        if max(values["auc"], 1.0 - values["auc"]) < weak_auc_floor
    }
    weak_metrics = apply_calibration(rows, calibration, abstain_datasets=weak)
    hard_metrics = apply_calibration(rows, calibration, abstain_datasets={"hc3", "sample"})

    lines = [
        "# Threshold And Abstention Experiment",
        "",
        "This report tests thresholding as an inference-layer problem. It keeps the same mantissa KS score and changes only orientation, threshold, and abstention policy.",
        "",
        f"- Score configuration: `{mode}` / `{window_size}` / `{statistic}`",
        "- Label convention: higher oriented score means more AI-like.",
        "- Per-dataset calibration is an oracle diagnostic, not a deployable estimate.",
        "",
        "## Strategy Comparison",
        "",
        "| Strategy | Coverage | Accuracy on covered | FPR | FNR | TP | TN | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        strategy_row("global zero threshold", zero),
        strategy_row(f"global Youden threshold {global_threshold:.4f}", global_metrics),
    ]
    if fpr_metrics is not None:
        lines.append(strategy_row(f"global FPR<=5% threshold {fpr_threshold:.4f}", fpr_metrics))
    lines.extend([
        strategy_row("per-dataset oracle orientation/threshold", oracle_metrics),
        strategy_row(f"oracle abstain weak regimes: {', '.join(sorted(weak)) or 'none'}", weak_metrics),
        strategy_row("oracle abstain HC3 + sample", hard_metrics),
        "",
        "## Per-Dataset Calibration",
        "",
        "| Dataset | AUC | Orientation | Threshold | Accuracy | FPR | FNR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset, values in sorted(calibration.items()):
        auc = values["auc"]
        threshold = values["threshold"]
        metric = values["metrics"]
        lines.append(
            f"| {dataset} | {auc:.3f} | {values['orientation']} | {threshold:.4f} | "
            f"{metric['accuracy']:.3f} | {metric['false_positive_rate']:.3f} | "
            f"{metric['false_negative_rate']:.3f} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- A single zero threshold is too conservative for AI detection: it preserves low FPR but misses most AI.",
        "- A single global threshold improves recall, but it still mixes incompatible regimes.",
        "- Per-regime orientation and thresholding improves the diagnostic upper bound substantially.",
        "- HC3 and bundled sample GPT behave like hard or inverted regimes under the production score.",
        "- The deployable version should represent short, prescriptive, formal, or otherwise regime-ambiguous text as weak evidence.",
        "",
        "## Product Rule",
        "",
        "Short formal/prescriptive text should be reported as weak authorship evidence rather than forced into a strong Human/AI verdict.",
    ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def strategy_row(name, values):
    """Render one metrics table row."""
    return (
        f"| {name} | {values['coverage']:.3f} | {values['accuracy']:.3f} | "
        f"{values['false_positive_rate']:.3f} | {values['false_negative_rate']:.3f} | "
        f"{values['tp']} | {values['tn']} | {values['fp']} | {values['fn']} |"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate thresholding and abstention policies")
    parser.add_argument("--scores", default="evaluation/renorm/static_contextual_hybrids/scores.csv")
    parser.add_argument("--mode", default="production")
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--statistic", default="pooled_norm")
    parser.add_argument("--weak-auc-floor", type=float, default=0.65)
    parser.add_argument("--output", default="reports/threshold_abstention_experiment.md")
    args = parser.parse_args()

    rows = load_rows(args.scores, args.mode, args.window_size, args.statistic)
    if not rows:
        raise SystemExit("No rows matched the requested configuration")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_report(
        rows,
        output_path,
        args.mode,
        args.window_size,
        args.statistic,
        args.weak_auc_floor,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
