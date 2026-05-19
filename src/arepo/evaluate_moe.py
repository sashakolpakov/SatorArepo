#!/usr/bin/env python3
"""Score benchmark datasets with the MoE engine and write ROC artifacts."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .corpus import get_demo_corpus
from .core import DEFAULT_MODELS
from .download import (
    load_sample_texts,
    load_cgtd,
    load_hc3,
    load_wiki,
    load_raid,
    load_mage,
    load_arepo_essays,
)
from .stats import _roc_auc, roc_curve
from .web import DEFAULT_MOE_MODEL, _classify, _extract_features_single, _get_device, _load_moe_model, _preload_models


DATASET_LOADERS = {
    "sample": load_sample_texts,
    "cgtd": load_cgtd,
    "hc3": load_hc3,
    "wiki": load_wiki,
    "raid": load_raid,
    "mage": load_mage,
    "arepo": load_arepo_essays,
}


def metrics_at_threshold(labels, scores, threshold):
    """Compute binary metrics for a P(AI) decision threshold."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= threshold).astype(int)

    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))
    total = len(labels)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def summarize_scores(records, thresholds=(0.5, 0.6)):
    """Summarize scored records by dataset and overall."""
    summary = {}
    groups = {"overall": records}
    for record in records:
        groups.setdefault(record["dataset"], []).append(record)

    for name, group in groups.items():
        labels = np.array([r["label"] for r in group], dtype=int)
        scores = np.array([r["p_ai"] for r in group], dtype=float)
        has_both_classes = len(np.unique(labels)) == 2
        summary[name] = {
            "n": int(len(group)),
            "n_human": int(np.sum(labels == 0)),
            "n_ai": int(np.sum(labels == 1)),
            "mean_p_ai_human": float(np.mean(scores[labels == 0])) if np.any(labels == 0) else None,
            "mean_p_ai_ai": float(np.mean(scores[labels == 1])) if np.any(labels == 1) else None,
            "auc_roc": float(_roc_auc(labels, scores)) if has_both_classes else None,
            "thresholds": [metrics_at_threshold(labels, scores, t) for t in thresholds],
        }
    return summary


def load_records(dataset_names, max_per_class):
    """Load labeled evaluation records."""
    records = []
    for name in dataset_names:
        if name == "demo":
            for item in get_demo_corpus():
                label = 0 if item["expected"] == "human" else 1
                for i, passage in enumerate(item["passages"], start=1):
                    records.append({
                        "dataset": "demo",
                        "record_id": f"{item['id']}:{i}",
                        "title": f"{item['title']} - {passage['title']}",
                        "label": label,
                        "text": passage["text"],
                    })
            continue

        loader = DATASET_LOADERS[name]
        if name == "sample":
            human_texts, ai_texts = loader()
            human_texts = human_texts[:max_per_class]
            ai_texts = ai_texts[:max_per_class]
        else:
            human_texts, ai_texts = loader(max_per_class=max_per_class)
        for i, text in enumerate(human_texts, start=1):
            records.append({
                "dataset": name,
                "record_id": f"{name}:human:{i}",
                "title": f"{name} human {i}",
                "label": 0,
                "text": text,
            })
        for i, text in enumerate(ai_texts, start=1):
            records.append({
                "dataset": name,
                "record_id": f"{name}:ai:{i}",
                "title": f"{name} ai {i}",
                "label": 1,
                "text": text,
            })
    return records


def score_records(records, model_path):
    """Score records with the serialized MoE model and transformer features."""
    device = _get_device()
    experts = _load_moe_model(model_path)
    loaded_models = _preload_models(DEFAULT_MODELS, device)

    scored = []
    for i, record in enumerate(records, start=1):
        if i == 1 or i % 10 == 0:
            print(f"  Scoring {i}/{len(records)}...", flush=True)
        features = _extract_features_single(record["text"], loaded_models, device)
        if np.any(np.isnan(features)):
            continue
        score = _classify(features, experts, text=record["text"])
        scored.append({
            "dataset": record["dataset"],
            "record_id": record["record_id"],
            "title": record["title"],
            "label": record["label"],
            "label_name": "ai" if record["label"] else "human",
            "p_human": score["p_human"],
            "p_ai": score["p_ai"],
            "geometric_confidence": score["geometric_confidence"],
            "evidence_class": score["evidence_class"],
            "chars": len(record["text"]),
        })
    return scored


def write_scores_csv(records, path):
    """Write per-record scores."""
    fields = ["dataset", "record_id", "title", "label", "label_name", "p_human", "p_ai", "chars"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_roc_plots(records, output_dir):
    """Write ROC plots for overall and each two-class dataset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = {"overall": records}
    for record in records:
        groups.setdefault(record["dataset"], []).append(record)

    for name, group in groups.items():
        labels = np.array([r["label"] for r in group], dtype=int)
        if len(np.unique(labels)) != 2:
            continue
        scores = np.array([r["p_ai"] for r in group], dtype=float)
        fpr, tpr = roc_curve(labels, scores)
        auc = _roc_auc(labels, scores)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"AUC {auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC - {name}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"roc_{name}.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate MoE scores and ROC curves")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MOE_MODEL),
        help="Path to MoE .npz model.",
    )
    parser.add_argument(
        "--datasets",
        default="demo,sample,wiki",
        help=(
            "Comma-separated datasets. Choices: demo,sample,cgtd,hc3,wiki,"
            "raid,mage,arepo,all"
        ),
    )
    parser.add_argument("--max-per-class", type=int, default=20)
    parser.add_argument("--output-dir", default="evaluation")
    args = parser.parse_args()

    dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if "all" in dataset_names:
        dataset_names = ["demo", "sample", "cgtd", "hc3", "wiki", "raid", "mage", "arepo"]
    valid = set(DATASET_LOADERS) | {"demo", "all"}
    unknown = sorted(set(dataset_names) - valid)
    if unknown:
        raise SystemExit(f"Unknown datasets: {', '.join(unknown)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading records: {', '.join(dataset_names)}")
    records = load_records(dataset_names, args.max_per_class)
    print(f"Loaded {len(records)} records")

    scored = score_records(records, Path(args.model))
    summary = summarize_scores(scored)

    write_scores_csv(scored, output_dir / "scores.csv")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_roc_plots(scored, output_dir)

    print(f"Wrote {output_dir / 'scores.csv'}")
    print(f"Wrote {output_dir / 'summary.json'}")
    for name, values in summary.items():
        auc = values["auc_roc"]
        auc_text = "n/a" if auc is None else f"{auc:.3f}"
        print(f"{name}: n={values['n']} AUC={auc_text}")


if __name__ == "__main__":
    main()
