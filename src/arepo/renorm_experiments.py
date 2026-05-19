#!/usr/bin/env python3
"""Renormalization experiments for the Benford/mantissa engine.

This module deliberately keeps the mathematical object small: four KS-derived
signals from the existing transformer embedding fields. It varies window size,
overlap, representation mode, and KS normalization to test stability of the
generative-vs-masked asymmetry.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from transformers import AutoModel, AutoTokenizer, T5EncoderModel

from .core import DEFAULT_MODELS, GENERATIVE_MODELS, MASKED_MODELS
from .evaluate_moe import load_records, metrics_at_threshold
from .stats import _roc_auc, extract_mantissas
from .web import _get_device


def parse_csv_arg(value, cast=str):
    """Parse a comma-separated CLI argument."""
    return [cast(v.strip()) for v in value.split(",") if v.strip()]


def window_ranges(n_tokens, window_size, overlap):
    """Return token window ranges for a sequence length."""
    if n_tokens <= 0:
        return []
    if n_tokens <= window_size:
        return [(0, n_tokens)]

    stride = max(1, int(round(window_size * (1.0 - overlap))))
    ranges = []
    start = 0
    while start < n_tokens:
        end = min(start + window_size, n_tokens)
        ranges.append((start, end))
        if end == n_tokens:
            break
        start += stride
    return ranges


def load_models(model_names, device):
    """Load model/tokenizer pairs once."""
    loaded = {}
    for name in model_names:
        print(f"  Loading {name}...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if "t5" in name.lower():
            model = T5EncoderModel.from_pretrained(name)
        else:
            model = AutoModel.from_pretrained(name)
        model = model.float().to(device)
        model.eval()
        loaded[name] = (tokenizer, model)
    return loaded


def mode_for_model(mode, model_name):
    """Resolve an experiment mode to the representation used for one model."""
    if mode in {"static", "contextual"}:
        return mode
    if mode == "production":
        return "static" if model_name in GENERATIVE_MODELS else "contextual"
    if mode == "swapped":
        return "contextual" if model_name in GENERATIVE_MODELS else "static"
    raise ValueError(f"Unknown representation mode: {mode}")


def extract_window_embedding(model, input_ids, attention_mask, mode):
    """Extract either static input embeddings or contextual hidden states."""
    with torch.no_grad():
        if mode == "static":
            embedding_layer = model.get_input_embeddings()
            embeddings = embedding_layer(input_ids)
        elif mode == "contextual":
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                embeddings = outputs.hidden_states[-1]
            elif hasattr(outputs, "last_hidden_state"):
                embeddings = outputs.last_hidden_state
            else:
                embeddings = outputs[0]
        else:
            raise ValueError(f"Unknown representation mode: {mode}")
    return embeddings.cpu()


def model_window_stats(text, tokenizer, model, device, mode, window_size, overlap):
    """Compute raw and normalized KS aggregates for one model/text/config."""
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    ranges = window_ranges(len(token_ids), window_size, overlap)
    if not ranges:
        return None

    raw_ks = []
    norm_ks = []
    mantissa_chunks = []
    n_values = []

    all_ids = torch.tensor(token_ids, dtype=torch.long)
    for start, end in ranges:
        ids = all_ids[start:end].unsqueeze(0).to(device)
        mask = torch.ones_like(ids).to(device)
        embeddings = extract_window_embedding(model, ids, mask, mode)
        mantissas = extract_mantissas(embeddings)
        if len(mantissas) == 0:
            continue
        ks, _ = stats.kstest(mantissas, "uniform")
        raw_ks.append(float(ks))
        norm_ks.append(float(np.sqrt(len(mantissas)) * ks))
        mantissa_chunks.append(mantissas)
        n_values.append(len(mantissas))

    if not raw_ks:
        return None

    pooled = np.concatenate(mantissa_chunks)
    pooled_ks, _ = stats.kstest(pooled, "uniform")
    return {
        "n_windows": len(raw_ks),
        "n_mantissas": int(np.sum(n_values)),
        "mean_raw": float(np.mean(raw_ks)),
        "mean_norm": float(np.mean(norm_ks)),
        "pooled_raw": float(pooled_ks),
        "pooled_norm": float(np.sqrt(len(pooled)) * pooled_ks),
    }


def score_record(record, loaded, device, modes, window_sizes, overlaps):
    """Score one labeled record across renormalization configs."""
    rows = []
    for mode in modes:
        for window_size in window_sizes:
            for overlap in overlaps:
                per_model = {}
                valid = True
                for model_name in DEFAULT_MODELS:
                    tokenizer, model = loaded[model_name]
                    model_mode = mode_for_model(mode, model_name)
                    stats_for_model = model_window_stats(
                        record["text"],
                        tokenizer,
                        model,
                        device,
                        model_mode,
                        window_size,
                        overlap,
                    )
                    if stats_for_model is None:
                        valid = False
                        break
                    per_model[model_name] = stats_for_model
                if not valid:
                    continue

                for statistic in ("mean_raw", "mean_norm", "pooled_raw", "pooled_norm"):
                    gen_vals = [per_model[m][statistic] for m in GENERATIVE_MODELS]
                    mask_vals = [per_model[m][statistic] for m in MASKED_MODELS]
                    gen_avg = float(np.mean(gen_vals))
                    masked_avg = float(np.mean(mask_vals))
                    rows.append({
                        "dataset": record["dataset"],
                        "record_id": record["record_id"],
                        "title": record["title"],
                        "label": record["label"],
                        "label_name": "ai" if record["label"] else "human",
                        "mode": mode,
                        "window_size": window_size,
                        "overlap": overlap,
                        "statistic": statistic,
                        "generative_avg": gen_avg,
                        "masked_avg": masked_avg,
                        "asymmetry_score": masked_avg - gen_avg,
                        "n_windows_min": min(v["n_windows"] for v in per_model.values()),
                        "n_windows_max": max(v["n_windows"] for v in per_model.values()),
                    })
    return rows


def summarize(rows):
    """Summarize rows by config and dataset."""
    groups = {}
    for row in rows:
        key = (
            row["dataset"],
            row["mode"],
            row["window_size"],
            row["overlap"],
            row["statistic"],
        )
        groups.setdefault(key, []).append(row)

    summary = []
    for (dataset, mode, window_size, overlap, statistic), group in sorted(groups.items()):
        labels = np.array([r["label"] for r in group], dtype=int)
        scores = np.array([r["asymmetry_score"] for r in group], dtype=float)
        gen = np.array([r["generative_avg"] for r in group], dtype=float)
        masked = np.array([r["masked_avg"] for r in group], dtype=float)
        has_both = len(np.unique(labels)) == 2
        human = labels == 0
        ai = labels == 1
        metrics_50 = metrics_at_threshold(labels, scores, 0.0) if has_both else None
        summary.append({
            "dataset": dataset,
            "mode": mode,
            "window_size": int(window_size),
            "overlap": float(overlap),
            "statistic": statistic,
            "n": int(len(group)),
            "n_human": int(np.sum(human)),
            "n_ai": int(np.sum(ai)),
            "auc": float(_roc_auc(labels, scores)) if has_both else None,
            "accuracy_at_zero": metrics_50["accuracy"] if metrics_50 else None,
            "fpr_at_zero": metrics_50["false_positive_rate"] if metrics_50 else None,
            "fnr_at_zero": metrics_50["false_negative_rate"] if metrics_50 else None,
            "gen_human_mean": float(np.mean(gen[human])) if np.any(human) else None,
            "gen_ai_mean": float(np.mean(gen[ai])) if np.any(ai) else None,
            "masked_human_mean": float(np.mean(masked[human])) if np.any(human) else None,
            "masked_ai_mean": float(np.mean(masked[ai])) if np.any(ai) else None,
            "score_human_mean": float(np.mean(scores[human])) if np.any(human) else None,
            "score_ai_mean": float(np.mean(scores[ai])) if np.any(ai) else None,
        })

    # Overall configs.
    config_groups = {}
    for row in rows:
        key = (row["mode"], row["window_size"], row["overlap"], row["statistic"])
        config_groups.setdefault(key, []).append(row)
    for (mode, window_size, overlap, statistic), group in sorted(config_groups.items()):
        labels = np.array([r["label"] for r in group], dtype=int)
        scores = np.array([r["asymmetry_score"] for r in group], dtype=float)
        metrics_50 = metrics_at_threshold(labels, scores, 0.0)
        summary.append({
            "dataset": "overall",
            "mode": mode,
            "window_size": int(window_size),
            "overlap": float(overlap),
            "statistic": statistic,
            "n": int(len(group)),
            "n_human": int(np.sum(labels == 0)),
            "n_ai": int(np.sum(labels == 1)),
            "auc": float(_roc_auc(labels, scores)),
            "accuracy_at_zero": metrics_50["accuracy"],
            "fpr_at_zero": metrics_50["false_positive_rate"],
            "fnr_at_zero": metrics_50["false_negative_rate"],
            "gen_human_mean": None,
            "gen_ai_mean": None,
            "masked_human_mean": None,
            "masked_ai_mean": None,
            "score_human_mean": float(np.mean(scores[labels == 0])),
            "score_ai_mean": float(np.mean(scores[labels == 1])),
        })
    return summary


def write_csv(path, rows):
    """Write dict rows to CSV."""
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_report(summary, output_path, datasets, modes, window_sizes, overlaps):
    """Write a markdown report of outcomes."""
    overall = [r for r in summary if r["dataset"] == "overall"]
    best = sorted(overall, key=lambda r: (r["auc"] if r["auc"] is not None else -1), reverse=True)[:10]

    lines = [
        "# Renormalization Experiment Report",
        "",
        "This report varies window size, overlap, representation mode, and KS normalization while keeping the mathematical signal fixed: generative-vs-masked mantissa KS asymmetry.",
        "",
        f"- Datasets: {', '.join(datasets)}",
        f"- Representation modes: {', '.join(modes)}",
        f"- Window sizes: {', '.join(str(w) for w in window_sizes)} tokens",
        f"- Overlaps: {', '.join(str(o) for o in overlaps)}",
        "- AI score used for ROC: `masked_avg - generative_avg`; higher means more AI-like under the asymmetry hypothesis.",
        "",
        "## Best Overall Configurations",
        "",
        "| Mode | Window | Overlap | Statistic | AUC | Acc@0 | FPR@0 | FNR@0 | Human score mean | AI score mean |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best:
        lines.append(
            f"| {row['mode']} | {row['window_size']} | {row['overlap']:.2f} | "
            f"{row['statistic']} | {row['auc']:.3f} | {row['accuracy_at_zero']:.3f} | "
            f"{row['fpr_at_zero']:.3f} | {row['fnr_at_zero']:.3f} | "
            f"{row['score_human_mean']:.4f} | {row['score_ai_mean']:.4f} |"
        )

    lines.extend(["", "## Outcomes By Dataset", ""])
    for dataset in sorted({r["dataset"] for r in summary if r["dataset"] != "overall"}):
        rows = [r for r in summary if r["dataset"] == dataset]
        rows = sorted(rows, key=lambda r: (r["auc"] if r["auc"] is not None else -1), reverse=True)[:8]
        lines.extend([
            f"### {dataset}",
            "",
            "| Mode | Window | Overlap | Statistic | AUC | Gen H-A diff | Masked A-H diff | AI score mean - human score mean |",
            "|---|---:|---:|---|---:|---:|---:|---:|",
        ])
        for row in rows:
            gen_diff = None
            masked_diff = None
            score_diff = None
            if row["gen_human_mean"] is not None and row["gen_ai_mean"] is not None:
                gen_diff = row["gen_human_mean"] - row["gen_ai_mean"]
            if row["masked_human_mean"] is not None and row["masked_ai_mean"] is not None:
                masked_diff = row["masked_ai_mean"] - row["masked_human_mean"]
            if row["score_human_mean"] is not None and row["score_ai_mean"] is not None:
                score_diff = row["score_ai_mean"] - row["score_human_mean"]
            lines.append(
                f"| {row['mode']} | {row['window_size']} | {row['overlap']:.2f} | "
                f"{row['statistic']} | {row['auc']:.3f} | "
                f"{gen_diff:.4f} | {masked_diff:.4f} | {score_diff:.4f} |"
            )
        lines.append("")

    lines.extend([
        "## Interpretation Notes",
        "",
        "- `mean_raw` is the mean of local window KS distances.",
        "- `mean_norm` is the mean of local `sqrt(n) D_n`, a sample-size renormalized KS distance.",
        "- `pooled_raw` computes KS after pooling mantissas from all windows.",
        "- `pooled_norm` applies `sqrt(n)D_n` after pooling.",
        "- Overlap changes the effective action along the embedding path: it samples nearby path neighborhoods repeatedly, which can smooth discontinuities but also introduces dependence.",
        "- `static` uses input embeddings for every model.",
        "- `contextual` uses final hidden states for every model.",
        "- `production` matches the current public extractor: generative static embeddings and masked contextual hidden states.",
        "- `swapped` is the opposite control: generative contextual hidden states and masked static embeddings.",
    ])

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run renormalization experiments")
    parser.add_argument("--datasets", default="demo,wiki")
    parser.add_argument("--max-per-class", type=int, default=4)
    parser.add_argument("--window-sizes", default="64,128,256")
    parser.add_argument("--overlaps", default="0,0.5")
    parser.add_argument("--modes", default="static,contextual")
    parser.add_argument("--output-dir", default="evaluation/renorm")
    args = parser.parse_args()

    datasets = parse_csv_arg(args.datasets)
    window_sizes = parse_csv_arg(args.window_sizes, int)
    overlaps = parse_csv_arg(args.overlaps, float)
    modes = parse_csv_arg(args.modes)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading records for {', '.join(datasets)}")
    records = load_records(datasets, args.max_per_class)
    print(f"Loaded {len(records)} records")

    device = _get_device()
    loaded = load_models(DEFAULT_MODELS, device)

    rows = []
    for i, record in enumerate(records, start=1):
        print(f"Scoring record {i}/{len(records)}: {record['record_id']}", flush=True)
        rows.extend(score_record(record, loaded, device, modes, window_sizes, overlaps))

    summary = summarize(rows)
    write_csv(output_dir / "scores.csv", rows)
    write_csv(output_dir / "summary.csv", summary)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    render_report(
        summary,
        output_dir / "renormalization_report.md",
        datasets,
        modes,
        window_sizes,
        overlaps,
    )
    print(f"Wrote {output_dir / 'renormalization_report.md'}")


if __name__ == "__main__":
    main()
