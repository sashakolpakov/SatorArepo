#!/usr/bin/env python3
"""Generate markdown reports for document, passage, and window scores."""

import argparse
import re
from pathlib import Path

import numpy as np

from .corpus import get_demo_corpus
from .core import DEFAULT_MODELS
from .download import load_sample_texts, load_wiki
from .web import DEFAULT_MOE_MODEL, _classify, _extract_features_single, _get_device, _load_moe_model, _preload_models


def split_word_windows(text, window_words=80, stride_words=40, min_words=25):
    """Split text into overlapping word windows."""
    words = re.findall(r"\S+", text)
    if len(words) <= window_words:
        return [{
            "title": "Window 1",
            "text": " ".join(words),
            "start_word": 1,
            "end_word": len(words),
            "word_count": len(words),
        }]

    windows = []
    start = 0
    while start < len(words):
        end = min(start + window_words, len(words))
        if end - start >= min_words:
            windows.append({
                "title": f"Window {len(windows) + 1}",
                "text": " ".join(words[start:end]),
                "start_word": start + 1,
                "end_word": end,
                "word_count": end - start,
            })
        if end == len(words):
            break
        start += stride_words
    return windows


def score_text(text, loaded_models, device, experts):
    """Score one text through the same public path used by the web app."""
    features = _extract_features_single(text, loaded_models, device)
    if np.any(np.isnan(features)):
        raise ValueError("Feature extraction failed")
    score = _classify(features, experts, text=text)
    score["chars"] = len(text)
    return score


def score_segments(segments, loaded_models, device, experts):
    """Score named text segments."""
    scored = []
    for segment in segments:
        score = score_text(segment["text"], loaded_models, device, experts)
        scored.append({**segment, **score})
    return scored


def weighted_average(scored_segments):
    """Character-weighted average of segment probabilities."""
    weights = np.array([max(1, s["chars"]) for s in scored_segments], dtype=float)
    p_human = np.array([s["p_human"] for s in scored_segments], dtype=float)
    return float(np.average(p_human, weights=weights))


def expected_check(expected, verdict):
    """Return the expected-label error status for a public verdict."""
    if expected == "ai" and verdict == "likely_human":
        return "FALSE NEGATIVE: expected AI, classified human"
    if expected == "human" and verdict == "likely_ai":
        return "FALSE POSITIVE: expected human, classified AI"
    return "matches expected label"


def expected_check_prediction(expected, prediction):
    """Return the expected-label error status for a human/AI prediction."""
    if expected == "ai" and prediction == "human":
        return "FALSE NEGATIVE"
    if expected == "human" and prediction == "ai":
        return "FALSE POSITIVE"
    return "ok"


def window_weight_summary(scored_windows):
    """Summarize overlapping window scores with margin-weighted evidence."""
    word_weights = np.array(
        [max(1, s.get("word_count", s.get("end_word", 1) - s.get("start_word", 1) + 1))
         for s in scored_windows],
        dtype=float,
    )
    p_ai = np.array([s["p_ai"] for s in scored_windows], dtype=float)
    margins = np.array(
        [max(0.0, float(s.get("margin", s.get("confidence", abs(s["p_ai"] - 0.5) * 2.0))))
         for s in scored_windows],
        dtype=float,
    )
    verdict_ai = np.array([1.0 if s["verdict"] == "likely_ai" else 0.0 for s in scored_windows])
    posterior_ai = (p_ai >= 0.5).astype(float)
    low_band = np.array([1.0 if s.get("confidence_band") == "low" else 0.0 for s in scored_windows])

    posterior_avg_ai = float(np.average(p_ai, weights=word_weights))
    evidence_weights = word_weights * margins
    evidence_mass = float(evidence_weights.sum())
    if evidence_mass > 0.0:
        posterior_evidence_ai = float(np.average(posterior_ai, weights=evidence_weights))
        vote_evidence_ai = float(np.average(verdict_ai, weights=evidence_weights))
        normalized_evidence_weights = evidence_weights / evidence_mass
    else:
        posterior_evidence_ai = 0.5
        vote_evidence_ai = 0.5
        normalized_evidence_weights = np.ones_like(evidence_weights) / len(evidence_weights)

    return {
        "n_windows": len(scored_windows),
        "posterior_avg_ai": posterior_avg_ai,
        "posterior_avg_human": 1.0 - posterior_avg_ai,
        "posterior_ai_window_share": float(np.average(posterior_ai, weights=word_weights)),
        "vote_ai_window_share": float(np.average(verdict_ai, weights=word_weights)),
        "low_band_window_share": float(np.average(low_band, weights=word_weights)),
        "posterior_evidence_ai": posterior_evidence_ai,
        "posterior_evidence_human": 1.0 - posterior_evidence_ai,
        "vote_evidence_ai": vote_evidence_ai,
        "vote_evidence_human": 1.0 - vote_evidence_ai,
        "evidence_mass": evidence_mass,
        "normalized_evidence_weights": normalized_evidence_weights.tolist(),
    }


def scheme_decision(ai_score):
    """Map a scheme's AI-side score to a hard side for comparison."""
    return "ai" if float(ai_score) >= 0.5 else "human"


def margin_weighted_side_share(scored_segments, weight_key="chars", side="verdict"):
    """Return AI-side share after weighting each segment by length times margin."""
    weights = np.array([max(1, s.get(weight_key, s.get("chars", 1))) for s in scored_segments], dtype=float)
    margins = np.array(
        [max(0.0, float(s.get("margin", s.get("confidence", abs(s["p_ai"] - 0.5) * 2.0))))
         for s in scored_segments],
        dtype=float,
    )
    evidence_weights = weights * margins
    if float(evidence_weights.sum()) <= 0.0:
        return 0.5
    if side == "posterior":
        ai_side = np.array([1.0 if s["p_ai"] >= 0.5 else 0.0 for s in scored_segments], dtype=float)
    else:
        ai_side = np.array([1.0 if s["verdict"] == "likely_ai" else 0.0 for s in scored_segments], dtype=float)
    return float(np.average(ai_side, weights=evidence_weights))


def top_k_ai_average(scored_windows, k=3):
    """Average the strongest AI-side window posteriors."""
    if not scored_windows:
        return 0.5
    values = sorted((float(s["p_ai"]) for s in scored_windows), reverse=True)
    k = max(1, min(k, len(values)))
    return float(np.mean(values[:k]))


def scoring_scheme_rows(expected, whole, passage_scores, window_scores):
    """Compare candidate document-level scoring schemes."""
    passage_weights = np.array([max(1, s["chars"]) for s in passage_scores], dtype=float)
    passage_ai = np.array([s["p_ai"] for s in passage_scores], dtype=float)
    passage_vote_ai = np.array([1.0 if s["verdict"] == "likely_ai" else 0.0 for s in passage_scores])
    window_weights = np.array([max(1, s.get("word_count", 1)) for s in window_scores], dtype=float)
    window_ai = np.array([s["p_ai"] for s in window_scores], dtype=float)
    window_vote_ai = np.array([1.0 if s["verdict"] == "likely_ai" else 0.0 for s in window_scores])
    window_summary = window_weight_summary(window_scores)

    schemes = [
        (
            "Whole text: MoE decision",
            1.0 if whole["verdict"] == "likely_ai" else 0.0,
            "Current public document decision; one 4D feature vector scored by the MoE engine.",
        ),
        (
            "Whole text: posterior",
            whole["p_ai"],
            "Same whole-text feature vector, but hard side comes directly from mean posterior.",
        ),
        (
            "Passages: char-weighted posterior",
            float(np.average(passage_ai, weights=passage_weights)),
            "Average passage P(AI) by character count.",
        ),
        (
            "Passages: char-weighted vote share",
            float(np.average(passage_vote_ai, weights=passage_weights)),
            "Fraction of passage character mass whose MoE decision says AI.",
        ),
        (
            "Passages: margin-weighted posterior side",
            margin_weighted_side_share(passage_scores, weight_key="chars", side="posterior"),
            "Passage side of 50/50, weighted by chars times margin.",
        ),
        (
            "Passages: margin-weighted vote side",
            margin_weighted_side_share(passage_scores, weight_key="chars", side="verdict"),
            "Passage MoE-decision side, weighted by chars times margin.",
        ),
        (
            "Windows: word-weighted posterior",
            float(np.average(window_ai, weights=window_weights)),
            "Average overlapping-window P(AI) by word count.",
        ),
        (
            "Windows: word-weighted vote share",
            float(np.average(window_vote_ai, weights=window_weights)),
            "Fraction of overlapping-window word mass whose MoE decision says AI.",
        ),
        (
            "Windows: margin-weighted posterior side",
            window_summary["posterior_evidence_ai"],
            "Overlapping-window side of 50/50, weighted by words times margin.",
        ),
        (
            "Windows: margin-weighted vote side",
            window_summary["vote_evidence_ai"],
            "Overlapping-window MoE-decision side, weighted by words times margin.",
        ),
        (
            "Windows: top-3 AI posterior",
            top_k_ai_average(window_scores, k=3),
            "Mean of the strongest three AI-side window posteriors; useful for localized AI signal.",
        ),
    ]

    rows = []
    for name, ai_score, note in schemes:
        prediction = scheme_decision(ai_score)
        rows.append({
            "scheme": name,
            "ai_score": float(ai_score),
            "prediction": prediction,
            "status": expected_check_prediction(expected, prediction),
            "note": note,
        })
    return rows


def fmt_pct(value):
    """Format probability as a percent."""
    return f"{value * 100:.1f}%"


def excerpt(text, limit=150):
    """Short markdown-safe excerpt."""
    one_line = re.sub(r"\s+", " ", text).strip()
    if len(one_line) <= limit:
        return one_line
    return one_line[: limit - 3] + "..."


def render_segment_table(scored_segments, include_window_bounds=False):
    """Render markdown table for scored segments."""
    if include_window_bounds:
        summary = window_weight_summary(scored_segments)
        evidence_weights = summary["normalized_evidence_weights"]
        header = "| Segment | Words | Public verdict | Human | AI | Margin | Evidence wt. | Excerpt |\n"
        header += "|---|---:|---|---:|---:|---:|---:|---|\n"
        rows = []
        for s, evidence_weight in zip(scored_segments, evidence_weights):
            rows.append(
                f"| {s['title']} | {s['start_word']}-{s['end_word']} | "
                f"{s['label']} | {fmt_pct(s['p_human'])} | {fmt_pct(s['p_ai'])} | "
                f"{fmt_pct(s.get('margin', s.get('confidence', 0.0)))} | "
                f"{fmt_pct(evidence_weight)} | {excerpt(s['text'])} |"
            )
    else:
        header = "| Segment | Chars | Public verdict | Human | AI | Margin | Excerpt |\n"
        header += "|---|---:|---|---:|---:|---:|---|\n"
        rows = [
            (
                f"| {s['title']} | {s['chars']} | {s['label']} | "
                f"{fmt_pct(s['p_human'])} | {fmt_pct(s['p_ai'])} | "
                f"{fmt_pct(s.get('margin', s.get('confidence', 0.0)))} | {excerpt(s['text'])} |"
            )
            for s in scored_segments
        ]
    return header + "\n".join(rows) + "\n"


def render_scheme_table(expected, whole, passage_scores, window_scores):
    """Render side-by-side document scoring schemes."""
    header = "| Scheme | AI-side score | Decision | Expected-label check | Notes |\n"
    header += "|---|---:|---|---|---|\n"
    rows = []
    for row in scoring_scheme_rows(expected, whole, passage_scores, window_scores):
        rows.append(
            f"| {row['scheme']} | {fmt_pct(row['ai_score'])} | "
            f"{row['prediction']} | {row['status']} | {row['note']} |"
        )
    return header + "\n".join(rows) + "\n"


def render_document_report(title, source, expected, passages, loaded_models, device,
                           experts, window_words, stride_words):
    """Render a markdown section for one document."""
    full_text = "\n\n".join(p["text"] for p in passages)
    passage_scores = score_segments(passages, loaded_models, device, experts)
    window_segments = split_word_windows(full_text, window_words=window_words, stride_words=stride_words)
    window_scores = score_segments(window_segments, loaded_models, device, experts)
    whole = score_text(full_text, loaded_models, device, experts)
    avg_human = weighted_average(passage_scores)
    avg_ai = 1.0 - avg_human
    window_summary = window_weight_summary(window_scores)
    check = expected_check(expected, whole["verdict"])

    lines = [
        f"## {title}",
        "",
        f"- Source: {source}",
        f"- Expected label: {expected}",
        f"- Expected-label check: {check}",
        f"- Whole-document public verdict: {whole['label']}",
        f"- Whole-document scores: {fmt_pct(whole['p_human'])} human / {fmt_pct(whole['p_ai'])} AI",
        f"- Character-weighted passage score average: {fmt_pct(avg_human)} human / {fmt_pct(avg_ai)} AI",
        f"- Windowing: {window_words} words, stride {stride_words} words",
        f"- Window posterior average: {fmt_pct(window_summary['posterior_avg_human'])} human / {fmt_pct(window_summary['posterior_avg_ai'])} AI",
        f"- Window posterior-side share: {fmt_pct(1.0 - window_summary['posterior_ai_window_share'])} human-side / {fmt_pct(window_summary['posterior_ai_window_share'])} AI-side",
        f"- Window vote-side share: {fmt_pct(1.0 - window_summary['vote_ai_window_share'])} human-side / {fmt_pct(window_summary['vote_ai_window_share'])} AI-side",
        f"- Margin-weighted posterior evidence: {fmt_pct(window_summary['posterior_evidence_human'])} human-side / {fmt_pct(window_summary['posterior_evidence_ai'])} AI-side",
        f"- Margin-weighted MoE-decision evidence: {fmt_pct(window_summary['vote_evidence_human'])} human-side / {fmt_pct(window_summary['vote_evidence_ai'])} AI-side",
        f"- Low-margin window share: {fmt_pct(window_summary['low_band_window_share'])}",
        "",
        "### Scoring Scheme Comparison",
        "",
        render_scheme_table(expected, whole, passage_scores, window_scores),
        "### Passages",
        "",
        render_segment_table(passage_scores),
        "### Sliding Word Windows",
        "",
        render_segment_table(window_scores, include_window_bounds=True),
    ]
    return "\n".join(lines)


def demo_documents():
    """Return demo corpus documents in report format."""
    docs = []
    for item in get_demo_corpus():
        docs.append({
            "title": item["title"],
            "source": item["source"],
            "expected": item["expected"],
            "passages": item["passages"],
        })
    return docs


def sample_documents():
    """Return a few bundled sample documents for report inspection."""
    human, ai = load_sample_texts()
    docs = []
    for i, text in enumerate(human[:2], start=1):
        docs.append({
            "title": f"Bundled Human Sample {i}",
            "source": "src/arepo/data",
            "expected": "human",
            "passages": [{"title": "Full sample", "text": text}],
        })
    for i, text in enumerate(ai[:2], start=1):
        docs.append({
            "title": f"Bundled GPT Sample {i}",
            "source": "src/arepo/data",
            "expected": "ai",
            "passages": [{"title": "Full sample", "text": text}],
        })
    return docs


def wiki_documents():
    """Return a few Wiki human/generated documents for report inspection."""
    human, ai = load_wiki(max_per_class=2)
    docs = []
    for i, text in enumerate(human, start=1):
        docs.append({
            "title": f"Wiki Human Intro {i}",
            "source": "GPT-Wiki-Intro wiki_intro",
            "expected": "human",
            "passages": [{"title": "Full intro", "text": text}],
        })
    for i, text in enumerate(ai, start=1):
        docs.append({
            "title": f"Wiki Generated Intro {i}",
            "source": "GPT-Wiki-Intro generated_intro",
            "expected": "ai",
            "passages": [{"title": "Full intro", "text": text}],
        })
    return docs


def write_markdown_report(path, title, docs, loaded_models, device, experts,
                          window_words, stride_words):
    """Write one markdown report file."""
    sections = [
        f"# {title}",
        "",
        "Scores are MoE posteriors. Public verdicts use the same mixture-expert path as the web app. The sliding-window section scores each overlapping word window as its own text.",
        "",
        "The sliding-window section is a separate diagnostic: each overlapping word window is scored as its own text. `Evidence wt.` is the window's share of total margin-weighted evidence, computed from word count times distance from the 50/50 boundary. Margin-weighted posterior evidence uses the posterior side of 50/50; margin-weighted MoE-decision evidence uses the public MoE decision side. False negatives are called out explicitly when an expected-AI document is classified human.",
        "",
    ]
    for doc in docs:
        sections.append(render_document_report(
            doc["title"],
            doc["source"],
            doc["expected"],
            doc["passages"],
            loaded_models,
            device,
            experts,
            window_words,
            stride_words,
        ))
        sections.append("")
    path.write_text("\n".join(sections), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate markdown document/window score reports")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MOE_MODEL),
    )
    parser.add_argument("--output-dir", default="evaluation/reports")
    parser.add_argument("--window-words", type=int, default=80)
    parser.add_argument("--stride-words", type=int, default=40)
    parser.add_argument(
        "--reports",
        default="demo,sample,wiki",
        help="Comma-separated reports: demo,sample,wiki",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _get_device()
    experts = _load_moe_model(Path(args.model))
    loaded_models = _preload_models(DEFAULT_MODELS, device)

    report_map = {
        "demo": ("classics_and_controls.md", "Classics And Controls", demo_documents),
        "sample": ("bundled_samples.md", "Bundled Sample Documents", sample_documents),
        "wiki": ("wiki_intro_samples.md", "Wiki Intro Samples", wiki_documents),
    }

    for report_name in [r.strip() for r in args.reports.split(",") if r.strip()]:
        if report_name not in report_map:
            raise SystemExit(f"Unknown report: {report_name}")
        filename, title, loader = report_map[report_name]
        path = output_dir / filename
        write_markdown_report(
            path,
            title,
            loader(),
            loaded_models,
            device,
            experts,
            args.window_words,
            args.stride_words,
        )
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
