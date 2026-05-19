#!/usr/bin/env python3
"""Flask web app for Arepo AI text detection.

Usage:
    python -m arepo.web [--port 5000] [--host 0.0.0.0]
"""

import argparse
import os
import sys
import re
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from transformers import AutoModel, AutoTokenizer, T5EncoderModel

from .calibration import public_score
from .corpus import get_demo_corpus
from .core import GENERATIVE_MODELS, MASKED_MODELS, DEFAULT_MODELS
from .evidence import (
    aggregate_evidence,
    document_evidence_fields,
    ranked_evidence,
    scale_consistency,
)
from .stats import compute_ks_chunked

# MPS fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


DEFAULT_MOE_MODEL = Path(__file__).parent / "models" / "long_guardrail_4d_baseline_w120.npz"
MOE_SETTINGS = {
    "temperature": 2.0,
    "ai_veto_threshold": 0.89,
    "ai_veto_min_weight": 0.20,
    "human_veto_threshold": 0.95,
    "human_veto_min_weight": 0.20,
    "competence_metric": "plane",
    "competence_strength": 1.0,
    "alignment_threshold": 0.8,
}


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_moe_model(path):
    """Load the production MoE expert model."""
    from .mixture_experts import load_experts

    return load_experts(path)


def _preload_models(model_names, device):
    """Preload all HuggingFace models and tokenizers into memory."""
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


def _extract_features_single(text, loaded_models, device):
    """Extract 4D KS feature vector for a single text using preloaded models.

    For texts longer than a model's max token window, splits into overlapping
    chunks, computes a KS statistic per chunk, and averages them.
    """
    features = []
    for model_name in DEFAULT_MODELS:
        tokenizer, model = loaded_models[model_name]
        is_gen = model_name in GENERATIVE_MODELS
        ks = compute_ks_chunked(text, tokenizer, model, device, is_gen)
        features.append(ks)
    return np.array(features)


def _public_score(
    avg_posterior,
    prediction=None,
    geometric_confidence=None,
):
    """Build a user-facing score object while keeping internals hidden."""
    return public_score(
        avg_posterior,
        prediction=prediction,
        geometric_confidence=geometric_confidence,
    )


def _classify(features, experts, text=None):
    """Run the MoE engine and return public-facing evidence fields."""
    from .mixture_experts import predict_mixture

    result = predict_mixture(features, experts, **MOE_SETTINGS)
    geometry = result["geometric_confidence"]
    return _public_score(
        result["p_human"],
        prediction=result["prediction"],
        geometric_confidence=geometry["confidence"],
    )


def _split_passages(text):
    """Split user text into substantial paragraph-like passages."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if len(paragraphs) <= 1:
        return [{"title": "Full text", "text": text.strip()}]
    return [
        {"title": f"Paragraph {i}", "text": paragraph}
        for i, paragraph in enumerate(paragraphs, start=1)
    ]


def _split_word_windows(text, window_words=120, stride_words=60, min_words=45):
    """Split text into overlapping word windows for evidence aggregation."""
    words = re.findall(r"\S+", text)
    if not words:
        return []
    if len(words) <= window_words:
        if len(words) < min_words:
            return []
        return [{
            "title": "Window 1",
            "text": " ".join(words),
            "start_word": 1,
            "end_word": len(words),
            "word_count": len(words),
            "scale_words": window_words,
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
                "scale_words": window_words,
            })
        if end == len(words):
            break
        start += stride_words
    return windows


def _score_text(text, loaded_models, device, experts):
    """Score full text through the same single-vector path as main."""
    features = _extract_features_single(text, loaded_models, device)
    if np.any(np.isnan(features)):
        raise ValueError("Feature extraction failed for some models")
    return _classify(features, experts, text=text)


def _public_passage_payload(passage):
    """Return public passage fields."""
    return dict(passage)


def _score_passages(passages, loaded_models, device, experts):
    """Score a list of passages and return public-facing results."""
    scored = []
    for i, passage in enumerate(passages, start=1):
        text = passage["text"].strip()
        features = _extract_features_single(text, loaded_models, device)
        if np.any(np.isnan(features)):
            raise ValueError("Feature extraction failed for some models")
        score = _classify(features, experts, text=text)
        score.update({
            "index": i,
            "title": passage.get("title") or f"Passage {i}",
            "chars": len(text),
        })
        scored.append(score)
    return scored


def _score_windows(
    text,
    loaded_models,
    device,
    experts,
    scales=(80, 120, 240),
    max_windows_per_scale=12,
):
    """Score overlapping word windows at multiple scales."""
    scored = []
    for scale_words in scales:
        stride_words = max(1, scale_words // 2)
        windows = _split_word_windows(
            text,
            window_words=scale_words,
            stride_words=stride_words,
            min_words=min(45, scale_words),
        )
        if max_windows_per_scale is not None:
            windows = windows[:max_windows_per_scale]
        for window in windows:
            window_text = window["text"]
            features = _extract_features_single(window_text, loaded_models, device)
            if np.any(np.isnan(features)):
                raise ValueError("Feature extraction failed for some models")
            score = _classify(features, experts, text=window_text)
            score.update({
                "index": len(scored) + 1,
                "title": f"{scale_words}w {window['title']}",
                "chars": len(window_text),
                "word_count": window["word_count"],
                "start_word": window["start_word"],
                "end_word": window["end_word"],
                "scale_words": scale_words,
            })
            scored.append(score)
    return scored


def _document_evidence_payload(document_score, passage_scores, window_scores):
    """Attach aggregate evidence summaries to a document score."""
    evidence_rows = window_scores if window_scores else passage_scores
    summary = aggregate_evidence(evidence_rows)
    document_evidence = document_evidence_fields(document_score, summary)
    return {
        **document_score,
        **document_evidence,
        "label": document_evidence["document_evidence_label"],
        "verdict": document_evidence["document_evidence_class"],
        "evidence_summary": summary,
        "evidence_rankings": ranked_evidence(evidence_rows),
        "scale_consistency": scale_consistency(window_scores),
    }


def create_app(model_path=None, preload_models=True):
    """Flask application factory."""
    app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))

    # Resolve MoE model path
    if model_path is None:
        model_path = DEFAULT_MOE_MODEL
    else:
        model_path = Path(model_path)

    # Load MoE experts
    if model_path.exists():
        print(f"Loading MoE model from {model_path}...")
        app.config["experts"] = _load_moe_model(model_path)
        app.config["moe_loaded"] = True
    else:
        print(f"WARNING: No MoE model found at {model_path}")
        print("  Run: python -m arepo.greedy_expert_selection")
        app.config["experts"] = None
        app.config["moe_loaded"] = False

    # Preload ML models
    device = _get_device()
    app.config["device"] = device
    app.config["models_loaded"] = False
    app.config["loaded_models"] = {}
    if preload_models:
        print(f"Preloading models on {device}...")
        app.config["loaded_models"] = _preload_models(DEFAULT_MODELS, device)
        app.config["models_loaded"] = True
        print("Models loaded.")

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/demo-corpus")
    def demo_corpus():
        response = jsonify({"corpus": get_demo_corpus()})
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.route("/analyze", methods=["POST"])
    def analyze():
        if not app.config["moe_loaded"]:
            return jsonify({"error": "MoE model not loaded. Train or provide --model."}), 503
        if not app.config["models_loaded"]:
            return jsonify({"error": "Transformer models not loaded."}), 503

        data = request.get_json(silent=True)
        if not data or ("text" not in data and "passages" not in data):
            return jsonify({"error": "Missing 'text' or 'passages' field"}), 400

        if "passages" in data:
            passages = data["passages"]
            if not isinstance(passages, list) or not passages:
                return jsonify({"error": "'passages' must be a non-empty list"}), 400
            passages = [
                {
                    "title": p.get("title", f"Passage {i}"),
                    "text": p.get("text", "").strip(),
                }
                for i, p in enumerate(passages, start=1)
                if isinstance(p, dict) and p.get("text", "").strip()
            ]
        else:
            text = data["text"].strip()
            passages = _split_passages(text)

        if not passages:
            return jsonify({"error": "No text to analyze"}), 400
        full_text = "\n\n".join(p["text"] for p in passages)

        short = [p["title"] for p in passages if len(p["text"]) < 100]
        if short:
            return jsonify({
                "error": "Each passage must be at least 100 characters",
                "short_passages": short,
            }), 400

        try:
            scored = _score_passages(
                passages,
                app.config["loaded_models"],
                app.config["device"],
                app.config["experts"],
            )
            scored_windows = _score_windows(
                full_text,
                app.config["loaded_models"],
                app.config["device"],
                app.config["experts"],
            )
            document_score = _score_text(
                full_text,
                app.config["loaded_models"],
                app.config["device"],
                app.config["experts"],
            )
            return jsonify({
                "document": _document_evidence_payload(document_score, scored, scored_windows),
                "passages": [_public_passage_payload(p) for p in scored],
                "windows": [_public_passage_payload(w) for w in scored_windows],
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/health")
    def health():
        return jsonify({
            "status": "ok",
            "moe_loaded": app.config["moe_loaded"],
            "models_loaded": app.config["models_loaded"],
            "device": str(app.config["device"]),
        })

    return app


def main():
    parser = argparse.ArgumentParser(description="Arepo web interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--model", type=str, default=None,
                        help="Path to MoE expert .npz model")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app(model_path=args.model)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
