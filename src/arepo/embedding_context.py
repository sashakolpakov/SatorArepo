"""Embedding-space context features for expert competence gating."""

from pathlib import Path

import numpy as np
import torch

from .core import DEFAULT_MODELS, GENERATIVE_MODELS
from .mixture_experts import stable_text_hash
from .stats import get_embeddings
from .web import _get_device, _preload_models


def mean_pooled_embedding(text, loaded_models, device, model_names=None, max_length=512):
    """Return concatenated mean-pooled embeddings from the configured models."""
    if model_names is None:
        model_names = DEFAULT_MODELS

    vectors = []
    for model_name in model_names:
        tokenizer, model = loaded_models[model_name]
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)
        embeddings = get_embeddings(
            model,
            encoded["input_ids"],
            encoded["attention_mask"],
            model_name in GENERATIVE_MODELS,
        )[0]
        mask = encoded["attention_mask"][0].detach().cpu().float()
        if float(mask.sum()) <= 0:
            return None
        pooled = (embeddings * mask[:, None]).sum(dim=0) / mask.sum()
        vectors.append(pooled.numpy())

    return np.concatenate(vectors).astype(float)


def extract_embedding_context_features_with_indices(texts, model_names=None, max_length=512):
    """Extract embedding context vectors and surviving row indices."""
    if model_names is None:
        model_names = DEFAULT_MODELS
    device = _get_device()
    loaded_models = _preload_models(model_names, device)

    features = []
    valid_indices = []
    for i, text in enumerate(texts):
        if i % 10 == 0:
            print(f"  embedding context {i + 1}/{len(texts)}...", flush=True)
        try:
            vector = mean_pooled_embedding(
                text,
                loaded_models,
                device,
                model_names=model_names,
                max_length=max_length,
            )
            if vector is not None and not np.any(np.isnan(vector)):
                features.append(vector)
                valid_indices.append(i)
        except Exception:
            continue

    del loaded_models
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    if not features:
        return np.empty((0, 0)), np.array([], dtype=int)
    return np.vstack(features), np.array(valid_indices, dtype=int)


def load_or_extract_cached_embedding_context(records, cache_path, max_length=512):
    """Load or extract cached embedding context vectors for records/rows."""
    cache_path = Path(cache_path)
    text_hashes = np.array([stable_text_hash(record["text"]) for record in records])
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if (
            "text_hashes" in cached
            and len(cached["text_hashes"]) == len(text_hashes)
            and np.array_equal(cached["text_hashes"], text_hashes)
        ):
            return cached["features"], cached["valid_indices"]

    texts = [record["text"] for record in records]
    features, valid_indices = extract_embedding_context_features_with_indices(
        texts,
        max_length=max_length,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        features=features,
        valid_indices=valid_indices,
        text_hashes=text_hashes,
    )
    return features, valid_indices
