#!/usr/bin/env python3
"""Mixture of oriented Gaussian experts over the 4D KS engine signal."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import expit

from .core import extract_4d_features_with_indices
from .guardrail_evaluation import expected_status, fmt_pct, guardrail_records
from .stats import _roc_auc, fit_gaussian_4d
from .training_data import add_dataset, load_historic_civic, load_training_records
from .web import _extract_features_single, _get_device, _preload_models
from .core import DEFAULT_MODELS


def stable_text_hash(text):
    """Return a stable hash for cache validation."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def civic_expert_records():
    """Return adjacent historic-civic human/AI records for a specialized expert."""
    records = []
    human_texts, ai_texts = load_historic_civic()
    add_dataset(records, "historic_civic", human_texts, ai_texts)
    return records


def logsumexp_pair(a, b):
    """Stable log(exp(a) + exp(b)) for arrays."""
    maximum = np.maximum(a, b)
    return maximum + np.log(np.exp(a - maximum) + np.exp(b - maximum))


def log_gaussian(model, data):
    """Return independent Gaussian log density."""
    return np.sum(stats.norm.logpdf(data, model["mean"], model["std"]), axis=1)


def direct_ai_score(model_human, model_ai, data):
    """Return direct-orientation P(AI) under one expert."""
    log_human = log_gaussian(model_human, data)
    log_ai = log_gaussian(model_ai, data)
    return expit(log_ai - log_human)


def pooled_context_model(features, n_components=2):
    """Fit the pooled source-context distribution for competence gating."""
    features = np.asarray(features, dtype=float)
    mean = features.mean(axis=0)
    std = np.maximum(features.std(axis=0), 1e-6)
    z = (features - mean) / std
    basis = np.zeros((features.shape[1], 0), dtype=float)
    if len(features) > 1 and n_components > 0:
        _, _, vt = np.linalg.svd(z, full_matrices=False)
        k = min(int(n_components), vt.shape[0], features.shape[1])
        basis = vt[:k].T
    return {
        "mean": mean,
        "std": std,
        "basis": basis,
    }


def require_context_model(expert):
    """Return an expert context model or fail hard."""
    context = expert.get("context_model")
    if context is None:
        raise ValueError(f"MoE expert {expert.get('group', '<unknown>')} is missing context_model")
    return context


def context_distance(expert, data):
    """Return RMS z-distance from an expert's pooled source context."""
    context = require_context_model(expert)
    z = (data - context["mean"]) / context["std"]
    return float(np.sqrt(np.mean(z ** 2)))


def context_projection(expert, data):
    """Return point-to-context-plane alignment diagnostics."""
    context = require_context_model(expert)
    z = ((data - context["mean"]) / context["std"]).reshape(-1)
    basis = np.asarray(context.get("basis", np.zeros((len(z), 0))), dtype=float)
    if basis.ndim != 2 or basis.shape[0] != len(z) or basis.shape[1] == 0:
        return {
            "plane_alignment": 0.0,
            "plane_residual": float(np.linalg.norm(z)),
        }
    projected = basis @ (basis.T @ z)
    norm = float(np.linalg.norm(z))
    projected_norm = float(np.linalg.norm(projected))
    residual = float(np.linalg.norm(z - projected))
    if norm <= 1e-12:
        alignment = 1.0
    else:
        alignment = max(0.0, min(1.0, projected_norm / norm))
    return {
        "plane_alignment": alignment,
        "plane_residual": residual,
    }


def context_competence_log(expert, data, metric):
    """Return log-competence and diagnostics for a chosen context metric."""
    projection = context_projection(expert, data)
    distance = context_distance(expert, data)
    if metric == "cosine":
        value = max(projection["plane_alignment"], 1e-6)
        return float(np.log(value)), distance, projection
    if metric == "plane":
        return -0.5 * projection["plane_residual"] ** 2, projection["plane_residual"], projection
    return -0.5 * distance ** 2, distance, projection


def reliability_bin(distance, edges):
    """Return the reliability-bin index for a context distance."""
    if edges is None or len(edges) == 0:
        return 0
    return int(np.searchsorted(np.asarray(edges, dtype=float), distance, side="right"))


def reliability_for_vote(expert, oriented_ai, distance, floor=0.05):
    """Return held-out reliability for an expert vote at this context distance."""
    table = expert.get("reliability")
    if table is None:
        return 1.0

    side = "ai" if oriented_ai >= 0.5 else "human"
    values = np.asarray(table.get(side, []), dtype=float)
    if len(values) == 0:
        return 1.0

    bin_idx = reliability_bin(distance, table.get("edges", []))
    bin_idx = min(max(bin_idx, 0), len(values) - 1)
    value = float(values[bin_idx])
    if not np.isfinite(value):
        value = float(table.get("overall", 1.0))
    return max(float(floor), min(1.0, value))


def geometric_confidence(expert_rows, p_ai, prediction, alignment_threshold=0.8):
    """Return 4D context-plane confidence diagnostics for a mixture vote.

    This is a reliability proxy, not a posterior. It measures how much of the
    winning-side expert mass is close to its own context plane, subtracts
    aligned opposing mass, and scales that by distance from the 50/50 posterior
    boundary.
    """
    winning_ai = prediction == "ai"
    posterior_margin = abs(float(p_ai) - 0.5) * 2.0
    majority_weight = 0.0
    minority_weight = 0.0
    aligned_majority_weight = 0.0
    aligned_minority_weight = 0.0
    aligned_majority_count = 0
    aligned_minority_count = 0
    threshold = float(alignment_threshold)

    for row in expert_rows:
        expert_ai = float(row["oriented_ai"]) >= 0.5
        same_side = expert_ai == winning_ai
        weight = float(row.get("weight", 0.0))
        aligned = float(row.get("plane_alignment", 0.0)) >= threshold
        if same_side:
            majority_weight += weight
            if aligned:
                aligned_majority_weight += weight
                aligned_majority_count += 1
        else:
            minority_weight += weight
            if aligned:
                aligned_minority_weight += weight
                aligned_minority_count += 1

    aligned_gap = max(0.0, aligned_majority_weight - aligned_minority_weight)
    return {
        "posterior_margin": float(posterior_margin),
        "alignment_threshold": threshold,
        "majority_weight": float(majority_weight),
        "minority_weight": float(minority_weight),
        "aligned_majority_weight": float(aligned_majority_weight),
        "aligned_minority_weight": float(aligned_minority_weight),
        "aligned_majority_count": int(aligned_majority_count),
        "aligned_minority_count": int(aligned_minority_count),
        "aligned_gap": float(aligned_gap),
        "confidence": float(posterior_margin * aligned_gap),
    }


def calibration_fold_indices(labels, n_folds, random_state=42):
    """Return deterministic stratified folds for reliability calibration."""
    rng = np.random.default_rng(random_state)
    labels = np.asarray(labels, dtype=int)
    folds = [[] for _ in range(n_folds)]
    for label in sorted(set(labels)):
        idx = np.flatnonzero(labels == label)
        idx = rng.permutation(idx)
        for i, row_idx in enumerate(idx):
            folds[i % n_folds].append(int(row_idx))
    return [np.array(sorted(fold), dtype=int) for fold in folds if fold]


def fit_expert_reliability(
    group,
    features,
    labels,
    random_state=42,
    n_bins=3,
    min_reliability_count=3,
    smoothing=2.0,
):
    """Estimate P(expert vote is correct | predicted side, context distance bin)."""
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=int)
    class_counts = [int(np.sum(labels == label)) for label in (0, 1)]
    n_folds = min(5, min(class_counts))
    if n_folds < 2:
        return None

    fold_rows = []
    for fold_idx, validation_idx in enumerate(
        calibration_fold_indices(labels, n_folds, random_state=random_state)
    ):
        train_mask = np.ones(len(labels), dtype=bool)
        train_mask[validation_idx] = False
        if len(np.unique(labels[train_mask])) < 2 or len(np.unique(labels[validation_idx])) < 2:
            continue
        fold_experts = train_expert_for_group(
            group,
            features[train_mask],
            labels[train_mask],
            random_state=random_state + fold_idx + 1,
            reliability=False,
        )
        if not fold_experts:
            continue
        fold_expert = fold_experts[0]
        for sample_idx in validation_idx:
            sample = features[sample_idx].reshape(1, -1)
            direct_ai = direct_ai_score(
                fold_expert["model_human"],
                fold_expert["model_ai"],
                sample,
            )[0]
            oriented_ai = float(direct_ai if fold_expert["orientation"] == 1 else 1.0 - direct_ai)
            prediction = 1 if oriented_ai >= 0.5 else 0
            fold_rows.append({
                "distance": context_distance(fold_expert, sample),
                "side": "ai" if prediction == 1 else "human",
                "correct": int(prediction == int(labels[sample_idx])),
            })

    if len(fold_rows) < max(4, min_reliability_count):
        return None

    distances = np.array([row["distance"] for row in fold_rows], dtype=float)
    if n_bins <= 1 or len(np.unique(distances)) < 2:
        edges = np.array([], dtype=float)
    else:
        quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
        edges = np.unique(np.quantile(distances, quantiles))
    n_table_bins = len(edges) + 1
    overall = (sum(row["correct"] for row in fold_rows) + smoothing * 0.5) / (
        len(fold_rows) + smoothing
    )

    table = {
        "edges": edges.tolist(),
        "overall": float(overall),
        "human": [],
        "ai": [],
        "human_counts": [],
        "ai_counts": [],
    }
    for side in ("human", "ai"):
        for bin_idx in range(n_table_bins):
            rows = [
                row for row in fold_rows
                if row["side"] == side and reliability_bin(row["distance"], edges) == bin_idx
            ]
            table[f"{side}_counts"].append(len(rows))
            if len(rows) < min_reliability_count:
                table[side].append(float(overall))
                continue
            correct = sum(row["correct"] for row in rows)
            table[side].append(float((correct + smoothing * overall) / (len(rows) + smoothing)))
    return table


def fit_reliability_from_labeled_rows(
    expert,
    features,
    labels,
    n_bins=3,
    min_reliability_count=3,
    smoothing=2.0,
):
    """Estimate reliability for an already-trained expert on labeled calibration rows."""
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=int)
    rows = []
    for feature, label in zip(features, labels):
        sample = feature.reshape(1, -1)
        direct_ai = direct_ai_score(
            expert["model_human"],
            expert["model_ai"],
            sample,
        )[0]
        oriented_ai = float(direct_ai if expert["orientation"] == 1 else 1.0 - direct_ai)
        prediction = 1 if oriented_ai >= 0.5 else 0
        rows.append({
            "distance": context_distance(expert, sample),
            "side": "ai" if prediction == 1 else "human",
            "correct": int(prediction == int(label)),
        })

    if len(rows) < max(4, min_reliability_count):
        return None

    distances = np.array([row["distance"] for row in rows], dtype=float)
    if n_bins <= 1 or len(np.unique(distances)) < 2:
        edges = np.array([], dtype=float)
    else:
        quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
        edges = np.unique(np.quantile(distances, quantiles))
    n_table_bins = len(edges) + 1
    overall = (sum(row["correct"] for row in rows) + smoothing * 0.5) / (
        len(rows) + smoothing
    )

    table = {
        "edges": edges.tolist(),
        "overall": float(overall),
        "human": [],
        "ai": [],
        "human_counts": [],
        "ai_counts": [],
    }
    for side in ("human", "ai"):
        for bin_idx in range(n_table_bins):
            bin_rows = [
                row for row in rows
                if row["side"] == side and reliability_bin(row["distance"], edges) == bin_idx
            ]
            table[f"{side}_counts"].append(len(bin_rows))
            if len(bin_rows) < min_reliability_count:
                table[side].append(float(overall))
                continue
            correct = sum(row["correct"] for row in bin_rows)
            table[side].append(float((correct + smoothing * overall) / (len(bin_rows) + smoothing)))
    return table


def load_or_extract_cached_features(records, cache_path, base_cache_path=None):
    """Load feature cache or extract and save it."""
    cache_path = Path(cache_path)
    text_hashes = np.array([stable_text_hash(r["text"]) for r in records])

    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if (
            "text_hashes" in cached
            and len(cached["text_hashes"]) == len(text_hashes)
            and np.array_equal(cached["text_hashes"], text_hashes)
        ):
            return (
                cached["features"],
                cached["labels"],
                cached["groups"].astype(str),
                cached["valid_indices"],
            )

    if base_cache_path is not None and Path(base_cache_path).exists():
        base_cached = np.load(base_cache_path, allow_pickle=False)
        if "text_hashes" in base_cached:
            base_hashes = base_cached["text_hashes"]
            base_valid = base_cached["valid_indices"]
            feature_by_hash = {
                str(base_hashes[source_idx]): feature
                for source_idx, feature in zip(base_valid, base_cached["features"])
            }
            if feature_by_hash:
                missing_records = []
                missing_positions = []
                cached_features = {}
                for position, record in enumerate(records):
                    text_hash = str(text_hashes[position])
                    if text_hash in feature_by_hash:
                        cached_features[position] = feature_by_hash[text_hash]
                    else:
                        missing_positions.append(position)
                        missing_records.append(record)

                if len(cached_features) > 0:
                    extracted_features = {}
                    if missing_records:
                        missing_texts = [record["text"] for record in missing_records]
                        missing_features, missing_valid = extract_4d_features_with_indices(missing_texts)
                        for local_idx, feature in zip(missing_valid, missing_features):
                            extracted_features[missing_positions[local_idx]] = feature

                    valid_positions = sorted(set(cached_features) | set(extracted_features))
                    features = np.vstack([
                        cached_features[position]
                        if position in cached_features
                        else extracted_features[position]
                        for position in valid_positions
                    ])
                    labels_all = np.array([record["label"] for record in records], dtype=int)
                    groups_all = np.array([record["group"] for record in records])
                    labels = labels_all[valid_positions]
                    groups = groups_all[valid_positions]
                    valid_indices = np.array(valid_positions, dtype=int)

                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(
                        cache_path,
                        features=features,
                        labels=labels,
                        groups=groups,
                        valid_indices=valid_indices,
                        text_hashes=text_hashes,
                    )
                    return features, labels, groups, valid_indices

            if (
                len(base_hashes) < len(text_hashes)
                and np.array_equal(text_hashes[:len(base_hashes)], base_hashes)
            ):
                extra_records = records[len(base_hashes):]
                extra_texts = [record["text"] for record in extra_records]
                extra_labels = np.array([record["label"] for record in extra_records], dtype=int)
                extra_groups = np.array([record["group"] for record in extra_records])
                extra_features, extra_valid = extract_4d_features_with_indices(extra_texts)
                features = np.vstack([base_cached["features"], extra_features])
                labels = np.concatenate([base_cached["labels"], extra_labels[extra_valid]])
                groups = np.concatenate([base_cached["groups"].astype(str), extra_groups[extra_valid]])
                valid_indices = np.concatenate([
                    base_cached["valid_indices"],
                    len(base_hashes) + extra_valid,
                ])
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    cache_path,
                    features=features,
                    labels=labels,
                    groups=groups,
                    valid_indices=valid_indices,
                    text_hashes=text_hashes,
                )
                return features, labels, groups, valid_indices

    texts = [record["text"] for record in records]
    labels = np.array([record["label"] for record in records], dtype=int)
    groups = np.array([record["group"] for record in records])

    features, valid_indices = extract_4d_features_with_indices(texts)
    labels = labels[valid_indices]
    groups = groups[valid_indices]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        features=features,
        labels=labels,
        groups=groups,
        valid_indices=valid_indices,
        text_hashes=text_hashes,
    )
    return features, labels, groups, valid_indices


def stratified_validation_indices(labels, validation_fraction=0.3, random_state=42):
    """Return deterministic stratified train/validation index arrays."""
    rng = np.random.default_rng(random_state)
    train_parts = []
    validation_parts = []
    for label in (0, 1):
        idx = np.flatnonzero(labels == label)
        idx = rng.permutation(idx)
        n_validation = max(1, int(round(len(idx) * validation_fraction)))
        if len(idx) - n_validation < 1:
            n_validation = max(0, len(idx) - 1)
        validation_parts.append(idx[:n_validation])
        train_parts.append(idx[n_validation:])
    train_idx = np.concatenate(train_parts)
    validation_idx = np.concatenate(validation_parts)
    return train_idx, validation_idx


def learn_group_orientation(features, labels, random_state=42):
    """Learn whether a group's direct likelihood ratio should be inverted."""
    if len(np.unique(labels)) < 2:
        return 1, None

    train_idx, validation_idx = stratified_validation_indices(labels, random_state=random_state)
    if len(train_idx) == 0 or len(validation_idx) == 0 or len(np.unique(labels[validation_idx])) < 2:
        train_idx = np.arange(len(labels))
        validation_idx = np.arange(len(labels))

    train_features = features[train_idx]
    train_labels = labels[train_idx]
    validation_features = features[validation_idx]
    validation_labels = labels[validation_idx]

    model_human = fit_gaussian_4d(train_features[train_labels == 0])
    model_ai = fit_gaussian_4d(train_features[train_labels == 1])
    scores = direct_ai_score(model_human, model_ai, validation_features)
    auc = _roc_auc(validation_labels, scores)
    orientation = -1 if auc < 0.5 else 1
    return orientation, float(auc)


def train_oriented_experts(features, labels, groups, random_state=42):
    """Train one oriented Gaussian expert per dataset/source group."""
    experts = []
    for group in sorted(set(groups)):
        mask = groups == group
        experts.extend(train_expert_for_group(
            group,
            features[mask],
            labels[mask],
            random_state=random_state,
        ))
    return experts


def train_expert_for_group(
    group,
    features,
    labels,
    context_features=None,
    random_state=42,
    reliability=False,
    reliability_bins=3,
    min_reliability_count=3,
):
    """Train one expert for one named feature subset."""
    if len(np.unique(labels)) < 2:
        return []

    orientation, direct_auc = learn_group_orientation(
        features,
        labels,
        random_state=random_state,
    )
    model_human = fit_gaussian_4d(features[labels == 0])
    model_ai = fit_gaussian_4d(features[labels == 1])
    context_source = features if context_features is None else context_features
    context_model = pooled_context_model(context_source)
    reliability_table = None
    if reliability:
        reliability_table = fit_expert_reliability(
            group,
            features,
            labels,
            random_state=random_state,
            n_bins=reliability_bins,
            min_reliability_count=min_reliability_count,
        )
    return [{
        "group": group,
        "model_human": model_human,
        "model_ai": model_ai,
        "context_model": context_model,
        "reliability": reliability_table,
        "orientation": orientation,
        "direct_auc": direct_auc,
        "n_human": int(np.sum(labels == 0)),
        "n_ai": int(np.sum(labels == 1)),
    }]


def kmeans_labels(features, n_clusters, random_state=42, n_iter=50):
    """Small deterministic k-means for 4D feature-space sub-experts."""
    rng = np.random.default_rng(random_state)
    features = np.asarray(features, dtype=float)
    if len(features) < n_clusters:
        return np.zeros(len(features), dtype=int)

    center_idx = rng.choice(len(features), size=n_clusters, replace=False)
    centers = features[center_idx].copy()
    labels = np.zeros(len(features), dtype=int)

    for _ in range(n_iter):
        distances = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
        next_labels = np.argmin(distances, axis=1)
        if np.array_equal(next_labels, labels):
            break
        labels = next_labels
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            if np.any(mask):
                centers[cluster_id] = features[mask].mean(axis=0)
            else:
                centers[cluster_id] = features[rng.integers(0, len(features))]
    return labels


def train_dataset_and_cluster_experts(
    features,
    labels,
    groups,
    context_features=None,
    random_state=42,
    clusters_per_group=0,
    min_cluster_class=8,
    dataset_union_pairs=0,
    min_dataset_union_class=None,
    union_cluster_pairs=0,
    min_union_class=None,
    reliability=False,
    reliability_bins=3,
    min_reliability_count=3,
):
    """Train dataset experts plus optional within-dataset feature-cluster experts."""
    experts = []
    dataset_subsets = []
    cluster_subsets = []
    for group in sorted(set(groups)):
        mask = groups == group
        group_indices = np.flatnonzero(mask)
        group_features = features[mask]
        group_labels = labels[mask]
        group_context = None if context_features is None else context_features[mask]
        dataset_subsets.append({
            "name": group,
            "indices": group_indices,
            "center": (group_features if group_context is None else group_context).mean(axis=0),
        })
        experts.extend(train_expert_for_group(
            group,
            group_features,
            group_labels,
            context_features=group_context,
            random_state=random_state,
            reliability=reliability,
            reliability_bins=reliability_bins,
            min_reliability_count=min_reliability_count,
        ))

        if clusters_per_group <= 0:
            continue
        if len(group_features) < clusters_per_group * min_cluster_class * 2:
            continue

        scale = np.maximum(group_features.std(axis=0), 1e-6)
        centered = (group_features - group_features.mean(axis=0)) / scale
        cluster_labels = kmeans_labels(
            centered,
            clusters_per_group,
            random_state=random_state,
        )
        for cluster_id in range(clusters_per_group):
            cluster_mask = cluster_labels == cluster_id
            cluster_features = group_features[cluster_mask]
            cluster_y = group_labels[cluster_mask]
            cluster_context = None if group_context is None else group_context[cluster_mask]
            if (
                np.sum(cluster_y == 0) < min_cluster_class
                or np.sum(cluster_y == 1) < min_cluster_class
            ):
                continue
            experts.extend(train_expert_for_group(
                f"{group}:c{cluster_id}",
                cluster_features,
                cluster_y,
                context_features=cluster_context,
                random_state=random_state,
                reliability=reliability,
                reliability_bins=reliability_bins,
                min_reliability_count=min_reliability_count,
            ))
            cluster_subsets.append({
                "name": f"{group}:c{cluster_id}",
                "indices": group_indices[cluster_mask],
                "center": (cluster_features if cluster_context is None else cluster_context).mean(axis=0),
            })
    if dataset_union_pairs > 0 and len(dataset_subsets) > 1:
        if min_dataset_union_class is None:
            min_dataset_union_class = min_cluster_class
        pair_rows = []
        for i, left in enumerate(dataset_subsets):
            for right in dataset_subsets[i + 1:]:
                distance = float(np.linalg.norm(left["center"] - right["center"]))
                pair_rows.append((distance, left, right))
        pair_rows.sort(key=lambda row: (row[0], row[1]["name"], row[2]["name"]))
        trained_pairs = 0
        for _, left, right in pair_rows:
            union_indices = np.unique(np.concatenate([left["indices"], right["indices"]]))
            union_labels = labels[union_indices]
            if (
                np.sum(union_labels == 0) < min_dataset_union_class
                or np.sum(union_labels == 1) < min_dataset_union_class
            ):
                continue
            union_context = None if context_features is None else context_features[union_indices]
            experts.extend(train_expert_for_group(
                f"dataset_union:{left['name']}+{right['name']}",
                features[union_indices],
                union_labels,
                context_features=union_context,
                random_state=random_state,
                reliability=reliability,
                reliability_bins=reliability_bins,
                min_reliability_count=min_reliability_count,
            ))
            trained_pairs += 1
            if trained_pairs >= dataset_union_pairs:
                break
    if union_cluster_pairs > 0 and len(cluster_subsets) > 1:
        if min_union_class is None:
            min_union_class = min_cluster_class
        pair_rows = []
        for i, left in enumerate(cluster_subsets):
            for right in cluster_subsets[i + 1:]:
                distance = float(np.linalg.norm(left["center"] - right["center"]))
                pair_rows.append((distance, left, right))
        pair_rows.sort(key=lambda row: (row[0], row[1]["name"], row[2]["name"]))
        trained_pairs = 0
        for _, left, right in pair_rows:
            union_indices = np.unique(np.concatenate([left["indices"], right["indices"]]))
            union_labels = labels[union_indices]
            if (
                np.sum(union_labels == 0) < min_union_class
                or np.sum(union_labels == 1) < min_union_class
            ):
                continue
            union_context = None if context_features is None else context_features[union_indices]
            experts.extend(train_expert_for_group(
                f"union:{left['name']}+{right['name']}",
                features[union_indices],
                union_labels,
                context_features=union_context,
                random_state=random_state,
                reliability=reliability,
                reliability_bins=reliability_bins,
                min_reliability_count=min_reliability_count,
            ))
            trained_pairs += 1
            if trained_pairs >= union_cluster_pairs:
                break
    return experts


def predict_mixture(
    features,
    experts,
    context_features=None,
    temperature=1.0,
    ai_veto_threshold=None,
    ai_veto_min_weight=0.0,
    human_veto_threshold=None,
    human_veto_min_weight=0.0,
    competence_metric="z_distance",
    competence_max_distance=None,
    competence_strength=0.0,
    reliability_strength=0.0,
    reliability_floor=0.05,
    alignment_threshold=0.8,
):
    """Predict P(AI) with likelihood/context-weighted oriented experts."""
    if not experts:
        raise ValueError("MoE prediction requires at least one expert")
    data = np.asarray(features, dtype=float).reshape(1, -1)
    context_data = data if context_features is None else np.asarray(context_features, dtype=float).reshape(1, -1)
    ai_scores = []
    log_weights = []
    expert_rows = []

    for expert in experts:
        model_human = expert["model_human"]
        model_ai = expert["model_ai"]
        log_human = log_gaussian(model_human, data)[0]
        log_ai = log_gaussian(model_ai, data)[0]
        direct_ai = float(expit(log_ai - log_human))
        oriented_ai = direct_ai if expert["orientation"] == 1 else 1.0 - direct_ai
        log_marginal = logsumexp_pair(log_human, log_ai) - np.log(2.0)
        competence_log, distance, projection = context_competence_log(
            expert,
            context_data,
            competence_metric,
        )
        reliability = reliability_for_vote(
            expert,
            oriented_ai,
            distance,
            floor=reliability_floor,
        )
        competent = (
            competence_max_distance is None
            or distance <= competence_max_distance
        )

        ai_scores.append(oriented_ai)
        if not competent:
            log_weights.append(-np.inf)
        else:
            log_weights.append(
                log_marginal
                + float(competence_strength) * competence_log
                + float(reliability_strength) * np.log(max(reliability, 1e-300))
            )
        expert_rows.append({
            "group": expert["group"],
            "direct_ai": float(direct_ai),
            "oriented_ai": float(oriented_ai),
            "log_marginal": float(log_marginal),
            "context_distance": float(distance),
            "competence": float(np.exp(competence_log)),
            "plane_alignment": float(projection["plane_alignment"]),
            "plane_residual": float(projection["plane_residual"]),
            "reliability": float(reliability),
            "competent": bool(competent),
            "orientation": int(expert["orientation"]),
        })

    log_weights = np.array(log_weights, dtype=float) / float(temperature)
    if not np.any(np.isfinite(log_weights)):
        raise ValueError("No competent MoE experts for input")
    log_weights = log_weights - np.max(log_weights)
    weights = np.exp(log_weights)
    weights = weights / weights.sum()
    ai_scores = np.array(ai_scores, dtype=float)
    p_ai = float(np.sum(weights * ai_scores))

    for row, weight in zip(expert_rows, weights):
        row["weight"] = float(weight)

    base_prediction = "ai" if p_ai >= 0.5 else "human"
    prediction = base_prediction
    override = None
    if ai_veto_threshold is not None and base_prediction == "human":
        veto_candidates = [
            row for row in expert_rows
            if row["weight"] >= ai_veto_min_weight
            and row["oriented_ai"] >= ai_veto_threshold
        ]
        if veto_candidates:
            strongest = max(veto_candidates, key=lambda row: (row["oriented_ai"], row["weight"]))
            prediction = "ai"
            override = {
                "rule": "ai_expert_disagreement",
                "expert": strongest["group"],
                "expert_ai": float(strongest["oriented_ai"]),
                "expert_weight": float(strongest["weight"]),
                "threshold": float(ai_veto_threshold),
                "min_weight": float(ai_veto_min_weight),
            }
    if human_veto_threshold is not None and override is None and base_prediction == "ai":
        veto_candidates = [
            row for row in expert_rows
            if row["weight"] >= human_veto_min_weight
            and (1.0 - row["oriented_ai"]) >= human_veto_threshold
        ]
        if veto_candidates:
            strongest = max(veto_candidates, key=lambda row: (1.0 - row["oriented_ai"], row["weight"]))
            prediction = "human"
            override = {
                "rule": "human_expert_disagreement",
                "expert": strongest["group"],
                "expert_human": float(1.0 - strongest["oriented_ai"]),
                "expert_weight": float(strongest["weight"]),
                "threshold": float(human_veto_threshold),
                "min_weight": float(human_veto_min_weight),
            }

    sorted_experts = sorted(expert_rows, key=lambda r: r["weight"], reverse=True)
    geometry = geometric_confidence(
        sorted_experts,
        p_ai,
        prediction,
        alignment_threshold=alignment_threshold,
    )

    return {
        "p_ai": p_ai,
        "p_human": 1.0 - p_ai,
        "prediction": prediction,
        "base_prediction": base_prediction,
        "override": override,
        "geometric_confidence": geometry,
        "experts": sorted_experts,
    }


def save_experts(path, experts, metadata=None):
    """Save oriented experts to .npz."""
    save_dict = {
        "n_experts": len(experts),
        "metadata": json.dumps(metadata or {}, sort_keys=True),
    }
    for i, expert in enumerate(experts):
        save_dict[f"expert{i}_group"] = expert["group"]
        save_dict[f"expert{i}_orientation"] = expert["orientation"]
        save_dict[f"expert{i}_direct_auc"] = -1.0 if expert["direct_auc"] is None else expert["direct_auc"]
        save_dict[f"expert{i}_n_human"] = expert["n_human"]
        save_dict[f"expert{i}_n_ai"] = expert["n_ai"]
        save_dict[f"expert{i}_human_mean"] = expert["model_human"]["mean"]
        save_dict[f"expert{i}_human_std"] = expert["model_human"]["std"]
        save_dict[f"expert{i}_ai_mean"] = expert["model_ai"]["mean"]
        save_dict[f"expert{i}_ai_std"] = expert["model_ai"]["std"]
        context = require_context_model(expert)
        save_dict[f"expert{i}_context_mean"] = context["mean"]
        save_dict[f"expert{i}_context_std"] = context["std"]
        save_dict[f"expert{i}_context_basis"] = context.get(
            "basis",
            np.zeros((len(context["mean"]), 0), dtype=float),
        )
        save_dict[f"expert{i}_reliability"] = json.dumps(
            expert.get("reliability"),
            sort_keys=True,
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **save_dict)


def load_experts(path):
    """Load oriented experts from .npz."""
    data = np.load(path, allow_pickle=False)
    experts = []
    for i in range(int(data["n_experts"])):
        direct_auc = float(data[f"expert{i}_direct_auc"])
        expert = {
            "group": str(data[f"expert{i}_group"]),
            "orientation": int(data[f"expert{i}_orientation"]),
            "direct_auc": None if direct_auc < 0 else direct_auc,
            "n_human": int(data[f"expert{i}_n_human"]),
            "n_ai": int(data[f"expert{i}_n_ai"]),
            "model_human": {
                "mean": data[f"expert{i}_human_mean"],
                "std": data[f"expert{i}_human_std"],
            },
            "model_ai": {
                "mean": data[f"expert{i}_ai_mean"],
                "std": data[f"expert{i}_ai_std"],
            },
        }
        if f"expert{i}_context_mean" not in data or f"expert{i}_context_std" not in data:
            raise ValueError(f"MoE model expert {i} is missing context geometry")
        expert["context_model"] = {
            "mean": data[f"expert{i}_context_mean"],
            "std": data[f"expert{i}_context_std"],
        }
        if f"expert{i}_context_basis" in data:
            expert["context_model"]["basis"] = data[f"expert{i}_context_basis"]
        if f"expert{i}_reliability" in data:
            raw_reliability = str(data[f"expert{i}_reliability"].item())
            expert["reliability"] = json.loads(raw_reliability)
        experts.append(expert)
    return experts


def evaluate_guardrails(
    experts,
    output_path,
    temperature=1.0,
    ai_veto_threshold=None,
    ai_veto_min_weight=0.0,
    human_veto_threshold=None,
    human_veto_min_weight=0.0,
    competence_metric="z_distance",
    competence_max_distance=None,
    competence_strength=0.0,
    reliability_strength=0.0,
    reliability_floor=0.05,
    suite="basic",
    max_wiki=2,
    holdout_start=500,
    holdout_per_class=3,
    window_words=None,
    stride_words=None,
    min_window_words=40,
):
    """Evaluate mixture experts on hard guardrails."""
    records = guardrail_records(
        max_wiki=max_wiki,
        suite=suite,
        holdout_start=holdout_start,
        holdout_per_class=holdout_per_class,
        window_words=window_words,
        stride_words=stride_words,
        min_window_words=min_window_words,
    )
    device = _get_device()
    loaded_models = _preload_models(DEFAULT_MODELS, device)
    rows = []

    for i, record in enumerate(records, start=1):
        print(f"  Guardrail {i}/{len(records)}: {record['title']}", flush=True)
        features = _extract_features_single(record["text"], loaded_models, device)
        result = predict_mixture(
            features,
            experts,
            temperature=temperature,
            ai_veto_threshold=ai_veto_threshold,
            ai_veto_min_weight=ai_veto_min_weight,
            human_veto_threshold=human_veto_threshold,
            human_veto_min_weight=human_veto_min_weight,
            competence_metric=competence_metric,
            competence_max_distance=competence_max_distance,
            competence_strength=competence_strength,
            reliability_strength=reliability_strength,
            reliability_floor=reliability_floor,
        )
        verdict = "likely_ai" if result["prediction"] == "ai" else "likely_human"
        status = expected_status(record["expected"], verdict)
        rows.append({**record, **result, "status": status})

    failures = [row for row in rows if row["status"] != "ok"]
    lines = [
        "# Mixture Expert Guardrail Evaluation",
        "",
        f"- Result: {'FAIL' if failures else 'PASS'}",
        f"- Failures: {len(failures)} / {len(rows)}",
        f"- Expert temperature: {temperature}",
        f"- AI disagreement threshold: {ai_veto_threshold if ai_veto_threshold is not None else 'disabled'}",
        f"- AI disagreement minimum expert weight: {ai_veto_min_weight}",
        f"- Human disagreement threshold: {human_veto_threshold if human_veto_threshold is not None else 'disabled'}",
        f"- Human disagreement minimum expert weight: {human_veto_min_weight}",
        f"- Competence metric: {competence_metric}",
        f"- Competence maximum distance: {competence_max_distance if competence_max_distance is not None else 'disabled'}",
        f"- Competence strength: {competence_strength}",
        f"- Reliability strength: {reliability_strength}",
        f"- Reliability floor: {reliability_floor}",
        f"- Guardrail suite: {suite}",
        f"- Wiki rows per class: {max_wiki}",
        f"- Holdout start per class: {holdout_start}",
        f"- Holdout rows per class: {holdout_per_class}",
        f"- Window words: {window_words if window_words else 'disabled'}",
        f"- Window stride words: {stride_words if stride_words else 'default'}",
        f"- Minimum window words: {min_window_words}",
        "",
        "## Experts",
        "",
        "| Expert | Orientation | Direct AUC | Human n | AI n |",
        "|---|---|---:|---:|---:|",
    ]
    for expert in experts:
        orientation = "direct" if expert["orientation"] == 1 else "inverted"
        auc = "n/a" if expert["direct_auc"] is None else f"{expert['direct_auc']:.3f}"
        lines.append(
            f"| {expert['group']} | {orientation} | {auc} | "
            f"{expert['n_human']} | {expert['n_ai']} |"
        )

    lines.extend([
        "",
        "## Failure Summary",
        "",
        "| Kind | Rows | Failures |",
        "|---|---:|---:|",
    ])
    kinds = sorted({row.get("kind", "unknown") for row in rows})
    for kind in kinds:
        kind_rows = [row for row in rows if row.get("kind", "unknown") == kind]
        kind_failures = [row for row in kind_rows if row["status"] != "ok"]
        lines.append(f"| {kind} | {len(kind_rows)} | {len(kind_failures)} |")

    lines.extend([
        "",
        "## Guardrails",
        "",
        "| ID | Kind | Expected | Prediction | Human | AI | Status | Rule | Top experts |",
        "|---|---|---|---|---:|---:|---|---|---|",
    ])
    for row in rows:
        top = ", ".join(
            f"{expert['group']} w={expert['weight'] * 100:.1f}% ai={expert['oriented_ai'] * 100:.1f}% d={expert['context_distance']:.2f} a={expert['plane_alignment']:.2f} r={expert['reliability']:.2f}"
            for expert in row["experts"][:3]
        )
        override = row.get("override")
        if override:
            if override["rule"] == "ai_expert_disagreement":
                rule = (
                    f"{override['expert']} ai={override['expert_ai'] * 100:.1f}% "
                    f"w={override['expert_weight'] * 100:.1f}%"
                )
            else:
                rule = (
                    f"{override['expert']} human={override['expert_human'] * 100:.1f}% "
                    f"w={override['expert_weight'] * 100:.1f}%"
                )
        else:
            rule = row["base_prediction"]
        lines.append(
            f"| {row['id']} | {row.get('kind', '')} | {row['expected']} | {row['prediction']} | "
            f"{fmt_pct(row['p_human'])} | {fmt_pct(row['p_ai'])} | "
            f"{row['status']} | {rule} | {top} |"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate oriented Gaussian mixture experts")
    parser.add_argument("--data", default="extended")
    parser.add_argument("--max-per-class", type=int, default=150)
    parser.add_argument("--cache", default="reports/feature_cache_extended_150.npz")
    parser.add_argument("--base-cache", default=None)
    parser.add_argument("--output", default="src/arepo/models/mixture_experts.extended.npz")
    parser.add_argument("--report", default="reports/mixture_expert_guardrails.md")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ai-veto-threshold", type=float, default=None)
    parser.add_argument("--ai-veto-min-weight", type=float, default=0.0)
    parser.add_argument("--human-veto-threshold", type=float, default=None)
    parser.add_argument("--human-veto-min-weight", type=float, default=0.0)
    parser.add_argument("--competence-metric", choices=["z_distance", "cosine", "plane"], default="z_distance")
    parser.add_argument("--competence-max-distance", type=float, default=None)
    parser.add_argument("--competence-strength", type=float, default=0.0)
    parser.add_argument("--reliability-strength", type=float, default=0.0)
    parser.add_argument("--reliability-floor", type=float, default=0.05)
    parser.add_argument("--reliability-bins", type=int, default=3)
    parser.add_argument("--min-reliability-count", type=int, default=3)
    parser.add_argument("--suite", choices=["basic", "expanded"], default="basic")
    parser.add_argument("--max-wiki", type=int, default=2)
    parser.add_argument("--holdout-start", type=int, default=500)
    parser.add_argument("--holdout-per-class", type=int, default=3)
    parser.add_argument("--window-words", type=int, default=0)
    parser.add_argument("--stride-words", type=int, default=None)
    parser.add_argument("--min-window-words", type=int, default=40)
    parser.add_argument("--clusters-per-group", type=int, default=0)
    parser.add_argument("--min-cluster-class", type=int, default=8)
    parser.add_argument("--dataset-union-pairs", type=int, default=0)
    parser.add_argument("--min-dataset-union-class", type=int, default=None)
    parser.add_argument("--union-cluster-pairs", type=int, default=0)
    parser.add_argument("--min-union-class", type=int, default=None)
    parser.add_argument("--include-civic-expert", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)
    if args.evaluate_only:
        experts = load_experts(output_path)
    else:
        records = load_training_records(args.data, args.max_per_class)
        if args.include_civic_expert:
            records = records + civic_expert_records()
        features, labels, groups, _ = load_or_extract_cached_features(
            records,
            args.cache,
            base_cache_path=args.base_cache,
        )
        experts = train_dataset_and_cluster_experts(
            features,
            labels,
            groups,
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
        save_experts(
            output_path,
            experts,
            metadata={
                "data": args.data,
                "max_per_class": args.max_per_class,
                "cache": args.cache,
                "random_state": args.random_state,
                "clusters_per_group": args.clusters_per_group,
                "min_cluster_class": args.min_cluster_class,
                "dataset_union_pairs": args.dataset_union_pairs,
                "min_dataset_union_class": args.min_dataset_union_class,
                "union_cluster_pairs": args.union_cluster_pairs,
                "min_union_class": args.min_union_class,
                "include_civic_expert": args.include_civic_expert,
            },
        )
        print(f"Saved {output_path}")

    evaluate_guardrails(
        experts,
        args.report,
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
        suite=args.suite,
        max_wiki=args.max_wiki,
        holdout_start=args.holdout_start,
        holdout_per_class=args.holdout_per_class,
        window_words=args.window_words,
        stride_words=args.stride_words,
        min_window_words=args.min_window_words,
    )
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
