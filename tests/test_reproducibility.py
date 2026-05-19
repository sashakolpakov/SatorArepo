"""Reproducibility checks for the MoE engine."""

import numpy as np
import pytest

pytestmark = pytest.mark.reproducibility


def test_moe_training_is_deterministic_on_fixed_features(sample_4d_data):
    from arepo.mixture_experts import predict_mixture, train_expert_for_group

    X, y = sample_4d_data
    left = train_expert_for_group("repro", X, y, random_state=123)
    right = train_expert_for_group("repro", X, y, random_state=123)

    for feature in X[:4]:
        left_score = predict_mixture(feature, left)
        right_score = predict_mixture(feature, right)
        assert left_score["p_ai"] == pytest.approx(right_score["p_ai"])
        assert left_score["prediction"] == right_score["prediction"]


def test_moe_separates_synthetic_4d_regimes(sample_4d_data):
    from arepo.mixture_experts import predict_mixture, train_expert_for_group

    X, y = sample_4d_data
    experts = train_expert_for_group("repro", X, y, random_state=42)
    predictions = np.array([
        1 if predict_mixture(feature, experts)["prediction"] == "ai" else 0
        for feature in X
    ])

    assert (predictions == y).mean() > 0.8
