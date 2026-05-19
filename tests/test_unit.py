"""Fast unit tests — no network, no models, pure math."""

import numpy as np
import pytest
from types import SimpleNamespace

pytestmark = pytest.mark.unit


def _unit_context(mean, std=None):
    mean = np.asarray(mean, dtype=float)
    std = np.ones_like(mean) if std is None else np.asarray(std, dtype=float)
    return {
        "mean": mean,
        "std": std,
        "basis": np.eye(len(mean), min(2, len(mean))),
    }


# ---------------------------------------------------------------------------
# extract_leading_digits
# ---------------------------------------------------------------------------

class TestExtractLeadingDigits:
    def test_known_values(self):
        from arepo.utils import extract_leading_digits
        arr = np.array([1.0, 2.5, 0.003, 7890.0])
        digits = extract_leading_digits(arr)
        assert list(digits) == [1, 2, 3, 7]

    def test_negative_values(self):
        from arepo.utils import extract_leading_digits
        arr = np.array([-4.5, -0.067])
        digits = extract_leading_digits(arr)
        assert list(digits) == [4, 6]

    def test_zeros_excluded(self):
        from arepo.utils import extract_leading_digits
        arr = np.array([0.0, 0.0, 5.0])
        digits = extract_leading_digits(arr)
        assert len(digits) == 1
        assert digits[0] == 5

    def test_range_1_to_9(self):
        from arepo.utils import extract_leading_digits
        rng = np.random.default_rng(0)
        arr = rng.standard_normal(500)
        digits = extract_leading_digits(arr)
        assert np.all((digits >= 1) & (digits <= 9))


# ---------------------------------------------------------------------------
# calculate_empirical_distribution
# ---------------------------------------------------------------------------

class TestEmpiricalDistribution:
    def test_sums_to_one(self):
        from arepo.utils import calculate_empirical_distribution
        digits = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        p = calculate_empirical_distribution(digits)
        assert pytest.approx(p.sum(), abs=1e-10) == 1.0

    def test_correct_counts(self):
        from arepo.utils import calculate_empirical_distribution
        digits = np.array([1, 1, 1, 2, 2, 3])
        p = calculate_empirical_distribution(digits)
        assert pytest.approx(p[0]) == 3.0 / 6  # digit 1
        assert pytest.approx(p[1]) == 2.0 / 6  # digit 2
        assert pytest.approx(p[2]) == 1.0 / 6  # digit 3

    def test_nine_bins(self):
        from arepo.utils import calculate_empirical_distribution
        digits = np.array([1])
        assert len(calculate_empirical_distribution(digits)) == 9


# ---------------------------------------------------------------------------
# compute_benford_indicators
# ---------------------------------------------------------------------------

class TestBenfordIndicators:
    def test_perfect_benford_low_mse(self, benford_distribution):
        from arepo.utils import compute_benford_indicators
        ind = compute_benford_indicators(benford_distribution)
        assert ind["MSE"] < 1e-6

    def test_uniform_deviates(self):
        from arepo.utils import compute_benford_indicators
        p_uniform = np.ones(9) / 9.0
        ind = compute_benford_indicators(p_uniform)
        assert ind["MSE"] > 1e-4  # clearly not Benford

    def test_kl_nonnegative(self, benford_distribution):
        from arepo.utils import compute_benford_indicators
        ind = compute_benford_indicators(benford_distribution)
        assert ind["KL"] >= -1e-10


# ---------------------------------------------------------------------------
# extract_mantissas
# ---------------------------------------------------------------------------

class TestExtractMantissas:
    def test_unit_interval(self, sample_embedding):
        from arepo.stats import extract_mantissas
        m = extract_mantissas(sample_embedding)
        assert np.all(m >= 0) and np.all(m < 1)

    def test_zeros_excluded(self):
        from arepo.stats import extract_mantissas
        arr = np.array([0.0, 0.0, 1.0])
        m = extract_mantissas(arr)
        assert len(m) == 1

    def test_known_values(self):
        from arepo.stats import extract_mantissas
        arr = np.array([10.0, 100.0, 1000.0])  # log10 = 1, 2, 3 -> mantissa 0
        m = extract_mantissas(arr)
        np.testing.assert_allclose(m, 0.0, atol=1e-10)

    def test_lognormal_approx_uniform(self):
        """Lognormal samples should have approximately uniform mantissas (Benford)."""
        from arepo.stats import extract_mantissas
        from scipy import stats as sp_stats
        rng = np.random.default_rng(42)
        samples = rng.lognormal(mean=0, sigma=5, size=10000)
        m = extract_mantissas(samples)
        ks_stat, p_value = sp_stats.kstest(m, "uniform")
        assert p_value > 0.01, f"KS p={p_value:.4f}, mantissas not uniform enough"


# ---------------------------------------------------------------------------
# compute_mantissa_multi_stats
# ---------------------------------------------------------------------------

class TestComputeMantissaMultiStats:
    def test_returns_five_keys(self, sample_embedding):
        from arepo.stats import extract_mantissas, compute_mantissa_multi_stats
        m = extract_mantissas(sample_embedding)
        result = compute_mantissa_multi_stats(m)
        assert set(result.keys()) == {'ks', 'cvm', 'ad', 'benford_mse', 'benford_kl'}

    def test_all_finite(self, sample_embedding):
        from arepo.stats import extract_mantissas, compute_mantissa_multi_stats
        m = extract_mantissas(sample_embedding)
        result = compute_mantissa_multi_stats(m)
        for k, v in result.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"

    def test_empty_returns_nan(self):
        from arepo.stats import compute_mantissa_multi_stats
        result = compute_mantissa_multi_stats(np.array([]))
        for v in result.values():
            assert np.isnan(v)

    def test_ks_matches_standalone(self, sample_embedding):
        from arepo.stats import extract_mantissas, compute_mantissa_multi_stats
        from scipy import stats as sp_stats
        m = extract_mantissas(sample_embedding)
        result = compute_mantissa_multi_stats(m)
        ks_direct, _ = sp_stats.kstest(m, 'uniform')
        assert result['ks'] == pytest.approx(ks_direct, abs=1e-10)


# ---------------------------------------------------------------------------
# fit_gaussian_4d
# ---------------------------------------------------------------------------

class TestFitGaussian4d:
    def test_mean_and_std(self):
        from arepo.stats import fit_gaussian_4d
        data = np.array([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=float)
        model = fit_gaussian_4d(data)
        np.testing.assert_allclose(model["mean"], [1, 2, 3, 4])
        # std is 0 but clamped to min_std
        assert np.all(model["std"] >= 0.001)

    def test_min_std_regularization(self):
        from arepo.stats import fit_gaussian_4d
        data = np.ones((5, 4))
        model = fit_gaussian_4d(data, min_std=0.01)
        assert np.all(model["std"] >= 0.01)


class TestMixtureExperts:
    def test_predict_mixture_uses_expert_orientation(self):
        from arepo.mixture_experts import predict_mixture

        experts = [
            {
                "group": "direct",
                "orientation": 1,
                "direct_auc": 1.0,
                "n_human": 2,
                "n_ai": 2,
                "model_human": {
                    "mean": np.array([0.0, 0.0]),
                    "std": np.array([1.0, 1.0]),
                },
                "model_ai": {
                    "mean": np.array([3.0, 3.0]),
                    "std": np.array([1.0, 1.0]),
                },
                "context_model": _unit_context([1.5, 1.5]),
            },
            {
                "group": "inverted",
                "orientation": -1,
                "direct_auc": 0.0,
                "n_human": 2,
                "n_ai": 2,
                "model_human": {
                    "mean": np.array([3.0, 3.0]),
                    "std": np.array([1.0, 1.0]),
                },
                "model_ai": {
                    "mean": np.array([0.0, 0.0]),
                    "std": np.array([0.9, 0.9]),
                },
                "context_model": _unit_context([1.5, 1.5]),
            },
        ]

        result = predict_mixture(np.array([3.0, 3.0]), experts, temperature=1.0)

        assert result["prediction"] == "ai"
        assert result["p_ai"] > 0.9

    def test_learn_group_orientation_detects_inversion(self):
        from arepo.mixture_experts import learn_group_orientation

        rng = np.random.default_rng(123)
        human = rng.normal(3.0, 0.1, size=(10, 2))
        ai = rng.normal(0.0, 0.1, size=(10, 2))
        features = np.vstack([human, ai])
        labels = np.array([0] * len(human) + [1] * len(ai))

        orientation, auc = learn_group_orientation(features, labels, random_state=123)

        assert orientation == 1
        assert auc > 0.9

    def test_ai_veto_overrides_human_weighted_mixture(self):
        from arepo.mixture_experts import predict_mixture

        experts = [
            {
                "group": "weak-human",
                "orientation": 1,
                "direct_auc": 1.0,
                "n_human": 2,
                "n_ai": 2,
                "model_human": {
                    "mean": np.array([0.0, 0.0]),
                    "std": np.array([0.1, 0.1]),
                },
                "model_ai": {
                    "mean": np.array([3.0, 3.0]),
                    "std": np.array([0.1, 0.1]),
                },
                "context_model": _unit_context([1.5, 1.5]),
            },
            {
                "group": "strong-ai",
                "orientation": 1,
                "direct_auc": 1.0,
                "n_human": 2,
                "n_ai": 2,
                "model_human": {
                    "mean": np.array([1.0, 1.0]),
                    "std": np.array([1.0, 1.0]),
                },
                "model_ai": {
                    "mean": np.array([0.0, 0.0]),
                    "std": np.array([0.9, 0.9]),
                },
                "context_model": _unit_context([0.5, 0.5]),
            },
        ]

        result = predict_mixture(
            np.array([0.0, 0.0]),
            experts,
            ai_veto_threshold=0.7,
            ai_veto_min_weight=0.01,
        )

        assert result["prediction"] == "ai"
        assert result["base_prediction"] == "human"
        assert result["override"]["rule"] == "ai_expert_disagreement"

    def test_human_veto_overrides_ai_weighted_mixture(self):
        from arepo.mixture_experts import predict_mixture

        experts = [
            {
                "group": "strong-ai",
                "orientation": 1,
                "direct_auc": 1.0,
                "n_human": 2,
                "n_ai": 2,
                "model_human": {
                    "mean": np.array([3.0, 3.0]),
                    "std": np.array([1.0, 1.0]),
                },
                "model_ai": {
                    "mean": np.array([0.0, 0.0]),
                    "std": np.array([0.9, 0.9]),
                },
                "context_model": _unit_context([1.5, 1.5]),
            },
            {
                "group": "strong-human",
                "orientation": 1,
                "direct_auc": 1.0,
                "n_human": 2,
                "n_ai": 2,
                "model_human": {
                    "mean": np.array([0.0, 0.0]),
                    "std": np.array([1.0, 1.0]),
                },
                "model_ai": {
                    "mean": np.array([4.0, 4.0]),
                    "std": np.array([1.0, 1.0]),
                },
                "context_model": _unit_context([2.0, 2.0]),
            },
        ]

        result = predict_mixture(
            np.array([0.0, 0.0]),
            experts,
            human_veto_threshold=0.9,
            human_veto_min_weight=0.2,
        )

        assert result["prediction"] == "human"
        assert result["base_prediction"] == "ai"
        assert result["override"]["rule"] == "human_expert_disagreement"

    def test_human_veto_does_not_undo_ai_veto_recovery(self):
        from arepo.mixture_experts import predict_mixture

        experts = [
            {
                "group": "aggregate-human",
                "orientation": 1,
                "direct_auc": 1.0,
                "n_human": 2,
                "n_ai": 2,
                "model_human": {
                    "mean": np.array([0.0, 0.0]),
                    "std": np.array([1.0, 1.0]),
                },
                "model_ai": {
                    "mean": np.array([3.5, 3.5]),
                    "std": np.array([1.0, 1.0]),
                },
                "context_model": _unit_context([1.75, 1.75]),
            },
            {
                "group": "specialist-ai",
                "orientation": 1,
                "direct_auc": 1.0,
                "n_human": 2,
                "n_ai": 2,
                "model_human": {
                    "mean": np.array([2.5, 2.5]),
                    "std": np.array([1.0, 1.0]),
                },
                "model_ai": {
                    "mean": np.array([0.0, 0.0]),
                    "std": np.array([1.0, 1.0]),
                },
                "context_model": _unit_context([1.25, 1.25]),
            },
        ]

        result = predict_mixture(
            np.array([0.0, 0.0]),
            experts,
            ai_veto_threshold=0.9,
            ai_veto_min_weight=0.2,
            human_veto_threshold=0.9,
            human_veto_min_weight=0.2,
        )

        assert result["base_prediction"] == "human"
        assert result["prediction"] == "ai"
        assert result["override"]["rule"] == "ai_expert_disagreement"

    def test_competence_gate_excludes_off_context_expert(self):
        from arepo.mixture_experts import predict_mixture

        experts = [
            {
                "group": "on-context-ai",
                "orientation": 1,
                "direct_auc": 1.0,
                "n_human": 2,
                "n_ai": 2,
                "model_human": {
                    "mean": np.array([2.0, 2.0]),
                    "std": np.array([1.0, 1.0]),
                },
                "model_ai": {
                    "mean": np.array([0.0, 0.0]),
                    "std": np.array([1.0, 1.0]),
                },
                "context_model": {
                    "mean": np.array([0.0, 0.0]),
                    "std": np.array([1.0, 1.0]),
                },
            },
            {
                "group": "off-context-human",
                "orientation": 1,
                "direct_auc": 1.0,
                "n_human": 2,
                "n_ai": 2,
                "model_human": {
                    "mean": np.array([0.0, 0.0]),
                    "std": np.array([0.1, 0.1]),
                },
                "model_ai": {
                    "mean": np.array([2.0, 2.0]),
                    "std": np.array([0.1, 0.1]),
                },
                "context_model": {
                    "mean": np.array([10.0, 10.0]),
                    "std": np.array([1.0, 1.0]),
                },
            },
        ]

        ungated = predict_mixture(np.array([0.0, 0.0]), experts)
        gated = predict_mixture(
            np.array([0.0, 0.0]),
            experts,
            competence_max_distance=3.0,
        )

        assert ungated["prediction"] == "human"
        assert gated["prediction"] == "ai"
        assert gated["experts"][0]["group"] == "on-context-ai"
        assert gated["experts"][1]["weight"] == 0.0

    def test_plane_competence_uses_context_subspace_alignment(self):
        from arepo.mixture_experts import predict_mixture

        experts = [
            {
                "group": "aligned-ai",
                "orientation": 1,
                "direct_auc": 1.0,
                "n_human": 2,
                "n_ai": 2,
                "model_human": {
                    "mean": np.array([2.0, 0.0]),
                    "std": np.array([1.0, 1.0]),
                },
                "model_ai": {
                    "mean": np.array([1.0, 0.0]),
                    "std": np.array([1.0, 1.0]),
                },
                "context_model": {
                    "mean": np.array([0.0, 0.0]),
                    "std": np.array([1.0, 1.0]),
                    "basis": np.array([[1.0], [0.0]]),
                },
            },
            {
                "group": "misaligned-human",
                "orientation": 1,
                "direct_auc": 1.0,
                "n_human": 2,
                "n_ai": 2,
                "model_human": {
                    "mean": np.array([1.0, 0.0]),
                    "std": np.array([1.0, 1.0]),
                },
                "model_ai": {
                    "mean": np.array([2.0, 0.0]),
                    "std": np.array([1.0, 1.0]),
                },
                "context_model": {
                    "mean": np.array([0.0, 0.0]),
                    "std": np.array([1.0, 1.0]),
                    "basis": np.array([[0.0], [1.0]]),
                },
            },
        ]

        result = predict_mixture(
            np.array([1.0, 0.0]),
            experts,
            competence_metric="plane",
            competence_strength=10.0,
        )

        assert result["prediction"] == "ai"
        assert result["experts"][0]["group"] == "aligned-ai"
        assert result["experts"][0]["plane_alignment"] == pytest.approx(1.0)

    def test_saved_experts_include_context_model(self, tmp_path):
        from arepo.mixture_experts import load_experts, save_experts, train_expert_for_group

        features = np.array([
            [0.0, 0.0],
            [0.2, 0.0],
            [2.0, 2.0],
            [2.2, 2.0],
        ])
        labels = np.array([0, 0, 1, 1])
        expert = train_expert_for_group("x", features, labels)[0]
        path = tmp_path / "experts.npz"

        save_experts(path, [expert])
        loaded = load_experts(path)

        assert "context_model" in loaded[0]
        np.testing.assert_allclose(loaded[0]["context_model"]["mean"], features.mean(axis=0))

    def test_reliability_downweights_unreliable_vote(self):
        from arepo.mixture_experts import predict_mixture

        experts = [
            {
                "group": "reliable-ai",
                "orientation": 1,
                "direct_auc": 1.0,
                "n_human": 2,
                "n_ai": 2,
                "model_human": {
                    "mean": np.array([2.0, 2.0]),
                    "std": np.array([1.0, 1.0]),
                },
                "model_ai": {
                    "mean": np.array([0.0, 0.0]),
                    "std": np.array([1.0, 1.0]),
                },
                "context_model": {
                    "mean": np.array([0.0, 0.0]),
                    "std": np.array([1.0, 1.0]),
                },
                "reliability": {
                    "edges": [],
                    "overall": 0.9,
                    "human": [0.9],
                    "ai": [0.9],
                },
            },
            {
                "group": "unreliable-human",
                "orientation": 1,
                "direct_auc": 1.0,
                "n_human": 2,
                "n_ai": 2,
                "model_human": {
                    "mean": np.array([0.0, 0.0]),
                    "std": np.array([0.1, 0.1]),
                },
                "model_ai": {
                    "mean": np.array([2.0, 2.0]),
                    "std": np.array([0.1, 0.1]),
                },
                "context_model": {
                    "mean": np.array([0.0, 0.0]),
                    "std": np.array([1.0, 1.0]),
                },
                "reliability": {
                    "edges": [],
                    "overall": 0.1,
                    "human": [0.1],
                    "ai": [0.1],
                },
            },
        ]

        ungated = predict_mixture(np.array([0.0, 0.0]), experts)
        reliable = predict_mixture(
            np.array([0.0, 0.0]),
            experts,
            reliability_strength=5.0,
        )

        assert ungated["prediction"] == "human"
        assert reliable["prediction"] == "ai"
        assert reliable["experts"][0]["group"] == "reliable-ai"

    def test_train_expert_can_fit_reliability_table(self):
        from arepo.mixture_experts import train_expert_for_group

        rng = np.random.default_rng(123)
        human = rng.normal(0.0, 0.1, size=(12, 2))
        ai = rng.normal(2.0, 0.1, size=(12, 2))
        features = np.vstack([human, ai])
        labels = np.array([0] * len(human) + [1] * len(ai))

        expert = train_expert_for_group(
            "x",
            features,
            labels,
            reliability=True,
            reliability_bins=2,
            min_reliability_count=2,
            random_state=123,
        )[0]

        assert expert["reliability"] is not None
        assert len(expert["reliability"]["human"]) == 2
        assert len(expert["reliability"]["ai"]) == 2


class TestSetCoverageSelection:
    def test_geometric_confidence_counts_aligned_majority_mass(self):
        from arepo.mixture_experts import geometric_confidence

        confidence = geometric_confidence(
            [
                {"weight": 0.6, "oriented_ai": 0.9, "plane_alignment": 0.9},
                {"weight": 0.2, "oriented_ai": 0.8, "plane_alignment": 0.5},
                {"weight": 0.2, "oriented_ai": 0.1, "plane_alignment": 0.9},
            ],
            p_ai=0.8,
            prediction="ai",
            alignment_threshold=0.8,
        )

        assert confidence["posterior_margin"] == pytest.approx(0.6)
        assert confidence["majority_weight"] == pytest.approx(0.8)
        assert confidence["aligned_majority_weight"] == pytest.approx(0.6)
        assert confidence["aligned_minority_weight"] == pytest.approx(0.2)
        assert confidence["confidence"] == pytest.approx(0.24)

    def test_geometric_confidence_gate_marks_low_alignment_confidence(self):
        from arepo.greedy_expert_selection import (
            acceptance_summary,
            score_guardrails,
        )

        experts = []
        for group, center in (("left", -1.0), ("right", 1.0)):
            experts.append({
                "group": group,
                "orientation": 1,
                "direct_auc": 1.0,
                "n_human": 2,
                "n_ai": 2,
                "model_human": {
                    "mean": np.array([center]),
                    "std": np.array([1.0]),
                },
                "model_ai": {
                    "mean": np.array([center + 2.0]),
                    "std": np.array([1.0]),
                },
                "context_model": {
                    "mean": np.array([center]),
                    "std": np.array([1.0]),
                },
            })
        args = SimpleNamespace(
            temperature=10.0,
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
            min_geometric_confidence=0.75,
            context_space="feature",
        )
        rows = [{"id": "x", "expected": "ai", "kind": "unit", "text": "x"}]
        scored = score_guardrails(np.array([[0.0]]), rows, experts, args)

        assert scored[0]["accepted"] is False
        assert scored[0]["acceptance_gate"]["confidence"] < 0.75
        assert acceptance_summary(scored)["accepted"] == 0

    def test_set_coverage_rewards_fixed_failures_and_penalizes_breaks(self):
        from arepo.greedy_expert_selection import set_coverage_delta

        args = SimpleNamespace(
            false_negative_weight=2.0,
            false_positive_weight=1.0,
            break_penalty=0.75,
            protected_break_penalty=1.0,
            protected_kind_tokens="public-domain,wiki",
        )
        current = [
            {"id": "a", "expected": "ai", "kind": "synthetic", "status": "FAIL false negative"},
            {"id": "b", "expected": "human", "kind": "wiki", "status": "ok"},
            {"id": "c", "expected": "human", "kind": "mixed-domain", "status": "ok"},
        ]
        trial = [
            {"id": "a", "expected": "ai", "kind": "synthetic", "status": "ok"},
            {"id": "b", "expected": "human", "kind": "wiki", "status": "FAIL false positive"},
            {"id": "c", "expected": "human", "kind": "mixed-domain", "status": "FAIL false positive"},
        ]

        delta = set_coverage_delta(current, trial, args)

        assert delta["fixed"] == 1
        assert delta["broken"] == 2
        assert delta["protected_broken"] == 1
        assert delta["utility"] == pytest.approx(0.25)

    def test_set_coverage_initial_round_counts_all_correct_rows_as_fixed(self):
        from arepo.greedy_expert_selection import set_coverage_delta

        args = SimpleNamespace()
        trial = [
            {"id": "a", "status": "ok"},
            {"id": "b", "status": "FAIL false positive"},
            {"id": "c", "status": "ok"},
        ]

        delta = set_coverage_delta(None, trial, args)

        assert delta["fixed"] == 2
        assert delta["utility"] == pytest.approx(2.0)

    def test_set_coverage_can_target_false_negatives_with_hard_protected_cap(self):
        from arepo.greedy_expert_selection import set_coverage_delta

        args = SimpleNamespace(
            false_negative_weight=2.0,
            false_positive_weight=1.0,
            break_penalty=0.5,
            protected_break_penalty=1.0,
            protected_kind_tokens="wiki",
            coverage_target="false_negatives",
            max_breaks=-1,
            max_protected_breaks=0,
        )
        current = [
            {"id": "a", "expected": "ai", "kind": "generated", "status": "FAIL false negative"},
            {"id": "b", "expected": "human", "kind": "wiki", "status": "ok"},
            {"id": "c", "expected": "human", "kind": "other", "status": "FAIL false positive"},
        ]
        trial = [
            {"id": "a", "expected": "ai", "kind": "generated", "status": "ok"},
            {"id": "b", "expected": "human", "kind": "wiki", "status": "FAIL false positive"},
            {"id": "c", "expected": "human", "kind": "other", "status": "ok"},
        ]

        delta = set_coverage_delta(current, trial, args)

        assert delta["fixed"] == 2
        assert delta["fixed_weight"] == pytest.approx(2.0)
        assert delta["protected_broken"] == 1
        assert delta["utility"] == float("-inf")

    def test_cluster_experts_add_specific_groups(self):
        from arepo.mixture_experts import train_dataset_and_cluster_experts

        rng = np.random.default_rng(123)
        human_a = rng.normal(0.0, 0.05, size=(10, 2))
        ai_a = rng.normal(0.2, 0.05, size=(10, 2))
        human_b = rng.normal(3.0, 0.05, size=(10, 2))
        ai_b = rng.normal(3.2, 0.05, size=(10, 2))
        features = np.vstack([human_a, ai_a, human_b, ai_b])
        labels = np.array([0] * 10 + [1] * 10 + [0] * 10 + [1] * 10)
        groups = np.array(["x"] * len(labels))

        experts = train_dataset_and_cluster_experts(
            features,
            labels,
            groups,
            clusters_per_group=2,
            min_cluster_class=3,
            random_state=123,
        )

        names = {expert["group"] for expert in experts}

        assert "x" in names
        assert any(name.startswith("x:c") for name in names)

    def test_union_cluster_experts_add_pair_groups(self):
        from arepo.mixture_experts import train_dataset_and_cluster_experts

        rng = np.random.default_rng(123)
        human_a = rng.normal(0.0, 0.05, size=(10, 2))
        ai_a = rng.normal(0.2, 0.05, size=(10, 2))
        human_b = rng.normal(3.0, 0.05, size=(10, 2))
        ai_b = rng.normal(3.2, 0.05, size=(10, 2))
        features = np.vstack([human_a, ai_a, human_b, ai_b])
        labels = np.array([0] * 10 + [1] * 10 + [0] * 10 + [1] * 10)
        groups = np.array(["x"] * len(labels))

        experts = train_dataset_and_cluster_experts(
            features,
            labels,
            groups,
            clusters_per_group=2,
            min_cluster_class=3,
            union_cluster_pairs=1,
            random_state=123,
        )

        names = {expert["group"] for expert in experts}

        assert any(name.startswith("union:x:c") for name in names)

    def test_dataset_union_experts_add_pair_groups(self):
        from arepo.mixture_experts import train_dataset_and_cluster_experts

        rng = np.random.default_rng(123)
        x_human = rng.normal(0.0, 0.05, size=(8, 2))
        x_ai = rng.normal(0.4, 0.05, size=(8, 2))
        y_human = rng.normal(1.0, 0.05, size=(8, 2))
        y_ai = rng.normal(1.4, 0.05, size=(8, 2))
        features = np.vstack([x_human, x_ai, y_human, y_ai])
        labels = np.array([0] * 8 + [1] * 8 + [0] * 8 + [1] * 8)
        groups = np.array(["x"] * 16 + ["y"] * 16)

        experts = train_dataset_and_cluster_experts(
            features,
            labels,
            groups,
            dataset_union_pairs=1,
            min_dataset_union_class=3,
            random_state=123,
        )

        names = {expert["group"] for expert in experts}

        assert "dataset_union:x+y" in names

    def test_civic_expert_records_have_both_classes(self):
        from arepo.mixture_experts import civic_expert_records

        records = civic_expert_records()
        labels = [record["label"] for record in records]

        assert labels.count(0) >= 2
        assert labels.count(1) >= 2
        assert {record["group"] for record in records} == {"historic_civic"}

    def test_feature_cache_reuses_base_rows_by_hash(self, tmp_path, monkeypatch):
        import arepo.mixture_experts as mixture

        base_records = [
            {"text": "alpha", "label": 0, "group": "a"},
            {"text": "beta", "label": 1, "group": "a"},
        ]
        base_cache = tmp_path / "base.npz"
        np.savez(
            base_cache,
            features=np.array([[1.0, 2.0], [3.0, 4.0]]),
            labels=np.array([0, 1]),
            groups=np.array(["a", "a"]),
            valid_indices=np.array([0, 1]),
            text_hashes=np.array([mixture.stable_text_hash(r["text"]) for r in base_records]),
        )

        calls = []

        def fake_extract(texts):
            calls.append(list(texts))
            assert texts == ["gamma"]
            return np.array([[5.0, 6.0]]), np.array([0])

        monkeypatch.setattr(mixture, "extract_4d_features_with_indices", fake_extract)

        records = [
            {"text": "beta", "label": 1, "group": "a"},
            {"text": "gamma", "label": 0, "group": "b"},
        ]
        features, labels, groups, valid_indices = mixture.load_or_extract_cached_features(
            records,
            tmp_path / "grown.npz",
            base_cache_path=base_cache,
        )

        assert calls == [["gamma"]]
        assert features.tolist() == [[3.0, 4.0], [5.0, 6.0]]
        assert labels.tolist() == [1, 0]
        assert groups.tolist() == ["a", "b"]
        assert valid_indices.tolist() == [0, 1]


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_perfect_predictions(self):
        from arepo.stats import evaluate
        y = np.array([0, 0, 1, 1])
        m = evaluate(y, y)
        assert m["accuracy"] == 1.0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0

    def test_all_wrong(self):
        from arepo.stats import evaluate
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        m = evaluate(y_pred, y_true)
        assert m["accuracy"] == 0.0

    def test_auc_roc_perfect(self):
        from arepo.stats import evaluate
        y = np.array([0, 0, 1, 1])
        posteriors = np.array([0.9, 0.8, 0.2, 0.1])  # high=human, low=AI
        m = evaluate(y, y, posteriors=posteriors)
        assert m["auc_roc"] == pytest.approx(1.0, abs=1e-10)

    def test_auc_roc_nan_without_posteriors(self):
        from arepo.stats import evaluate
        y = np.array([0, 0, 1, 1])
        m = evaluate(y, y)
        assert np.isnan(m["auc_roc"])

    def test_auc_roc_nan_single_class(self):
        from arepo.stats import evaluate
        y = np.array([0, 0, 0, 0])
        posteriors = np.array([0.9, 0.8, 0.7, 0.6])
        m = evaluate(y, y, posteriors=posteriors)
        assert np.isnan(m["auc_roc"])

    def test_confusion_keys(self):
        from arepo.stats import evaluate
        y = np.array([0, 1, 1, 0])
        m = evaluate(y, y)
        assert set(m["confusion"].keys()) == {"tp", "tn", "fp", "fn"}
        assert m["confusion"]["tp"] + m["confusion"]["tn"] == len(y)


# ---------------------------------------------------------------------------
# _roc_auc
# ---------------------------------------------------------------------------

class TestRocAuc:
    def test_perfect_separation(self):
        from arepo.stats import _roc_auc
        y = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert _roc_auc(y, scores) == pytest.approx(1.0, abs=1e-10)

    def test_random_around_half(self):
        from arepo.stats import _roc_auc
        rng = np.random.default_rng(42)
        y = np.array([0]*100 + [1]*100)
        scores = rng.uniform(0, 1, 200)
        auc = _roc_auc(y, scores)
        assert 0.3 < auc < 0.7

    def test_nan_single_class(self):
        from arepo.stats import _roc_auc
        y = np.zeros(10, dtype=int)
        scores = np.random.default_rng(0).uniform(0, 1, 10)
        assert np.isnan(_roc_auc(y, scores))


# ---------------------------------------------------------------------------
# roc_curve
# ---------------------------------------------------------------------------

class TestRocCurve:
    def test_endpoints(self):
        from arepo.stats import roc_curve
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.4, 0.6, 0.9])
        fpr, tpr = roc_curve(y, scores)
        assert fpr[0] == 0.0 and tpr[0] == 0.0
        assert fpr[-1] == pytest.approx(1.0) and tpr[-1] == pytest.approx(1.0)

    def test_monotonic(self):
        from arepo.stats import roc_curve
        rng = np.random.default_rng(7)
        y = np.array([0]*50 + [1]*50)
        scores = rng.uniform(0, 1, 100)
        fpr, tpr = roc_curve(y, scores)
        assert np.all(np.diff(fpr) >= -1e-10)
        assert np.all(np.diff(tpr) >= -1e-10)


# permutation_test
# ---------------------------------------------------------------------------

class TestPermutationTest:
    def test_identical_not_significant(self):
        from arepo.stats import permutation_test
        a = np.ones(30)
        b = np.ones(30)
        p = permutation_test(a, b, n_permutations=500)
        assert p > 0.05

    def test_different_significant(self):
        from arepo.stats import permutation_test
        rng = np.random.default_rng(42)
        a = rng.normal(0, 0.1, 50)
        b = rng.normal(5, 0.1, 50)
        p = permutation_test(a, b, n_permutations=500)
        assert p < 0.05


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_contains_true_mean(self):
        from arepo.stats import bootstrap_ci
        rng = np.random.default_rng(42)
        data = rng.normal(10.0, 1.0, 200)
        lo, hi = bootstrap_ci(data, n_bootstrap=2000)
        assert lo < 10.0 < hi

    def test_narrow_with_large_sample(self):
        from arepo.stats import bootstrap_ci
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 5000)
        lo, hi = bootstrap_ci(data, n_bootstrap=2000)
        assert (hi - lo) < 0.2
