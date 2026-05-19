"""Integration tests — require transformer model downloads."""

import numpy as np
import pytest

pytestmark = pytest.mark.slow


class TestGenerateEmbeddings:
    def test_returns_dict_with_finite_values(self, sample_texts):
        from arepo.core import generate_embeddings
        human_texts, _ = sample_texts
        emb = generate_embeddings(human_texts[0])
        assert isinstance(emb, dict)
        assert len(emb) > 0
        for name, val in emb.items():
            if val is not None:
                arr = val.numpy() if hasattr(val, "numpy") else np.asarray(val)
                assert np.all(np.isfinite(arr)), f"{name} has non-finite values"


class TestConstructFeatureVector:
    def test_shape_and_finite(self, sample_texts):
        from arepo.core import construct_feature_vector, DEFAULT_MODELS
        human_texts, _ = sample_texts
        fv = construct_feature_vector(human_texts[0])
        assert fv.shape == (len(DEFAULT_MODELS) * 4,)
        assert np.all(np.isfinite(fv))


class TestExtract4dFeatures:
    def test_output_shape_and_ks_range(self, sample_texts):
        from arepo.core import extract_4d_features, DEFAULT_MODELS
        human_texts, _ = sample_texts
        feats = extract_4d_features(human_texts[:2])
        assert feats.ndim == 2
        assert feats.shape[1] == len(DEFAULT_MODELS)
        assert np.all(feats >= 0) and np.all(feats <= 1)


class TestFullPipeline:
    def test_extract_train_moe_predict(self, sample_texts):
        from arepo.core import extract_4d_features
        from arepo.mixture_experts import predict_mixture, train_expert_for_group
        human_texts, ai_texts = sample_texts
        X_h = extract_4d_features(human_texts[:3])
        X_a = extract_4d_features(ai_texts[:3])
        if len(X_h) < 2 or len(X_a) < 2:
            pytest.skip("Not enough valid features extracted")
        X = np.vstack([X_h, X_a])
        y = np.array([0] * len(X_h) + [1] * len(X_a))
        experts = train_expert_for_group("integration", X, y, random_state=0)
        result = predict_mixture(X[0], experts)
        assert 0 <= result["p_ai"] <= 1
        assert result["prediction"] in {"human", "ai"}


class TestVisualizationOutput:
    def test_matplotlib_save(self, tmp_path, sample_texts):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        ax.set_title("smoke test")
        out = tmp_path / "test_plot.png"
        fig.savefig(out)
        plt.close(fig)
        assert out.exists() and out.stat().st_size > 0
