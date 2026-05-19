"""Fast tests for the public web boundary."""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_public_score_uses_50_50_rule_with_low_confidence_band():
    from arepo.web import _public_score

    assert _public_score(0.60, geometric_confidence=0.3)["verdict"] == "likely_human"
    assert _public_score(0.40, geometric_confidence=0.3)["verdict"] == "likely_ai"
    assert _public_score(0.50, geometric_confidence=0.0)["verdict"] == "likely_ai"
    assert _public_score(0.50, geometric_confidence=0.0)["confidence_band"] == "low"
    assert _public_score(0.54, geometric_confidence=0.1)["verdict"] == "likely_human"
    assert _public_score(0.54, geometric_confidence=0.1)["confidence_band"] == "low"
    assert _public_score(0.44, geometric_confidence=0.2)["verdict"] == "likely_ai"
    assert _public_score(0.44, geometric_confidence=0.2)["confidence_band"] == "normal"


def test_public_score_rejects_removed_calibration_kwargs():
    from arepo.web import _public_score

    with pytest.raises(TypeError):
        _public_score(0.90, asymmetry_score=-1.0, geometric_confidence=0.8)
    with pytest.raises(TypeError):
        _public_score(0.90, reference_regime="sample", geometric_confidence=0.8)


def test_vote_prediction_overrides_posterior_side_to_match_main():
    from arepo.web import _public_score

    score = _public_score(0.90, prediction="ai", geometric_confidence=0.8)

    assert score["p_human"] == pytest.approx(0.9)
    assert score["verdict"] == "likely_ai"
    assert score["calibration_basis"] == "vote_majority"


def test_split_passages_keeps_paragraph_order():
    from arepo.web import _split_passages

    text = "First paragraph has enough content.\n\nSecond paragraph follows it."
    passages = _split_passages(text)

    assert [p["title"] for p in passages] == ["Paragraph 1", "Paragraph 2"]
    assert passages[0]["text"].startswith("First")
    assert passages[1]["text"].startswith("Second")


def test_public_passage_payload_keeps_public_fields():
    from arepo.web import _public_passage_payload

    payload = _public_passage_payload({
        "title": "x",
        "p_ai": 0.9,
        "margin": 0.8,
    })

    assert payload == {"title": "x", "p_ai": 0.9, "margin": 0.8}


def test_moe_result_hides_internal_votes():
    from arepo.web import _classify

    experts = [{
        "group": "unit",
        "orientation": 1,
        "model_human": {"mean": np.zeros(4), "std": np.ones(4)},
        "model_ai": {"mean": np.ones(4), "std": np.ones(4)},
        "context_model": {
            "mean": np.zeros(4),
            "std": np.ones(4),
            "basis": np.eye(4, 2),
        },
    }]
    result = _classify(np.array([0.1, 0.1, 0.0, 0.0]), experts)

    assert {
        "verdict",
        "label",
        "p_human",
        "p_ai",
        "confidence",
        "margin",
        "geometric_confidence",
        "evidence_class",
    } <= set(result)
    assert "vote_human" not in result
    assert "vote_ai" not in result
    assert "experts" not in result


def test_document_evidence_label_uses_local_evidence_not_posterior_verdict():
    from arepo.web import _document_evidence_payload

    document_score = {
        "label": "Likely AI",
        "verdict": "likely_ai",
        "p_human": 0.45,
        "p_ai": 0.55,
        "geometric_confidence": 0.05,
        "posterior_margin": 0.1,
    }
    window_scores = [
        {
            "index": 1,
            "title": "80w Window 1",
            "p_human": 0.44,
            "p_ai": 0.56,
            "posterior_margin": 0.12,
            "geometric_confidence": 0.0,
            "evidence_class": "ambiguous",
            "word_count": 80,
        },
        {
            "index": 2,
            "title": "120w Window 1",
            "p_human": 0.43,
            "p_ai": 0.57,
            "posterior_margin": 0.14,
            "geometric_confidence": 0.04,
            "evidence_class": "ambiguous",
            "word_count": 120,
        },
    ]

    payload = _document_evidence_payload(document_score, [], window_scores)

    assert payload["posterior_label"] == "Likely AI"
    assert payload["document_evidence_label"] == "Ambiguous evidence"
    assert payload["label"] == "Ambiguous evidence"
    assert payload["verdict"] == "ambiguous"
    assert payload["evidence_summary"]["ambiguous_fraction"] == pytest.approx(1.0)
    assert payload["evidence_summary"]["margin_weighted_ai_evidence"] == pytest.approx(0.5)
    assert payload["evidence_summary"]["geometric_confidence_weighted_ai_evidence"] == pytest.approx(0.5)


def test_regime_is_no_op_stub():
    from arepo.regime import infer_text_regime
    from arepo.web import _public_score

    text = (
        "First, define the objective and review the relevant constraints. "
        "Second, consider the appropriate implementation steps and ensure "
        "the process has a clear outcome."
    )

    regime = infer_text_regime(text)
    score = _public_score(0.1, geometric_confidence=0.8)

    assert regime["abstain"] is False
    assert regime["regime"] == "unclassified"
    assert score["verdict"] == "likely_ai"
    assert score["p_ai"] == pytest.approx(0.9)


def test_long_general_text_keeps_score_verdict():
    from arepo.regime import infer_text_regime
    from arepo.web import _public_score

    text = (
        "The old road climbed through a stand of maple trees before it reached "
        "the ridge above town. " * 20
    )

    regime = infer_text_regime(text)
    score = _public_score(0.1, geometric_confidence=0.8)

    assert regime["abstain"] is False
    assert score["verdict"] == "likely_ai"


def test_demo_corpus_route_does_not_require_models():
    from arepo.web import create_app

    app = create_app(preload_models=False)
    client = app.test_client()

    resp = client.get("/demo-corpus")
    data = resp.get_json()

    assert resp.status_code == 200
    assert resp.headers["Cache-Control"] == "no-store"
    assert len(data["corpus"]) >= 4
    assert {item["id"] for item in data["corpus"]} >= {
        "declaration",
        "moby-dick",
        "jane-eyre",
        "llm-ambiguous",
        "llm-hard",
    }
    counts = {item["id"]: len(item["passages"]) for item in data["corpus"]}
    assert counts["declaration"] == 3
    assert counts["moby-dick"] == 4
    assert counts["jane-eyre"] == 4
    assert counts["llm-ambiguous"] == 4
    assert counts["llm-hard"] == 1


def test_load_hc3_reads_local_jsonl(tmp_path):
    from arepo.download import load_hc3

    data_dir = tmp_path / "hc3"
    data_dir.mkdir()
    long_human = "Human answer. " * 20
    long_ai = "ChatGPT answer. " * 20
    (data_dir / "all.jsonl").write_text(
        '{"question":"Q","human_answers":["%s"],"chatgpt_answers":["%s"]}\n'
        % (long_human, long_ai),
        encoding="utf-8",
    )

    human, ai = load_hc3(
        max_per_class=1,
        benchmarks_dir=tmp_path,
        auto_download=False,
    )

    assert len(human) == 1
    assert len(ai) == 1
    assert human[0].startswith("Q\n\nHuman answer.")
    assert ai[0].startswith("Q\n\nChatGPT answer.")


def test_load_hc3_skip_per_class_reads_holdout(tmp_path):
    from arepo.download import load_hc3

    data_dir = tmp_path / "hc3"
    data_dir.mkdir()
    lines = []
    for i in range(3):
        lines.append(
            '{"question":"Q%d","human_answers":["%s"],"chatgpt_answers":["%s"]}'
            % (i, f"Human {i}. " * 20, f"ChatGPT {i}. " * 20)
        )
    (data_dir / "all.jsonl").write_text("\n".join(lines), encoding="utf-8")

    human, ai = load_hc3(
        max_per_class=1,
        benchmarks_dir=tmp_path,
        auto_download=False,
        skip_per_class=2,
    )

    assert len(human) == 1
    assert len(ai) == 1
    assert human[0].startswith("Q2\n\nHuman 2.")
    assert ai[0].startswith("Q2\n\nChatGPT 2.")


def test_external_loader_label_mappings(monkeypatch):
    import builtins
    import arepo.download as download

    long_text = "Long enough text. " * 12

    def fake_load_dataset(repo, split="train", streaming=True):
        assert streaming is True
        if repo == "liamdugan/raid":
            return [
                {"model": "human", "generation": long_text + "human"},
                {"model": "gpt4", "generation": long_text + "ai"},
            ]
        if repo == "yaful/MAGE":
            return [
                {"label": 1, "text": long_text + "human"},
                {"label": 0, "text": long_text + "ai"},
            ]
        if repo == "polsci/ghostbuster-essay-cleaned":
            return [
                {"label": 0, "text": long_text + "human"},
                {"label": 3, "text": long_text + "ai"},
            ]
        raise AssertionError(repo)

    original_import = builtins.__import__

    def fake_import(module_name, globals=None, locals=None, fromlist=(), level=0):
        if module_name == "datasets" and "load_dataset" in fromlist:
            class FakeDatasets:
                load_dataset = staticmethod(fake_load_dataset)
            return FakeDatasets()
        return original_import(module_name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    for loader in (
        download.load_raid,
        download.load_mage,
        download.load_arepo_essays,
    ):
        human, ai = loader(max_per_class=1)
        assert len(human) == 1
        assert len(ai) == 1
        assert human[0].endswith("human")
        assert ai[0].endswith("ai")


def test_external_loader_skip_per_class(monkeypatch):
    import builtins
    import arepo.download as download

    long_text = "Long enough text. " * 12

    def fake_load_dataset(repo, split="train", streaming=True):
        assert repo == "liamdugan/raid"
        return [
            {"model": "human", "generation": long_text + "human0"},
            {"model": "gpt4", "generation": long_text + "ai0"},
            {"model": "human", "generation": long_text + "human1"},
            {"model": "gpt4", "generation": long_text + "ai1"},
        ]

    original_import = builtins.__import__

    def fake_import(module_name, globals=None, locals=None, fromlist=(), level=0):
        if module_name == "datasets" and "load_dataset" in fromlist:
            class FakeDatasets:
                load_dataset = staticmethod(fake_load_dataset)
            return FakeDatasets()
        return original_import(module_name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    human, ai = download.load_raid(max_per_class=1, skip_per_class=1)

    assert human[0].endswith("human1")
    assert ai[0].endswith("ai1")


def test_expanded_guardrails_include_holdout_and_kinds(monkeypatch):
    import arepo.guardrail_evaluation as guardrails

    def fake_wiki(max_per_class=2, skip_per_class=0):
        return ["Wiki human. " * 20], ["Wiki ai. " * 20]

    def fake_holdout(max_per_class=3, skip_per_class=500):
        assert skip_per_class == 500
        return ["Holdout human. " * 20], ["Holdout ai. " * 20]

    monkeypatch.setattr(guardrails, "load_wiki", fake_wiki)
    monkeypatch.setattr(guardrails, "load_hc3", fake_holdout)
    monkeypatch.setattr(guardrails, "load_raid", fake_holdout)
    monkeypatch.setattr(guardrails, "load_mage", fake_holdout)
    monkeypatch.setattr(guardrails, "load_arepo_essays", fake_holdout)

    records = guardrails.guardrail_records(suite="expanded", holdout_start=500)
    ids = {record["id"] for record in records}

    assert "declaration-passage-1" in ids
    assert "formal-human-email" in ids
    assert "ai-formal-email" in ids
    assert "hc3-holdout-human-1" in ids
    assert all("kind" in record for record in records)


def test_guardrails_can_add_word_windows(monkeypatch):
    import arepo.guardrail_evaluation as guardrails

    monkeypatch.setattr(guardrails, "load_wiki", lambda max_per_class=2, skip_per_class=0: ([], []))

    records = guardrails.guardrail_records(
        suite="basic",
        window_words=20,
        stride_words=10,
        min_window_words=10,
    )

    window_rows = [record for record in records if "window" in record.get("kind", "")]

    assert window_rows
    assert all("parent_id" in record for record in window_rows)
    assert all(record["word_count"] >= 10 for record in window_rows)


def test_metrics_at_threshold_tracks_error_rates():
    from arepo.evaluate_moe import metrics_at_threshold

    metrics = metrics_at_threshold(
        labels=[0, 0, 1, 1],
        scores=[0.2, 0.7, 0.8, 0.4],
        threshold=0.6,
    )

    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["false_positive_rate"] == pytest.approx(0.5)
    assert metrics["false_negative_rate"] == pytest.approx(0.5)


def test_summarize_scores_computes_auc_by_dataset():
    from arepo.evaluate_moe import summarize_scores

    records = [
        {"dataset": "a", "label": 0, "p_ai": 0.1},
        {"dataset": "a", "label": 0, "p_ai": 0.2},
        {"dataset": "a", "label": 1, "p_ai": 0.8},
        {"dataset": "a", "label": 1, "p_ai": 0.9},
        {"dataset": "guardrail", "label": 0, "p_ai": 0.4},
    ]

    summary = summarize_scores(records)

    assert summary["a"]["auc_roc"] == pytest.approx(1.0)
    assert summary["guardrail"]["auc_roc"] is None
    assert summary["overall"]["n"] == 5


def test_split_word_windows_overlaps():
    from arepo.document_report import split_word_windows

    text = " ".join(f"w{i}" for i in range(1, 121))
    windows = split_word_windows(text, window_words=50, stride_words=25, min_words=20)

    assert len(windows) == 4
    assert windows[0]["start_word"] == 1
    assert windows[0]["end_word"] == 50
    assert windows[1]["start_word"] == 26
    assert windows[-1]["end_word"] == 120


def test_weighted_average_uses_chars():
    from arepo.document_report import weighted_average

    avg = weighted_average([
        {"p_human": 0.0, "chars": 1},
        {"p_human": 1.0, "chars": 3},
    ])

    assert avg == pytest.approx(0.75)


def test_window_weight_summary_tracks_margin_weighted_evidence():
    from arepo.document_report import window_weight_summary

    summary = window_weight_summary([
        {
            "p_ai": 0.9,
            "verdict": "likely_ai",
            "margin": 0.8,
            "word_count": 10,
            "confidence_band": "normal",
        },
        {
            "p_ai": 0.4,
            "verdict": "likely_human",
            "margin": 0.2,
            "word_count": 10,
            "confidence_band": "normal",
        },
    ])

    assert summary["posterior_avg_ai"] == pytest.approx(0.65)
    assert summary["posterior_evidence_ai"] == pytest.approx(0.8)
    assert summary["vote_evidence_ai"] == pytest.approx(0.8)


def test_scoring_scheme_rows_flags_false_negative():
    from arepo.document_report import scoring_scheme_rows

    whole = {"verdict": "likely_human", "p_ai": 0.2}
    segments = [
        {
            "p_ai": 0.2,
            "verdict": "likely_human",
            "margin": 0.6,
            "chars": 100,
            "word_count": 10,
            "confidence_band": "normal",
        }
    ]

    rows = scoring_scheme_rows("ai", whole, segments, segments)

    assert rows[0]["scheme"] == "Whole text: MoE decision"
    assert rows[0]["status"] == "FALSE NEGATIVE"


def test_renorm_window_ranges_overlap():
    from arepo.renorm_experiments import window_ranges

    assert window_ranges(10, 64, 0.0) == [(0, 10)]
    assert window_ranges(130, 64, 0.0) == [(0, 64), (64, 128), (128, 130)]
    assert window_ranges(130, 64, 0.5) == [
        (0, 64),
        (32, 96),
        (64, 128),
        (96, 130),
    ]


def test_renorm_mode_for_model_controls_hybrid_modes():
    from arepo.core import GENERATIVE_MODELS, MASKED_MODELS
    from arepo.renorm_experiments import mode_for_model

    gen = GENERATIVE_MODELS[0]
    masked = MASKED_MODELS[0]

    assert mode_for_model("static", gen) == "static"
    assert mode_for_model("static", masked) == "static"
    assert mode_for_model("contextual", gen) == "contextual"
    assert mode_for_model("contextual", masked) == "contextual"
    assert mode_for_model("production", gen) == "static"
    assert mode_for_model("production", masked) == "contextual"
    assert mode_for_model("swapped", gen) == "contextual"
    assert mode_for_model("swapped", masked) == "static"


def test_threshold_experiment_can_invert_bad_regime():
    from arepo.threshold_experiments import (
        best_threshold,
        metrics,
        per_dataset_calibration,
        apply_calibration,
    )

    rows = [
        {"dataset": "x", "label": 0, "score": 0.8},
        {"dataset": "x", "label": 0, "score": 0.7},
        {"dataset": "x", "label": 1, "score": 0.2},
        {"dataset": "x", "label": 1, "score": 0.1},
    ]

    threshold, direct = best_threshold(rows, orientation=1)
    calibration = per_dataset_calibration(rows)
    inverted = apply_calibration(rows, calibration)

    assert threshold is not None
    assert metrics(rows, [1 if r["score"] >= threshold else 0 for r in rows])["accuracy"] < 1.0
    assert calibration["x"]["orientation"] == -1
    assert inverted["accuracy"] == pytest.approx(1.0)
