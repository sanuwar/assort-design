"""Tests for app.ml_router — MLRouter load/predict/explain using a real sklearn pipeline."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from app.ml_router import MLRouter, CLASSES


# ── Helpers ───────────────────────────────────────────────────────────────────

def _train_and_save(artifact_dir: Path) -> None:
    """Train a minimal sklearn pipeline and save artifacts to artifact_dir."""
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    texts = [
        "sales market pricing commercial revenue forecast",
        "sales market commercial product launch strategy",
        "clinical trial safety patient endpoint efficacy data",
        "patient safety adverse events clinical endpoint data",
        "experiment assay protocol method research laboratory",
        "laboratory protocol assay research method gene silencing",
    ]
    labels = [
        "commercial", "commercial",
        "medical_affairs", "medical_affairs",
        "r_and_d", "r_and_d",
    ]

    vect = TfidfVectorizer(ngram_range=(1, 2), max_features=500, min_df=1)
    clf = LogisticRegression(max_iter=200, solver="lbfgs")
    pipe = Pipeline([("vectorizer", vect), ("classifier", clf)])
    pipe.fit(texts, labels)

    joblib.dump(pipe.named_steps["vectorizer"], artifact_dir / "vectorizer.pkl")
    joblib.dump(pipe.named_steps["classifier"], artifact_dir / "classifier.pkl")

    metadata = {
        "trained_at": "2026-01-01T00:00:00+00:00",
        "n_docs": 6,
        "labels": sorted(set(labels)),
        "accuracy": 1.0,
        "per_class_metrics": {},
    }
    with (artifact_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f)


# ── load() ────────────────────────────────────────────────────────────────────

def test_load_returns_false_when_artifacts_missing(tmp_path):
    router = MLRouter()
    with patch("app.ml_router.ARTIFACTS_DIR", tmp_path):
        result = router.load()
    assert result is False
    assert router._loaded is False


def test_load_returns_true_with_valid_artifacts(tmp_path):
    _train_and_save(tmp_path)
    router = MLRouter()
    with patch("app.ml_router.ARTIFACTS_DIR", tmp_path):
        result = router.load()
    assert result is True
    assert router._loaded is True


def test_load_is_idempotent(tmp_path):
    _train_and_save(tmp_path)
    router = MLRouter()
    with patch("app.ml_router.ARTIFACTS_DIR", tmp_path):
        router.load()
        result = router.load()  # second call — should not reload
    assert result is True


def test_reload_resets_and_reloads(tmp_path):
    _train_and_save(tmp_path)
    router = MLRouter()
    with patch("app.ml_router.ARTIFACTS_DIR", tmp_path):
        router.load()
        result = router.reload()
    assert result is True


# ── predict() ─────────────────────────────────────────────────────────────────

def test_predict_raises_if_not_loaded():
    router = MLRouter()
    with pytest.raises(RuntimeError, match="not loaded"):
        router.predict("some text", threshold=0.5, margin=0.1)


def test_predict_returns_commercial(tmp_path):
    _train_and_save(tmp_path)
    router = MLRouter()
    with patch("app.ml_router.ARTIFACTS_DIR", tmp_path):
        router.load()
    result = router.predict(
        "sales market pricing commercial revenue", threshold=0.3, margin=0.05
    )
    assert result["audience"] == "commercial"
    assert result["routing_source"] == "ml"
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["candidates"], list)
    assert isinstance(result["top_signals"], list)


def test_predict_returns_medical_affairs(tmp_path):
    _train_and_save(tmp_path)
    router = MLRouter()
    with patch("app.ml_router.ARTIFACTS_DIR", tmp_path):
        router.load()
    result = router.predict(
        "clinical trial patient safety endpoint adverse", threshold=0.3, margin=0.05
    )
    assert result["audience"] == "medical_affairs"


def test_predict_returns_rnd(tmp_path):
    _train_and_save(tmp_path)
    router = MLRouter()
    with patch("app.ml_router.ARTIFACTS_DIR", tmp_path):
        router.load()
    result = router.predict(
        "experiment assay laboratory protocol method", threshold=0.3, margin=0.05
    )
    assert result["audience"] == "r_and_d"


def test_predict_cross_functional_on_low_confidence(tmp_path):
    _train_and_save(tmp_path)
    router = MLRouter()
    with patch("app.ml_router.ARTIFACTS_DIR", tmp_path):
        router.load()
    # Force low confidence by using a very high threshold
    result = router.predict("sales market pricing", threshold=0.99, margin=0.0)
    assert result["audience"] == "cross_functional"
    assert result["fallback_reason"] is not None


def test_predict_cross_functional_on_small_margin(tmp_path):
    _train_and_save(tmp_path)
    router = MLRouter()
    with patch("app.ml_router.ARTIFACTS_DIR", tmp_path):
        router.load()
    # Force ambiguous by requiring huge margin
    result = router.predict("sales market pricing", threshold=0.0, margin=0.99)
    assert result["audience"] == "cross_functional"
    assert "ambiguous" in (result["fallback_reason"] or "")


def test_predict_router_version_matches_metadata(tmp_path):
    _train_and_save(tmp_path)
    router = MLRouter()
    with patch("app.ml_router.ARTIFACTS_DIR", tmp_path):
        router.load()
    result = router.predict("sales pricing market", threshold=0.3, margin=0.05)
    assert result["router_version"] == "2026-01-01T00:00:00+00:00"


# ── explain() ─────────────────────────────────────────────────────────────────

def test_explain_returns_none_when_not_loaded():
    router = MLRouter()
    assert router.explain("some text") is None


def test_explain_returns_contributions(tmp_path):
    _train_and_save(tmp_path)
    router = MLRouter()
    with patch("app.ml_router.ARTIFACTS_DIR", tmp_path):
        router.load()
    result = router.explain("sales market pricing commercial")
    assert result is not None
    assert "predicted_class" in result
    assert "contributions" in result
    assert "all_probs" in result
    assert isinstance(result["contributions"], list)


# ── get_model_stats() ─────────────────────────────────────────────────────────

def test_get_model_stats_returns_none_when_not_loaded():
    router = MLRouter()
    assert router.get_model_stats() is None


def test_get_model_stats_returns_dict(tmp_path):
    _train_and_save(tmp_path)
    router = MLRouter()
    with patch("app.ml_router.ARTIFACTS_DIR", tmp_path):
        router.load()
        # get_model_stats re-reads ARTIFACTS_DIR for metadata — keep patch active
        stats = router.get_model_stats()
    assert stats is not None
    assert "vocab_size" in stats
    assert "top_terms_per_class" in stats
    assert stats["n_docs"] == 6
    assert stats["accuracy"] == 1.0
