from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
CLASSES = ["commercial", "medical_affairs", "r_and_d"]


class MLRouter:
    """TF-IDF + Logistic Regression audience router loaded lazily from artifacts/."""
    # load model once then reuse it
    
    def __init__(self) -> None:
        self._pipeline = None  # sklearn Pipeline (vectorizer + classifier)
        self._version: Optional[str] = None 
        self._loaded: bool = False

    def reload(self) -> bool:
        """Drop cached model and re-load from disk.  Returns True on success."""
        self._pipeline = None
        self._version = None
        self._loaded = False
        return self.load()

    def load(self) -> bool:
        """Attempt to load artifacts. Returns True on success, False if missing/corrupt."""
        if self._loaded:
            return True

        vectorizer_path = ARTIFACTS_DIR / "vectorizer.pkl"
        classifier_path = ARTIFACTS_DIR / "classifier.pkl"
        metadata_path = ARTIFACTS_DIR / "metadata.json"

        if not (vectorizer_path.exists() and classifier_path.exists() and metadata_path.exists()):
            return False

        try:
            import joblib

            vectorizer = joblib.load(vectorizer_path)
            classifier = joblib.load(classifier_path)

            # Reconstruct pipeline manually so we can access components.
            from sklearn.pipeline import Pipeline

            self._pipeline = Pipeline([
                ("vectorizer", vectorizer),
                ("classifier", classifier),
            ])

            with metadata_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            self._version = meta.get("trained_at", "unknown")
            self._loaded = True
            return True
        except Exception as exc:
            logger.warning("MLRouter: failed to load artifacts — %s", exc)
            self._loaded = False
            self._pipeline = None
            return False

    def predict(self, text: str, threshold: float, margin: float) -> Dict:
        """
        Classify text and apply decision rules.

        Returns a dict with keys:
          audience, confidence, candidates, top_signals, routing_source, router_version,
          fallback_reason
        """
        if not self._loaded or self._pipeline is None:
            raise RuntimeError("MLRouter: model not loaded — call load() first.")

        vectorizer = self._pipeline.named_steps["vectorizer"]
        classifier = self._pipeline.named_steps["classifier"]

        X = vectorizer.transform([text])
        proba = classifier.predict_proba(X)[0]  # shape (3,)

        # Map probabilities to class names (classifier.classes_ order)
        cls_order: List[str] = list(classifier.classes_)
        prob_map = {cls: float(proba[i]) for i, cls in enumerate(cls_order)}

        sorted_classes = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)
        p1_label, p1 = sorted_classes[0]
        _, p2 = sorted_classes[1]

        candidates = [c for c, _ in sorted_classes[:3]]

        # Decision rules
        fallback_reason: Optional[str] = None
        if p1 < threshold:
            audience = "cross_functional"
            fallback_reason = f"low confidence ({int(p1 * 100)}% < {int(threshold * 100)}%)"
        elif (p1 - p2) < margin:
            audience = "cross_functional"
            fallback_reason = (
                f"ambiguous ({p1_label} {int(p1 * 100)}% vs "
                f"{sorted_classes[1][0]} {int(p2 * 100)}%)"
            )
        else:
            audience = p1_label

        # Top TF-IDF signals for predicted primary class
        class_idx = cls_order.index(p1_label)
        signals = self._top_features(vectorizer, classifier, class_idx, n=5)

        return {
            "audience": audience,
            "confidence": round(p1, 4),
            "candidates": candidates,
            "top_signals": signals,
            "routing_source": "ml",
            "router_version": self._version,
            "fallback_reason": fallback_reason,
        }

    def _top_features(self, vectorizer, classifier, class_idx: int, n: int = 5) -> List[str]:
        """Return the top n TF-IDF feature names for a given class index."""
        try:
            coef = classifier.coef_[class_idx]
            feature_names = vectorizer.get_feature_names_out()
            top_indices = coef.argsort()[::-1][:n]
            return [str(feature_names[i]) for i in top_indices]
        except Exception:
            return []

    def get_model_stats(self, top_n: int = 15) -> Optional[Dict]:
        """Return model-level stats for the insights page."""
        if not self._loaded or self._pipeline is None:
            return None

        vectorizer = self._pipeline.named_steps["vectorizer"]
        classifier = self._pipeline.named_steps["classifier"]
        feature_names = vectorizer.get_feature_names_out()

        # Metadata from JSON
        metadata_path = ARTIFACTS_DIR / "metadata.json"
        meta: Dict = {}
        if metadata_path.exists():
            try:
                with metadata_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                pass

        # Top IDF terms
        top_idf_terms: List[str] = []
        try:
            import numpy as np

            idf = vectorizer.idf_
            top_idf_idx = np.argsort(idf)[::-1][:20]
            top_idf_terms = [str(feature_names[i]) for i in top_idf_idx]
        except Exception:
            pass

        # Top positive-weight terms per class
        cls_order: List[str] = list(classifier.classes_)
        top_terms_per_class: Dict[str, List[Dict]] = {}
        try:
            for i, label in enumerate(cls_order):
                coef = classifier.coef_[i]
                top_idx = coef.argsort()[::-1][:top_n]
                top_terms_per_class[label] = [
                    {"term": str(feature_names[j]), "weight": round(float(coef[j]), 4)}
                    for j in top_idx
                    if coef[j] > 0
                ]
        except Exception:
            pass

        return {
            "trained_at": meta.get("trained_at", "unknown"),
            "n_docs": meta.get("n_docs", 0),
            "accuracy": meta.get("accuracy"),
            "labels": meta.get("labels", cls_order),
            "per_class_metrics": meta.get("per_class_metrics", {}),
            "vocab_size": len(feature_names),
            "top_idf_terms": top_idf_terms,
            "top_terms_per_class": top_terms_per_class,
        }

    def explain(self, text: str, top_n: int = 10) -> Optional[Dict]:
        """Explain routing for a single document — per-term contributions."""
        if not self._loaded or self._pipeline is None:
            return None

        vectorizer = self._pipeline.named_steps["vectorizer"]
        classifier = self._pipeline.named_steps["classifier"]

        X = vectorizer.transform([text])
        proba = classifier.predict_proba(X)[0]
        feature_names = vectorizer.get_feature_names_out()

        cls_order: List[str] = list(classifier.classes_)
        prob_map = {cls: round(float(proba[i]), 4) for i, cls in enumerate(cls_order)}

        sorted_classes = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)
        predicted_class, predicted_prob = sorted_classes[0]
        runner_up_class, runner_up_prob = sorted_classes[1]

        def _term_contributions(class_idx: int) -> List[Dict]:
            coef = classifier.coef_[class_idx]
            nz_indices = X[0].nonzero()[1]
            contributions = []
            for j in nz_indices:
                tfidf_val = float(X[0, j])
                weight = float(coef[j])
                contrib = tfidf_val * weight
                contributions.append({
                    "term": str(feature_names[j]),
                    "tfidf": round(tfidf_val, 4),
                    "weight": round(weight, 4),
                    "contribution": round(contrib, 4),
                })
            contributions.sort(key=lambda c: c["contribution"], reverse=True)
            return contributions[:top_n]

        predicted_idx = cls_order.index(predicted_class)
        runner_up_idx = cls_order.index(runner_up_class)

        return {
            "predicted_class": predicted_class,
            "predicted_prob": predicted_prob,
            "runner_up_class": runner_up_class,
            "runner_up_prob": runner_up_prob,
            "all_probs": prob_map,
            "contributions": _term_contributions(predicted_idx),
            "runner_up_contributions": _term_contributions(runner_up_idx),
        }


# Module-level singleton — loaded lazily on first use.
_ml_router = MLRouter()
