"""
CLI + programmatic training for the TF-IDF + Logistic Regression audience router.

CLI usage:
    python -m app.train_router [--min-samples N] [--output-dir PATH]

Programmatic usage (from inside the running app):
    from app.train_router import train_model
    result = train_model(min_samples=10)   # returns dict or None
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Ensure project root is on path when run as __main__
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "artifacts"


def train_model(
    min_samples: int = 10,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Train the ML router from labelled jobs in the DB.

    Returns a metadata dict on success, or None if there aren't enough samples.
    Raises on unexpected errors (DB, sklearn, IO).
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data from DB ────────────────────────────────────────────────
    from app.db import engine
    from sqlalchemy import text

    VALID_LABELS = {"commercial", "medical_affairs", "r_and_d"}

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT d.content, j.audience
                FROM job j
                JOIN document d ON d.id = j.document_id
                WHERE j.status = 'completed'
                  AND j.audience IN ('commercial', 'medical_affairs', 'r_and_d')
                ORDER BY j.created_at DESC
                """
            )
        ).fetchall()

    texts = []
    labels = []
    for content, audience in rows:
        if content and content.strip() and audience in VALID_LABELS:
            texts.append(content.strip())
            labels.append(audience)

    n = len(texts)
    if verbose:
        print(f"Found {n} labelled examples.")

    if n < min_samples:
        msg = f"Need at least {min_samples} samples to train (got {n})."
        if verbose:
            print(f"ERROR: {msg}", file=sys.stderr)
        logger.info("train_model skipped: %s", msg)
        return None

    # ── 2. Build pipeline ───────────────────────────────────────────────────
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        (
            "vectorizer",
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=30000,
                min_df=2,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
            ),
        ),
    ])

    # ── 3. Train / evaluate ─────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )

    unique_labels = sorted(set(labels))
    stratify = labels if all(labels.count(l) >= 2 for l in unique_labels) else None

    if n >= 5:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=stratify
        )
    else:
        X_train, y_train = texts, labels
        X_test, y_test = [], []

    pipeline.fit(X_train, y_train)

    accuracy = None
    per_class: dict = {}

    if X_test:
        y_pred = pipeline.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))
        if verbose:
            print(f"\nAccuracy: {accuracy * 100:.1f}%")
            print("\nClassification report:")
            print(classification_report(y_test, y_pred, labels=unique_labels))
            print("Confusion matrix (rows=true, cols=pred):")
            cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
            print("Labels:", unique_labels)
            print(cm)

        from sklearn.metrics import precision_recall_fscore_support
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=unique_labels, average=None, zero_division=0
        )
        for i, lbl in enumerate(unique_labels):
            per_class[lbl] = {
                "precision": round(float(prec[i]), 3),
                "recall": round(float(rec[i]), 3),
                "f1": round(float(f1[i]), 3),
            }
    elif verbose:
        print("(No test split — trained on all data, evaluation skipped.)")

    # ── 4. Save artifacts ───────────────────────────────────────────────────
    import joblib

    vectorizer = pipeline.named_steps["vectorizer"]
    classifier = pipeline.named_steps["classifier"]

    joblib.dump(vectorizer, output_dir / "vectorizer.pkl")
    joblib.dump(classifier, output_dir / "classifier.pkl")

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_docs": n,
        "labels": unique_labels,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "per_class_metrics": per_class,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(
            f"\nTrained on {n} docs."
            + (f" Accuracy: {accuracy * 100:.1f}%." if accuracy is not None else "")
            + f" Saved to {output_dir}/"
        )

    logger.info(
        "train_model complete: %d docs, accuracy=%s, saved to %s",
        n,
        f"{accuracy:.1%}" if accuracy is not None else "N/A",
        output_dir,
    )
    return metadata


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train the ML audience router.")
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum number of labelled examples required to train (default: 10).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save artifacts (default: artifacts/).",
    )
    args = parser.parse_args()

    result = train_model(
        min_samples=args.min_samples,
        output_dir=Path(args.output_dir),
        verbose=True,
    )
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
