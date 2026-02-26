"""
CLI inspection for the ML audience router artifacts.

Usage:
    python -m app.inspect_router                     # print model stats + top terms
    python -m app.inspect_router --text "some text"  # explain routing for text
    python -m app.inspect_router --top-n 20          # control number of terms
    python -m app.inspect_router --verbose           # show threshold/margin info
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the ML audience router.")
    parser.add_argument(
        "--text", type=str, default=None,
        help="Text to explain routing for.",
    )
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="Number of top terms to show (default: 10).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show threshold, margin, and all class probabilities.",
    )
    args = parser.parse_args()

    from app.ml_router import _ml_router

    if not _ml_router.load():
        print("ERROR: Could not load ML router artifacts from artifacts/.", file=sys.stderr)
        print("Train first: python -m app.train_router", file=sys.stderr)
        sys.exit(1)

    stats = _ml_router.get_model_stats(top_n=args.top_n)
    if stats:
        print(f"Model trained: {stats['trained_at']}")
        print(f"Training docs: {stats['n_docs']}")
        acc = stats["accuracy"]
        print(f"Accuracy:      {acc * 100:.1f}%" if acc is not None else "Accuracy:      N/A")
        print(f"Labels:        {', '.join(stats['labels'])}")
        print(f"Vocab size:    {stats['vocab_size']}")
        print()

        for label in stats["labels"]:
            terms = stats["top_terms_per_class"].get(label, [])
            if not terms:
                continue
            print(f"── {label} ──")
            max_w = max(t["weight"] for t in terms) if terms else 1
            for t in terms:
                bar_len = int(t["weight"] / max_w * 30) if max_w > 0 else 0
                bar = "\u2588" * bar_len
                print(f"  {t['term']:>25s}  {t['weight']:+.4f}  {bar}")
            print()

    if args.text:
        explanation = _ml_router.explain(args.text, top_n=args.top_n)
        if not explanation:
            print("ERROR: explain() returned None.", file=sys.stderr)
            sys.exit(1)

        print(f"Predicted: {explanation['predicted_class']} "
              f"({explanation['predicted_prob']:.1%})")
        print(f"Runner-up: {explanation['runner_up_class']} "
              f"({explanation['runner_up_prob']:.1%})")

        if args.verbose:
            from app.config import get_ml_router_config

            ml_cfg = get_ml_router_config()
            threshold = ml_cfg["ml_router_threshold"]
            margin_cfg = ml_cfg["ml_router_margin"]
            actual_margin = explanation["predicted_prob"] - explanation["runner_up_prob"]
            print(f"Threshold: {threshold}")
            print(f"Margin:    {margin_cfg}")
            print(f"Actual margin: {actual_margin:.4f}")
            print(f"All probs: {explanation['all_probs']}")

        print(f"\nTop terms for {explanation['predicted_class']}:")
        print(f"  {'Term':>25s}  {'TF-IDF':>8s}  {'Weight':>8s}  {'Contrib':>8s}")
        for c in explanation["contributions"]:
            print(f"  {c['term']:>25s}  {c['tfidf']:8.4f}  {c['weight']:8.4f}  {c['contribution']:8.4f}")

        print(f"\nTop terms for {explanation['runner_up_class']} (runner-up):")
        print(f"  {'Term':>25s}  {'TF-IDF':>8s}  {'Weight':>8s}  {'Contrib':>8s}")
        for c in explanation["runner_up_contributions"]:
            print(f"  {c['term']:>25s}  {c['tfidf']:8.4f}  {c['weight']:8.4f}  {c['contribution']:8.4f}")


if __name__ == "__main__":
    main()
