from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "data" / "sample_title_data.json"
DEFAULT_OUTPUT = BASE_DIR / "outputs"
SEED = 20260429


@dataclass
class TitleSample:
    sample_id: str
    article: str
    title: str


class PretrainedTitleRoute:
    def describe(self) -> dict[str, str]:
        return {
            "status": "placeholder",
            "message": "Future extension point: replace heuristic title generation with a pretrained title generation model when datasets and dependencies are available.",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab08 title generation starter workflow.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to local article-title dataset.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Directory for metrics and prediction outputs.")
    parser.add_argument("--random-seed", type=int, default=SEED, help="Random seed for deterministic split.")
    return parser.parse_args()


def load_dataset(path: Path) -> list[TitleSample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples: list[TitleSample] = []
    for raw_sample in payload["samples"]:
        sample = TitleSample(sample_id=raw_sample["id"], article=raw_sample["article"], title=raw_sample["title"])
        validate_sample(sample)
        samples.append(sample)
    if len(samples) < 4:
        raise ValueError("Dataset must contain at least four samples for starter validation.")
    return samples


def validate_sample(sample: TitleSample) -> None:
    if not sample.article.strip():
        raise ValueError(f"Sample {sample.sample_id}: article is empty.")
    if not sample.title.strip():
        raise ValueError(f"Sample {sample.sample_id}: title is empty.")


def split_samples(samples: list[TitleSample], seed: int) -> tuple[list[TitleSample], list[TitleSample]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    validation_size = max(2, len(shuffled) // 3)
    return shuffled[validation_size:], shuffled[:validation_size]


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。；：！？,.!?]", "", text)
    return text


def generate_title(article: str) -> str:
    first_clause = re.split(r"[，。；：！？]", article)[0].strip()
    compact = normalize_text(first_clause)
    if len(compact) <= 14:
        return compact
    return compact[:14]


def char_f1(prediction: str, gold: str) -> float:
    pred_chars = list(prediction)
    gold_chars = list(gold)
    if not pred_chars or not gold_chars:
        return 0.0
    overlap = 0
    gold_pool = gold_chars.copy()
    for char in pred_chars:
        if char in gold_pool:
            overlap += 1
            gold_pool.remove(char)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_chars)
    recall = overlap / len(gold_chars)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if prediction == gold else 0.0


def evaluate(samples: list[TitleSample]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    em_scores: list[float] = []
    f1_scores: list[float] = []
    predictions: list[dict[str, Any]] = []

    for sample in samples:
        predicted = generate_title(sample.article)
        em = exact_match(predicted, sample.title)
        f1 = char_f1(predicted, sample.title)
        em_scores.append(em)
        f1_scores.append(f1)
        predictions.append(
            {
                "id": sample.sample_id,
                "article": sample.article,
                "gold_title": sample.title,
                "predicted_title": predicted,
                "exact_match": em,
                "f1": f1,
            }
        )

    return {
        "exact_match": sum(em_scores) / len(em_scores),
        "f1": sum(f1_scores) / len(f1_scores),
    }, predictions


def main() -> None:
    args = parse_args()
    samples = load_dataset(args.dataset)
    train_samples, validation_samples = split_samples(samples, args.random_seed)

    held_out_metrics, held_out_predictions = evaluate(validation_samples)
    replay_metrics, replay_predictions = evaluate(samples)
    placeholder = PretrainedTitleRoute().describe()

    summary = {
        "dataset": str(args.dataset),
        "train_sample_count": len(train_samples),
        "validation_sample_count": len(validation_samples),
        "placeholder_route": placeholder,
        "held_out_metrics": held_out_metrics,
        "starter_replay_metrics": replay_metrics,
    }

    print(f"Dataset: {args.dataset}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(validation_samples)}")
    print(f"Held-out EM: {held_out_metrics['exact_match']:.4f}")
    print(f"Held-out F1: {held_out_metrics['f1']:.4f}")
    print(f"Starter replay EM: {replay_metrics['exact_match']:.4f}")
    print(f"Starter replay F1: {replay_metrics['f1']:.4f}")
    if replay_predictions:
        print("Sample prediction:")
        print(replay_predictions[0])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "sample_predictions.json").write_text(
        json.dumps(
            {
                "held_out_predictions": held_out_predictions,
                "starter_replay_predictions": replay_predictions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
