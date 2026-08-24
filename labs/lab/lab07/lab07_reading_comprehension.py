from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "data" / "sample_qa.json"
DEFAULT_OUTPUT = BASE_DIR / "outputs"
SEED = 20260429


@dataclass
class Answer:
    text: str
    answer_start: int


@dataclass
class QaSample:
    sample_id: str
    context: str
    question: str
    answers: list[Answer]


class PretrainedQaRoute:
    def describe(self) -> dict[str, str]:
        return {
            "status": "placeholder",
            "message": "Future extension point: replace heuristic extraction with a pretrained extractive QA model when dependencies and datasets are available.",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab07 extractive QA starter workflow.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to local QA JSON dataset.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Directory for metrics and sample predictions.")
    parser.add_argument("--random-seed", type=int, default=SEED, help="Random seed for deterministic split.")
    return parser.parse_args()


def load_dataset(path: Path) -> list[QaSample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples: list[QaSample] = []
    for raw_sample in payload["samples"]:
        answers = [Answer(text=item["text"], answer_start=int(item["answer_start"])) for item in raw_sample["answers"]]
        sample = QaSample(
            sample_id=raw_sample["id"],
            context=raw_sample["context"],
            question=raw_sample["question"],
            answers=answers,
        )
        validate_sample(sample)
        samples.append(sample)
    if len(samples) < 4:
        raise ValueError("Dataset must contain at least four samples for starter validation.")
    return samples


def validate_sample(sample: QaSample) -> None:
    if not sample.answers:
        raise ValueError(f"Sample {sample.sample_id} must contain at least one answer.")
    for answer in sample.answers:
        start = answer.answer_start
        end = start + len(answer.text)
        if start < 0 or end > len(sample.context):
            raise ValueError(f"Sample {sample.sample_id}: answer span is out of context bounds.")
        if sample.context[start:end] != answer.text:
            raise ValueError(f"Sample {sample.sample_id}: answer text does not match context span.")


def split_samples(samples: list[QaSample], seed: int) -> tuple[list[QaSample], list[QaSample]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    validation_size = max(2, len(shuffled) // 3)
    return shuffled[validation_size:], shuffled[:validation_size]


def question_keywords(question: str) -> list[str]:
    stop_chars = set("？?，。；：！的了在是吗哪多少什么哪里谁如何")
    return [char for char in question if char not in stop_chars and char.strip()]


def heuristic_extract_answer(context: str, question: str) -> str:
    keywords = question_keywords(question)
    if not keywords:
        return context[: min(8, len(context))]

    best_start = 0
    best_end = min(8, len(context))
    best_score = -1

    for start in range(len(context)):
        for end in range(start + 1, min(len(context), start + 14) + 1):
            span = context[start:end]
            score = sum(1 for key in keywords if key in span)
            score += min(len(span), 10) * 0.01
            if score > best_score:
                best_score = score
                best_start = start
                best_end = end

    candidate = context[best_start:best_end].strip("，。；： ")
    return candidate if candidate else context[best_start:best_end]


def exact_match(prediction: str, gold_answer: str) -> float:
    return 1.0 if prediction == gold_answer else 0.0


def f1_char(prediction: str, gold_answer: str) -> float:
    pred_chars = list(prediction)
    gold_chars = list(gold_answer)
    if not pred_chars or not gold_chars:
        return 0.0
    overlap = 0
    remaining = gold_chars.copy()
    for char in pred_chars:
        if char in remaining:
            overlap += 1
            remaining.remove(char)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_chars)
    recall = overlap / len(gold_chars)
    return 2 * precision * recall / (precision + recall)


def evaluate(samples: list[QaSample]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    em_scores: list[float] = []
    f1_scores: list[float] = []
    predictions: list[dict[str, Any]] = []

    for sample in samples:
        prediction = heuristic_extract_answer(sample.context, sample.question)
        gold = sample.answers[0].text
        em = exact_match(prediction, gold)
        f1 = f1_char(prediction, gold)
        em_scores.append(em)
        f1_scores.append(f1)
        predictions.append(
            {
                "id": sample.sample_id,
                "question": sample.question,
                "context": sample.context,
                "gold_answer": gold,
                "predicted_answer": prediction,
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
    placeholder = PretrainedQaRoute().describe()

    held_out_metrics, held_out_predictions = evaluate(validation_samples)
    replay_metrics, replay_predictions = evaluate(samples)

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
