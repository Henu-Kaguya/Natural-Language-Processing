from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "data" / "spam_samples.csv"
DEFAULT_OUTPUT = BASE_DIR / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab04 naive Bayes text classification baseline.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="CSV dataset with label,text columns.")
    parser.add_argument("--test-size", type=float, default=0.3, help="Test split ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducible splitting.")
    parser.add_argument("--predict-text", type=str, default="免费领取内部资料并加入股票群", help="Extra text used for a post-training prediction demo.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Directory for metrics and prediction outputs.")
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    normalized = re.sub(r"\s+", "", text)
    return [token for token in jieba.lcut(normalized) if token.strip()]


def load_dataset(dataset_path: Path) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    labels: list[int] = []
    with dataset_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            label = row["label"].strip().lower()
            texts.append(row["text"].strip())
            labels.append(1 if label == "spam" else 0)
    if len(set(labels)) < 2:
        raise ValueError("Dataset must contain at least two classes.")
    return texts, labels


def main() -> None:
    args = parse_args()
    texts, labels = load_dataset(args.dataset)

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels,
    )

    vectorizer = CountVectorizer(tokenizer=tokenize, token_pattern=None)
    train_matrix = vectorizer.fit_transform(x_train)
    test_matrix = vectorizer.transform(x_test)

    model = MultinomialNB()
    model.fit(train_matrix, y_train)
    predictions = model.predict(test_matrix)

    matrix = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=["ham", "spam"], output_dict=True, zero_division=0)
    demo_prediction = model.predict(vectorizer.transform([args.predict_text]))[0]

    print(f"Dataset: {args.dataset}")
    print(f"Samples: {len(texts)}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print("Confusion matrix:")
    print(matrix)
    print("Classification report:")
    print(classification_report(y_test, predictions, target_names=["ham", "spam"], zero_division=0))
    print(f"Demo prediction: {args.predict_text} -> {'spam' if demo_prediction == 1 else 'ham'}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "sample_count": len(texts),
                "vocabulary_size": len(vectorizer.vocabulary_),
                "confusion_matrix": matrix.tolist(),
                "classification_report": report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "demo_prediction.json").write_text(
        json.dumps(
            {
                "text": args.predict_text,
                "predicted_label": "spam" if demo_prediction == 1 else "ham",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()