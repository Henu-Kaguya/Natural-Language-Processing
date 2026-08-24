from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


try:
    import paddle
    import paddle.nn as nn
    import paddle.nn.functional as F
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: paddlepaddle. Install it with 'pip install -r lab03/requirements.txt' first."
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS_PATH = BASE_DIR / "data" / "skipgram_corpus.txt"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"
DEFAULT_SIMILARITY_PAIRS = [
    ("king", "queen"),
    ("she", "her"),
    ("topic", "theme"),
    ("woman", "game"),
    ("one", "name"),
]
SEED = 20260429


@dataclass
class TrainConfig:
    corpus_path: Path
    output_dir: Path
    embedding_dim: int = 32
    window_size: int = 2
    epochs: int = 160
    batch_size: int = 16
    learning_rate: float = 0.03
    min_count: int = 1
    validation_ratio: float = 0.1
    log_every: int = 20
    device: str = "auto"


class SkipGram(nn.Layer):
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        # self.embedding stores the dense vector for each word id and is the core
        # representation that we later reuse for cosine-similarity queries.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, center_words: paddle.Tensor) -> paddle.Tensor:
        hidden = self.embedding(center_words)
        return self.output(hidden)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a PaddlePaddle Skip-gram model for lab03.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH, help="Path to the training corpus.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for saved outputs.")
    parser.add_argument("--embedding-dim", type=int, default=32, help="Embedding vector size.")
    parser.add_argument("--window-size", type=int, default=2, help="Context window size for Skip-gram pairs.")
    parser.add_argument("--epochs", type=int, default=160, help="Training epochs for the default CPU run.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size used for training.")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="Optimizer learning rate.")
    parser.add_argument("--min-count", type=int, default=1, help="Minimum token frequency kept in the vocabulary.")
    parser.add_argument("--validation-ratio", type=float, default=0.1, help="Hold-out ratio for validation loss.")
    parser.add_argument("--log-every", type=int, default=20, help="Print loss every N epochs.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Use cpu by default, or request gpu when Paddle is built with CUDA.",
    )
    args = parser.parse_args()
    return TrainConfig(
        corpus_path=args.corpus,
        output_dir=args.output_dir,
        embedding_dim=args.embedding_dim,
        window_size=args.window_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        min_count=args.min_count,
        validation_ratio=args.validation_ratio,
        log_every=args.log_every,
        device=args.device,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    paddle.seed(seed)


def choose_device(requested: str) -> str:
    if requested == "cpu":
        paddle.set_device("cpu")
        return "cpu"
    if requested == "gpu":
        if not paddle.is_compiled_with_cuda():
            raise SystemExit("GPU was requested, but the installed Paddle build does not support CUDA.")
        paddle.set_device("gpu")
        return "gpu"
    if paddle.is_compiled_with_cuda():
        paddle.set_device("gpu")
        return "gpu"
    paddle.set_device("cpu")
    return "cpu"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def load_tokens(corpus_path: Path) -> list[str]:
    text = corpus_path.read_text(encoding="utf-8")
    tokens = tokenize(text)
    if not tokens:
        raise ValueError(f"No valid tokens were found in {corpus_path}.")
    return tokens


def build_vocab(tokens: Sequence[str], min_count: int) -> tuple[list[str], dict[str, int], list[int], Counter[str]]:
    counter = Counter(tokens)
    vocab = [token for token, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])) if count >= min_count]
    if not vocab:
        raise ValueError("The filtered vocabulary is empty. Reduce --min-count or use a richer corpus.")
    word_to_id = {token: index for index, token in enumerate(vocab)}
    token_ids = [word_to_id[token] for token in tokens if token in word_to_id]
    return vocab, word_to_id, token_ids, counter


def build_data(token_ids: Sequence[int], window_size: int) -> list[tuple[int, int]]:
    # build_data converts the full token-id stream into supervised Skip-gram
    # training pairs shaped as (center_word_id, context_word_id).
    training_pairs: list[tuple[int, int]] = []
    for center_index, center_word_id in enumerate(token_ids):
        left = max(0, center_index - window_size)
        right = min(len(token_ids), center_index + window_size + 1)
        for context_index in range(left, right):
            if context_index == center_index:
                continue
            training_pairs.append((center_word_id, token_ids[context_index]))
    if not training_pairs:
        raise ValueError("No Skip-gram pairs were created. Check the corpus length and window size.")
    return training_pairs


def split_pairs(pairs: Sequence[tuple[int, int]], validation_ratio: float) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    shuffled = list(pairs)
    random.shuffle(shuffled)
    validation_size = max(1, int(len(shuffled) * validation_ratio)) if len(shuffled) >= 10 else 0
    if validation_size == 0:
        return shuffled, []
    return shuffled[validation_size:], shuffled[:validation_size]


def iterate_batches(pairs: Sequence[tuple[int, int]], batch_size: int) -> Iterable[tuple[paddle.Tensor, paddle.Tensor]]:
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        centers = paddle.to_tensor([item[0] for item in batch], dtype="int64")
        contexts = paddle.to_tensor([item[1] for item in batch], dtype="int64")
        yield centers, contexts


def evaluate_loss(model: SkipGram, pairs: Sequence[tuple[int, int]], batch_size: int) -> float | None:
    if not pairs:
        return None
    model.eval()
    losses: list[float] = []
    with paddle.no_grad():
        for centers, contexts in iterate_batches(pairs, batch_size):
            logits = model(centers)
            loss = F.cross_entropy(logits, contexts)
            losses.append(float(loss.numpy().item()))
    model.train()
    return sum(losses) / len(losses)


def train_model(
    model: SkipGram,
    train_pairs: Sequence[tuple[int, int]],
    validation_pairs: Sequence[tuple[int, int]],
    config: TrainConfig,
) -> list[dict[str, float | int | None]]:
    optimizer = paddle.optimizer.Adam(learning_rate=config.learning_rate, parameters=model.parameters())
    history: list[dict[str, float | int | None]] = []

    for epoch in range(1, config.epochs + 1):
        shuffled = list(train_pairs)
        random.shuffle(shuffled)
        batch_losses: list[float] = []

        for centers, contexts in iterate_batches(shuffled, config.batch_size):
            logits = model(centers)
            loss = F.cross_entropy(logits, contexts)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            batch_losses.append(float(loss.numpy().item()))

        train_loss = sum(batch_losses) / len(batch_losses)
        validation_loss = evaluate_loss(model, validation_pairs, config.batch_size)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        }
        history.append(epoch_record)

        if epoch == 1 or epoch % config.log_every == 0 or epoch == config.epochs:
            validation_text = f", val_loss={validation_loss:.4f}" if validation_loss is not None else ""
            print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f}{validation_text}")

    return history


def get_embedding_matrix(model: SkipGram) -> paddle.Tensor:
    return model.embedding.weight.detach()


def cosine_similarity(embedding_matrix: paddle.Tensor, word_to_id: dict[str, int], first: str, second: str) -> float:
    if first not in word_to_id or second not in word_to_id:
        raise KeyError(f"Cannot compute similarity because '{first}' or '{second}' is missing from the vocabulary.")
    first_vector = embedding_matrix[word_to_id[first]]
    second_vector = embedding_matrix[word_to_id[second]]
    numerator = float(paddle.dot(first_vector, second_vector).numpy().item())
    denominator = math.sqrt(float(paddle.dot(first_vector, first_vector).numpy().item())) * math.sqrt(
        float(paddle.dot(second_vector, second_vector).numpy().item())
    )
    return numerator / denominator if denominator else 0.0


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    config = parse_args()
    device = choose_device(config.device)
    set_seed(SEED)

    tokens = load_tokens(config.corpus_path)
    vocab, word_to_id, token_ids, token_counter = build_vocab(tokens, config.min_count)
    train_pairs = build_data(token_ids, config.window_size)
    actual_train_pairs, validation_pairs = split_pairs(train_pairs, config.validation_ratio)

    print(f"Using device: {device}")
    print(f"Loaded {len(tokens)} tokens from {config.corpus_path}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Skip-gram pairs: train={len(actual_train_pairs)}, validation={len(validation_pairs)}")

    model = SkipGram(vocab_size=len(vocab), embedding_dim=config.embedding_dim)
    history = train_model(model, actual_train_pairs, validation_pairs, config)
    embedding_matrix = get_embedding_matrix(model)

    similarity_results: list[dict[str, str | float]] = []
    for first, second in DEFAULT_SIMILARITY_PAIRS:
        similarity = cosine_similarity(embedding_matrix, word_to_id, first, second)
        similarity_results.append({"word1": first, "word2": second, "cosine": similarity})
        print(f"word1={first:<6} word2={second:<6} cosine={similarity:.6f}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config.output_dir / "run_config.json", {**asdict(config), "corpus_path": str(config.corpus_path), "output_dir": str(config.output_dir), "device": device})
    save_json(config.output_dir / "vocab.json", {"vocab": vocab, "token_frequency": dict(token_counter)})
    save_json(config.output_dir / "training_metrics.json", history)
    save_json(config.output_dir / "similarity_results.json", similarity_results)


if __name__ == "__main__":
    main()