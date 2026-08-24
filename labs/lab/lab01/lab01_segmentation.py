from __future__ import annotations

import argparse
from pathlib import Path
import sys

import jieba


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def load_dict(file_path: str | Path) -> tuple[set[str], int]:
    dictionary_path = Path(file_path)
    words: set[str] = set()

    with dictionary_path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            word = stripped.split()[0]
            if word:
                words.add(word)

    if not words:
        raise ValueError(f"词典为空: {dictionary_path}")

    max_word_len = max(len(word) for word in words)
    return words, max_word_len


def fmm(dictionary: set[str], max_word_len: int, sentence: str) -> list[str]:
    result: list[str] = []
    start = 0

    while start < len(sentence):
        end = min(len(sentence), start + max_word_len)
        while end > start:
            candidate = sentence[start:end]
            if candidate in dictionary or len(candidate) == 1:
                result.append(candidate)
                start = end
                break
            end -= 1

    return result


def rmm(dictionary: set[str], max_word_len: int, sentence: str) -> list[str]:
    result: list[str] = []
    end = len(sentence)

    while end > 0:
        start = max(0, end - max_word_len)
        while start < end:
            candidate = sentence[start:end]
            if candidate in dictionary or len(candidate) == 1:
                result.insert(0, candidate)
                end = start
                break
            start += 1

    return result


def bm(dictionary: set[str], max_word_len: int, sentence: str) -> list[str]:
    fmm_result = fmm(dictionary, max_word_len, sentence)
    rmm_result = rmm(dictionary, max_word_len, sentence)

    if len(fmm_result) != len(rmm_result):
        return fmm_result if len(fmm_result) < len(rmm_result) else rmm_result

    if fmm_result == rmm_result:
        return fmm_result

    fmm_single_count = sum(1 for word in fmm_result if len(word) == 1)
    rmm_single_count = sum(1 for word in rmm_result if len(word) == 1)
    return fmm_result if fmm_single_count < rmm_single_count else rmm_result


def format_result(words: list[str]) -> str:
    return " / ".join(words)


def get_default_dict_path() -> Path:
    return Path(jieba.__file__).resolve().parent / "dict.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验一：基于词表的三种中文分词算法")
    parser.add_argument("--dict", dest="dict_path", type=Path, default=get_default_dict_path(), help="词典文件路径")
    parser.add_argument("--text", dest="text", help="待分词文本；未提供时进入交互输入")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dictionary, max_word_len = load_dict(args.dict_path)
    text = args.text if args.text is not None else input("请输入需要分词的段落：")
    sentence = text.strip()

    if not sentence:
        print("输入内容为空，请重新运行程序后输入待分词文本。")
        return

    fmm_result = fmm(dictionary, max_word_len, sentence)
    rmm_result = rmm(dictionary, max_word_len, sentence)
    bm_result = bm(dictionary, max_word_len, sentence)

    print(f"词典路径: {Path(args.dict_path).resolve()}")
    print(f"词典词条数: {len(dictionary)}")
    print(f"词典最大词长: {max_word_len}")
    print(f"待分词文本: {sentence}")
    print(f"FMM 分词结果: {format_result(fmm_result)}")
    print(f"RMM 分词结果: {format_result(rmm_result)}")
    print(f"BM 分词结果: {format_result(bm_result)}")


if __name__ == "__main__":
    main()
