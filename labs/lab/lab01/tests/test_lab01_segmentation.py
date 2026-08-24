from pathlib import Path
import subprocess
import sys

import jieba
import pytest


def test_load_dict_extracts_words_and_max_length(tmp_path: Path):
    dict_file = tmp_path / "mini_dict.txt"
    dict_file.write_text("研究生 10 n\n研究 8 v\n生命 6 n\n起源 5 n\n", encoding="utf-8")

    from lab01_segmentation import load_dict

    dictionary, max_word_len = load_dict(dict_file)

    assert dictionary == {"研究生", "研究", "生命", "起源"}
    assert max_word_len == 3


def test_fmm_rmm_bm_accept_explicit_dictionary_and_max_length():
    from lab01_segmentation import bm, fmm, rmm

    dictionary = {"研究生", "研究", "生命", "命", "起源"}
    max_word_len = 3
    sentence = "研究生命起源"

    assert fmm(dictionary, max_word_len, sentence) == ["研究生", "命", "起源"]
    assert rmm(dictionary, max_word_len, sentence) == ["研究", "生命", "起源"]
    assert bm(dictionary, max_word_len, sentence) == ["研究", "生命", "起源"]


def test_default_jieba_dictionary_can_be_located():
    from lab01_segmentation import get_default_dict_path

    dict_path = get_default_dict_path()

    assert dict_path.exists()
    assert dict_path.name == "dict.txt"
    assert Path(jieba.__file__).resolve().parent == dict_path.parent


def test_cli_prints_expected_sections(tmp_path: Path):
    dict_file = tmp_path / "mini_dict.txt"
    dict_file.write_text("研究生 10 n\n研究 8 v\n生命 6 n\n起源 5 n\n命 1 n\n", encoding="utf-8")

    command = [
        sys.executable,
        "lab01_segmentation.py",
        "--dict",
        str(dict_file),
        "--text",
        "研究生命起源",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=True)

    assert "FMM 分词结果" in completed.stdout
    assert "RMM 分词结果" in completed.stdout
    assert "BM 分词结果" in completed.stdout
    assert "研究生 / 命 / 起源" in completed.stdout
    assert "研究 / 生命 / 起源" in completed.stdout
