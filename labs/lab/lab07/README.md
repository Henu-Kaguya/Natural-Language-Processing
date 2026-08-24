# Lab07 Reading Comprehension Starter

本实验提供一个本地可运行的抽取式阅读理解 starter。目标是先固定 SQuAD 风格输入输出契约，打通从上下文和问题到答案输出的最小流程，并保留预训练模型扩展接口。

## 目录结构

- `lab07_reading_comprehension.py`：默认入口脚本
- `data/sample_qa.json`：本地样例 QA 数据
- `outputs/metrics.json`：held-out 与 replay 指标
- `outputs/sample_predictions.json`：预测明细
- `requirements.txt`：依赖说明（当前 starter 仅标准库）

## 数据 schema

`samples` 中每条样本包含：

- `id`
- `context`
- `question`
- `answers`：至少一个答案，包含 `text` 与 `answer_start`

脚本会校验答案文本是否与 context 指定 span 完全一致。

## 运行方式

在工作区根目录执行：

```powershell
f:/Study/ComputerScience/Computer_2026_Spring_Term/Natural_Language_Processing/lab/.venv/Scripts/python.exe lab07/lab07_reading_comprehension.py
```

## 当前验证结果

- 训练样本数：4
- 验证样本数：2
- Held-out EM/F1：0.0000 / 0.4444
- Starter replay EM/F1：0.0000 / 0.3517

说明：
- 当前是启发式抽取器，主要用于验证结构与流程，不代表正式模型效果。
- 指标偏低符合预期，后续应接入预训练模型路线提升效果。

## 扩展点

`PretrainedQaRoute` 是预训练模型占位接口。拿到课程数据和依赖后，可将启发式抽取替换为正式模型推理或训练。
