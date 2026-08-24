# 实验三 Skip-gram 词向量实验

## 目录说明

- `lab03_skipgram.py`：实验3的主脚本，负责数据准备、模型训练、验证损失统计和词语余弦相似度计算。
- `data/skipgram_corpus.txt`：默认训练语料。为了对应题目中的英文词对示例，这里提供了一份可直接运行的小型英文语料。
- `outputs/`：脚本运行后生成的配置、词表、训练指标和相似度结果。
- `docs/lab03_report.md`：实验分析、题目问答和运行说明。

## 依赖安装

建议在工作区虚拟环境中执行：

```powershell
f:/Study/ComputerScience/Computer_2026_Spring_Term/Natural_Language_Processing/lab/.venv/Scripts/python.exe -m pip install -r lab03/requirements.txt
```

默认只要求 CPU 可运行。若本机安装的是支持 CUDA 的 Paddle 版本，可以在运行时传入 `--device gpu`。

## 运行命令

```powershell
f:/Study/ComputerScience/Computer_2026_Spring_Term/Natural_Language_Processing/lab/.venv/Scripts/python.exe lab03/lab03_skipgram.py
```

常用可调参数：

```powershell
f:/Study/ComputerScience/Computer_2026_Spring_Term/Natural_Language_Processing/lab/.venv/Scripts/python.exe lab03/lab03_skipgram.py --epochs 80 --embedding-dim 48 --window-size 3 --device cpu
```

## 最小验证点

- 终端应先打印加载到的 token 数、词表大小和训练样本数。
- 训练过程中应持续打印 `train_loss`，若存在验证集还会打印 `val_loss`。
- 训练结束后应输出 5 组词对的余弦相似度。
- `outputs/` 目录下应生成 `run_config.json`、`vocab.json`、`training_metrics.json` 和 `similarity_results.json`。

## 默认语料选择说明

题目示例中的词对是英文词汇，如 `king/queen`、`topic/theme`。为了保证仓库内可以直接复现实验流程，默认语料选择了一份小型英文合成语料，把这些目标词放在可重复的上下文里，便于在短时间训练后观察到合理的相似度趋势。
