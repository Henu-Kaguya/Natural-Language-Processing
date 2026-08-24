# Lab05 基于BiLSTM+CRF的命名实体识别

本实验实现基于 PyTorch 的 BiLSTM+CRF 命名实体识别模型，使用 CoNLL2003 英文数据集进行训练和评估。

## 目录结构

```
lab05/
├── vocab.py            # Vocabulary 和 TokenVocabulary 词表类
├── vectorizer.py       # ConllVectorizer 向量化器
├── dataset.py          # ConllDataset 数据集 + batch 生成
├── pre_data.py         # CoNLL 格式数据读取
├── model.py            # BiLSTM_CRF 模型
├── train.py            # 训练脚本（含 early stopping）
├── toy_version.py      # 教学用最小 BiLSTM-CRF 示例
├── conlleval.pl        # CoNLL 评估脚本
├── data/
│   └── conll2003/      # CoNLL2003 数据集
│       ├── train.txt
│       ├── valid.txt
│       └── test.txt
├── model_storage/      # 模型检查点保存目录
├── README.md
└── requirements.txt
```

## 环境依赖

```bash
pip install -r lab05/requirements.txt
```

## 运行方式

### 训练模型

```bash
cd lab05
python train.py
```

训练完成后会生成 `result.txt`，包含测试集的预测结果。

### 评估

```bash
perl conlleval.pl < result.txt
```

### 可选参数

在 `train.py` 中修改 `args` 可调整：
- `batch_size`: 批大小（默认 32）
- `num_epochs`: 训练轮数（默认 50）
- `embedding_dim`: 词嵌入维度（默认 100）
- `hidden_dim`: LSTM 隐藏层维度（默认 50）
- `use_glove`: 是否使用 GloVe 预训练词向量（默认 False）
- `glove_filepath`: GloVe 文件路径（需自行下载 glove.6B.100d.txt）

### Toy Version

运行简化版示例理解 BiLSTM-CRF 过程：

```bash
python toy_version.py
```

## 模型架构

- **Embedding 层**: 将 token 映射为稠密向量
- **BiLSTM 层**: 双向 LSTM 编码上下文信息
- **线性层**: 将 LSTM 输出映射到标签空间（emission scores）
- **CRF 层**: 转移矩阵 + 前向算法 + Viterbi 解码

## 数据格式

CoNLL2003 格式，每行一个 token 和其 BIO 标签，空行分隔句子：

```
EU B-ORG
rejects O
German B-MISC
call O
```
