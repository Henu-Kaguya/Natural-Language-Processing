# Lab06 基于预训练模型的实体关系联合抽取

本实验实现基于 BERT 预训练模型的联合实体关系抽取，同时完成：
1. **序列标注**（token-level）：BIO 标签预测实体边界和类型
2. **多标签分类**（sentence-level）：谓词关系多标签预测
3. **三元组生成**：从预测结果还原 SPO 三元组

## 模型架构

```
Input Text
    ↓
BERT Encoder (bert-base-chinese)
    ↓
┌─────────────────────┬──────────────────────┐
│ [CLS] pooled output │ sequence output      │
│                     │                      │
│ Predicate Head      │ Token Label Head     │
│ (Linear + Sigmoid)  │ (Linear + Softmax)   │
│                     │                      │
│ Multi-label         │ BIO tags per token   │
│ relations           │                      │
└─────────────────────┴──────────────────────┘
```

## 目录结构

```
lab06/
├── data_processor.py     # 数据读取、BERT分词、BIO标注、谓词编码
├── model.py              # JointModel: BERT + 双任务输出头
├── train.py              # 训练脚本 (dual loss)
├── evaluate.py           # 评估: 实体F1 + 关系F1
├── triples_generation.py # 预测结果 → SPO三元组还原
├── data/
│   ├── train.jsonl       # 训练数据 (25条)
│   └── valid.jsonl       # 验证数据 (8条)
├── outputs/              # 模型输出 (metrics, predictions)
├── README.md
└── requirements.txt
```

## 数据格式

JSONL 格式，每行一个样本：

```json
{
  "text": "李雷在华为公司工作",
  "spo_list": [
    {
      "subject": "李雷",
      "subject_type": "PER",
      "predicate": "works_for",
      "object": "华为公司",
      "object_type": "ORG"
    }
  ]
}
```

### 实体类型
- `PER`: 人名
- `ORG`: 机构名
- `LOC`: 地点名

### 关系类型
- `works_for`: 就职于
- `educated_at`: 毕业于
- `lives_in`: 居住在
- `located_in`: 位于
- `visits`: 参观
- `founder_of`: 创立

## 环境依赖

```bash
pip install -r lab06/requirements.txt
```

## 运行方式

```bash
cd lab06
python train.py
```

### 可选参数

```bash
python train.py \
    --bert-model bert-base-chinese \
    --epochs 10 \
    --batch-size 8 \
    --learning-rate 2e-5 \
    --max-seq-length 128
```

## 输出

训练完成后在 `outputs/` 目录生成：
- `best_model.pt`: 最佳模型权重
- `metrics.json`: 评估指标（token级和predicate级 precision/recall/F1）
- `predictions.json`: 验证集预测结果（含实体和关系三元组）

## 实验结果说明

本实验基于 chapter08/8.2.3 的 tagging schema 联合抽取思路，使用 PyTorch + HuggingFace transformers 实现。核心创新在于：
- 单个模型同时做实体识别（序列标注）和关系分类（多标签分类）
- 共享 BERT 编码器，双任务联合训练
- 通过 BIO 标签和谓词预测还原完整 SPO 三元组
