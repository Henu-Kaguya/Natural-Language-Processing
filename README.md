# 自然语言处理 (Natural Language Processing) 课程与实验全集

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)
![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)

本项目为**自然语言处理 (Natural Language Processing)** 课程的完整学习归档与开源实验项目集。包含核心课件、课后编程作业与案例分析、大作业论文复现 (SDFT) 全套报告与周报，以及 8 大经典 NLP 实验的全套可运行代码、测试用例、图表与标准化实验报告。

---

## 📚 仓库目录结构

```
Natural_Language_Processing/
├── README.md                                 # 仓库主页与实验索引
├── LICENSE                                   # CC BY-NC-SA 4.0 开源协议
├── .gitignore                                # Git 忽略规则配置
│
├── slides/                                   # 教学课件与幻灯片
│   ├── 第1章_自然语言处理概述.pptx
│   └── 第4章_中文分词.pptx
│
├── exercises/                                # 课后习题与编程作业
│   └── Homework/
│       ├── CH05/                             # 第5章 TF-IDF 特征提取与词频分析
│       ├── CH06/                             # 第6章 Transformer 机器翻译代码解析与实现
│       └── CH07/                             # 第7章 文本分类 (SVM 与 Logistic Regression)
│
├── labs/                                     # 核心实验体系与课程大作业
│   ├── lab/                                  # 8 大基础/进阶实验
│   │   ├── lab01/                            # 实验1：中文分词 (HMM 与 Viterbi 算法)
│   │   ├── lab02/                            # 实验2：关键词提取 (TextRank 与 TF-IDF)
│   │   ├── lab03/                            # 实验3：词向量技术 (Skip-gram 模型)
│   │   ├── lab04/                            # 实验4：垃圾邮件分类 (多项式朴素贝叶斯)
│   │   ├── lab05/                            # 实验5：命名实体识别 (BiLSTM-CRF)
│   │   ├── lab06/                            # 实验6：实体关系联合抽取 (Tagging Schema)
│   │   ├── lab07/                            # 实验7：抽取式机器阅读理解 (Extractive QA)
│   │   ├── lab08/                            # 实验8：文本标题自动生成 (Seq2Seq)
│   │   ├── chapter07/                        # 第7章 进阶文本分类源码与模型
│   │   ├── chapter08/                        # 第8章 序列标注与关系抽取扩展资源
│   │   └── 《自然语言处理》实验指导书.pdf        # 课程配套实验指导书
│   └── Project/                              # 期末大作业：ACL 论文复现 (SDFT)
│       ├── 自然语言处理_SDFT论文复现_总结报告.md
│       ├── weekly_reports/                   # 8 周复现进展周报
│       └── NLP研究领域常见问题与关键词中英文对照.md
│
└── review/                                   # 结课复习与答辩汇报
    ├── notes/                                # SDFT 答辩复习指南
    └── slides/                               # SDFT 论文复现汇报幻灯片
```

---

## 🧪 8 大核心实验概览

| 实验编号 | 实验主题 | 核心算法 / 架构 | 数据集 / 语料 | 关键技术与成果 | 实验报告 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Lab 01** | 中文分词 | Bigram 正向最大匹配、HMM + Viterbi 算法 | 人民日报分词语料 | 状态转移概率矩阵构建与 Viterbi 动态规划最优路径解码 | [Lab01 报告](labs/lab/lab01/docs/lab01_report.md) |
| **Lab 02** | 关键词提取 | TextRank (网络图模型)、TF-IDF | 新闻文本语料、通用停用词表 | 对比 jieba 与 TextRank4zh 在不同滑动窗口参数下的提取效果 | [Lab02 报告](labs/lab/lab02/docs/lab02_report.md) |
| **Lab 03** | 词向量表示 | Skip-gram 模型、负采样 (Negative Sampling) | 文本语料 (维基/新闻) | 训练稠密词向量，实现 Cosine 相似度检索与词向量空间可视化 | [Lab03 报告](labs/lab/lab03/docs/lab03_report.md) |
| **Lab 04** | 文本分类 | 多项式朴素贝叶斯 (Multinomial NB) | 中文垃圾邮件分类数据集 (150+ 样本) | 构建词袋特征，绘制混淆矩阵与 ROC 曲线 (Acc: 83.3%) | [Lab04 报告](labs/lab/lab04/docs/lab04_report.md) |
| **Lab 05** | 命名实体识别 | BiLSTM + CRF 序列标注 | CoNLL-2003 NER 标注语料 | 发射矩阵与转移矩阵联合优化，使用 conlleval 评估 P/R/F1 | [Lab05 报告](labs/lab/lab05/docs/lab05_report.md) |
| **Lab 06** | 关系抽取 | Tagging Schema 联合抽取模型 | 实体关系抽取标注数据集 | 端到端联合识别实体对与关系类型 (Replay F1: 1.0) | [Lab06 报告](labs/lab/lab06/docs/lab06_report.md) |
| **Lab 07** | 机器阅读理解 | 抽取式阅读理解 (Span Selection) | QA 上下文问答对数据集 | 基于注意力机制预测答案片段起始与终止位置 | [Lab07 报告](labs/lab/lab07/docs/lab07_report.md) |
| **Lab 08** | 文本标题生成 | Encoder-Decoder (Seq2Seq) | 长文本-标题对数据集 | 自回归解码生成紧凑摘要标题 | [Lab08 报告](labs/lab/lab08/docs/lab08_report.md) |

---

## 📝 课后作业与案例分析

- **CH05（TF-IDF 与特征提取）**：包含 `5.1.3` 与 `5.2.3` 案例代码，实现 TF-IDF 矩阵计算与停用词过滤。
- **CH06（Transformer 机器翻译代码解析）**：精读 PyTorch 官方 `language_translation` 与 *The Annotated Transformer*，包含详细执行流与逐行源码注解。
- **CH07（基于 SVM 与 Logistic Regression 的文本分类）**：实现基于多种传统机器学习模型的邮件分类与超参数调优对比。

---

## 🚀 快速上手与环境配置

### 1. 克隆仓库
```bash
git clone https://github.com/Henu-Kaguya/Natural-Language-Processing.git
cd Natural-Language-Processing
```

### 2. 环境配置
推荐使用 Python 3.10+ 环境：
```bash
python -m venv .venv
# Windows 激活
.venv\Scripts\activate
# Linux / macOS 激活
source .venv/bin/activate

# 安装基础依赖
pip install jieba numpy scikit-learn torch torchvision transformers
```

### 3. 运行示例 (以 Lab04 垃圾邮件分类为例)
```bash
cd labs/lab/lab04
python lab04_text_classification.py
```

---

## 📄 许可声明与安全声明

- 本项目采用 [CC BY-NC-SA 4.0 (知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议)](LICENSE) 进行许可。
- 本仓库所涉及的所有个人姓名、学号与院校隐私均已做严格脱敏处理。

## 大型资料下载

课程课件、高阶实验与实验指导书已移至 [archive-2026 Release](https://github.com/Henu-Kaguya/Natural-Language-Processing/releases/tag/archive-2026)：

| Release 资产 | 说明 |
|---|---|
| `nlp-slides.zip` | NLP 核心教学课件（PPTX 共 2 个文件） |
| `nlp-lab-chapter07.zip` | 第 7 章 自适应决策边界模型与文本分类工程代码与数据 |
| `nlp-lab-chapter08.zip` | 第 8 章 基于 LSTM 的命名实体识别模型权重与大型标注数据集 |
| `nlp-lab-manual.pdf` | 《自然语言处理》全套实验指导书规范手册（原生 PDF） |

`lab01` 至 `lab08` 基础实验代码与 `exercises/` 继续在 Git 中维护。
