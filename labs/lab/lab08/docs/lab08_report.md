# 实验八：基于 Seq2Seq 的文本标题自动生成

实验地点:计算机大楼606

实验目的：    完成一个可本地执行的标题生成 starter，明确文章输入与标题输出的实验契约，并形成最小验证结果。

实验环境（硬件和软件）  Windows 11，Python 3.12 虚拟环境，标准库 JSON 处理

实验内容：

本实验围绕文本标题生成任务，构建了一个启发式 starter，用于把输入文章压缩为一个较短的候选标题。

程序使用 sample_title_data.json 作为本地样例数据，对文章先做基础归一化，再取首个分句生成不超过 14 字的候选标题。

在评估阶段，脚本分别计算 held-out 与 replay 的 exact match 和字符级 F1，并保存每条样本的预测标题，便于分析生成器是否覆盖了文章中的关键信息。

实验分析重点是理解启发式标题生成器的工作方式，以及为什么当前结果经常出现 EM 为 0 但字符级 F1 仍较高。

实验步骤：

### 1. 文本归一化：先去掉空白和常见标点，减少标题生成时的无效字符干扰。

```python
def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。；：！？,.!?]", "", text)
    return text
```

### 2. 启发式标题生成：从文章首个分句抽取主要信息，并把候选标题限制在 14 个字以内。

```python
def generate_title(article: str) -> str:
    first_clause = re.split(r"[，。；：！？]", article)[0].strip()
    compact = normalize_text(first_clause)
    if len(compact) <= 14:
        return compact
    return compact[:14]
```

### 3. 指标评估：使用 exact match 和字符级 F1 评估生成标题与金标标题之间的重合程度。

```python
def evaluate(samples: list[TitleSample]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    em_scores: list[float] = []
    f1_scores: list[float] = []
    predictions: list[dict[str, Any]] = []

    for sample in samples:
        predicted = generate_title(sample.article)
        em = exact_match(predicted, sample.title)
        f1 = char_f1(predicted, sample.title)
        em_scores.append(em)
        f1_scores.append(f1)
        predictions.append(
            {
                "id": sample.sample_id,
                "article": sample.article,
                "gold_title": sample.title,
                "predicted_title": predicted,
                "exact_match": em,
                "f1": f1,
            }
        )

    return {
        "exact_match": sum(em_scores) / len(em_scores),
        "f1": sum(f1_scores) / len(f1_scores),
    }, predictions
```

- **（1）启发式标题生成器是怎样生成候选标题的？**

程序首先把文章按中文标点切分，选择首个分句作为最主要的信息来源，再对该分句进行空白和标点归一化，得到紧凑文本。

如果归一化后的文本长度不超过 14 个字，就直接作为标题；如果过长，则截取前 14 个字作为候选标题。这是一种简单但可复现的最小标题生成策略。

- **（2）为什么 EM 为 0，但字符级 F1 仍然较高？**

EM 要求预测标题与金标标题完全一致，而标题生成任务本身允许多种合理表达，因此只要措辞稍有不同，EM 就会变成 0。

字符级 F1 更关注预测标题与金标标题在关键词层面的重合度。当前启发式方法虽然没有很好完成语义压缩，但通常能覆盖新闻主体、地点或事件关键词，所以 F1 仍然保持在较高水平。

实验数据记录：

### 1. 数据集为 lab08/data/sample_title_data.json，训练样本数 4，验证样本数 2。

### 2. Held-out EM=0.0000，Held-out F1=0.7409；Starter replay EM=0.0000，Starter replay F1=0.6209。

### 3. 示例“上海发布人工智能产业新政策...”的金标标题为“上海发布人工智能产业新政”，模型预测为“上海发布人工智能产业新政策”，字符级 F1 达到 0.96。

### 4. 示例“韩梅梅加入华为云团队后...”的金标标题为“韩梅梅推进华为云推理优化”，模型预测为“韩梅梅加入华为云团队后”，说明生成器覆盖了主体信息，但缺少更强的压缩与概括能力。

### 5. 总体来看，启发式方法已经能稳定输出标题，但还不足以达到高质量新闻标题生成的要求。

问题讨论：

问题：当前标题生成器能够覆盖关键信息，但语义压缩能力不足。

现象描述：exact match 始终为 0，而字符级 F1 在 held-out 集合上达到 0.7409，说明生成标题通常和金标有较多重合字符，但表达方式不够精炼。

原因分析：模型只是取首句并做截断，没有真正学习新闻摘要或标题压缩规律，因此容易保留冗余背景信息，而无法生成更概括、更符合新闻风格的标题。

解决方法：后续可以引入预训练标题生成模型，并增加 Rouge 等指标，继续以当前 starter 的输入输出结构作为本地实验基线。
