# 实验七：抽取式机器阅读理解 (MRC) 系统

实验地点:计算机大楼606

实验目的：    建立可本地执行的抽取式阅读理解流程，明确上下文、问题与答案片段的输入输出格式，并完成最小验证。

实验环境（硬件和软件）  Windows 11，Python 3.12 虚拟环境，标准库 JSON 处理

实验内容：

本实验围绕抽取式阅读理解任务，构建了一个基于启发式规则的 starter 版本，用于完成问题到答案片段的最小抽取。

程序使用本地 sample_qa.json 组织 SQuAD 风格样本，先校验 answer_start 与答案文本在 context 中的一致性，再根据问题关键词在上下文中搜索候选答案 span。

最终脚本分别输出 held-out 和 replay 两组 EM、字符级 F1 指标，并保存详细预测结果，便于分析模型是“完全命中”还是“部分覆盖”答案。

实验分析重点是理解启发式抽取器的工作方式，以及为什么当前结果经常出现 EM 为 0 而 F1 仍大于 0 的现象。

实验步骤：

### 1. 数据校验：脚本先检查每个样本的答案片段是否真的与 context 中给定位置对齐，保证输入数据合法。

```python
def validate_sample(sample: QaSample) -> None:
    if not sample.answers:
        raise ValueError(f"Sample {sample.sample_id} must contain at least one answer.")
    for answer in sample.answers:
        start = answer.answer_start
        end = start + len(answer.text)
        if start < 0 or end > len(sample.context):
            raise ValueError(f"Sample {sample.sample_id}: answer span is out of context bounds.")
        if sample.context[start:end] != answer.text:
            raise ValueError(f"Sample {sample.sample_id}: answer text does not match context span.")
```

### 2. 启发式答案抽取：先从问题中提取关键词，再在上下文中枚举候选片段并选择得分最高的 span 作为预测答案。

```python
def heuristic_extract_answer(context: str, question: str) -> str:
    keywords = question_keywords(question)
    if not keywords:
        return context[: min(8, len(context))]

    best_start = 0
    best_end = min(8, len(context))
    best_score = -1

    for start in range(len(context)):
        for end in range(start + 1, min(len(context), start + 14) + 1):
            span = context[start:end]
            score = sum(1 for key in keywords if key in span)
            score += min(len(span), 10) * 0.01
            if score > best_score:
                best_score = score
                best_start = start
                best_end = end

    candidate = context[best_start:best_end].strip("，。；： ")
    return candidate if candidate else context[best_start:best_end]
```

### 3. 指标评估：程序分别计算 exact match 和字符级 F1，用于衡量答案是否完全匹配或部分覆盖了金标答案。

```python
def evaluate(samples: list[QaSample]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    em_scores: list[float] = []
    f1_scores: list[float] = []
    predictions: list[dict[str, Any]] = []

    for sample in samples:
        prediction = heuristic_extract_answer(sample.context, sample.question)
        gold = sample.answers[0].text
        em = exact_match(prediction, gold)
        f1 = f1_char(prediction, gold)
        em_scores.append(em)
        f1_scores.append(f1)
        predictions.append(
            {
                "id": sample.sample_id,
                "question": sample.question,
                "context": sample.context,
                "gold_answer": gold,
                "predicted_answer": prediction,
                "exact_match": em,
                "f1": f1,
            }
        )

    return {
        "exact_match": sum(em_scores) / len(em_scores),
        "f1": sum(f1_scores) / len(f1_scores),
    }, predictions
```

- **（1）启发式抽取器是如何生成候选答案的？**

程序先从问题中提取若干关键词，忽略“什么、哪里、多少”等停用字符，然后在 context 中枚举一系列长度有限的候选片段。

对于每个候选片段，脚本统计其中包含多少问题关键词，并结合片段长度给出一个启发式得分，最终选取得分最高的片段作为预测答案。

- **（2）为什么本实验中 F1 常常高于 EM？**

EM 要求预测答案与金标答案完全一致，只要多出一个字或少一个字，就会记为 0，因此对边界误差非常敏感。

字符级 F1 则允许部分重合。当前启发式方法虽然经常把答案边界拉长，但通常还能覆盖金标答案中的一部分核心字符，所以 F1 往往高于 EM。

实验数据记录：

### 1. 数据集为 lab07/data/sample_qa.json，训练样本数 4，验证样本数 2。

### 2. Held-out EM=0.0000，Held-out F1=0.4444，说明在未见样本上预测答案经常无法完全命中，但能部分覆盖金标答案。

### 3. Starter replay EM=0.0000，Starter replay F1=0.3517，说明整体启发式规则仍然偏粗糙。

### 4. 例如“张三来自哪里？”的金标答案是“上海”，模型预测为“张三来自上海，目前在”；“韩梅梅加入了哪个部门？”的金标答案是“华为云部门”，预测为“韩梅梅毕业后加入华为云部门”。

### 5. 从这些结果可以看出，模型更擅长找到与问题相关的句段，而不是精确截取答案边界。

问题讨论：

问题：当前启发式阅读理解模型无法精确定位答案边界。

现象描述：验证和 replay 的 exact match 都为 0，但字符级 F1 保持在 0.35 到 0.44 之间，说明预测片段通常相关但偏长。

原因分析：模型没有真正的语义匹配或起止位置建模能力，只是依靠关键词覆盖和简单长度偏置来打分，因此容易把答案所在句子的较长片段整体截取出来。

解决方法：后续可以接入预训练抽取式问答模型，并继续沿用当前数据校验、评估和结果落盘结构作为实验基础。
