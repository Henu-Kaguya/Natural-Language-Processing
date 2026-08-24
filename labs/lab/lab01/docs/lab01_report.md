# 实验一：中文分词技术 (HMM 与 Viterbi 算法)

实验目的：1. 理解并掌握基于词表的 FMM、RMM 和 BM 三种中文分词算法。

### 2. 掌握使用 load_dict(file_path) 加载结巴自带词典并统计最大词长的方法；

### 3. 掌握在本机 venv 环境下完成实验、运行程序并分析结果。

实验环境（硬件和软件）win11,python3.12.8

实验内容：

### 1. 阅读 chapter04/4.1.4.py，理解正向最大匹配、逆向最大匹配和双向最大匹配的基本流程。

### 2. 编写 load_dict(file_path) 函数，加载 jieba 自带 dict.txt 的首列词语，构造词典并统计最大词长。

### 3. 改造 FMM、RMM 和 BM，使其能够接收 dictionary 与 max_word_len 两个参数，适配大规模词典。

### 4. 编写 lab01_segmentation.py，在控制台显示“请输入需要分词的段落：”，并输出三种分词结果。

### 5. 关键代码：load_dict() 逐行读取词典首列；bm() 在 FMM 与 RMM 结果之间按分词数和单字数选择更优结果。

```python
关键代码1：load_dict(file_path)
def load_dict(file_path):
    words = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().split()[0]
            words.add(word)
    max_word_len = max(len(word) for word in words)
    return words, max_word_len
```

实验步骤：

### 1. 执行 python -m venv .venv 创建本地虚拟环境。

### 2. 执行 .venv\Scripts\pip install -r requirements.txt 安装实验依赖。

### 3. 执行 .venv\Scripts\python -m pytest tests/test_lab01_segmentation.py -q 验证词典加载、算法和命令行输出。

### 4. 执行 .venv\Scripts\python lab01_segmentation.py --text "在野生动物园玩，我们喜欢观赏日出" 获取实际分词结果。

```python
关键代码2：FMM / RMM / BM 的改造
def fmm(dictionary, max_word_len, sentence):
    ... # 从左到右按最大词长匹配
def rmm(dictionary, max_word_len, sentence):
    ... # 从右到左按最大词长匹配
def bm(dictionary, max_word_len, sentence):
    ... # 比较 FMM 与 RMM，选取更优结果
```

实验数据记录：

### 1. 词典来源：

.venv\Lib\site-packages\jieba\dict.txt

### 2. 词典词条数：349045；最大词长：16。

### 3. 自动化测试结果：4 passed in 0.11s。

### 4. 输入文本：在野生动物园玩，我们喜欢观赏日出。

### 5. FMM 分词结果：在野 / 生动 / 物 / 园 / 玩 / ， / 我们 / 喜欢 / 观赏 / 日出。

### 6. RMM 分词结果：在 / 野生 / 动物园 / 玩 / ， / 我们 / 喜欢 / 观赏 / 日出。

### 7. BM 分词结果：在 / 野生 / 动物园 / 玩 / ， / 我们 / 喜欢 / 观赏 / 日出。

关键代码3：控制台交互输出
text = input('请输入需要分词的段落：')
print('FMM 分词结果:', ' / '.join(fmm_result))
print('RMM 分词结果:', ' / '.join(rmm_result))
print('BM 分词结果:', ' / '.join(bm_result))

问题讨论：

### 1. FMM 采用从左到右的最长匹配策略，在大词典下容易因为局部最优而出现语义不自然的切分，例如本实验样例中出现“在野 / 生动 / 物 / 园 / 玩”。
2. RMM 从右向左匹配，得到“在 / 野生 / 动物园 / 玩”，语义更合理。
3. BM 综合比较 FMM 与 RMM 的分词数以及单字数，在本样例中最终选择了 RMM 的结果。
4. 实验说明：仅依赖词表可以完成基本分词，但面对歧义切分时仍需要统计模型或语义信息进一步优化。
