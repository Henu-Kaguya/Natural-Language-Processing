# 实验二：基于 TextRank 与 TF-IDF 的关键词提取

实验目的：理解并掌握TextRank关键字提取算法。

实验环境（硬件和软件）win11,python3.12.8

实验内容：

TextRank算法是一种基于图的排序算法，由PageRank算法改进而来。它利用一篇文档内部词语间的共现信息来抽取关键词，可以抽取出文本的关键词、关键词组和关键句。TextRank算法的基本思想是将文档看作一个词的网络，网络中的链接表示词与词之间的语义关系。

本实验分别利用jieba工具和TextRank4zh两种方法实现TextRank算法。

实验步骤：

任务一：提取燕山大学简介的关键词

读懂jieba和TextRank4zh两种算法，为主要代码做注释，编写代码输出燕山大学简介这段话的关键词和权重。

代码如下：

```python
import jieba.analyse
from textrank4zh import TextRank4Keyword

text = "燕山大学是河北省人民政府、教育部、工业和信息化部、国家国防科技工业局四方共建的全国重点大学，河北省重点支持的国家一流大学和世界一流学科建设高校，北京高科大学联盟成员。"

# 使用jieba的TF-IDF方法提取关键词，topK=10表示提取前10个，withWeight=True表示同时返回权重
print("jieba TF-IDF 提取关键词：")
tfidf_result = jieba.analyse.extract_tags(text, topK=10, withWeight=True)
for word, weight in tfidf_result:
    print(f"  {word} : {weight:.4f}")

# 使用jieba的TextRank方法提取关键词
print("jieba TextRank 提取关键词：")
textrank_result = jieba.analyse.textrank(text, topK=10, withWeight=True)
for word, weight in textrank_result:
    print(f"  {word} : {weight:.4f}")

# 使用TextRank4zh提取关键词
# 先创建TextRank4Keyword对象
tr4w = TextRank4Keyword()
# analyze方法进行分析，lower=True表示转小写，window=5表示共现窗口大小为5
tr4w.analyze(text=text, lower=True, window=5)
# get_keywords获取关键词，10表示取前10个，word_min_len=2表示词最短2个字
print("TextRank4zh 提取关键词：")
for item in tr4w.get_keywords(10, word_min_len=2):
    print(f"  {item['word']} : {item['weight']:.4f}")
```

运行结果：

jieba TF-IDF 提取关键词：

河北省人民政府 : 0.4742
  一流大学 : 0.4635
  燕山大学 : 0.4428
  学科建设 : 0.4296
  大学 : 0.4233
  国防科技 : 0.4004
  工业局 : 0.3901
  重点 : 0.3891
  共建 : 0.3466
  高科 : 0.3149

jieba TextRank 提取关键词：

重点 : 1.0000
  大学 : 0.9636
  国家 : 0.9273
  河北省 : 0.6236
  共建 : 0.5434
  全国 : 0.5341
  信息化 : 0.5235
  北京 : 0.5018
  一流 : 0.4987
  高校 : 0.4982

TextRank4zh 提取关键词：

国家 : 0.0771
  大学 : 0.0630
  重点 : 0.0521
  信息化 : 0.0468
  北京 : 0.0444
  高校 : 0.0436
  一流 : 0.0430
  学科建设 : 0.0429
  世界 : 0.0426
  一流大学 : 0.0425

任务二：提取[院校名称]新闻网新闻的关键词

从[院校名称]新闻网上找一篇新闻，保存到记事本文件中，在代码中读取这个文件，分别使用jieba和TextRank4zh提取关键词，尝试输出5个和10个。

新闻内容是一篇关于"行进与变迁——当代山水画的学术梳理与呈现"全国高校巡展在[院校名称]美术学院举行的报道。

链接: https://w2022.henu.edu.cn/info/1083/162588.htm

代码如下：

```python
import jieba.analyse
from textrank4zh import TextRank4Keyword

# 读取新闻文件
with open("docs/[院校名称]新闻网新闻.txt", "r", encoding="utf-8") as f:
    news_text = f.read()

# jieba TextRank 提取5个关键词
print("jieba TextRank 提取 5 个关键词：")
result_5 = jieba.analyse.textrank(news_text, topK=5, withWeight=True)
for word, weight in result_5:
    print(f"  {word} : {weight:.4f}")

# jieba TextRank 提取10个关键词
print("jieba TextRank 提取 10 个关键词：")
result_10 = jieba.analyse.textrank(news_text, topK=10, withWeight=True)
for word, weight in result_10:
    print(f"  {word} : {weight:.4f}")

# TextRank4zh 提取5个关键词
tr4w = TextRank4Keyword()
tr4w.analyze(text=news_text, lower=True, window=5)
print("TextRank4zh 提取 5 个关键词：")
for item in tr4w.get_keywords(5, word_min_len=2):
    print(f"  {item['word']} : {item['weight']:.4f}")

# TextRank4zh 提取10个关键词
print("TextRank4zh 提取 10 个关键词：")
for item in tr4w.get_keywords(10, word_min_len=2):
    print(f"  {item['word']} : {item['weight']:.4f}")
```

运行结果：

jieba TextRank 提取 5 个关键词：

山水画 : 1.0000
  高校 : 0.6333
  巡展 : 0.6143
  创作 : 0.5881
  学术 : 0.5307

jieba TextRank 提取 10 个关键词：

山水画 : 1.0000
  高校 : 0.6333
  巡展 : 0.6143
  创作 : 0.5881
  学术 : 0.5307
  教学 : 0.4374
  呈现 : 0.3546
  山水 : 0.3420
  中原 : 0.3356
  开幕式 : 0.3351

TextRank4zh 提取 5 个关键词：

山水画 : 0.0254
  美术学院 : 0.0234
  高校 : 0.0223
  当代 : 0.0176
  创作 : 0.0165

TextRank4zh 提取 10 个关键词：

山水画 : 0.0254
  美术学院 : 0.0234
  高校 : 0.0223
  当代 : 0.0176
  创作 : 0.0165
  学术 : 0.0157
  巡展 : 0.0156
  山水 : 0.0135
  教学 : 0.0123
  [院校名称] : 0.0122

任务三：停用词过滤后的关键词提取

编写停用词过滤算法，读取停用词表，对新闻文档进行停用词过滤后再用TextRank4zh进行关键词抽取，并尝试设置不同的window参数。

代码如下：

```python
import codecs
import jieba
from textrank4zh import TextRank4Keyword

# 读取新闻文件
with open("docs/[院校名称]新闻网新闻.txt", "r", encoding="utf-8") as f:
    news_text = f.read()

# 读取停用词表，把每个停用词存到一个集合里
stopword_path = "chapter05/5.2.3/data/stopWord.txt"
stop_words = set()
for line in codecs.open(stopword_path, "r", encoding="utf-8"):
    stop_words.add(line.strip())

# 停用词过滤函数：先分词，然后去掉在停用词表里的词，最后拼回去
def filter_stopwords(text, stop_words):
    words = jieba.lcut(text)
    filtered = [w for w in words if w.strip() and w not in stop_words]
    return "".join(filtered)

# 过滤停用词
filtered_text = filter_stopwords(news_text, stop_words)

# 用不同的window参数提取关键词
for window_size in [2, 3, 5, 8]:
    print(f"window = {window_size} 时的关键词：")
    tr4w = TextRank4Keyword()
    tr4w.analyze(text=filtered_text, lower=True, window=window_size)
    for item in tr4w.get_keywords(10, word_min_len=2):
        print(f"  {item['word']} : {item['weight']:.4f}")
```

运行结果：

window = 2 时的关键词：

山水画 : 0.0234
  美术学院 : 0.0215
  高校 : 0.0194
  学术 : 0.0173
  巡展 : 0.0157
  当代 : 0.0147
  [院校名称] : 0.0135
  创作 : 0.0130
  中原 : 0.0119
  教学 : 0.0113

window = 3 时的关键词：

山水画 : 0.0253
  美术学院 : 0.0222
  高校 : 0.0190
  巡展 : 0.0189
  学术 : 0.0175
  [院校名称] : 0.0171
  当代 : 0.0167
  创作 : 0.0148
  山水 : 0.0123
  中原 : 0.0110

window = 5 时的关键词：

山水画 : 0.0250
  美术学院 : 0.0211
  高校 : 0.0196
  巡展 : 0.0181
  当代 : 0.0175
  创作 : 0.0175
  [院校名称] : 0.0158
  学术 : 0.0151
  山水 : 0.0132
  教学 : 0.0116

window = 8 时的关键词：

山水画 : 0.0257
  巡展 : 0.0201
  美术学院 : 0.0177
  创作 : 0.0177
  当代 : 0.0176
  高校 : 0.0161
  [院校名称] : 0.0157
  学术 : 0.0133
  山水 : 0.0125
  教学 : 0.0123

可以看到window参数不同，关键词的排序和权重都会有变化。window越小的时候，只看很近的词之间的关系，提取出来的关键词偏向于局部出现频率高的词。window越大，考虑的词语共现范围就越广，排序会发生变化，比如window=8时"巡展"排到了第二位，而window=2时它只排在第五位。总体来看前几个核心关键词变化不大，都是"山水画""美术学院""高校"这几个，但是后面的词排名变动比较明显。

任务四：jieba和TextRank4zh两种方法的差异

### 1. jieba的textrank方法底层先用jieba分词，然后构建词图，最后用PageRank算法迭代计算每个词的权重。它默认只保留名词和动词，输出的权重值经过了归一化，最高的关键词权重为1.0。

### 2. TextRank4zh也是基于TextRank算法，但它自己内部做了分词处理，不完全依赖jieba。它可以通过window参数来控制共现窗口的大小，可以通过word_min_len参数控制最短词长。输出的权重值没有归一化，数值比较小。

### 3. 从实际结果来看，两种方法提取出的关键词大致相同，都能抓住文本的主题。但是在排序上有差别，比如TextRank4zh能提取出"美术学院""当代"这样的词，而jieba的TextRank更侧重于"呈现""开幕式"这类词。

### 4. jieba还提供了extract_tags方法，这个用的是TF-IDF算法而不是TextRank，提取出的关键词和TextRank会有不同，比如TF-IDF能提取出"燕山大学"这种专有名词，而TextRank更倾向于提取通用的高频共现词。

实验数据记录：

实验中使用的新闻来源于[院校名称]新闻网，是一篇关于"行进与变迁——当代山水画的学术梳理与呈现"全国高校巡展在[院校名称]美术学院举行的报道。

各任务的运行结果已在上方给出。

问题讨论：

通过本次实验，我了解了TextRank算法的基本原理。TextRank和PageRank的思想类似，把词语看作图中的节点，词语之间的共现关系看作边，然后通过迭代计算每个节点的重要性。实验中同时用了jieba和TextRank4zh两种工具，发现它们提取的关键词虽然大致相同，但在细节上有不同，比如权重的计算方式和词语的筛选策略都不一样。停用词过滤能有效去掉一些没有实际意义的词，让提取结果更准确。window参数的大小会影响关键词的排序，窗口越大考虑的上下文范围越广。
