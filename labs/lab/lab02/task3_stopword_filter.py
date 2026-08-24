# -*- coding: utf-8 -*-
"""
任务三：停用词过滤 + TextRank4zh 关键词提取
尝试不同的 window 参数，观察关键词变化
"""
import codecs

from textrank_compat import patch_networkx_for_textrank4zh

patch_networkx_for_textrank4zh()

from textrank4zh import TextRank4Keyword

# 读取新闻文件
with open("docs/campus_news.txt", "r", encoding="utf-8") as f:
    news_text = f.read()

# 读取停用词表
stopword_path = "chapter05/5.2.3/data/stopWord.txt"
stop_words = set()
for line in codecs.open(stopword_path, "r", encoding="utf-8"):
    stop_words.add(line.strip())

# 停用词过滤函数
def filter_stopwords(text, stop_words):
    """
    读取停用词表，对文本进行停用词过滤
    """
    import jieba
    words = jieba.lcut(text)
    filtered = [w for w in words if w.strip() and w not in stop_words]
    return "".join(filtered)

# 过滤停用词
filtered_text = filter_stopwords(news_text, stop_words)
print("过滤停用词后的文本：")
print(filtered_text[:200] + "...")
print()

# 尝试不同的 window 参数
for window_size in [2, 3, 5, 8]:
    print("=" * 50)
    print(f"window = {window_size} 时的关键词：")
    print("=" * 50)

    tr4w = TextRank4Keyword()
    tr4w.analyze(text=filtered_text, lower=True, window=window_size)

    for item in tr4w.get_keywords(10, word_min_len=2):
        print(f"  {item['word']} : {item['weight']:.4f}")
    print()
