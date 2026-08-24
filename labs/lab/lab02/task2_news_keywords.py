# -*- coding: utf-8 -*-
"""
任务二：从[高校名称]新闻网新闻中提取关键词
分别使用jieba和TextRank4zh提取5个和10个关键词
"""
import jieba.analyse

from textrank_compat import patch_networkx_for_textrank4zh

patch_networkx_for_textrank4zh()

from textrank4zh import TextRank4Keyword

# 读取新闻文件
with open("docs/campus_news.txt", "r", encoding="utf-8") as f:
    news_text = f.read()

print("新闻内容：")
print(news_text[:100] + "...")
print()

# ========== 方法一：jieba TextRank ==========
print("=" * 50)
print("方法一：jieba TextRank 提取关键词")
print("=" * 50)

print("\n提取 5 个关键词：")
result_5 = jieba.analyse.textrank(news_text, topK=5, withWeight=True)
for word, weight in result_5:
    print(f"  {word} : {weight:.4f}")

print("\n提取 10 个关键词：")
result_10 = jieba.analyse.textrank(news_text, topK=10, withWeight=True)
for word, weight in result_10:
    print(f"  {word} : {weight:.4f}")

# ========== 方法二：TextRank4zh ==========
print("\n" + "=" * 50)
print("方法二：TextRank4zh 提取关键词")
print("=" * 50)

tr4w = TextRank4Keyword()
tr4w.analyze(text=news_text, lower=True, window=5)

print("\n提取 5 个关键词：")
for item in tr4w.get_keywords(5, word_min_len=2):
    print(f"  {item['word']} : {item['weight']:.4f}")

print("\n提取 10 个关键词：")
for item in tr4w.get_keywords(10, word_min_len=2):
    print(f"  {item['word']} : {item['weight']:.4f}")
