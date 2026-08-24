# -*- coding: utf-8 -*-
"""
任务一：使用jieba工具和TextRank4zh两种方法提取关键词
输入文本：燕山大学简介
"""
import jieba.analyse

from textrank_compat import patch_networkx_for_textrank4zh

patch_networkx_for_textrank4zh()

from textrank4zh import TextRank4Keyword

text = "燕山大学是河北省人民政府、教育部、工业和信息化部、国家国防科技工业局四方共建的全国重点大学，河北省重点支持的国家一流大学和世界一流学科建设高校，北京高科大学联盟成员。"

# ========== 方法一：jieba的TextRank ==========
print("=" * 50)
print("方法一：jieba的TextRank关键词提取")
print("=" * 50)

# extract_tags 使用的是 TF-IDF 方法
print("\njieba TF-IDF 提取关键词：")
tfidf_result = jieba.analyse.extract_tags(text, topK=10, withWeight=True)
for word, weight in tfidf_result:
    print(f"  {word} : {weight:.4f}")

# textrank 使用的是 TextRank 方法
print("\njieba TextRank 提取关键词：")
textrank_result = jieba.analyse.textrank(text, topK=10, withWeight=True)
for word, weight in textrank_result:
    print(f"  {word} : {weight:.4f}")

# ========== 方法二：TextRank4zh ==========
print("\n" + "=" * 50)
print("方法二：TextRank4zh 关键词提取")
print("=" * 50)

tr4w = TextRank4Keyword()
tr4w.analyze(text=text, lower=True, window=5)

print("\nTextRank4zh 提取关键词：")
for item in tr4w.get_keywords(10, word_min_len=2):
    print(f"  {item['word']} : {item['weight']:.4f}")
