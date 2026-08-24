# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import jieba
import jieba.analyse
import jieba.posseg
#jieba.load_userdict("user.txt")

text = ''
s = '我来到北京清华大学'
ss = "此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，增资后，吉林欧亚置业注册资本由7000万元增加到5亿元。吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。目前在建吉林欧亚城市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。"
print('精确模式:',jieba.lcut(s))
print('全模式:',jieba.lcut(s,cut_all=True))
print('搜索引擎模式',jieba.lcut_for_search(s))
print('精确模式，新词发现',jieba.lcut('他来到了网易杭研大厦'))
print('精确模式',jieba.lcut('他来到了网易杭研大厦',HMM=False))
print('搜索引擎模式',jieba.lcut_for_search('小明硕士毕业于中国科学院计算所，后在日本京都大学深造'))

print('词性标注:',jieba.posseg.lcut(ss))
print('关键词提取:',jieba.analyse.textrank(ss, topK=10, withWeight=True))
print('关键词提取:',jieba.analyse.textrank(ss, topK=10, withWeight=True, allowPOS=('n',)))

print('用户词典:',jieba.lcut(('小红烧肉，王大师兄'),HMM=False))

print('调整词典:',jieba.lcut('如果放到post中将出错。'))
#jieba.suggest_freq(('中', '将'), True)
#print('调整词典:',jieba.lcut('如果放到post中将出错。'))
print(jieba.lcut('「台中」正确应该不会被切开', HMM=False))
jieba.suggest_freq('台中', True)
print(jieba.lcut('「台中」正确应该不会被切开', HMM=False))