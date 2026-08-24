def FMM(dictionary, max_len, sentence):
    """
    正向最大匹配（Forward Maximum Matching）
    dictionary: 外部传入的词典集合
    max_len: 词典中最大词长
    sentence: 待分词句子
    """

    result = []
    start = 0

    # 从句子左侧开始扫描，直到所有字符都被切分
    while start < len(sentence):
        # 每次优先尝试“最长词长”的片段
        end = min(len(sentence), start + max_len)

        # 如果最长片段不在词典中，就逐步缩短片段长度
        while end > start:
            word = sentence[start:end]

            # 命中词典，或者已经退化到单字时，都可以切分输出
            if word in dictionary or len(word) == 1:
                result.append(word)
                start = end
                break

            end -= 1

    return result


def RMM(dictionary, max_len, sentence):
    """
    逆向最大匹配（Reverse Maximum Matching）
    dictionary: 外部传入的词典集合
    max_len: 词典中最大词长
    sentence: 待分词句子
    """

    result = []
    end = len(sentence)

    # 从句子右侧开始扫描，直到句首位置
    while end > 0:
        start = max(0, end - max_len)

        # 也是优先取最长片段，不命中时逐渐右移 start 缩短词长
        while start < end:
            word = sentence[start:end]

            # 命中词典或只剩单字时，插入到结果最前面
            if word in dictionary or len(word) == 1:
                result.insert(0, word)
                end = start
                break

            start += 1

    return result


def BM(dictionary, max_len, sentence):
    """
    双向最大匹配（Bidirectional Matching）
    先分别执行 FMM 和 RMM，再根据规则选出更优结果。
    """

    fmm_result = FMM(dictionary, max_len, sentence)
    rmm_result = RMM(dictionary, max_len, sentence)

    # 规则1：如果切分词数不同，优先选择词数较少的结果
    if len(fmm_result) != len(rmm_result):
        return fmm_result if len(fmm_result) < len(rmm_result) else rmm_result

    # 规则2：如果两者完全相同，直接返回任意一个即可
    if fmm_result == rmm_result:
        return fmm_result

    # 规则3：词数相同时，优先选择单字词更少的结果
    fmm_single = sum(1 for word in fmm_result if len(word) == 1)
    rmm_single = sum(1 for word in rmm_result if len(word) == 1)

    return fmm_result if fmm_single < rmm_single else rmm_result


if __name__ == "__main__":
    dictionary = {
        '今日', '阳光明媚', '光明', '明媚', '阳光', '我们', '在', '在野', '生动', '野生',
        '动物园', '野生动物园', '物', '园', '玩'
    }
    sentence = '在野生动物园玩'
    max_len = max(len(word) for word in dictionary)

    print("the results of FMM:\n", FMM(dictionary, max_len, sentence))
    print("the results of RMM:\n", RMM(dictionary, max_len, sentence))
    print("the results of BM:\n", BM(dictionary, max_len, sentence))
