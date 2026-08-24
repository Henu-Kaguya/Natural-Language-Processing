import json


# Viterbi算法求测试集最优状态序列
def Viterbi(sentence, array_pi, array_a, array_b, STATES):
    weight = [{}]  # 动态规划表
    path = {}

    if sentence[0] not in array_b['B']:
        for state in STATES:
            if state == 'S':
                array_b[state][sentence[0]] = 0
            else:
                array_b[state][sentence[0]] = -3.14e+100

    for state in STATES:
        weight[0][state] = array_pi[state] + array_b[state][sentence[0]]
        path[state] = [state]

    # 置分词开始和结束标志
    for state in STATES:
        if state == 'B':
            array_b[state]['begin'] = 0
        else:
            array_b[state]['begin'] = -3.14e+100
    for state in STATES:
        if state == 'E':
            array_b[state]['end'] = 0
        else:
            array_b[state]['end'] = -3.14e+100

    for i in range(1, len(sentence)):
        weight.append({})
        new_path = {}

        for state0 in STATES:  # state0表示sentence[i]的状态
            items = []
            for state1 in STATES:  # state1表示sentence[i-1]的状态
                prob = weight[i - 1][state1] + array_a[state1][state0] + array_b[state0][sentence[i]]
                items.append((prob, state1))
            best = max(items, key=lambda x: x[0])
            weight[i][state0] = best[0]
            new_path[state0] = path[best[1]] + [state0]
        path = new_path

    prob, state = max(
        [(weight[len(sentence) - 1][state], state) for state in STATES],
        key=lambda x: x[0]
    )
    return path[state]


# 根据状态序列进行分词
def tag_seg(sentence, tag):
    word_list = []
    start = -1
    started = False

    if len(tag) != len(sentence):
        return None
    if len(tag) == 1:
        word_list.append(sentence[0])
    else:
        if tag[-1] == 'B' or tag[-1] == 'M':
            if tag[-2] == 'B' or tag[-2] == 'M':
                tag[-1] = 'E'
            else:
                tag[-1] = 'S'
        for i in range(len(tag)):
            if tag[i] == 'S':
                word_list.append(sentence[i])
            elif tag[i] == 'B':
                if started:
                    word_list.append(sentence[start:i])
                start = i
                started = True
            elif tag[i] == 'E':
                started = False
                word = sentence[start:i + 1]
                word_list.append(word)
            elif tag[i] == 'M':
                continue
    return word_list


if __name__ == '__main__':
    pramater = json.load(open('hmm_states.txt', encoding='utf-8'))
    array_A = pramater['states_matrix']
    array_B = pramater['observation_matrix']
    array_Pi = pramater['init_states']
    STATES = ['B', 'M', 'E', 'S']

    test = "成员A在[高校名称]计算机学院开发了一个小程序"
    tag = Viterbi(test, array_Pi, array_A, array_B, STATES)
    print("状态序列:", tag)
    seg = tag_seg(test, tag)
    print("分词结果:", '/ '.join(seg))
