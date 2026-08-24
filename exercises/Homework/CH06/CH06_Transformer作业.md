# 第6章作业：Transformer 机器翻译代码解析

Transformer 执行过程分析

—— 基于 PyTorch 官方 language_translation 示例的执行流程与代码注释说明

代码来源：PyTorch 官方 examples 仓库中的 language_translation 示例

代码链接：https://github.com/pytorch/examples/tree/main/language_translation

## 一、整体架构概览

这次我选的案例是 PyTorch 官方提供的 Transformer 翻译示例。从代码结构看，这个案例主要分成三部分：数据准备、模型前向传播、训练与推理。

整体执行链可以概括为：

Multi30k 数据集 -> 分词 -> 词表构建 -> 加入 <bos>/<eos> -> pad 补齐

-> DataLoader 输出 batch -> Translator.forward -> Embedding + PositionalEncoding

-> nn.Transformer -> 线性层 ff -> 输出词表概率 -> loss 计算 -> 反向传播更新参数

推理阶段则是另一条更短的链：

输入句子 -> model.encode 得到 memory -> 以 <bos> 作为起点

-> 循环执行 model.decode -> 预测下一个词 -> 拼接回结果序列 -> 直到生成 <eos>

如果把这个案例和课堂上的原始 Transformer 结构对应起来，那么可以看到：get_data 负责把自然语言变成模型输入，Translator.forward 负责完整前向传播，train 负责参数学习，greedy_decode 负责一步一步生成最终译文。

## 二、逐组件执行过程详解

### 2.1 输入处理：get_data 与词表构建

【get_data 函数】

这个函数在 data.py 中，是整个数据准备阶段的入口。程序先读取 Multi30k 平行语料，然后分别给源语言和目标语言创建 tokenizer。分词之后，代码会使用 build_vocab_from_iterator 根据训练集建立词表，并且额外加入 <unk>、<pad>、<bos>、<eos> 四个特殊符号。

执行流程：

Step 1: 读取训练集和验证集，并按语言对取出 src 与 tgt 句子。

Step 2: 使用 spacy 分词器把原始句子切分成 token。

Step 3: 根据训练集统计词频并建立词表。

Step 4: 在每个句子前后补上 <bos> 和 <eos>，让模型知道句子边界。

Step 5: 通过 pad_sequence 把同一个 batch 中长短不一的句子补齐。

这一步的意义是把原始文本统一变成模型可以处理的整数序列。如果没有词表、起止符和 padding，后面的 Transformer 就无法按 batch 接收输入，更无法正确构造掩码。

### 2.2 掩码构造：create_mask

【create_mask 函数】

create_mask 的作用是生成训练时必须使用的各种掩码。Transformer 和 RNN 不一样，它会同时看到整段序列，所以如果不做掩码处理，解码器在训练时就会提前看到未来词，相当于直接偷看答案。

执行流程：

Step 1: 先读取源序列和目标序列的长度。

Step 2: 通过 generate_square_subsequent_mask 生成目标端的下三角掩码。

Step 3: 根据 pad_idx 找出 src 和 tgt 中所有补齐位置。

Step 4: 把这些信息分别传给编码器、解码器和 memory 交互部分。

这里最关键的是 tgt_mask。它保证当前位置只能看到自己和前面的词，不能看到后面的真实答案，所以 Transformer 才能保持自回归生成的特点。src_padding_mask 和 tgt_padding_mask 则是防止模型把 <pad> 当成正常单词参与注意力计算。

### 2.3 模型主体：Translator.forward

【Translator.forward】

Translator 类定义在 model.py 中，是整个案例的模型主体。它先分别给源语言和目标语言建立嵌入层，然后再通过 PositionalEncoding 注入位置信息，最后调用 nn.Transformer 完成编码和解码。

执行流程：

Step 1: src 经过 src_embedding 变成词向量。

Step 2: trg 经过 tgt_embedding 变成目标端词向量。

Step 3: 两边都再加上位置编码，补足顺序信息。

Step 4: 把 src_emb、tgt_emb 和各类 mask 一起送入 nn.Transformer。

Step 5: 输出经过线性层 ff，映射到目标词表大小，得到 logits。

虽然这个示例没有手写多头注意力的底层细节，但从 forward 的结构还是能看出标准的 Encoder-Decoder 逻辑：编码器先抽取源句上下文表示，解码器再结合当前目标序列和 memory 预测下一个词。

### 2.4 训练过程：train

【train 函数】

train 函数是最能体现模型学习过程的部分。每次读取一个 batch 之后，程序会先把目标序列分成解码器输入 tgt_input 和监督标签 tgt_out，这实际上就是 teacher forcing。

执行流程：

Step 1: 将 src、tgt 移到 DEVICE 上。

Step 2: 用 tgt[:-1, :] 作为解码器输入，用 tgt[1:, :] 作为预测目标。

Step 3: 调用 create_mask 生成 src_mask、tgt_mask 和 padding mask。

Step 4: 执行 model(src, tgt_input, ...) 得到每个位置的预测结果。

Step 5: 用 CrossEntropyLoss 比较 logits 和 tgt_out，得到损失。

Step 6: zero_grad、backward、step，完成一次参数更新。

我觉得这里最值得注意的是 teacher forcing 的处理方式。训练时模型并不是完全依赖自己前一步预测出来的词，而是利用真实目标序列向右错开一位后的结果作为输入，这样更容易稳定训练。

### 2.5 推理过程：greedy_decode

【greedy_decode 函数】

推理和训练最大的区别在于，推理阶段没有真实目标序列可以参考，所以模型只能根据已经生成出来的词一步一步往后预测。这个过程就在 greedy_decode 里完成。

执行流程：

Step 1: 先调用 model.encode 对源句子编码，得到 memory。

Step 2: 用 <bos> 初始化结果序列 ys。

Step 3: 循环构造当前长度对应的 tgt_mask。

Step 4: 调用 model.decode(ys, memory, tgt_mask) 预测下一个词。

Step 5: 取最后一个时间步的输出，通过 ff 得到词表概率。

Step 6: 选出概率最大的词接到 ys 后面，直到出现 <eos>。

这种方法叫贪心解码，优点是实现简单、速度快，缺点是每一步都只选局部最优结果，所以最终译文不一定是全局最优。即便如此，它已经足够清楚地展示 Transformer 在推理阶段是如何逐词生成输出的。

## 三、关键代码片段注释

下面结合源码中的几个关键片段，做一个更直接的注释说明。

### 3.1 get_data 核心片段

train_iterator = Multi30k(split='train', language_pair=(src_lang, tgt_lang))

src_tokenizer = get_tokenizer('spacy', src_lang)

tgt_tokenizer = get_tokenizer('spacy', tgt_lang)

src_vocab = build_vocab_from_iterator(...)

tgt_vocab = build_vocab_from_iterator(...)

```python
def _tensor_transform(token_ids): return torch.cat(([<bos>], token_ids, [<eos>]))
```

src_batch = pad_sequence(src_batch, padding_value=special_symbols['<pad>'])

这段代码说明数据预处理的主线就是：先读取数据，再建词表，最后把长短不一样的句子补齐。这样做以后，DataLoader 输出的 src 和 tgt 就已经是适合模型直接使用的张量了。

### 3.2 create_mask 核心片段

src_seq_len = src.shape[0]

tgt_seq_len = tgt.shape[0]

tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)

src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

src_padding_mask = (src == pad_idx).transpose(0, 1)

tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)

这里可以直接看出两类掩码的区别：一种是为了挡住未来词，一种是为了挡住 padding。前者保证生成顺序正确，后者保证注意力不会浪费在补齐符号上。

### 3.3 Translator.forward 核心片段

src_emb = self.pos_enc(self.src_embedding(src))

tgt_emb = self.pos_enc(self.tgt_embedding(trg))

outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, ...)

```python
return self.ff(outs)
```

这一段基本就是整个前向传播的主干。先做词嵌入，再做位置编码，再进入 Transformer 主体，最后映射到词表维度输出预测分数。

### 3.4 PositionalEncoding 核心片段

den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)

pos = torch.arange(0, maxlen).reshape(maxlen, 1)

pos_embedding[:, 0::2] = torch.sin(pos * den)

pos_embedding[:, 1::2] = torch.cos(pos * den)

```python
return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
```

这里展示的是位置编码的核心实现。因为 Transformer 本身没有循环结构，所以它需要把位置信息显式加到词向量里。这个案例使用的就是经典的正弦和余弦位置编码。

### 3.5 train 核心片段

tgt_input = tgt[:-1, :]

src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(...)

logits = model(src, tgt_input, ...)

tgt_out = tgt[1:, :]

loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

loss.backward()

optim.step()

这几行代码非常集中地体现了训练逻辑：目标序列拆分、计算损失、反向传播、参数更新。课程里讲到的监督学习流程，在这里可以直接看到完整实现。

### 3.6 greedy_decode 核心片段

memory = model.encode(src, src_mask)

ys = torch.ones(1, 1).fill_(start_symbol)

tgt_mask = generate_square_subsequent_mask(ys.size(0), DEVICE)

out = model.decode(ys, memory, tgt_mask)

prob = model.ff(out.transpose(0, 1)[:, -1])

_, next_word = torch.max(prob, dim=1)

ys = torch.cat([ys, next_word], dim=0)

这部分体现的就是 Transformer 推理时一步一步生成输出的过程。模型不会一次性把完整结果全部给出来，而是每次只预测下一个词，再把它拼回到序列中继续往后算。
