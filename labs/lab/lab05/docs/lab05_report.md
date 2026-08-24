# 实验五：基于 BiLSTM-CRF 的命名实体识别 (NER)

实验地点:计算机大楼606

实验目的：    理解并实现基于 BiLSTM+CRF 的命名实体识别流程，掌握 CRF 层在实体边界约束中的作用。

实验环境（硬件和软件）  Windows 11，Python 3.12，PyTorch

实验内容：

本实验基于 CoNLL2003 英文数据集，实现了完整的 BiLSTM+CRF 命名实体识别流水线。

整体流程包括：读取 CoNLL 格式数据、构建词表和向量化器、使用 Embedding + BiLSTM + CRF 完成序列标注，最后通过 Viterbi 解码输出最优标签序列。

CRF 层的核心作用是通过转移矩阵约束标签之间的合法转移关系，例如 B-PER 后只能接 I-PER 或 O，而不能接 B-LOC，从而避免产生不合法的标签序列。

模型训练采用 early stopping 和学习率衰减策略，评估使用 CoNLL 标准工具 conlleval.pl 统计 precision、recall 和 F1。

实验步骤：

### 1. 数据读取：使用 Conll03Reader 读取 CoNLL2003 格式的 train/valid/test 数据。

```python
class Conll03Reader:
    def read(self, data_path):
        data_parts = ['train', 'valid', 'test']
        extension = '.txt'
        dataset = {}
        for data_part in tqdm(data_parts):
            file_path = os.path.join(data_path, data_part + extension)
            dataset[data_part] = self.read_file(str(file_path))
        return dataset

    def read_file(self, file_path):
        samples = []
        tokens = ['<start>']
        tag = ['<start>']
        with open(file_path, 'r', encoding='utf-8') as fb:
            for line in fb:
                line = line.strip('\n')

                if line == '-DOCSTART- -X- -X- O':
                    pass
                elif line == '':
                    if len(tokens) > 1:
                        samples.append((tokens + ['<end>'], tag + ['<end>']))
                        tokens = ['<start>']
                        tag = ['<start>']
                else:
                    contents = line.split(' ')
                    tokens.append(contents[0])
                    tag.append(contents[-1])
        return samples

def predata(input_path="./data/conll2003"):
    ds_rd = Conll03Reader()
    condata = ds_rd.read(input_path)
    return condata
```

### 2. 模型定义：BiLSTM+CRF 模型，包含 Embedding 层、BiLSTM 编码层、线性层（emission scores）和 CRF 转移矩阵。

```python
class BiLSTM_CRF(nn.Module):

    def __init__(self, token_vocab, tag_vocab, batch_size,
                 dropout=0.5, embedding_dim=256,
                 hidden_dim=256, pretrained_embedding=None,
                 padding_idx=0, num_layers=1):
        super(BiLSTM_CRF, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.token_vocab = token_vocab
        self.tag_vocab = tag_vocab
        self.pad = self.token_vocab.pad_token

        self.tagset_size = len(tag_vocab)
        self.begin_tag_idx = tag_vocab.lookup_token('<start>')
        self.end_tag_idx = tag_vocab.lookup_token('<end>')

        if pretrained_embedding is None:
            self.word_embeds = nn.Embedding(len(self.token_vocab), embedding_dim)
        else:
            self.word_embeds = nn.Embedding(len(self.token_vocab), embedding_dim,
                                            _weight=pretrained_embedding)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=num_layers, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.transition = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))
        self.transition.data[self.begin_tag_idx, :] = -10000
        self.transition.data[:, self.end_tag_idx] = -10000

        self.hidden = self.init_hidden(num_layers, batch_size)

    def init_hidden(self, num_layers, batch_size):
        return (torch.randn(2 * num_layers, batch_size, self.hidden_dim // 2, device=self.device),
                torch.randn(2 * num_layers, batch_size, self.hidden_dim // 2, device=self.device))

    def _forward_alg(self, feats, mask):
        """Forward algorithm for CRF partition function

        Args:
            feats: [b_s, seq_len, tag_size]
            mask: [b_s, seq_len]
        Returns:
            [b_s] partition function scores
        """
        init_alphas = torch.full((feats.size(0), self.tagset_size), -10000., device=self.device)
        init_alphas[:, self.begin_tag_idx] = 0.

        forward_var_list = []
        forward_var_list.append(init_alphas)
        d = torch.unsqueeze(feats[:, 0], dim=1)
        for feat_index in range(1, feats.size(1)):
            n_unfinish = mask[:, feat_index].sum()
            d_uf = d[:n_unfinish]
            emit_and_transition = feats[:n_unfinish, feat_index].unsqueeze(dim=1) + self.transition
            log_sum = d_uf.transpose(1, 2) + emit_and_transition
            max_v = log_sum.max(dim=1)[0].unsqueeze(dim=1)
            log_sum = log_sum - max_v
            d_uf = max_v + torch.logsumexp(log_sum, dim=1).unsqueeze(dim=1)
            d = torch.cat((d_uf, d[n_unfinish:]), dim=0)
        d = d.squeeze(dim=1)
        max_d = d.max(dim=-1)[0]
        d = max_d + torch.logsumexp(d - max_d.unsqueeze(dim=1), dim=1)
        return d

    def _get_lstm_features(self, embedded_vec, seq_len):
        """Get emission scores from BiLSTM

        Args:
            embedded_vec: [max_seq_len, b_s, e_d]
            seq_len: [b_s]
        Returns:
            [b_s, seq_len, tag_size]
        """
        pack_seq = pack_padded_sequence(embedded_vec, seq_len)
        lstm_out, self.hidden = self.lstm(pack_seq)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_feats = self.hidden2tag(lstm_out)
        lstm_feats = self.dropout(lstm_feats)
        return lstm_feats

    def _score_sentence(self, feats, tags, mask):
        """Score the gold tag sequence

        Args:
            feats: [b_s, seq_len, tag_size]
            tags: [b_s, seq_len]
            mask: [b_s, seq_len]
        Returns:
            [b_s] gold path scores
        """
        score = torch.gather(feats, dim=2, index=tags.unsqueeze(dim=2)).squeeze(dim=2)
        score[:, 1:] += self.transition[tags[:, :-1], tags[:, 1:]]
        total_score = (score * mask.type(torch.float)).sum(dim=1)
        return total_score

    def _viterbi_decode(self, feats, mask, seq_len):
        """Viterbi decoding for finding best tag sequence

        Args:
            feats: [b_s, seq_len, tag_size]
            mask: [b_s, seq_len]
            seq_len: [b_s]
        Returns:
            scores, tag_sequences
        """
        batch_size = feats.size(0)
        tags = [[[i] for i in range(len(self.tag_vocab))]] * batch_size
        d = torch.unsqueeze(feats[:, 0], dim=1)
        for i in range(1, seq_len[0]):
            n_unfinished = mask[:, i].sum()
            d_uf = d[:n_unfinished]
            emit_and_transition = self.transition + feats[:n_unfinished, i].unsqueeze(dim=1)
            new_d_uf = d_uf.transpose(1, 2) + emit_and_transition
            d_uf, max_idx = torch.max(new_d_uf, dim=1)
            max_idx = max_idx.tolist()
            tags[:n_unfinished] = [[tags[b][k] + [j] for j, k in enumerate(max_idx[b])] for b in range(n_unfinished)]
            d = torch.cat((torch.unsqueeze(d_uf, dim=1), d[n_unfinished:]), dim=0)
        d = d.squeeze(dim=1)
        score, max_idx = torch.max(d, dim=1)
        max_idx = max_idx.tolist()
        tags = [tags[b][k] for b, k in enumerate(max_idx)]
        return score, tags

    def neg_log_likelihood(self, token_vec, tag_vec, seq_len):
        """Compute negative log likelihood loss"""
        mask = (token_vec != self.token_vocab.lookup_token(self.pad)).to(self.device)
        token_vec = token_vec.transpose(0, 1)
        embedded_vec = self.word_embeds(token_vec)
        feats = self._get_lstm_features(embedded_vec, seq_len)

        forward_score = self._forward_alg(feats, mask)
        gold_score = self._score_sentence(feats, tag_vec, mask)
        return forward_score - gold_score

    def forward(self, token_vec, tag_vec, seq_len):
        """Forward pass: Viterbi decoding to find best path

        Args:
            token_vec: [b_s, max_seq_len]
            tag_vec: [b_s, max_seq_len]
            seq_len: [b_s]
        Returns:
            scores, tag_sequences
        """
        mask = (token_vec != self.token_vocab.lookup_token(self.pad)).to(self.device)
        token_vec = token_vec.transpose(0, 1)
        embedded_vec = self.word_embeds(token_vec)
        lstm_feats = self._get_lstm_features(embedded_vec, seq_len)

        mask = mask[:, :lstm_feats.size(1)]
        score, tag_seq = self._viterbi_decode(lstm_feats, mask, seq_len)
        return score, tag_seq

    @property
    def device(self):
        return self.word_embeds.weight.device
```

### 3. CRF 前向算法：计算配分函数 Z(x)，用于归一化所有可能路径的得分之和。

```python
def _forward_alg(self, feats, mask):
        """Forward algorithm for CRF partition function

        Args:
            feats: [b_s, seq_len, tag_size]
            mask: [b_s, seq_len]
        Returns:
            [b_s] partition function scores
        """
        init_alphas = torch.full((feats.size(0), self.tagset_size), -10000., device=self.device)
        init_alphas[:, self.begin_tag_idx] = 0.

        forward_var_list = []
        forward_var_list.append(init_alphas)
        d = torch.unsqueeze(feats[:, 0], dim=1)
        for feat_index in range(1, feats.size(1)):
            n_unfinish = mask[:, feat_index].sum()
            d_uf = d[:n_unfinish]
            emit_and_transition = feats[:n_unfinish, feat_index].unsqueeze(dim=1) + self.transition
            log_sum = d_uf.transpose(1, 2) + emit_and_transition
            max_v = log_sum.max(dim=1)[0].unsqueeze(dim=1)
            log_sum = log_sum - max_v
            d_uf = max_v + torch.logsumexp(log_sum, dim=1).unsqueeze(dim=1)
            d = torch.cat((d_uf, d[n_unfinish:]), dim=0)
        d = d.squeeze(dim=1)
        max_d = d.max(dim=-1)[0]
        d = max_d + torch.logsumexp(d - max_d.unsqueeze(dim=1), dim=1)
        return d
```

- **（1）CRF 层在本实验中的作用是什么？**

CRF 层通过可学习的转移矩阵，对相邻标签之间的转移概率进行建模。例如，B-PER 到 I-PER 的转移得分会较高，而 B-PER 到 I-LOC 的转移得分会被压低。

这使得模型在解码时能够利用全局信息找到最优标签序列，而非逐 token 独立决策，从而有效提升实体边界的准确性。

- **（2）Viterbi 解码与贪心解码的区别是什么？**

贪心解码对每个位置独立选择概率最高的标签，不考虑标签之间的依赖关系，容易产生不合法的标签序列。

Viterbi 解码则通过动态规划在所有可能的标签序列中找到全局最优解，保证输出序列在转移矩阵约束下的得分最大。

实验数据记录：

### 1. 数据集为 CoNLL2003，包含 train/valid/test 三个划分。

### 2. 模型参数：embedding_dim=100，hidden_dim=50，batch_size=32，num_epochs=50。

### 3. 使用 Adam 优化器，初始学习率 0.001，ReduceLROnPlateau 衰减策略。

### 4. Early stopping 在验证损失连续 5 个 epoch 不改善时触发。

### 5. 测试结果通过 conlleval.pl 评估 precision、recall 和 F1。

问题讨论：

问题：BiLSTM+CRF 模型相比纯 BiLSTM 的改进。

现象描述：CRF 层引入转移矩阵后，模型能够学习标签间的合法转移模式，避免产生 B-PER → I-LOC 等非法序列。

原因分析：纯 BiLSTM 对每个 token 独立预测标签，无法利用标签间的依赖关系；CRF 通过全局归一化建模了这种依赖。

解决方法：本实验已实现 BiLSTM+CRF 架构，后续可尝试引入 GloVe 预训练词向量进一步提升性能。
