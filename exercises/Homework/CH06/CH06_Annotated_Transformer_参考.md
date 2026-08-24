# 第6章参考：The Annotated Transformer 代码注解

2023xxxx04-同学D-第5章作业

Transformer 执行过程分析

—— 基于 "The Annotated Transformer" 代码实现的逐行注释详解

论文来源：Attention is All You Need (Vaswani et al., 2017)

代码来源：Harvard NLP Annotated Transformer (v2022)

代码链接：https://github.com/harvardnlp/annotated-transformer

## 一、整体架构概览

Transformer 的完整执行流程分为三个层次：

数据输入 → Embedding + PositionalEncoding → Encoder(×N) → Decoder(×N) → Generator → 输出概率分布

代码核心组件对应关系：

| 论文中的概念 | 代码类/函数 | 文件位置(行号) |
| --- | --- | --- |
| Scaled Dot-Product Attention | attention() | 第519行 |
| Multi-Head Attention | MultiHeadedAttention | 第589行 |
| Position-wise FFN | PositionwiseFeedForward | 第677行 |
| Positional Encoding | PositionalEncoding | 第748行 |
| Encoder | Encoder + EncoderLayer | 第292行 / 第367行 |
| Decoder | Decoder + DecoderLayer | 第390行 / 第413行 |
| Layer Normalization | LayerNorm | 第315行 |
| Residual Connection | SublayerConnection | 第344行 |
| Label Smoothing | LabelSmoothing | 第1148行 |
| Learning Rate Schedule | rate() | 第1058行 |
| Embedding | Embeddings | 第704行 |
| Output Generator | Generator | 第256行 |

## 二、逐组件执行过程详解

### 2.1 输入处理：Embedding + PositionalEncoding

【Embeddings — 第704行】

执行流程：

Step 1: 将输入的 token 索引 (batch, seq_len) 通过 nn.Embedding 查表，得到 (batch, seq_len, d_model)。

Step 2: 乘以 sqrt(d_model)，这是论文中的关键操作——防止 embedding 的方差过小。embedding 初始值很小，乘以 sqrt(512) ≈ 22.6 将其放大到与 positional encoding 同量级。

Step 3: 加上位置编码后，整体方差保持在一个合适的范围。

```python
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)  # 词嵌入表，(vocab_size, d_model)
        self.d_model = d_model                    # 模型维度，默认512

    def forward(self, x):
        # x: (batch_size, seq_len) —— token的整数索引
        # self.lut(x): (batch_size, seq_len, d_model) —— 查表得到稠密向量
        # math.sqrt(self.d_model): sqrt(512) ≈ 22.627
        # 缩放原因：embedding 初始化为 N(0,1)，方差为1；
        #   PositionalEncoding 的值在 [-1, 1] 之间，方差约 0.5；
        #   乘以 sqrt(d_model) 后 embedding 方差变为 d_model=512，
        #   这样相加后两者量级匹配，避免其中一个主导初始表示。
        return self.lut(x) * math.sqrt(self.d_model)
```

【PositionalEncoding — 第748行】

执行流程：

Step 1: 在 __init__ 中预计算位置编码矩阵 pe: (1, max_len, d_model)，形状为 (1, 5000, 512)。

Step 2: 使用正弦/余弦函数计算每个位置每个维度的编码值：

PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))

PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Step 3: forward 中将 pe 加到 embedding 上，对每个位置注入位置信息。

设计精妙之处：sin/cos 的性质使 PE(pos+k) 可表示为 PE(pos) 的线性组合，利于学习相对位置；不同维度有不同频率（波长从 2π 到 20000π），类似二进制编码，确保每个位置有唯一表示。

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 预计算位置编码矩阵
        pe = torch.zeros(max_len, d_model)              # (5000, 512) 全零初始化
        position = torch.arange(0, max_len).unsqueeze(1)  # (5000, 1) 位置索引

        # div_term: (256,) —— 计算 1/10000^(2i/d_model) 的序列
        # 用 exp-log 技巧替代直接指数运算，数值更稳定：
        #   exp(-(log(10000)/d_model) * (2i)) = 1/10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 偶数列 (0,2,4,...) 用 sin，奇数列 (1,3,5,...) 用 cos
        pe[:, 0::2] = torch.sin(position * div_term)   # (5000, 1)*(256,) → (5000, 256)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, 5000, 512) —— 增加 batch 维度用于广播
        self.register_buffer("pe", pe)  # 不作为可训练参数但随模型保存/加载

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        # 截取与输入序列等长的位置编码并相加
        # requires_grad_(False): 位置编码不参与梯度计算（固定值）
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)  # 加 dropout 防止过拟合
```

### 2.2 核心：Scaled Dot-Product Attention（第519行）

这是 Transformer 的核心计算单元，完整实现了 Attention(Q,K,V) = softmax(QK^T/√d_k) × V。

执行流程详解：

Step 1 — 计算相似度得分：scores = Q @ K^T，得到 (batch, h, seq_q, seq_k) 的矩阵。scores[i,j] 表示第 i 个 query 与第 j 个 key 的原始相似度（内积）。

Step 2 — 缩放：scores / √d_k。因为 d_k=64，√d_k=8。当 d_k 较大时点积值很大，会将 softmax 推到梯度极小的饱和区，缩放可缓解此问题。

Step 3 — 掩码：mask==0 的位置 score 设为 -1e9。Decoder 中屏蔽未来位置实现自回归；屏蔽 padding 位置防止无意义的填充词影响注意力。

Step 4 — Softmax：将相似度归一化为概率分布（注意力权重），每行之和为 1。

Step 5 — 加权求和：p_attn @ V，用注意力权重对 value 进行加权平均。每个位置的输出综合了所有位置的信息。

```python
def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'

    Q: "我要找什么"（查询向量）—— 描述当前位置需要什么信息
    K: "我是什么"（键向量）—— 描述每个位置提供了什么标签
    V: "我包含什么信息"（值向量）—— 每个位置的实际内容
    注意力权重 = softmax(QK^T/√d_k): 描述每个 query 应该关注每个 key 的程度
    """
    d_k = query.size(-1)  # 每个头的维度 = d_model/h = 512/8 = 64

    # (batch, h, seq_len, d_k) @ (batch, h, d_k, seq_len)
    # → scores: (batch, h, seq_len, seq_len)
    # 物理意义：scores[i,j] = query_i 与 key_j 的内积相似度
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # mask 中为 0 的位置 → scores 设为 -1e9
        # 原因：-1e9 经过 softmax 后 ≈ 0，等价于"不能关注该位置"
        # Decoder 中：将未来位置的 score 设为 -1e9 → 实现自回归约束
        # Padding 中：将填充位置的 score 设为 -1e9 → 防止 padding 影响注意力
        scores = scores.masked_fill(mask == 0, -1e9)

    # softmax 沿最后一维 (key方向) 归一化
    # 每个 query 对各个 key 的注意力权重之和为 1
    p_attn = scores.softmax(dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    # (batch, h, seq_len, seq_len) @ (batch, h, seq_len, d_k)
    # → (batch, h, seq_len, d_k)
    # 物理意义：每个位置的输出 = 对所有位置 V 的加权求和
    # 权重由注意力分布 p_attn 决定——越"相关"的位置贡献越大
    return torch.matmul(p_attn, value), p_attn
```

### 2.3 Multi-Head Attention（第589行）

执行流程：

Step 1 — 线性投影：将输入的 Q, K, V 分别通过 3 个不同的线性层（d_model → d_model），得到投影后的 Q', K', V'。

Step 2 — 拆分多头：将投影结果 reshape 为 (batch, seq, h=8, d_k=64)，再 transpose 为 (batch, h=8, seq, d_k=64)。每个头在 d_k=64 维的子空间中独立计算注意力。

Step 3 — 并行计算：对 8 个头各自调用 attention()，每个头学习不同的注意力模式。

Step 4 — Concat：将 8 个头的结果 (batch, h, seq, d_k) 拼接回 (batch, seq, d_model=512)。

Step 5 — 输出投影：通过第 4 个线性层 W_O 对拼接结果进行整合投影。

为什么需要多头？单头注意力的平均效应会抑制不同表示子空间的表达能力。多头允许模型同时关注不同位置的不同表示子空间——例如：头1学习"主谓关系"，头2学习"修饰关系"，头3学习"指代关系"，头4学习"语序信息"等。

```python
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0  # 必须能整除，如 512 % 8 = 0
        self.d_k = d_model // h  # 每个头的维度 512/8 = 64
        self.h = h               # 头数 = 8

        # 4 个线性层：
        #   前 3 个分别投影 Q, K, V（各自有独立的 W^Q, W^K, W^V）
        #   第 4 个是输出投影 W^O
        # clones 深拷贝同一个模块 N 次——注意每个 Linear 有独立权重，不共享
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None  # 保存最后的注意力权重（用于可视化）
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # query, key, value: 均为 (batch, seq, d_model)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, 1, seq) → (batch, 1, 1, seq)
        nbatches = query.size(0)

        # === 步骤1: 线性投影 + 拆分为多头 ===
        # 以 Q 为例：(batch, seq, d_model)
        #   → Linear(d_model, d_model) → (batch, seq, d_model)
        #   → view(nbatches, -1, h, d_k) → (batch, seq, 8, 64)
        #   → transpose(1, 2) → (batch, 8, seq, 64)
        # K, V 同理；三者在列表中分别通过 self.linears[0], [1], [2] 投影
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # === 步骤2: 并行计算 8 个头的注意力 ===
        # attention() 的输入输出维度：
        #   输入 Q,K,V: (batch, 8, seq, 64)
        #   输出 x: (batch, 8, seq, 64)
        #   输出 attn: (batch, 8, seq, seq)  注意力权重矩阵
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # === 步骤3: Concat（将8个头拼接回512维）===
        # (batch, 8, seq, 64) → transpose(1,2) → (batch, seq, 8, 64)
        # → contiguous() → view → (batch, seq, 512)
        # contiguous() 确保内存在转置后连续，view 要求连续内存布局
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query; del key; del value  # 手动释放中间变量，节省显存

        # === 步骤4: 最终线性投影 W^O ===
        # self.linears[-1] 即 self.linears[3]
        # (batch, seq, 512) → (batch, seq, 512)
        return self.linears[-1](x)
```

### 2.4 Position-wise Feed-Forward（第677行）

FFN(x) = max(0, xW1 + b1) × W2 + b2，即 512 → 2048 → ReLU → 512。"Position-wise" 意味着对每个位置独立应用相同的变换（等价于 kernel_size=1 的卷积）。第一层升维（512→2048）增加模型容量和非线性表达力，第二层降维还原（2048→512）。ReLU 引入非线性，使网络能拟合任意复杂函数。

```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)     # 升维：512 → 2048
        self.w_2 = nn.Linear(d_ff, d_model)     # 降维：2048 → 512
        self.dropout = nn.Dropout(dropout)       # Dropout 防止过拟合

    def forward(self, x):
        # x: (batch, seq, 512)
        # w_1(x): (batch, seq, 2048)
        # .relu(): 逐元素 max(0, x)
        # dropout: 随机关闭部分神经元
        # w_2: (batch, seq, 512) —— 还原到 d_model 维
        return self.w_2(self.dropout(self.w_1(x).relu()))
```

### 2.5 SublayerConnection：残差连接 + LayerNorm（第344行）

这是 EncoderLayer 和 DecoderLayer 内部的核心连接范式。公式为：LayerNorm(x + Dropout(Sublayer(x)))。注意代码实现是先 Norm 再 Sublayer（Pre-LN 范式），与论文原本的 Post-LN（先 Sublayer 再 Norm）不同。Pre-LN 的优势：训练更稳定，梯度流动更顺畅，不需要 warmup 也能收敛。

```python
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)         # 层归一化：沿 d_model 维标准化
        self.dropout = nn.Dropout(dropout)  # Dropout 正则化

    def forward(self, x, sublayer):
        # sublayer 是一个可调用的子层函数（Self-Attention 或 FFN）
        # 执行顺序（Pre-LN）：Norm → SubLayer → Dropout → 残差相加
        # x + Dropout(sublayer(Norm(x)))
        # 残差连接的作用：让梯度能直接传播到浅层，缓解深层网络的梯度消失
        return x + self.dropout(sublayer(self.norm(x)))

class LayerNorm(nn.Module):
    """沿最后一个维度（d_model=512）计算均值和标准差进行归一化"""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))   # 可学习的缩放参数 γ
        self.b_2 = nn.Parameter(torch.zeros(features))  # 可学习的平移参数 β
        self.eps = eps  # 防止除零的小常数

    def forward(self, x):
        # x: (batch, seq, d_model)
        # mean, std: (batch, seq, 1) —— keepdim=True 保证广播
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # y = γ * (x - μ)/σ + β —— 标准归一化后做仿射变换
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

### 2.6 EncoderLayer 和 Encoder

【EncoderLayer — 第367行】

一个 EncoderLayer 包含 2 个子层：(1) Multi-Head Self-Attention（Q=K=V=自身输入），(2) Position-wise FFN。每个子层都用 SublayerConnection 包裹（残差 + LayerNorm + Dropout）。

```python
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn          # MultiHeadedAttention 实例
        self.feed_forward = feed_forward    # PositionwiseFeedForward 实例
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size  # 512

    def forward(self, x, mask):
        # 子层1: Multi-Head Self-Attention
        # Q=K=V=x —— 输入序列中每个 token 都能关注所有其他 token（双向注意力）
        # mask 用于屏蔽 padding 位置
        # lambda 延迟调用：先由 SublayerConnection 执行 Norm(x)，再传入 sublayer
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))

        # 子层2: Position-wise Feed-Forward
        # 对每个位置独立进行非线性变换（不涉及位置间的信息交互）
        return self.sublayer[1](x, self.feed_forward)
```

【Encoder — 第292行】

完整编码器结构：Embedding+PE → EncoderLayer×6 → LayerNorm。逐层传递，每层的输出作为下一层的输入，逐步提取更高层的语义特征。

```python
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)    # N=6 个完全相同的 EncoderLayer
        self.norm = LayerNorm(layer.size)  # 最后的 LayerNorm（Post-LN 的残余）

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)  # 串行堆叠，逐层提炼特征
        return self.norm(x)
```

### 2.7 DecoderLayer 和 Decoder

【DecoderLayer — 第413行】

DecoderLayer 包含 3 个子层（比 Encoder 多一个 Cross-Attention），这是连接 Encoder 和 Decoder 的关键桥梁：

- **(1) Masked Multi-Head Self-Attention：只能看到当前位置及之前的内容（通过 subsequent_mask 实现因果约束）。**

- **(2) Cross-Attention：Q 来自 Decoder 当前输出，K, V 来自 Encoder 的 memory 输出。让 Decoder 能"查阅"源语言信息。**

- **(3) Position-wise FFN：同 Encoder。**

```python
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn        # Self-Attention（带 causal mask）
        self.src_attn = src_attn          # Cross-Attention（关注 Encoder 输出）
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory  # encoder 的输出，(batch, src_seq, d_model)

        # 子层1: Masked Self-Attention
        # Q=K=V=x，通过 tgt_mask 屏蔽未来位置（上三角 mask）
        # tgt_mask 确保位置 i 只能关注位置 < i 的内容，实现自回归生成
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))

        # 子层2: Cross-Attention（Encoder-Decoder 注意力）
        # Q 来自 Decoder（"当前翻译到哪了，需要什么信息"）
        # K, V 来自 Encoder 的 memory（"源语言每个词的含义"）
        # src_mask 屏蔽 encoder 输出中的 padding 位置
        # 这就是"对齐"过程——Decoder 学会关注源语言的相关部分
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))

        # 子层3: Feed-Forward（同 Encoder）
        return self.sublayer[2](x, self.feed_forward)
```

### 2.8 后续掩码（Subsequent Mask）—— 第441行

构造上三角矩阵实现因果掩码。例如 size=4 时的掩码矩阵：

[[1, 0, 0, 0],

[1, 1, 0, 0],

[1, 1, 1, 0],

[1, 1, 1, 1]]

第 i 行第 j 列 = 1 表示位置 i 可以关注位置 j。在 attention 中 mask==0 的位置 score 被设为 -1e9，softmax 后 ≈ 0，等价于完全不能关注。这样就实现了自回归约束——生成位置 i 时只能使用位置 0~i-1 的信息。

```python
def subsequent_mask(size):
    attn_shape = (1, size, size)  # (batch=1, seq, seq)
    # torch.triu(x, diagonal=1): 取严格上三角（不含对角线），非零元素=1
    # .type(torch.uint8): 转为无符号8位整数
    # 结果：上三角为1，下三角和对角线为0
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    # == 0: 翻转 —— 下三角和对角线为 True(可见)，上三角为 False(被屏蔽)
    return subsequent_mask == 0
    def subsequent_mask(size):
        attn_shape = (1, size, size)  # (batch=1, seq, seq)
        # torch.triu(x, diagonal=1): 取严格上三角（不含对角线），非零元素=1
        # .type(torch.uint8): 转为无符号8位整数
        # 结果：上三角为1，下三角和对角线为0
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
        # == 0: 翻转 —— 下三角和对角线为 True(可见)，上三角为 False(被屏蔽)
        return subsequent_mask == 0
```

### 2.9 顶层架构：EncoderDecoder + Generator

【EncoderDecoder — 第230行】

这是完整的 Transformer 模型类，将 Encoder、Decoder、Embedding、Generator 组合在一起：

```python
class EncoderDecoder(nn.Module):
```

```python
def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
```

super(EncoderDecoder, self).__init__()

self.encoder = encoder           # Encoder 实例，N=6 层

self.decoder = decoder           # Decoder 实例，N=6 层

self.src_embed = src_embed       # src: nn.Sequential(Embedding + PositionalEncoding)

self.tgt_embed = tgt_embed       # tgt: nn.Sequential(Embedding + PositionalEncoding)

self.generator = generator       # 输出层：Linear(d_model, vocab) + log_softmax

```python
def forward(self, src, tgt, src_mask, tgt_mask):
```

"""完整的前向传播：编码 → 解码"""

```python
return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
```

```python
def encode(self, src, src_mask):
```

# src: (batch, src_seq) 源语言 token 索引

# 1. src_embed = Embeddings → *√d_model → +PositionalEncoding → Dropout

# 2. Encoder 逐层处理：每层 Self-Attention(双向) + FFN，带残差连接

# 输出 memory: (batch, src_seq, d_model) —— 源语言每个位置的深层语义表示

```python
return self.encoder(self.src_embed(src), src_mask)
```

```python
def decode(self, memory, src_mask, tgt, tgt_mask):
```

# memory: Encoder 的输出，(batch, src_seq, d_model)

# tgt: (batch, tgt_seq) 目标语言 token 索引

# 1. tgt_embed = Embeddings → *√d_model → +PositionalEncoding → Dropout

# 2. Decoder 逐层处理：

#    - Masked Self-Attention（只能看已生成内容，保证自回归）

#    - Cross-Attention（Query=Decoder状态，Key/Value=Encoder memory）

#    - FFN

# 输出: (batch, tgt_seq, d_model)

```python
return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
```

【Generator — 第256行】

将 decoder 输出的 d_model 维向量投影到词表大小，再取 log_softmax 得到对数概率分布：

```python
class Generator(nn.Module):
```

```python
def __init__(self, d_model, vocab):
```

super(Generator, self).__init__()

self.proj = nn.Linear(d_model, vocab)  # 512 → vocab_size

```python
def forward(self, x):
```

# x: (batch, seq, d_model)

# self.proj(x): (batch, seq, vocab_size) —— 每个位置产生 vocab 个 logit

# log_softmax(dim=-1): 将 logit 转为对数概率

# 使用 log_softmax（而非 softmax）是为了配合 KLDivLoss / NLLLoss

```python
return log_softmax(self.proj(x), dim=-1)
```

### 2.10 模型构建工厂函数：make_model() — 第822行

将所有组件组装成完整 Transformer 模型，并进行 Xavier 参数初始化：

```python
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
```

"""从超参数构建完整 Transformer 模型"""

c = copy.deepcopy  # 深拷贝，确保每个层有独立的参数

# 构建可复用的基础组件

attn = MultiHeadedAttention(h, d_model)              # 多头注意力 (h=8头)

ff = PositionwiseFeedForward(d_model, d_ff, dropout) # FFN (512→2048→512)

position = PositionalEncoding(d_model, dropout)       # 正弦位置编码

model = EncoderDecoder(

# Encoder: 6 层，每层含 Self-Attn(双向) + FFN

Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),

# Decoder: 6 层，每层含 Self-Attn(因果) + Cross-Attn + FFN

# 注意：DecoderLayer 需要两个 attention 实例(一个self, 一个cross)

Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),

# src 端输入处理: Embedding + PositionalEncoding

nn.Sequential(Embeddings(d_model, src_vocab), c(position)),

# tgt 端输入处理: Embedding + PositionalEncoding

nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),

# 输出层: Linear(d_model, tgt_vocab) + log_softmax

Generator(d_model, tgt_vocab),

)

# Xavier(也称作 Glorot) 均匀初始化

# 对 dim>1 的张量（权重矩阵）进行初始化，偏置项(dim=1)保持原样

# 作用：保持前向和反向传播中各层的激活和梯度方差稳定

for p in model.parameters():

if p.dim() > 1:

nn.init.xavier_uniform_(p)

```python
return model
```

## 三、训练流程详解

### 3.1 Batch 构造（第907行）

Batch 类负责组织训练数据、构造各种掩码、处理 teacher forcing：

```python
class Batch:
```

```python
def __init__(self, src, tgt=None, pad=2):
```

self.src = src  # (batch, src_seq) 源语言序列

# src_mask: (batch, 1, src_seq) —— 扩展到4D后为 (batch, 1, 1, src_seq)

# 用于屏蔽 Encoder 输入中的 padding token

self.src_mask = (src != pad).unsqueeze(-2)

if tgt is not None:

# Teacher Forcing: 用真实的前缀预测下一个token

# 例如 tgt = [<BOS>, I, love, NLP, <EOS>]

#   self.tgt   = [<BOS>, I, love, NLP]      模型输入（去掉最后一个）

#   self.tgt_y = [I,    love, NLP, <EOS>]    预测目标（去掉第一个）

self.tgt = tgt[:, :-1]     # 去掉<EOS>，作为decoder输入

self.tgt_y = tgt[:, 1:]    # 去掉<BOS>，作为预测标签

# tgt_mask: 组合 mask = padding_mask & subsequent_mask

# 同时屏蔽 padding token 和未来 token

self.tgt_mask = self.make_std_mask(self.tgt, pad)

self.ntokens = (self.tgt_y != pad).data.sum()  # 有效token总数

@staticmethod

```python
def make_std_mask(tgt, pad):
```

# 两步 mask 组合（按位与 &）：

# 1. padding mask: (tgt != pad).unsqueeze(-2)

#    屏蔽所有 <blank> token，防止模型关注无意义的填充

# 2. subsequent mask: 上三角矩阵

#    屏蔽未来 token，保证自回归性质

tgt_mask = (tgt != pad).unsqueeze(-2)

tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)

```python
return tgt_mask
```

### 3.2 训练循环：run_epoch() — 第949行

完整的训练循环实现，包含前向传播、损失计算、反向传播、梯度累积、学习率调度和日志输出：

```python
def run_epoch(data_iter, model, loss_compute, optimizer, scheduler,
```

mode="train", accum_iter=1, train_state=TrainState()):

start = time.time()

total_tokens = 0; total_loss = 0; tokens = 0; n_accum = 0

for i, batch in enumerate(data_iter):

# === 前向传播 ===

# batch.src → Embedding+PE → Encoder(6层) → memory

# batch.tgt → Embedding+PE → Decoder(6层, memory) → 解码输出

out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)

# === 损失计算 ===

# out: (batch, tgt_seq, d_model)

# Generator → (batch, tgt_seq, vocab)

# 与 batch.tgt_y 计算 KL 散度损失（带 label smoothing）

loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)

if mode == "train" or mode == "train+log":

# === 反向传播 ===

loss_node.backward()  # 计算梯度（累积在 .grad 中）

# 梯度累积：每 accum_iter 步才更新一次参数

# 作用：在不增加显存消耗的情况下等效增大 batch_size

# 如 batch=32, accum_iter=10 → 等效 batch=320

if i % accum_iter == 0:

optimizer.step()                        # 用累积的梯度更新参数

optimizer.zero_grad(set_to_none=True)    # 清空梯度（set_to_none 更高效）

n_accum += 1

# 学习率调度：每个 step 都更新（warmup + inverse sqrt decay）

scheduler.step()

# 统计累积

total_loss += loss

total_tokens += batch.ntokens

tokens += batch.ntokens

# 每 40 个 batch 打印一次训练日志

if i % 40 == 1 and (mode == "train" or mode == "train+log"):

lr = optimizer.param_groups[0]["lr"]

elapsed = time.time() - start

print("Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "

"| Tokens / Sec: %7.1f | Learning Rate: %6.1e"

% (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr))

start = time.time(); tokens = 0

del loss; del loss_node  # 释放计算图，节省显存

```python
return total_loss / total_tokens, train_state
```

### 3.3 学习率调度：rate() — 第1058行

论文公式：lrate = d_model^(-0.5) × min(step_num^(-0.5), step_num × warmup^(-1.5))

阶段1 (warmup)：lr ∝ step，线性增长，避免训练初期梯度不稳定导致发散。

阶段2 (decay)：lr ∝ 1/√step，逐步减小以精细收敛到最优解。

举例：d_model=512, warmup=4000, step=4000 时：warmup 阶段增长到顶点 lr ≈ 1/√512 × 4000 × 4000^(-1.5) ≈ 0.0007，此后按 1/√step 衰减。

```python
def rate(step, model_size, factor, warmup):
```

"""学习率调度函数，实现 warmup + inverse sqrt decay"""

if step == 0:

step = 1  # 避免 0 的负指数（0^(-0.5) 无定义）

# factor = 1.0

# model_size^(-0.5): 基准缩放因子

#   d_model=512 → 1/√512 ≈ 0.044

# min(...): 在 warmup 段和 decay 段之间取较小值

#   warmup 段: step*warmup^(-1.5)  — 线性增长

#   decay 段:  step^(-0.5)         — 反比于 sqrt(step)

```python
return factor * (
```

model_size ** (-0.5)

* min(step ** (-0.5), step * warmup ** (-1.5))

)

### 3.4 Label Smoothing（第1148行）

标签平滑通过 KL 散度损失实现——不直接用 one-hot 目标分布，而是构造一个"软化"的目标分布：正确类别置信度为 1-ε=0.9，其余 ε=0.1 的质量均匀分配给所有非正确类别。效果：迫使模型不那么"自信"，防止过拟合，提升泛化能力和 BLEU 分数。

```python
class LabelSmoothing(nn.Module):
```

```python
def __init__(self, size, padding_idx, smoothing=0.0):
```

super(LabelSmoothing, self).__init__()

self.criterion = nn.KLDivLoss(reduction="sum")  # KL 散度作为损失函数

self.padding_idx = padding_idx

self.confidence = 1.0 - smoothing  # 正确类的概率（如 0.9）

self.smoothing = smoothing         # 平滑系数 ε（如 0.1）

self.size = size                   # 词表大小

```python
def forward(self, x, target):
```

# x: 模型输出的 log 概率 (batch*seq, vocab_size)

# target: 真实标签 (batch*seq,)

true_dist = x.data.clone()

# 所有非正确、非 padding 类别分配平滑质量

true_dist.fill_(self.smoothing / (self.size - 2))

# 正确类别赋值 confidence

true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

# padding 类别设为 0（不参与损失计算）

true_dist[:, self.padding_idx] = 0

mask = torch.nonzero(target.data == self.padding_idx)

if mask.dim() > 0:

true_dist.index_fill_(0, mask.squeeze(), 0.0)

# KL(P||Q) = Σ_i P(i) * [log P(i) - log Q(i)]

# 其中 P 是平滑后的目标分布，Q 是模型预测的分布

```python
return self.criterion(x, true_dist.clone().detach())
```

## 四、推理：Greedy Decoding（第1313行）

自回归生成——Encoder 一次性编码源序列，Decoder 逐 token 生成目标序列：

```python
def greedy_decode(model, src, src_mask, max_len, start_symbol):
```

# Step 1: Encoder 一次性编码整个源序列

memory = model.encode(src, src_mask)  # (1, src_seq, d_model)

# Step 2: 从 <BOS> token 开始，逐 token 生成

ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)  # (1, 1) = [<BOS>]

for i in range(max_len - 1):

# Step 3: 对已生成序列做全量解码（每次都要重新计算 attention）

# subsequent_mask 确保只能看到已生成的 token

out = model.decode(

memory, src_mask,

ys,  # 当前已生成的所有 token

subsequent_mask(ys.size(1)).type_as(src.data)  # 因果掩码

)

# Step 4: 只取最后一个位置的输出预测下一个 token

# out[:, -1]: (1, d_model) —— 最新位置的表示

prob = model.generator(out[:, -1])  # (1, vocab_size) log 概率

# Step 5: 贪心选择概率最高的 token

_, next_word = torch.max(prob, dim=1)

next_word = next_word.data[0]

# Step 6: 将新 token 拼接到已有序列后面

ys = torch.cat(

[ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)],

dim=1

)

```python
return ys  # (1, max_len) —— 生成的完整序列
```

## 五、完整数据流总结

下面是 Transformer 一次完整前向传播的数据流动全貌：

输入：

src = [2, 5, 8, 1, 3]          (源语言 token 序列)

tgt = [<BOS>, 输, 出, 序, 列]   (目标语言 token 序列)

【Step 1: 输入编码】

src → Embeddings(src) → *√d_model → +PositionalEncoding

→ (batch, src_seq, 512)          —— 融入词义+位置信息的源语言表示

tgt → Embeddings(tgt) → *√d_model → +PositionalEncoding

→ (batch, tgt_seq, 512)          —— 融入词义+位置信息的目标语言表示

【Step 2: Encoder 处理（×6 层）】

每层处理 (x, src_mask):

sublayer[0]: x → LayerNorm → MultiHeadSelfAttention(Q=K=V=x, mask)

→ Dropout → +x (残差连接)

sublayer[1]: x → LayerNorm → FFN(512→2048→512)

→ Dropout → +x (残差连接)

最终输出 memory: (batch, src_seq, 512)

—— 融入了全局上下文信息的源语言深层语义表示

【Step 3: Decoder 处理（×6 层）】

每层处理 (x, memory, src_mask, tgt_mask):

sublayer[0]: x → LayerNorm → MaskedMultiHeadSelfAttention(Q=K=V=x, tgt_mask)

→ Dropout → +x

sublayer[1]: x → LayerNorm → CrossAttention(Q=x, K=memory, V=memory, src_mask)

→ Dropout → +x

sublayer[2]: x → LayerNorm → FFN(512→2048→512)

→ Dropout → +x

最终输出: (batch, tgt_seq, 512)

—— 融入了源语言信息和已生成前缀信息的目标语言表示

【Step 4: 输出生成】

Decoder 输出 → Generator(Linear + log_softmax)

→ (batch, tgt_seq, vocab_size)

—— 每个位置输出词表上的对数概率分布

【Step 5: 损失计算（训练阶段）】

Generator 输出 + LabelSmoothing(平滑目标分布)

→ KLDivLoss

→ 标量损失值

→ 反向传播 → 更新所有参数

## 六、关键超参数汇总

| 参数 | 值 | 说明 |
| --- | --- | --- |
| N | 6 | Encoder / Decoder 层数 |
| d_model | 512 | 模型所有子层的输出维度 |
| d_ff | 2048 | FFN 内部隐藏层维度 |
| h | 8 | 多头注意力头数 |
| d_k = d_v | 64 (= 512/8) | 每个注意力头的维度 |
| dropout | 0.1 | Dropout 丢弃比例 |
| batch_size | 32 | 每批样本数 |
| accum_iter | 10 | 梯度累积步数（等效 batch = 320） |
| warmup | 3000 | 学习率预热步数 |
| base_lr | 1.0 | 基础学习率 |
| label_smoothing | 0.1 | 标签平滑系数 ε |
| optimizer | Adam(β1=0.9, β2=0.98, ε=1e-9) | 优化器配置 |
| epochs | 8 | 训练轮数（Multi30k 任务） |
| max_padding | 72 | 最大填充长度 |

## 七、总结

Transformer 的核心创新在于完全用自注意力机制替代了 RNN/CNN，实现了以下几个关键突破：

### 1. 并行计算：所有位置的表示可以同时计算，不需要像 RNN 那样串行处理，训练速度大幅提升。

### 2. 长距离依赖：任意两个位置之间的信息传递只需要 O(1) 步操作（通过注意力权重直接连接），克服了 RNN 中信息随距离衰减的问题。

### 3. 多头机制：多个注意力头并行工作，允许模型在不同的表示子空间中学习不同类型的依赖关系。

### 4. 位置编码：通过正弦/余弦函数注入位置信息，使模型能够感知序列顺序，同时支持外推到训练时未见过的序列长度。

### 5. 残差连接 + LayerNorm：确保深层网络的稳定训练和高效梯度传播。
