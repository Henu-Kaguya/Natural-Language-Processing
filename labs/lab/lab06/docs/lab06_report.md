# 实验六：基于预训练模型的实体关系联合抽取

实验地点:计算机大楼606

实验目的：    理解并实现基于预训练模型的实体关系联合抽取方法，掌握 tagging schema 联合抽取思路。

实验环境（硬件和软件）  Windows 11，Python 3.12，PyTorch，HuggingFace Transformers

实验内容：

本实验基于 tagging schema 的联合抽取思路，使用 BERT 预训练模型实现了实体关系联合抽取系统。

核心思想是用单个模型同时完成两个任务：token 级的 BIO 序列标注（实体识别）和句子级的多标签分类（关系预测），两个任务共享 BERT 编码器。

与 pipeline 方法（先识别实体再抽取关系）不同，联合抽取让实体识别和关系分类在训练时相互促进，避免了级联错误传播。

训练完成后，模型能够从输入文本中同时识别出实体和它们之间的关系，通过三元组生成模块输出完整的 SPO 结构化信息。

实验步骤：

### 1. 数据处理：读取 JSONL 格式的 SPO 数据，使用 BERT tokenizer 分词，并生成 BIO 标签和谓词多标签。

```python
def _process_example(self, data):
        """Convert a single JSONL example to model inputs.

        Follows the reference code's BIO labeling approach:
        1. Tokenize text with BERT tokenizer
        2. Label entity spans with B-I-O tags
        3. Encode predicate labels as multi-hot vector
        """
        text = data['text']
        spo_list = data.get('spo_list', [])

        # Tokenize with BERT
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_length - 2:  # Reserve for [CLS] and [SEP]
            tokens = tokens[:self.max_seq_length - 2]

        # Initialize token labels as O
        token_labels = ['O'] * len(tokens)

        # Label entities with BIO tags
        for spo in spo_list:
            for role, entity_text, entity_type in [
                ('subject', spo['subject'], spo['subject_type']),
                ('object', spo['object'], spo['object_type'])
            ]:
                entity_tokens = self.tokenizer.tokenize(entity_text)
                # Find entity tokens in text tokens
                start_idx = self._find_sublist(tokens, entity_tokens)
                if start_idx is not None:
                    token_labels[start_idx] = "B-" + entity_type
                    for j in range(1, len(entity_tokens)):
                        token_labels[start_idx + j] = "I-" + entity_type

        # Mark WordPiece continuation tokens
        for idx, token in enumerate(tokens):
            if token.startswith("##"):
                token_labels[idx] = "[##WordPiece]"

        # Build input IDs with [CLS] and [SEP]
        final_tokens = ['[CLS]'] + tokens + ['[SEP]']
        final_labels = ['[CLS]'] + token_labels + ['[SEP]']

        input_ids = self.tokenizer.convert_tokens_to_ids(final_tokens)
        attention_mask = [1] * len(input_ids)

        # Token label IDs
        token_label_ids = [self.token_label_map.get(l, self.token_label_map['O']) for l in final_labels]

        # Pad
        padding_length = self.max_seq_length - len(input_ids)
        input_ids += [0] * padding_length
        attention_mask += [0] * padding_length
        token_label_ids += [self.token_label_map['[Padding]']] * padding_length

        # Predicate multi-hot encoding
        predicate_ids = [0] * len(self.predicate_label_list)
        for spo in spo_list:
            pred = spo['predicate']
            if pred in self.predicate_label_map:
                predicate_ids[self.predicate_label_map[pred]] = 1

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_label_ids': torch.tensor(token_label_ids, dtype=torch.long),
            'predicate_ids': torch.tensor(predicate_ids, dtype=torch.float),
            'text': text,
            'tokens': tokens,
            'spo_list': spo_list,
```

### 2. 模型定义：BERT 编码器 + 双任务输出头（token label head 和 predicate head）。

```python
class JointExtractionModel(nn.Module):
    """Joint model for entity recognition and relation extraction.

    Two output heads sharing a BERT encoder:
    1. Token label head: sequence labeling for entity BIO tags
    2. Predicate head: multi-label classification for relation types
    """

    def __init__(self, bert_model_name, num_token_labels, num_predicates,
                 dropout=0.1):
        super(JointExtractionModel, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        # Token label head: sequence labeling (BIO tags)
        self.token_label_classifier = nn.Linear(hidden_size, num_token_labels)

        # Predicate head: multi-label classification (relation types)
        self.predicate_classifier = nn.Linear(hidden_size, num_predicates)

        # Loss functions
        self.token_label_criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore [Padding]
        self.predicate_criterion = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, token_label_ids=None,
                predicate_ids=None):
        """
        Args:
            input_ids: [batch_size, seq_length]
            attention_mask: [batch_size, seq_length]
            token_label_ids: [batch_size, seq_length] (for training)
            predicate_ids: [batch_size, num_predicates] (for training)

        Returns:
            dict with 'token_label_logits', 'predicate_logits', and optionally 'loss'
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs.last_hidden_state   # [batch, seq_len, hidden]
        pooled_output = outputs.pooler_output          # [batch, hidden]

        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        # Token label prediction
        token_label_logits = self.token_label_classifier(sequence_output)
        # [batch, seq_len, num_token_labels]

        # Predicate prediction
        predicate_logits = self.predicate_classifier(pooled_output)
        # [batch, num_predicates]

        result = {
            'token_label_logits': token_label_logits,
            'predicate_logits': predicate_logits,
        }

        # Compute loss if labels provided
        if token_label_ids is not None and predicate_ids is not None:
            # Token label loss
            active_loss = attention_mask.view(-1) == 1
            active_logits = token_label_logits.view(-1, token_label_logits.size(-1))[active_loss]
            active_labels = token_label_ids.view(-1)[active_loss]
            token_label_loss = self.token_label_criterion(active_logits, active_labels)

            # Predicate loss
            predicate_loss = self.predicate_criterion(predicate_logits, predicate_ids)

            # Combined loss
            total_loss = token_label_loss + predicate_loss
            result['loss'] = total_loss
            result['token_label_loss'] = token_label_loss.item()
            result['predicate_loss'] = predicate_loss.item()

        return result
```

### 3. 三元组生成：从预测的 BIO 标签提取实体，与预测的关系类型匹配生成 SPO 三元组。

```python
def extract_entities(tokens, token_labels, token_label_list):
    """Extract entities from BIO-tagged token sequence.

    Args:
        tokens: list of token strings
        token_labels: list of predicted label IDs (integers)
        token_label_list: mapping from ID to label string

    Returns:
        list of entity dicts with text, type, start, end
    """
    entities = []
    current_entity = None
    o_label = 'O'

    for i, (token, label_id) in enumerate(zip(tokens, token_labels)):
        if i >= len(tokens):
            break
        label = token_label_list[label_id] if label_id < len(token_label_list) else o_label

        # Skip special tokens
        if label in SPECIAL_TOKENS or label == '[##WordPiece]':
            continue

        if label.startswith("B-"):
            # Save previous entity
            if current_entity is not None:
                entities.append(current_entity)
            entity_type = label[2:]
            current_entity = {
                'text': token.replace('##', ''),
                'type': entity_type,
                'start': i,
                'end': i,
            }
        elif label.startswith("I-") and current_entity is not None:
            entity_type = label[2:]
            if entity_type == current_entity['type']:
                current_entity['text'] += token.replace('##', '')
                current_entity['end'] = i
            else:
                entities.append(current_entity)
                current_entity = None
        else:
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None

    if current_entity is not None:
        entities.append(current_entity)

    return entities
```

- **（1）联合抽取相比 pipeline 方法有什么优势？**

Pipeline 方法先做实体识别再做关系分类，两个模型独立训练，实体识别的错误会级联传播到关系抽取阶段。

联合抽取让实体识别和关系分类共享编码器，在训练时相互促进。例如，关系标签的学习可以帮助模型更好地区分不同语境下的实体边界。

- **（2）为什么谓词预测使用多标签分类而非多分类？**

一个句子中可能同时存在多种关系类型（如'李雷在华为工作，毕业于北京大学'同时含 works_for 和 educated_at），因此需要对每个谓词独立做二分类判断。

模型使用 Sigmoid + BCEWithLogitsLoss，对每个谓词输出 0/1 预测，而非 Softmax + CrossEntropy 的互斥多分类。

实验数据记录：

### 1. 训练数据 25 条，验证数据 8 条，JSONL 格式。

### 2. 实体类型 3 种：PER（人名）、ORG（机构）、LOC（地点）。

### 3. 关系类型 6 种：works_for、educated_at、lives_in、located_in、visits、founder_of。

### 4. 模型使用 bert-base-chinese 作为编码器，max_seq_length=128。

### 5. 训练参数：epochs=10，batch_size=8，learning_rate=2e-5，AdamW + linear warmup。

问题讨论：

问题：联合抽取模型在小规模数据上的效果受限。

现象描述：当前训练数据仅 25 条，模型学到的实体和关系模式有限，对新样本的泛化能力不足。

原因分析：BERT 预训练模型虽然具有强大的语言理解能力，但微调阶段仍需要足够的任务数据来学习特定的标注模式。

解决方法：扩充训练数据规模，增加更多实体和关系类型，使用更大的预训练模型（如 RoBERTa）或引入数据增强策略。
