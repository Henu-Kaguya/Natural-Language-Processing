"""
Data processor for joint entity-relation extraction.
Based on the tagging schema approach from chapter08/8.2.3.
Handles: JSONL reading, BERT tokenization, BIO labeling, predicate encoding.
"""
import json
import os
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


# Entity types and relation predicates (matching training data)
ENTITY_TYPES = ['PER', 'ORG', 'LOC']
PREDICATE_TYPES = ['works_for', 'educated_at', 'lives_in', 'located_in', 'visits', 'founder_of']

# BIO token labels
SPECIAL_TOKENS = ['[Padding]', '[##WordPiece]', '[CLS]', '[SEP]']


def get_token_label_list():
    """Build BIO label list: special tokens + B/I for each entity type + O"""
    labels = list(SPECIAL_TOKENS)
    for etype in ENTITY_TYPES:
        labels.append("B-" + etype)
        labels.append("I-" + etype)
    labels.append("O")
    return labels


def get_predicate_label_list():
    return list(PREDICATE_TYPES)


class JointExtractionDataset(Dataset):
    """Dataset for joint entity relation extraction."""

    def __init__(self, data_dir, split, tokenizer, max_seq_length=128):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.token_label_list = get_token_label_list()
        self.predicate_label_list = get_predicate_label_list()

        self.token_label_map = {label: i for i, label in enumerate(self.token_label_list)}
        self.predicate_label_map = {label: i for i, label in enumerate(self.predicate_label_list)}

        self.examples = []
        self._load_data(data_dir, split)

    def _load_data(self, data_dir, split):
        """Load JSONL data and convert to model inputs."""
        filepath = os.path.join(data_dir, f"{split}.jsonl")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                example = self._process_example(data)
                if example is not None:
                    self.examples.append(example)

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
        }

    def _find_sublist(self, tokens, entity_tokens):
        """Find entity_tokens as a contiguous sublist within tokens."""
        if not entity_tokens:
            return None
        for i in range(len(tokens) - len(entity_tokens) + 1):
            if tokens[i:i + len(entity_tokens)] == entity_tokens:
                return i
        return None

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def create_data_loader(dataset, batch_size, shuffle=True):
    """Create a DataLoader with collate function for joint extraction."""
    from torch.utils.data import DataLoader

    def collate_fn(batch):
        return {
            'input_ids': torch.stack([b['input_ids'] for b in batch]),
            'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
            'token_label_ids': torch.stack([b['token_label_ids'] for b in batch]),
            'predicate_ids': torch.stack([b['predicate_ids'] for b in batch]),
            'texts': [b['text'] for b in batch],
            'tokens_list': [b['tokens'] for b in batch],
            'spo_lists': [b['spo_list'] for b in batch],
        }

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
