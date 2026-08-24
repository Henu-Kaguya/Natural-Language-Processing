"""
Joint entity-relation extraction model.
Based on the tagging schema from chapter08/8.2.3, adapted for PyTorch.

Architecture:
    BERT Encoder
        ├── [CLS] pooled output → Predicate Head (multi-label classification)
        └── sequence output → Token Label Head (BIO sequence labeling)
"""
import torch
import torch.nn as nn
from transformers import BertModel


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
