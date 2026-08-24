"""
Evaluation for joint entity-relation extraction.
Computes token-level entity metrics and predicate-level relation metrics.
"""
import torch
import numpy as np
from data_processor import SPECIAL_TOKENS


def evaluate_model(model, data_loader, device, token_label_list, predicate_label_list):
    """Evaluate model on a dataset.

    Returns dict with precision/recall/F1 for both token labels and predicates.
    """
    # Token label metrics (entity recognition)
    token_tp = 0
    token_fp = 0
    token_fn = 0

    # Predicate metrics (relation extraction)
    pred_tp = 0
    pred_fp = 0
    pred_fn = 0

    # Indices to ignore for token labels (special tokens and O)
    ignore_indices = {token_label_list.index(t) for t in SPECIAL_TOKENS}
    o_index = token_label_list.index('O')
    ignore_indices.add(o_index)

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_label_ids = batch['token_label_ids'].to(device)
            predicate_ids = batch['predicate_ids'].to(device)

            outputs = model(input_ids, attention_mask)
            token_logits = outputs['token_label_logits']
            pred_logits = outputs['predicate_logits']

            # Token label predictions
            token_preds = torch.argmax(token_logits, dim=-1)  # [batch, seq_len]
            for i in range(input_ids.size(0)):
                seq_len = attention_mask[i].sum().item()
                for j in range(int(seq_len)):
                    gold = token_label_ids[i, j].item()
                    pred = token_preds[i, j].item()
                    if gold in ignore_indices:
                        continue
                    if pred == gold:
                        token_tp += 1
                    elif pred in ignore_indices:
                        token_fn += 1
                    else:
                        token_fp += 1
                        token_fn += 1  # missed the correct label

            # Predicate predictions (multi-label, threshold=0.5)
            pred_sigmoid = torch.sigmoid(pred_logits)
            pred_preds = (pred_sigmoid > 0.5).float()
            for i in range(predicate_ids.size(0)):
                gold_set = set(predicate_ids[i].nonzero(as_tuple=True)[0].tolist())
                pred_set = set(pred_preds[i].nonzero(as_tuple=True)[0].tolist())
                pred_tp += len(gold_set & pred_set)
                pred_fp += len(pred_set - gold_set)
                pred_fn += len(gold_set - pred_set)

    # Compute metrics
    token_precision = token_tp / (token_tp + token_fp) if (token_tp + token_fp) > 0 else 0.0
    token_recall = token_tp / (token_tp + token_fn) if (token_tp + token_fn) > 0 else 0.0
    token_f1 = (2 * token_precision * token_recall / (token_precision + token_recall)
                if (token_precision + token_recall) > 0 else 0.0)

    pred_precision = pred_tp / (pred_tp + pred_fp) if (pred_tp + pred_fp) > 0 else 0.0
    pred_recall = pred_tp / (pred_tp + pred_fn) if (pred_tp + pred_fn) > 0 else 0.0
    pred_f1 = (2 * pred_precision * pred_recall / (pred_precision + pred_recall)
               if (pred_precision + pred_recall) > 0 else 0.0)

    return {
        'token_precision': token_precision,
        'token_recall': token_recall,
        'token_f1': token_f1,
        'predicate_precision': pred_precision,
        'predicate_recall': pred_recall,
        'predicate_f1': pred_f1,
    }
