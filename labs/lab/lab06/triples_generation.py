"""
SPO triples generation from model predictions.
Based on the triples_generation.py logic from chapter08/8.2.3.

Process:
1. From token label predictions, extract entity spans (BIO → entities)
2. From predicate predictions, get relation types
3. Pair entities with predicates to form SPO triples
"""
import torch
from data_processor import ENTITY_TYPES, PREDICATE_TYPES, SPECIAL_TOKENS


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


def decode_predictions(model, data_loader, device, token_label_list,
                       predicate_label_list, tokenizer, predicate_threshold=0.5):
    """Decode model predictions into structured SPO triples.

    Args:
        model: trained JointExtractionModel
        data_loader: validation data loader
        device: torch device
        token_label_list: list of token label strings
        predicate_label_list: list of predicate strings
        tokenizer: BERT tokenizer
        predicate_threshold: sigmoid threshold for predicate prediction

    Returns:
        list of prediction dicts with text, gold_spo_list, pred_spo_list
    """
    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask)
            token_logits = outputs['token_label_logits']
            pred_logits = outputs['predicate_logits']

            token_preds = torch.argmax(token_logits, dim=-1)
            pred_sigmoid = torch.sigmoid(pred_logits)
            pred_preds = (pred_sigmoid > predicate_threshold).float()

            for i in range(input_ids.size(0)):
                text = batch['texts'][i]
                tokens = batch['tokens_list'][i]
                gold_spo_list = batch['spo_lists'][i]

                # Extract entities from token labels
                seq_len = attention_mask[i].sum().item()
                token_pred_ids = token_preds[i, 1:seq_len - 1].tolist()  # exclude [CLS] and [SEP]
                entities = extract_entities(tokens, token_pred_ids, token_label_list)

                # Get predicted predicates
                pred_indices = pred_preds[i].nonzero(as_tuple=True)[0].tolist()
                pred_labels = [predicate_label_list[idx] for idx in pred_indices]

                # Generate SPO triples by pairing entities with predicted predicates
                pred_spo_list = generate_spo_triples(entities, pred_labels)

                predictions.append({
                    'text': text,
                    'gold_spo_list': gold_spo_list,
                    'pred_spo_list': pred_spo_list,
                    'entities': entities,
                    'predicted_predicates': pred_labels,
                })

    return predictions


def generate_spo_triples(entities, predicates):
    """Generate SPO triples by pairing entities with predicates.

    Simplified rule: for each predicate, find the most likely subject/object
    entity pair based on entity types.
    """
    spo_list = []

    # Group entities by type
    entities_by_type = {}
    for e in entities:
        entities_by_type.setdefault(e['type'], []).append(e)

    # Define expected subject/object types for each predicate
    predicate_type_map = {
        'works_for': ('PER', 'ORG'),
        'educated_at': ('PER', 'ORG'),
        'lives_in': ('PER', 'LOC'),
        'located_in': ('PER', 'LOC'),
        'visits': ('PER', 'ORG'),
        'founder_of': ('PER', 'ORG'),
    }

    for pred in predicates:
        if pred not in predicate_type_map:
            continue
        subj_type, obj_type = predicate_type_map[pred]
        subjects = entities_by_type.get(subj_type, [])
        objects = entities_by_type.get(obj_type, [])

        for subj in subjects:
            for obj in objects:
                if subj['text'] != obj['text']:
                    spo_list.append({
                        'subject': subj['text'],
                        'subject_type': subj_type,
                        'predicate': pred,
                        'object': obj['text'],
                        'object_type': obj_type,
                    })

    return spo_list
