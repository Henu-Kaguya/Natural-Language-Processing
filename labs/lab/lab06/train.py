"""
Training script for joint entity-relation extraction.
Based on chapter08/8.2.3 tagging schema approach, adapted for PyTorch.
"""
import os
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from data_processor import (JointExtractionDataset, create_data_loader,
                            get_token_label_list, get_predicate_label_list)
from model import JointExtractionModel
from evaluate import evaluate_model
from triples_generation import decode_predictions


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Joint entity-relation extraction training.")
    parser.add_argument("--data-dir", type=str, default=os.path.join(BASE_DIR, "data"),
                        help="Directory containing train.jsonl and valid.jsonl")
    parser.add_argument("--output-dir", type=str, default=os.path.join(BASE_DIR, "outputs"),
                        help="Directory for model outputs")
    parser.add_argument("--bert-model", type=str, default="bert-base-chinese",
                        help="Pretrained BERT model name")
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build label lists
    token_label_list = get_token_label_list()
    predicate_label_list = get_predicate_label_list()
    print(f"Token labels ({len(token_label_list)}): {token_label_list}")
    print(f"Predicate labels ({len(predicate_label_list)}): {predicate_label_list}")

    # Load tokenizer
    print(f"Loading BERT tokenizer: {args.bert_model}")
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    # Load datasets
    print("Loading datasets...")
    train_dataset = JointExtractionDataset(args.data_dir, "train", tokenizer, args.max_seq_length)
    valid_dataset = JointExtractionDataset(args.data_dir, "valid", tokenizer, args.max_seq_length)
    print(f"Train examples: {len(train_dataset)}")
    print(f"Valid examples: {len(valid_dataset)}")

    train_loader = create_data_loader(train_dataset, args.batch_size, shuffle=True)
    valid_loader = create_data_loader(valid_dataset, args.batch_size, shuffle=False)

    # Initialize model
    print(f"Initializing model with BERT: {args.bert_model}")
    model = JointExtractionModel(
        bert_model_name=args.bert_model,
        num_token_labels=len(token_label_list),
        num_predicates=len(predicate_label_list),
        dropout=args.dropout,
    )
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training loop
    best_valid_loss = float('inf')
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_token_loss_sum = 0.0
        train_predicate_loss_sum = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_label_ids = batch['token_label_ids'].to(device)
            predicate_ids = batch['predicate_ids'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, token_label_ids, predicate_ids)
            loss = outputs['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss_sum += loss.item()
            train_token_loss_sum += outputs['token_label_loss']
            train_predicate_loss_sum += outputs['predicate_loss']
            num_batches += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss_sum / num_batches
        avg_token_loss = train_token_loss_sum / num_batches
        avg_pred_loss = train_predicate_loss_sum / num_batches
        print(f"Epoch {epoch} Train: loss={avg_train_loss:.4f} "
              f"(token={avg_token_loss:.4f}, predicate={avg_pred_loss:.4f})")

        # Validate
        model.eval()
        valid_loss_sum = 0.0
        num_valid_batches = 0

        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Epoch {epoch}/{args.epochs} [Valid]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_label_ids = batch['token_label_ids'].to(device)
                predicate_ids = batch['predicate_ids'].to(device)

                outputs = model(input_ids, attention_mask, token_label_ids, predicate_ids)
                valid_loss_sum += outputs['loss'].item()
                num_valid_batches += 1

        avg_valid_loss = valid_loss_sum / num_valid_batches
        print(f"Epoch {epoch} Valid: loss={avg_valid_loss:.4f}")

        # Save best model
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            save_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model (loss={best_valid_loss:.4f})")

    # Final evaluation with best model
    print("\n--- Final Evaluation ---")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt"),
                                      map_location=device))
    model.eval()

    metrics = evaluate_model(model, valid_loader, device, token_label_list, predicate_label_list)
    print(f"Validation Metrics:")
    print(f"  Token Label Precision: {metrics['token_precision']:.4f}")
    print(f"  Token Label Recall:    {metrics['token_recall']:.4f}")
    print(f"  Token Label F1:        {metrics['token_f1']:.4f}")
    print(f"  Predicate Precision:   {metrics['predicate_precision']:.4f}")
    print(f"  Predicate Recall:      {metrics['predicate_recall']:.4f}")
    print(f"  Predicate F1:          {metrics['predicate_f1']:.4f}")

    # Save metrics
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Generate sample predictions
    predictions = decode_predictions(model, valid_loader, device,
                                     token_label_list, predicate_label_list, tokenizer)
    pred_path = os.path.join(args.output_dir, "predictions.json")
    with open(pred_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"Predictions saved to {pred_path}")


if __name__ == "__main__":
    main()
