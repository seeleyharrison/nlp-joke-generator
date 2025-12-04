'''
    Evaluation script for fine-tuned GPT-2 joke model
    Calculates: Perplexity, Token Accuracy, Precision, Recall, F1
'''

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import argparse
import os

from gpt2 import (
    JokeDataset,
    prepare_joke_data,
    MODEL_DIR,
    MAX_LENGTH,
    BATCH_SIZE
)


def calculate_perplexity(model, dataloader, device):
    """
    Calculate perplexity - primary metric for language models
    Lower perplexity = better model
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    print("\nCalculating perplexity...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            # Count only non-padded tokens
            num_tokens = attention_mask.sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity, avg_loss


def calculate_token_metrics(model, dataloader, device, tokenizer):
    """
    Calculate token-level accuracy, precision, recall, and F1
    These metrics show how well the model predicts the next token
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    print("\nCalculating token-level metrics...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get predicted tokens (argmax of logits)
            predictions = torch.argmax(logits, dim=-1)
            
            # Flatten and filter out padding tokens
            for i in range(len(predictions)):
                mask = attention_mask[i].bool()
                pred_tokens = predictions[i][mask].cpu().numpy()
                label_tokens = labels[i][mask].cpu().numpy()
                
                # Skip the last prediction since there's no label for it
                if len(pred_tokens) > 1:
                    all_predictions.extend(pred_tokens[:-1])
                    all_labels.extend(label_tokens[1:])
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Precision, Recall, F1 (macro average across all tokens)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, 
        all_predictions, 
        average='macro',
        zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_tokens': len(all_labels)
    }


def evaluate_model(model_path, batch_size=BATCH_SIZE, max_samples=None):
    """
    Main evaluation function
    """
    print("=" * 70)
    print("GPT-2 Joke Model Evaluation")
    print("=" * 70)
    
    # Load model and tokenizer
    print(f"\nLoading model from {model_path}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model loaded on device: {device}")
    
    # Prepare evaluation data
    print("\nPreparing evaluation data...")
    jokes = prepare_joke_data()
    
    # Use last 5% as test set (matching your train/eval split)
    split_idx = int(len(jokes) * 0.95)
    test_jokes = jokes[split_idx:]
    
    if max_samples:
        test_jokes = test_jokes[:max_samples]
    
    print(f"Test set size: {len(test_jokes)} jokes")
    
    # Create dataset and dataloader
    test_dataset = JokeDataset(test_jokes, tokenizer, max_length=MAX_LENGTH)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Calculate perplexity
    print("\n" + "=" * 70)
    print("PERPLEXITY EVALUATION")
    print("=" * 70)
    perplexity, avg_loss = calculate_perplexity(model, test_dataloader, device)
    print(f"\nAverage Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print("\nInterpretation:")
    print(f"  - Lower perplexity is better")
    print(f"  - Perplexity ~{perplexity:.1f} means the model is ~{perplexity:.1f}x uncertain per token")
    
    # Calculate token-level metrics
    print("\n" + "=" * 70)
    print("TOKEN-LEVEL METRICS")
    print("=" * 70)
    metrics = calculate_token_metrics(model, test_dataloader, device, tokenizer)
    
    print(f"\nTotal tokens evaluated: {metrics['total_tokens']:,}")
    print(f"\nAccuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    
    print("\nInterpretation:")
    print(f"  - Accuracy: {metrics['accuracy']*100:.2f}% of tokens predicted correctly")
    print(f"  - F1: Harmonic mean of precision and recall")
    
    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Perplexity:      {perplexity:.4f}")
    print(f"Token Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"F1 Score:        {metrics['f1']:.4f}")
    print(f"Precision:       {metrics['precision']:.4f}")
    print(f"Recall:          {metrics['recall']:.4f}")
    print("=" * 70)
    
    return {
        'perplexity': perplexity,
        'loss': avg_loss,
        **metrics
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate GPT-2 joke model')
    parser.add_argument('--model_path', type=str, default=MODEL_DIR,
                       help='Path to fine-tuned model')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of test samples (for quick testing)')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_model(
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )