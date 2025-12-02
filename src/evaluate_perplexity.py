'''Calculate perplexity for a fine-tuned GPT-2 joke model.'''

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import math
import argparse

def calculate_perplexity(model, tokenizer, jokes, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    results = []
    
    for joke in jokes:
        inputs = tokenizer(joke + tokenizer.eos_token, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            num_tokens = inputs["input_ids"].shape[1]
            
            total_loss += loss * num_tokens
            total_tokens += num_tokens
            results.append({'joke': joke, 'loss': loss, 'ppl': math.exp(loss)})
    
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss), avg_loss, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../finalModel2')
    parser.add_argument('--jokes', type=str, nargs='+', default=None)
    args = parser.parse_args()
    
    model = GPT2LMHeadModel.from_pretrained(args.model)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    test_jokes = args.jokes or [
        "Why did the chicken cross the road? To get to the other side!",
        "A man walks into a bar. Ouch.",
        "What do you call a fish without eyes? A fsh.",
        "I told my wife she was drawing her eyebrows too high. She looked surprised.",
        "Why don't scientists trust atoms? Because they make up everything!",
    ]
    
    avg_ppl, avg_loss, results = calculate_perplexity(model, tokenizer, test_jokes, device)
    
    for r in results:
        preview = r['joke'][:50] + '...' if len(r['joke']) > 50 else r['joke']
        print(f"Loss: {r['loss']:.4f} | PPL: {r['ppl']:>8.2f} | {preview}")
    
    print(f"\nAverage Perplexity: {avg_ppl:.2f}")


if __name__ == "__main__":
    main()
