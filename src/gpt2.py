'''
    This file fine-tunes GPT-2 for joke generation using transfer learning.
    It handles the following steps:

    1) Load the preprocessed joke data
    2) Prepare data in the format expected by GPT-2
    3) Fine-tune GPT-2 on the joke corpus
    4) Save the fine-tuned model
    5) Provide text generation functionality
'''

import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    GPT2Config,
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from prepare_data import (
    read_kjokes_data, 
    read_rjokes_data,
    DATA_DIR,
    MODEL_DIR
)
import numpy as np

EPOCHS = 8
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
MAX_LENGTH = 192  # Maximum sequence length for GPT-2 (longer jokes)
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 16 * 2 = 32

# Data quality filtering so nothing is bad 
MIN_JOKE_LENGTH = 20
MAX_JOKE_LENGTH = 400

# Make sure pytorch and cuda works
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version (PyTorch): {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

class JokeDataset(Dataset):
    '''
    Custom Dataset for joke text that tokenizes on-the-fly
    '''
    def __init__(self, jokes, tokenizer, max_length=MAX_LENGTH):
        self.jokes = jokes
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.jokes)
    
    def __getitem__(self, idx):
        joke = self.jokes[idx]
        
        #registered special tokens for proper tokenization honestly i oprobably got to change this because training was bad and i shouldn't add new tokens
        bos = self.tokenizer.bos_token if self.tokenizer.bos_token else ''
        eos = self.tokenizer.eos_token if self.tokenizer.eos_token else ''
        text = f"{bos}{joke}{eos}"
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Return as dict with both input_ids and labels
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze()
        }

def prepare_joke_data():
    '''
    Load and prepare joke data from both sources
    Returns:
        List of joke strings (filtered by quality)
    '''
    print("Loading joke data...")
    
    #load kaggle jokes
    kaggle_tokens = read_kjokes_data()
    kaggle_jokes = []
    for tokens in kaggle_tokens:
        # Remove special tokens and reconstruct text not needed
        joke = ' '.join([t for t in tokens if t not in ['<s>', '</s>', '_']])
        joke = joke.replace(' _ ', ' ')  # Handle space character
        kaggle_jokes.append(joke)
    
    # Load reddit jokes (more samples for better training) get tokens
    _, rjokes_tokens = read_rjokes_data("train.tsv", max_jokes=100000)
    rjokes = []
    for tokens in rjokes_tokens:
        joke = ' '.join([t for t in tokens if t not in ['<s>', '</s>', '_']])
        joke = joke.replace(' _ ', ' ')
        rjokes.append(joke)
    
    all_jokes = kaggle_jokes + rjokes
    print(f"Total jokes loaded (before filtering): {len(all_jokes)}")
    
    # Filter jokes by length for quality (remove too short/long)
    filtered_jokes = [j for j in all_jokes if MIN_JOKE_LENGTH <= len(j) <= MAX_JOKE_LENGTH]
    print(f"Jokes after quality filtering ({MIN_JOKE_LENGTH}-{MAX_JOKE_LENGTH} chars): {len(filtered_jokes)}")
    
    return filtered_jokes

def fine_tune_model(model, tokenizer, train_dataset, eval_dataset, output_dir):
    '''
    Fine-tune the GPT-2 model on joke data with validation
    '''
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT-2 uses causal LM, not masked LM so i changed this
    )
    
    # (optimized) for cuda
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        # Optimized scheduler settings TODO CHANGE SMOOTHING FACTOR
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,  # 10% warmup (better than fixed steps) 
        # Regularization
        label_smoothing_factor=0.1,  # Prevents overconfidence
        max_grad_norm=1.0,  # Gradient clipping for stability
        # Logging and saving
        logging_steps=50,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=3,
        # Evaluation for overfitting detection
        eval_strategy="steps",
        eval_steps=250,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Performance
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to="none",
        logging_dir=f"{output_dir}/logs",
    )
    
    # Initialize trainer with eval dataset
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Train
    print("\nStarting fine-tuning...")
    start = time.time()
    trainer.train()
    end = time.time()
    
    elapsed = end - start
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(f"\nTraining complete! Took {minutes}m {seconds:.1f}s")
    
    return trainer

def generate_joke(model, tokenizer, prompt, max_length=100, temperature=0.8, top_k=50, top_p=0.95, num_return_sequences=1, repetition_penalty=1.2):
    '''
    Generate jokes using the fine-tuned model
    Args:
        model: Fine-tuned GPT-2 model
        tokenizer: GPT-2 tokenizer
        prompt: Starting text for generation
        max_length: Maximum length of generated text
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        num_return_sequences: Number of jokes to generate
        repetition_penalty: Penalty for repeating tokens (>1.0 = less repetition)
    Returns:
        List of generated jokes
    '''
    model.eval()
    
    # Encode prompt using the registered bos_token
    input_text = f"{tokenizer.bos_token}{prompt}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    # Move to same device as model
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    # Generate with improved coherence settings
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # Coherence improvements
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=3,  # Prevent 3-gram repetition e.g. the dog dog dog ran
        )
    
    # Decode outputs
    generated_jokes = []
    for sequence in output:
        text = tokenizer.decode(sequence, skip_special_tokens=True)
        # Remove the prompt from the output
        if prompt in text:
            text = text.replace(prompt, '').strip()
        generated_jokes.append(text)
    
    return generated_jokes

if __name__ == "__main__":
    print("=" * 70)
    print("GPT-2 Joke Generator - Transfer Learning (Optimized)")
    print("=" * 70)
    
    # parse arguments for running script
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', 
                       help='Force retraining even if model exists')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Custom prompt for generation')
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist in the gpu repo area e.g. models/
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    model_path = f"{MODEL_DIR}/gpt2-jokes"
    
    # Check if model exists gpu
    if os.path.exists(model_path) and not args.train:
        print("\n" + "=" * 70)
        print("Loading existing fine-tuned model...")
        print("=" * 70)
        
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"Model loaded on device: {device}")
        
    else:
        print("\n" + "=" * 70)
        print("Training new model with transfer learning...")
        print("=" * 70)
        
        # Load pre-trained GPT-2 (full model, 124M params) loading
        print("\nLoading GPT-2 base model (124M params)...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        
        # TODO: CHANGE THIS BECAUSE MESSED UP TRAIING
        special_tokens = {
            'bos_token': '<|startoftext|>',
            'eos_token': '<|endoftext|>',
            'pad_token': '<|pad|>'
        }
        num_added = tokenizer.add_special_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))
        print(f"Added {num_added} special tokens, vocab size: {len(tokenizer)}")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"Training on device: {device}")
        
        # Prepare data
        jokes = prepare_joke_data()
        
        # Split into train/eval (5% for validation) MIGHT DO 90 AND 10 TODO
        split_idx = int(len(jokes) * 0.95)
        train_jokes = jokes[:split_idx]
        eval_jokes = jokes[split_idx:]
        print(f"\nTrain/Eval split: {len(train_jokes)} train, {len(eval_jokes)} eval")
        
        # Create datasets
        print("Creating datasets...")
        train_dataset = JokeDataset(train_jokes, tokenizer)
        eval_dataset = JokeDataset(eval_jokes, tokenizer)
        print(f"Train dataset: {len(train_dataset)} jokes")
        print(f"Eval dataset: {len(eval_dataset)} jokes")
        
        # Fine-tune with validation
        trainer = fine_tune_model(model, tokenizer, train_dataset, eval_dataset, model_path)
        
        # Save final model
        print("\n" + "=" * 70)
        print("Saving fine-tuned model...")
        print("=" * 70)
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)
        print(f"Model saved to {model_path}")
    
    # Generate sample jokes
    print("\n" + "=" * 70)
    print("Generating sample jokes...")
    print("=" * 70)
    
    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = [
            "Why did the chicken",
            "A man walks into a bar",
            "What do you call a",
            "How many programmers does it take",
            "My wife told me"
        ]
    
    temperatures = [0.7, 1.0, 1.3]
    #TEST TEMPERATURES
    for temp in temperatures:
        print(f"\n{'=' * 70}")
        print(f"Temperature: {temp}")
        print('=' * 70)
        
        for prompt in prompts:
            jokes = generate_joke(
                model, 
                tokenizer, 
                prompt,
                max_length=80,
                temperature=temp,
                num_return_sequences=1
            )
            
            print(f"\nPrompt: '{prompt}'")
            for i, joke in enumerate(jokes, 1):
                print(f"Generated: {joke}")
    
    print("\n" + "=" * 70)
    print("Joke generation complete!")
    print("=" * 70)