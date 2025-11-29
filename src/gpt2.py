'''
    This file fine-tunes DistilGPT-2 for joke generation using transfer learning.
    It handles the following steps:

    1) Load the preprocessed joke data
    2) Prepare data in the format expected by GPT-2
    3) Fine-tune DistilGPT-2 on the joke corpus
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

# Training parameters
EPOCHS = 1
LEARNING_RATE = 5e-5
BATCH_SIZE = 8
MAX_LENGTH = 128  # Maximum sequence length for GPT-2
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 8 * 4 = 32

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
        
        # Add special tokens for better learning
        text = f"<|startoftext|>{joke}<|endoftext|>"
        
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
        List of joke strings
    '''
    print("Loading joke data...")
    
    # Load kaggle jokes
    kaggle_tokens = read_kjokes_data()
    kaggle_jokes = []
    for tokens in kaggle_tokens:
        # Remove special tokens and reconstruct text
        joke = ' '.join([t for t in tokens if t not in ['<s>', '</s>', '_']])
        joke = joke.replace(' _ ', ' ')  # Handle space character
        kaggle_jokes.append(joke)
    
    # Load reddit jokes (more samples for better training)
    _, rjokes_tokens = read_rjokes_data("train.tsv", max_jokes=50000)
    rjokes = []
    for tokens in rjokes_tokens:
        joke = ' '.join([t for t in tokens if t not in ['<s>', '</s>', '_']])
        joke = joke.replace(' _ ', ' ')
        rjokes.append(joke)
    
    all_jokes = kaggle_jokes + rjokes
    print(f"Total jokes loaded: {len(all_jokes)}")
    
    return all_jokes

def fine_tune_model(model, tokenizer, train_dataset, output_dir):
    '''
    Fine-tune the GPT-2 model on joke data
    '''
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT-2 uses causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=500,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        report_to="none",  # Disable wandb/tensorboard
        logging_dir=f"{output_dir}/logs",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
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

def generate_joke(model, tokenizer, prompt, max_length=100, temperature=0.8, top_k=50, top_p=0.95, num_return_sequences=1):
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
    Returns:
        List of generated jokes
    '''
    model.eval()
    
    # Encode prompt
    input_text = f"<|startoftext|>{prompt}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    # Move to same device as model
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    # Generate
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
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
    print("DistilGPT-2 Joke Generator - Transfer Learning")
    print("=" * 70)
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', 
                       help='Force retraining even if model exists')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Custom prompt for generation')
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    model_path = f"{MODEL_DIR}/distilgpt2-jokes"
    
    # Check if model exists
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
        
        # Load pre-trained DistilGPT-2
        print("\nLoading DistilGPT-2 base model...")
        tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        model = GPT2LMHeadModel.from_pretrained('distilgpt2')
        
        # Add padding token
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"Training on device: {device}")
        
        # Prepare data
        jokes = prepare_joke_data()
        
        # Create dataset
        print("\nCreating dataset...")
        train_dataset = JokeDataset(jokes, tokenizer)
        print(f"Dataset size: {len(train_dataset)} jokes")
        
        # Fine-tune
        trainer = fine_tune_model(model, tokenizer, train_dataset, model_path)
        
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