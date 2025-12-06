'''
    This file trains and fine-tunes GPT-2 for joke generation using transfer learning.
    It handles the following steps:

    1) Load the preprocessed joke data
        - Includes the step of filtering out reddit posts with small upvote counts,
        this helps filter out sloppy data
    2) Prepare data in the format expected by GPT-2
    3) Fine-tune GPT-2 on the joke corpus
    4) Save the fine-tuned model
    5) Evaluate performance of fine-tuned model
    6) Provide text generation functionality (through a terminal app or direct prompts)

    RUN INSTRUCTIONS:
    NOTE: All commands should be run from the root of the repository

    1) Train a new model from scratch
        python src/train_model.py --train
    
    2) Train with grid search
        python train_model.py --grid-search
    
    3) Load pre-trained model and generates sample jokes
        python train_model.py
    
    4) Custom prompt generation:
        python train_model.py --prompt "Your custom prompt here"
    
    5) Evaluate model perplexity, loss, and accuracy on test set
        python train_model.py --eval
    
    6) Interactive terminal app
        python train_model.py --app

    Options 2-5 will be run with the existing model by default. If you
    would like to instead train a new model to run any of these options,
    add the --train flag to the command. WARNING, THIS WILL OVERWRITE
    THE EXISTING MODEL AND REPLACE IT WITH THE NEW MODEL YOU TRAIN
    
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
import pandas as pd
import re
import json
from sklearn.metrics import accuracy_score
import math
import numpy as np

DATA_DIR = "compressed_data"
MODEL_DIR = "model3"

MIN_SCORE = 10  # Filters out low quality jokes, a joke must have a score greater than 10 to be included
MAX_JOKES = None 
EPOCHS = 3
LEARNING_RATE = 5e-5  # Slightly increased for faster convergence
BATCH_SIZE = 24
MAX_LENGTH = 192  # Maximum sequence length for GPT-2 (longer jokes)
GRADIENT_ACCUMULATION_STEPS = 3  # Effective batch size = 24 * 3 * 3 = 216

# Data quality filtering so nothing is bad 
MIN_JOKE_LENGTH = 20
MAX_JOKE_LENGTH = 400

# Make sure pytorch and cuda works, some of us had nvidia gpus
# so this was helpful for us in terms of speeding up training time
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version (PyTorch): {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

class JokeDataset(Dataset):
    '''
    Custom Dataset for joke text that tokenizes on-the-fly, this is mainly used
    because extending the Dataset class allows for seamless integration with
    PyTorch's trainer class that is great for transfer learning training.

    This is essentially a variant of our data loader
    
    Args:
        jokes (list): List of joke strings to be tokenized
        tokenizer (GPT2Tokenizer): Tokenizer instance for encoding text
        max_length (int): Maximum sequence length for truncation (default: MAX_LENGTH)
    '''
    def __init__(self, jokes, tokenizer, max_length=MAX_LENGTH):
        self.jokes = jokes
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        '''
        Returns:
            int: Total number of jokes in the dataset
        '''
        return len(self.jokes)
    
    def __getitem__(self, idx):
        '''
        Retrieves and tokenizes a single joke
        
        Args:
            idx (int): Index of the joke to retrieve
        
        Returns:
            dict: Dictionary containing 'input_ids' and 'attention_mask' tensors
        '''
        joke = self.jokes[idx]
        
        # Add eos token to end of the joke for gpt2 compatibilty
        text = f"{joke}{self.tokenizer.eos_token}"
        
        # Debug print for first item to check data integrity
        if idx == 0:
            print(f"\nDEBUG: Sample Text (idx=0): '{text}'\n")
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze()
        }

def prepare_joke_data():
    '''
    Load jokes from CSV and TSV files with score filtering
    
    Loads joke data from multiple sources (Kaggle CSV, rJokes TSVs, fullrjokes.json),
    applies quality filters (minimum score, length constraints), cleans text,
    and removes duplicates.
    
    Returns:
        list: List of cleaned, filtered, and deduplicated joke strings
    '''
    print("Loading joke data...")
    jokes = []
    
    def clean_text(text):
        '''Standardize and clean text'''
        text = str(text)
        # Remove deleted/removed markers
        if '[removed]' in text or '[deleted]' in text:
            return None
        # Clean HTML entities (including manual variations)
        text = re.sub(r'&amp;|&#x200B;|#x200B;|&[a-zA-Z]+;', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text if text else None

    # Load Kaggle CSV (title=setup, selftext=punchline)
    csv_path = f"{DATA_DIR}/one-million-reddit-jokes.csv"
    df = pd.read_csv(csv_path)
    df = df[df['score'] >= MIN_SCORE]
    
    # Combine title and selftext (vectorized for speed)
    print("Processing CSV...")
    df['joke'] = df['title'].fillna('').astype(str) + ' ' + df['selftext'].fillna('').astype(str)
    df = df[~df['joke'].str.contains(r'\[removed\]|\[deleted\]', regex=True, na=False)]
    
    for text in df['joke']:
        cleaned = clean_text(text)
        if cleaned:
            jokes.append(cleaned)
            
    print(f"Kaggle CSV: {len(jokes)} jokes (score >= {MIN_SCORE})")
    
    # Load rJokes TSVs (train + dev + test)
    tsv_count = 0
    print("Processing TSVs...")
    for tsv in ['rJokesData/train.tsv', 'rJokesData/dev.tsv']:
        path = f"{DATA_DIR}/{tsv}"
        if not os.path.exists(path): continue
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    try:
                        # Ensure score is valid integer
                        if int(parts[0]) >= MIN_SCORE:
                            cleaned = clean_text(parts[1])
                            if cleaned:
                                jokes.append(cleaned)
                                tsv_count += 1
                    except: pass
    print(f"rJokes TSV: {tsv_count} jokes (score >= {MIN_SCORE})")

    # Load fullrjokes.json (additional data source)
    json_path = f"{DATA_DIR}/rJokesData/fullrjokes.json"
    json_count = 0
    if os.path.exists(json_path):
        print("Processing fullrjokes.json...")
        with open(json_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get('score', 0) >= MIN_SCORE:
                        raw_joke = f"{data.get('title', '')} {data.get('selftext', '')}"
                        cleaned = clean_text(raw_joke)
                        if cleaned:
                            jokes.append(cleaned)
                            json_count += 1
                except: pass
        print(f"fullrjokes.json: {json_count} jokes (score >= {MIN_SCORE})")
    
    # Length filter
    jokes = [j for j in jokes if MIN_JOKE_LENGTH <= len(j) <= MAX_JOKE_LENGTH]
    print(f"After length filter: {len(jokes)}")
    
    # Deduplication (preserves order)
    before_dedup = len(jokes)
    jokes = list(dict.fromkeys(jokes))
    print(f"After deduplication: {len(jokes)} (removed {before_dedup - len(jokes)} duplicates)")
    
    # Apply max limit if set
    if MAX_JOKES and len(jokes) > MAX_JOKES:
        jokes = jokes[:MAX_JOKES]
        print(f"Limited to MAX_JOKES: {MAX_JOKES}")
    
    return jokes

def fine_tune_model(model, tokenizer, train_dataset, eval_dataset, output_dir):
    '''
    Fine-tune the GPT-2 model on joke data with validation
    
    Sets up training arguments with optimized hyperparameters, including cosine
    learning rate scheduling, gradient clipping, and early stopping based on
    evaluation loss.
    
    Args:
        model (GPT2LMHeadModel): Pre-trained GPT-2 model to fine-tune
        tokenizer (GPT2Tokenizer): Tokenizer for text encoding
        train_dataset (JokeDataset): Training dataset
        eval_dataset (JokeDataset): Evaluation dataset for validation
        output_dir (str): Directory path to save model checkpoints and logs
    
    Returns:
        Trainer: Trained Trainer instance with the fine-tuned model
    '''
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT-2 uses causal LM, not masked LM so i changed this
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        # Regularization
        weight_decay=0.02,  
        # Optimized scheduler settings learned cosine in class
        lr_scheduler_type="cosine",
        warmup_ratio=0.10,  # Better warmup for convergence

        label_smoothing_factor=0.02,  # Low for clearer loss signal
        max_grad_norm=1.0,  # Gradient clipping for stability
        # Logging and saving TODO: probably increase the steps because we dont needa do that many saves 
        # NOTE: must be equal savesteps and eval steps when running because of cuda
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

def grid_search_lr(model_class, tokenizer, train_dataset, eval_dataset, output_dir):
    '''
    Simple grid search over learning rates (1-2 epochs each)
    
    Tests multiple learning rate values and selects the one with the lowest
    evaluation loss. Each learning rate is tested with a fresh model instance.
    
    Args:
        model_class (type): Model class to instantiate (GPT2LMHeadModel)
        tokenizer (GPT2Tokenizer): Tokenizer for text encoding
        train_dataset (JokeDataset): Training dataset
        eval_dataset (JokeDataset): Evaluation dataset
        output_dir (str): Directory path for temporary grid search outputs
    
    Returns:
        float: Best learning rate based on evaluation loss
    '''
    lr_options = [1e-5, 2e-5, 3e-5]
    best_lr = LEARNING_RATE
    best_loss = float('inf')
    
    print("\n" + "=" * 70)
    print("Running grid search over learning rates...")
    print("=" * 70)
    
    for lr in lr_options:
        print(f"\nTesting LR={lr}...")
        
        # Fresh model for each test
        model = model_class.from_pretrained('gpt2')
        model.config.attn_pdrop = 0.15
        model.config.resid_pdrop = 0.15
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        
        # Quick 1-epoch training
        args = TrainingArguments(
            output_dir=f"{output_dir}/grid_search",
            num_train_epochs=1,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=lr,
            weight_decay=0.02,
            warmup_ratio=0.06,
            eval_strategy="epoch",
            save_strategy="no",
            fp16=torch.cuda.is_available(),
            report_to="none",
        )
        
        trainer = Trainer(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        trainer.train()
        eval_result = trainer.evaluate()
        eval_loss = eval_result['eval_loss']
        
        print(f"LR={lr} -> Eval Loss: {eval_loss:.4f}")
        
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_lr = lr
        
        # Cleanup
        del model, trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"\nBest LR: {best_lr} (loss: {best_loss:.4f})")
    return best_lr


def generate_joke(model, tokenizer, prompt, max_length=100, temperature=0.8, top_k=50, top_p=0.95, num_return_sequences=1, repetition_penalty=1.2, use_beam_search=True):
    '''
    Generate jokes using the fine-tuned model
    
    Supports two generation modes:
    - Beam search: More coherent but less varied outputs
    - Sampling: More creative but potentially less coherent outputs
    
    Args:
        model (GPT2LMHeadModel): Fine-tuned GPT-2 model
        tokenizer (GPT2Tokenizer): GPT-2 tokenizer
        prompt (str): Starting text for generation
        max_length (int): Maximum length of generated text (default: 100)
        temperature (float): Sampling temperature, higher = more random (default: 0.8)
        top_k (int): Top-k sampling parameter (default: 50)
        top_p (float): Nucleus sampling parameter (default: 0.95)
        num_return_sequences (int): Number of jokes to generate (default: 1)
        repetition_penalty (float): Penalty for repeating tokens, >1.0 = less repetition (default: 1.2)
        use_beam_search (bool): If True, use beam search; if False, use sampling (default: True)
    
    Returns:
        list: List of generated joke strings
    '''
    
    # Encode prompt WITH attention mask
    inputs = tokenizer(prompt, return_tensors='pt')
    
    # Move to same device as model
    device = next(model.parameters()).device
    input_ids = inputs['input_ids'].to(device)
    
    # Generate with beam search for coherent punchlines, or sampling for variety
    with torch.no_grad():
        if use_beam_search:
            # Beam search: explores multiple paths for logical outputs
            # Note: temperature is NOT used with beam search
            output = model.generate(
                input_ids,
                max_length=max_length,
                num_beams=4,
                num_return_sequences=num_return_sequences,
                do_sample=False,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=2,
            )
        else:
            # Sampling: more variety but potentially less coherent
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
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=2,
            )
    
    # Decode outputs
    generated_jokes = []
    for sequence in output:
        text = tokenizer.decode(sequence, skip_special_tokens=True)
        if prompt in text:
            text = text.replace(prompt, '').strip()
        generated_jokes.append(text)
    
    return generated_jokes


def evaluate(model, tokenizer, jokes, device):
    '''
    Evaluates the model via its overall perplexity score, loss, and accuracy
    
    Processes each joke individually, computing token-level predictions and
    comparing them to ground truth labels. Aggregates metrics across all jokes.
    
    Args:
        model (GPT2LMHeadModel): Fine-tuned GPT-2 model
        tokenizer (GPT2Tokenizer): GPT-2 tokenizer
        jokes (list): Test jokes set to evaluate on
        device (torch.device): CPU or GPU device for computation
    
    Returns:
        dict: Dictionary containing 'perplexity', 'loss', and 'accuracy' metrics
    '''
    model.eval()
    total_loss = 0
    total_tokens = 0
    results = []
    all_predictions = []
    all_labels = []
    
    for joke in jokes:
        inputs = tokenizer(joke + tokenizer.eos_token, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            num_tokens = inputs["input_ids"].shape[1]
            
            # Get predictions (argmax of logits)
            logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
            predictions = torch.argmax(logits, dim=-1)  # Shape: [batch_size, seq_len]
            pred_tokens = predictions[:, :-1].flatten().cpu().numpy()
            label_tokens = inputs["input_ids"][:, 1:].flatten().cpu().numpy()
            
            # Calculate per-joke accuracy
            joke_accuracy = (pred_tokens == label_tokens).mean()
            
            # Store for global metrics
            all_predictions.extend(pred_tokens.tolist())
            all_labels.extend(label_tokens.tolist())
            
            # Update totals
            total_loss += loss * num_tokens
            total_tokens += num_tokens
            
            results.append({
                'joke': joke,
                'loss': loss,
                'ppl': math.exp(loss),
                'accuracy': joke_accuracy,
                'num_tokens': num_tokens
            })
    
    # Calculate overall metrics
    avg_loss = total_loss / total_tokens
    avg_ppl = math.exp(avg_loss)
    overall_accuracy = accuracy_score(all_labels, all_predictions)
    
    # Compile summary metrics
    summary = {
        'perplexity': avg_ppl,
        'loss': avg_loss,
        'accuracy': overall_accuracy,
    }
    
    return summary


'''
    Main program/entry point to run evaluation and text generation on our model
'''
if __name__ == "__main__":
    print()
    print("=" * 70)
    print("The Joke Generator")
    print("=" * 70)
    
    # parse arguments for running script
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', 
                       help='Force retraining even if model exists')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Custom prompt for generation')
    parser.add_argument('--grid-search', action='store_true',
                       help='Run grid search for learning rate before training')
    parser.add_argument('--app', action='store_true',
                       help='Terminal application')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate loaded or trained model')
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist in the gpu repo area e.g. models/
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    model_path = f"{MODEL_DIR}/gpt2-jokes-v3"

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if model exists gpu
    if os.path.exists(MODEL_DIR) and not args.train:

        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
        model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = model.to(device)
        # print(f"Model loaded on device: {device}")        
    else:
        print("\n" + "=" * 70)
        print("Training new model with transfer learning...")
        print("=" * 70)
        
        # Load pre-trained GPT-2 (full model, 124M params) loading
        print("\nLoading GPT-2 base model (124M params)...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        
        # Reuse existing pre-trained token (no random embeddings)
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Using existing eos_token as pad_token, vocab size: {len(tokenizer)}")
        
        # Add dropout for regularization (no new deps, built into GPT2) optimizing
        model.config.attn_pdrop = 0.15
        model.config.resid_pdrop = 0.15
        print(f"Dropout set: attn={model.config.attn_pdrop}, resid={model.config.resid_pdrop}")
        
        # Freeze first 6 of 12 transformer layers (transfer learning optimization)
        # This preserves general language knowledge while training top layers for jokes
        frozen_layers = 0
        for i, layer in enumerate(model.transformer.h):
            if i < frozen_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Layer freezing: {frozen_layers}/12 layers frozen, {trainable:,}/{total:,} params trainable")

        model = model.to(device)
        print(f"Training on device: {device}")
        
        # Prepare data
        jokes = prepare_joke_data()
        
        # Split into train/eval (5% for validation) TODO: maybe change to 10%
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
        
        #  grid search for learning rate if you do gridsearch arg in script
        if args.grid_search:
            best_lr = grid_search_lr(GPT2LMHeadModel, tokenizer, train_dataset, eval_dataset, model_path)
            LEARNING_RATE = best_lr
            # Reload fresh model after grid search
            model = GPT2LMHeadModel.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
            model.config.attn_pdrop = 0.15
            model.config.resid_pdrop = 0.15
            # Re-freeze layers after reload
            for i, layer in enumerate(model.transformer.h):
                if i < frozen_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
            model = model.to(device)
        
        # Fine-tune with validation
        trainer = fine_tune_model(model, tokenizer, train_dataset, eval_dataset, model_path)
        
        # Save final model
        print("\n" + "=" * 70)
        print("Saving fine-tuned model...")
        print("=" * 70)
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)
        print(f"Model saved to {model_path}")

    # Evaluate model if prompted
    if args.eval:
        jokes = prepare_joke_data()
        split_idx = int(len(jokes) * 0.95)
        eval_jokes = jokes[split_idx:]
        evaluation = evaluate(model, tokenizer, eval_jokes, device)
        print("\n" + "=" * 70)
        print("Model Evaluation")
        print(evaluation)
        print("=" * 70)

    # Terminal Program
    if args.app:
        while True:
            print("\n" + "=" * 70)
            prompt = input("Enter a prompt for a new joke!\n\n")
            temperature = float(input("\nEnter a temperature for your joke: "))
            generation = generate_joke(model, tokenizer, prompt, max_length=30, temperature=temperature, num_return_sequences=1)[0]
            print()
            print((prompt + " " + generation.strip()))
            cont = int(input("\nEnter 1 to continue or 2 to exit: "))
            print("" + "=" * 70)
            if cont == 2:
                exit(0)
    
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
    
    # Test with beam search (coherent)
    print(f"\n{'=' * 70}")
    print("Mode: Beam Search (coherent)")
    print('=' * 70)
    
    for prompt in prompts:
        jokes = generate_joke(
            model, 
            tokenizer, 
            prompt,
            max_length=120,
            num_return_sequences=1,
            use_beam_search=True
        )
        print(f"\nPrompt: '{prompt}'")
        for joke in jokes:
            print(f"Generated: {joke}")
    
    # Test with sampling at different temperatures (variety)
    temperatures = [0.7, 1.0, 1.3]
    for temp in temperatures:
        print(f"\n{'=' * 70}")
        print(f"Mode: Sampling (temperature={temp})")
        print('=' * 70)
        
        for prompt in prompts:
            jokes = generate_joke(
                model, 
                tokenizer, 
                prompt,
                max_length=80,
                temperature=temp,
                num_return_sequences=1,
                use_beam_search=False  # Use sampling with temperature
            )
            
            print(f"\nPrompt: '{prompt}'")
            for joke in jokes:
                print(f"Generated: {joke}")
    
    print("\n" + "=" * 70)
    print("Joke generation complete!")
    print("=" * 70)