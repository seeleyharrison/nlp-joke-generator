"""
toy_lstm_joke_generator.py

A minimal, fully self-contained joke-generation LSTM.
Uses only the first 2000 rows of a Reddit jokes CSV located at:

    DATA_DIR = "../../data"

The file must contain columns: "title" and "selftext".

This script:
  1. Loads + cleans jokes
  2. Tokenizes using Keras Tokenizer
  3. Builds n-gram training sequences
  4. Trains a tiny LSTM model
  5. Generates jokes with temperature sampling

This is a simple toy example to scale up later with more of the data, also to figure out RNN hyperparameters.
This was our first attempt at the RNN, and is not runable. Our final version of our LSTM RNN can be
seen/run in the lstm_rnn.py file. Long story short, this is just here to show you our initial steps
despite this code not actually ever being used.
"""

import os
import numpy as np
import pandas as pd
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical

# Config
DATA_DIR = "../../data"
CSV_FILE = "one-million-reddit-jokes.csv"
MAX_LINES = 2000
NGRAM = 5   # Sequence length (predict the 6th word)
EPOCHS = 5
BATCH_SIZE = 32
EMBED_DIM = 64
LSTM_UNITS = 128


# Load + clean jokes
def load_reddit_jokes():
    """
    Loads the first MAX_LINES jokes from the Kaggle Reddit jokes CSV.
    Returns a list of cleaned sentences.
    """
    print("Loading CSV...")
    df = pd.read_csv(os.path.join(DATA_DIR, CSV_FILE), nrows=MAX_LINES)

    df["all_text"] = df["title"].astype(str) + " " + df["selftext"].astype(str)

    # remove [deleted]/[removed]
    df = df[~df["all_text"].str.contains(r"\[deleted\]|\[removed\]", regex=True)]

    jokes = df["all_text"].tolist()

    cleaned = []
    for j in jokes:
        j = j.lower()
        j = re.sub(r"http\S+", "", j)
        j = re.sub(r"[^a-z0-9\s.,!?']", " ", j)
        j = re.sub(r"\s+", " ", j).strip()
        cleaned.append(j)

    print(f"Loaded {len(cleaned)} cleaned jokes.")
    return cleaned



# Tokenize and build training data
def build_sequences(jokes, ngram=NGRAM):
    """
    Converts jokes → token sequences for LSTM training.
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(jokes)
    sequences = []

    for line in tokenizer.texts_to_sequences(jokes):
        # slide window of size ngram+1 to create training pairs
        for i in range(ngram, len(line)):
            seq = line[i-ngram:i+1]
            sequences.append(seq)

    sequences = np.array(sequences)
    print(f"Created {len(sequences)} training sequences.")

    X = sequences[:, :-1]
    y = sequences[:, -1]
    y = to_categorical(y, num_classes=len(tokenizer.word_index) + 1)

    return X, y, tokenizer


# Build LSTM model
def build_model(vocab_size):
    """
    Simple 1-layer LSTM for next-word prediction.
    """
    model = Sequential([
        Embedding(vocab_size, EMBED_DIM, input_length=NGRAM),
        LSTM(LSTM_UNITS),
        Dense(vocab_size, activation="softmax")
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam"
    )

    model.summary()
    return model


# Sampling + joke generation
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp = np.exp(preds)
    preds = exp / np.sum(exp)
    return np.random.choice(len(preds), p=preds)


def generate_joke(model, tokenizer, seed_text, length=30, temperature=0.9):
    """
    Generates a sentence continuing from seed_text.
    """
    result = seed_text.split()
    for _ in range(length):
        encoded = tokenizer.texts_to_sequences([" ".join(result[-NGRAM:])])
        encoded = pad_sequences(encoded, maxlen=NGRAM, truncating="pre")

        preds = model.predict(encoded, verbose=0)[0]
        next_id = sample(preds, temperature)
        word = tokenizer.index_word.get(next_id, "")

        result.append(word)

    return " ".join(result)

# Main execution
if __name__ == "__main__":
    jokes = load_reddit_jokes()
    X, y, tokenizer = build_sequences(jokes)
    vocab_size = len(tokenizer.word_index) + 1

    model = build_model(vocab_size)

    print("\nTraining model...\n")
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)

    print("\n--- Joke Generation Examples ---")
    seeds = [
        "why did the",
        "my girlfriend",
        "the bartender said",
        "i walked into"
    ]

    for s in seeds:
        print("\nSeed:", s)
        print(generate_joke(model, tokenizer, s, temperature=0.8))
