'''
    This file trains an RNN (LSTM) model for text generation using the
    prepared joke data. It handles the following steps:

    1) Load the preprocessed data from prepare_data.py
    2) Build an LSTM-based neural network
    3) Train the model on the joke corpus
    4) Save the trained model and tokenizer
    5) Provide text generation functionality
'''

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)  # Suppress gensim NumPy warnings

import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import pickle
from prepare_data import (
    read_kjokes_data, 
    read_rjokes_data, 
    create_word_embeddings,
    encode_as_indices,
    generate_ngram_training_samples,
    isolate_labels,
    read_embeddings,
    custom_data_generator,
    NGRAM,
    BATCH_SIZE,
    MODEL_DIR,
    DATA_DIR
)

# Training parameters
EPOCHS = 3  # increased from 2 for better learning
LEARNING_RATE = 0.001  # turned fown for more stable training
LSTM_UNITS = 64  # Reduced from 128 for faster training
DROPOUT_RATE = 0.2

def build_model(timesteps, embedding_dim, vocab_size, lstm_units=LSTM_UNITS, dropout_rate=DROPOUT_RATE):
    '''
    Builds and compiles the LSTM model for text generation
    Args:
        timesteps (int): number of timesteps (ngram - 1)
        embedding_dim (int): dimension of each word embedding
        vocab_size (int): size of vocabulary (output dimension)
        lstm_units (int): number of LSTM units
        dropout_rate (float): dropout rate for regularization
    Returns:
        Compiled Keras model
    '''
    model = Sequential()
    
    # First LSTM layer with return sequences for stacking
    model.add(LSTM(lstm_units, input_shape=(timesteps, embedding_dim), return_sequences=True))
    model.add(Dropout(dropout_rate))
    
    # Second LSTM layer
    model.add(LSTM(lstm_units))
    model.add(Dropout(dropout_rate))
    
    # Dense output layer with softmax for word prediction
    model.add(Dense(vocab_size, activation='softmax'))
    
    # Compile model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

def train_model(model, data_gen, steps_per_epoch, epochs=EPOCHS):
    '''
    Trains the model with callbacks for checkpointing and early stopping
    Args:
        model: Keras model to train
        data_gen: data generator yielding batches
        steps_per_epoch (int): number of batches per epoch
        epochs (int): number of training epochs
    Returns:
        Training history
    '''
    # Create checkpoint to save best model
    # Use .h5 format for compatibility with Keras 3.x
    checkpoint = ModelCheckpoint(
        f"{MODEL_DIR}/best_model.h5",
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    
    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(
        monitor='loss',
        patience=3,  # 5->3 for more reasonable training time
        verbose=1,
        mode='min'
    )
    
    # Train the model
    history = model.fit(
        data_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )
    
    return history

def sample_with_temperature(preds, temperature=1.0):
    '''
    Sample from preds using temperature for diversity
    higher temp = more random, kower temp = more deterministic
    Args:
        preds: prediction probabilities from model
        temperature: sampling temp
    Returns:
        sampled index
    '''
    preds = np.asarray(preds).astype('float64')
    if temperature == 0:
        return np.argmax(preds)
    # add temperature scaling to make it more random or deterministic
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    # sample from the distribution
    return np.random.choice(len(preds), p=preds)

def generate_text(model, tokenizer, index_to_embedding, seed_text, vocab_size, num_words=50, ngram=NGRAM, temperature=0.8):
    '''
    Generates text using the trained model with temperature sampling
    Args:
        model: trained Keras model
        tokenizer: fitted tokenizer from prepare_data
        index_to_embedding: mapping from word index to embedding
        seed_text (str): initial text to start generation
        num_words (int): number of words to generate
        ngram (int): ngram size used in training
        temperature (float): sampling temperature (0.7-1.2 works well)
    Returns:
        Generated text string
    '''
    from prepare_data import tokenize_line, SENTENCE_BEGIN, SENTENCE_END
    
    # Tokenize seed text
    tokens = tokenize_line(seed_text.lower(), ngram, by_char=False, space_char="_")
    
    generated_text = seed_text
    
    for i in range(num_words):
        # Get the last (ngram-1) tokens
        if len(tokens) < ngram - 1:
            # Pad with sentence begin tokens if needed
            context = [SENTENCE_BEGIN] * (ngram - 1 - len(tokens)) + tokens
        else:
            context = tokens[-(ngram-1):]
        
        # Convert tokens to indices
        encoded_context = tokenizer.texts_to_sequences([context])[0]
        
        # If empty sequence, try using OOV token
        if not encoded_context:
            encoded_context = [1] * (ngram - 1)  # Use OOV tokens
        
        # Filter out padding tokens (0) and ensure we have enough context
        encoded_context = [idx for idx in encoded_context if idx > 0]  # Remove padding tokens
        
        # Handle unknown words (not in vocabulary)
        if len(encoded_context) < ngram - 1:
            # Pad with OOV token (1) if needed
            encoded_context = [1] * (ngram - 1 - len(encoded_context)) + encoded_context
        
        # Ensure we have exactly ngram-1 tokens
        encoded_context = encoded_context[-(ngram-1):]
        
        # Convert indices to embeddings (3D: batch_size=1, timesteps, embedding_dim)
        embeddings = []
        for idx in encoded_context:
            if idx in index_to_embedding:
                embeddings.append(index_to_embedding[idx])
            elif idx == 1:  # OOV token
                # Use a zero vector for OOV
                if len(embeddings) > 0:
                    embeddings.append(np.zeros_like(embeddings[0]))
                else:
                    embeddings.append(np.zeros(100))  # Default embedding dim
            else:
                # Unknown index, use OOV
                if len(embeddings) > 0:
                    embeddings.append(np.zeros_like(embeddings[0]))
                else:
                    embeddings.append(np.zeros(100))
        
        if len(embeddings) < ngram - 1:
            break
        
        # Shape: (1, timesteps, embedding_dim)
        embedding_sequence = np.array([embeddings])
        
        # Predict next word with temperature sampling
        predictions = model.predict(embedding_sequence, verbose=0)
        predicted_index = sample_with_temperature(predictions[0], temperature)
        
        # Handle padding token (index 0) - skip it
        if predicted_index == 0:
            # Try to sample again, excluding index 0
            probs = predictions[0].copy()
            probs[0] = 0  # Set padding probability to 0
            probs = probs / np.sum(probs)  # Renormalize
            predicted_index = sample_with_temperature(probs, temperature)
        
        # Ensure predicted_index is within valid range
        if predicted_index >= vocab_size or predicted_index < 0:
            break
        
        # Get the word for this index using reverse mapping
        # Create reverse index mapping if it doesn't exist
        if not hasattr(tokenizer, 'index_word'):
            tokenizer.index_word = {index: word for word, index in tokenizer.word_index.items()}
        
        # Look up the word
        predicted_word = tokenizer.index_word.get(predicted_index, None)
        
        if predicted_word is None:
            break
        
        if predicted_word == SENTENCE_END or predicted_word == SENTENCE_BEGIN:
            break
        
        # Handle space character replacement
        if predicted_word == "_":
            generated_text += " "
        elif predicted_word != SENTENCE_BEGIN:
            generated_text += " " + predicted_word
        
        tokens.append(predicted_word)
    
    return generated_text

if __name__ == "__main__":
    print("=" * 50)
    print("Loading and preparing data...")
    print("=" * 50)
    
    # Load data (limited for faster training)
    kaggle_jokes = read_kjokes_data()
    rJokes_scores, rJokes = read_rjokes_data("train.tsv", max_jokes=2000)  # Limit to 2000 jokes
    all_jokes = kaggle_jokes + rJokes
    
    # Create embeddings
    word_embeddings = create_word_embeddings(all_jokes, save=True, fp=MODEL_DIR)
    
    # Encode words (limit to 10000 most frequent words)
    encoded_words, word_tokenizer = encode_as_indices(all_jokes, max_vocab_size=10000)
    # Ensure vocab_size matches the limited vocabulary
    vocab_size = len(word_tokenizer.word_index) + 1  # +1 for padding token at index 0
    print(f"\nVocabulary size: {vocab_size} (limited from full vocabulary)")
    
    # Generate training samples
    training_samples = generate_ngram_training_samples(encoded_words, NGRAM)
    training_samples_xy = isolate_labels(training_samples, NGRAM)
    
    print(f"Total training samples: {len(training_samples_xy[0])}")
    
    # Load embeddings
    word_to_embedding, word_index_to_embedding = read_embeddings(
        f"{MODEL_DIR}/word_embeddings.txt", 
        word_tokenizer
    )
    
    # Get embedding dimension
    sample_embedding = list(word_index_to_embedding.values())[0]
    embedding_dim = sample_embedding.shape[0]
    timesteps = NGRAM - 1  # Number of context words
    
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Timesteps (context words): {timesteps}")
    
    # Create data generator
    data_gen = custom_data_generator(
        training_samples_xy[0], 
        training_samples_xy[1], 
        BATCH_SIZE, 
        word_index_to_embedding, 
        vocab_size
    )
    
    # Calculate steps per epoch
    steps_per_epoch = len(training_samples_xy[0]) // BATCH_SIZE
    
    print("\n" + "=" * 50)
    print("Building model...")
    print("=" * 50)
    
    # Build model
    model = build_model(timesteps, embedding_dim, vocab_size)
    model.summary()
    
    print("\n" + "=" * 50)
    print("Training model...")
    print("=" * 50)
    
    # Train model
    history = train_model(model, data_gen, steps_per_epoch, epochs=EPOCHS)
    
    # Save final model (use .h5 for compatibility)
    model.save(f"{MODEL_DIR}/final_model.h5")
    print(f"\nModel saved to {MODEL_DIR}/final_model.keras")
    
    # Save tokenizer for later use
    with open(f"{MODEL_DIR}/tokenizer.pkl", 'wb') as f:
        pickle.dump(word_tokenizer, f)
    print(f"Tokenizer saved to {MODEL_DIR}/tokenizer.pkl")
    
    print("\n" + "=" * 50)
    print("Generating sample text...")
    print("=" * 50)
    
    # Test text generation WITH different temperatures
    seed_texts = ["why did the", "a man walks into", "what do you call"]
    
    print("\nGenerating with temperature=0.7 (more focused):")
    for seed in seed_texts:
        generated = generate_text(
            model, 
            word_tokenizer, 
            word_index_to_embedding, 
            seed,
            vocab_size,
            num_words=25,
            temperature=0.7
        )
        print(f"\nSeed: '{seed}'")
        print(f"Generated: {generated}")
    
    print("\n" + "-" * 50)
    print("Generating with temperature=1.0 (balanced):")
    for seed in seed_texts:
        generated = generate_text(
            model, 
            word_tokenizer, 
            word_index_to_embedding, 
            seed,
            vocab_size,
            num_words=25,
            temperature=1.0
        )
        print(f"\nSeed: '{seed}'")
        print(f"Generated: {generated}")
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)