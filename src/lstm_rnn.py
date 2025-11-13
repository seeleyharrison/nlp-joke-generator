'''
    This file trains an RNN (LSTM) model for text generation using the
    prepared joke data. It handles the following steps:

    1) Load the preprocessed data from prepare_data.py
    2) Build an LSTM-based neural network
    3) Train the model on the joke corpus
    4) Save the trained model and tokenizer
    5) Provide text generation functionality
'''

import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.utils import to_categorical
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
EPOCHS = 2
LEARNING_RATE = 0.0015
LSTM_UNITS = 128
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
    checkpoint = ModelCheckpoint(
        f"{MODEL_DIR}/best_model.keras",
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    
    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(
        monitor='loss',
        patience=5,
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

def generate_text(model, tokenizer, index_to_embedding, seed_text, num_words=50, ngram=NGRAM):
    '''
    Generates text using the trained model
    Args:
        model: trained Keras model
        tokenizer: fitted tokenizer from prepare_data
        index_to_embedding: mapping from word index to embedding
        seed_text (str): initial text to start generation
        num_words (int): number of words to generate
        ngram (int): ngram size used in training
    Returns:
        Generated text string
    '''
    from prepare_data import tokenize_line, SENTENCE_BEGIN, SENTENCE_END
    
    # Tokenize seed text
    tokens = tokenize_line(seed_text.lower(), ngram, by_char=False, space_char="_")
    
    generated_text = seed_text
    
    for _ in range(num_words):
        # Get the last (ngram-1) tokens
        if len(tokens) < ngram - 1:
            # Pad with sentence begin tokens if needed
            context = [SENTENCE_BEGIN] * (ngram - 1 - len(tokens)) + tokens
        else:
            context = tokens[-(ngram-1):]
        
        # Convert tokens to indices
        encoded_context = tokenizer.texts_to_sequences([context])[0]
        
        # Handle unknown words (not in vocabulary)
        if len(encoded_context) < ngram - 1:
            # If some words are unknown, we can't make a prediction
            break
        
        # Convert indices to embeddings (3D: batch_size=1, timesteps, embedding_dim)
        embeddings = [index_to_embedding[idx] for idx in encoded_context if idx in index_to_embedding]
        if len(embeddings) < ngram - 1:
            break
        
        # Shape: (1, timesteps, embedding_dim)
        embedding_sequence = np.array([embeddings])
        
        # Predict next word
        predictions = model.predict(embedding_sequence, verbose=0)
        predicted_index = np.argmax(predictions[0])
        
        # Get the word for this index
        predicted_word = None
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                predicted_word = word
                break
        
        if predicted_word is None or predicted_word == SENTENCE_END:
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
    
    # Load data
    kaggle_jokes = read_kjokes_data()
    rJokes_scores, rJokes = read_rjokes_data("lightweight.tsv")
    all_jokes = kaggle_jokes + rJokes
    
    # Create embeddings
    word_embeddings = create_word_embeddings(all_jokes, save=True, fp=MODEL_DIR)
    
    # Encode words
    encoded_words, word_tokenizer = encode_as_indices(all_jokes)
    vocab_size = len(word_tokenizer.word_index) + 1
    print(f"\nVocabulary size: {vocab_size}")
    
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
    
    # Save final model
    model.save(f"{MODEL_DIR}/final_model.keras")
    print(f"\nModel saved to {MODEL_DIR}/final_model.keras")
    
    # Save tokenizer for later use
    with open(f"{MODEL_DIR}/tokenizer.pkl", 'wb') as f:
        pickle.dump(word_tokenizer, f)
    print(f"Tokenizer saved to {MODEL_DIR}/tokenizer.pkl")
    
    print("\n" + "=" * 50)
    print("Generating sample text...")
    print("=" * 50)
    
    # Test text generation
    seed_texts = ["why did the", "a man walks into", "what do you call"]
    
    for seed in seed_texts:
        generated = generate_text(
            model, 
            word_tokenizer, 
            word_index_to_embedding, 
            seed, 
            num_words=20
        )
        print(f"\nSeed: '{seed}'")
        print(f"Generated: {generated}")
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)