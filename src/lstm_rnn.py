import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)  # Suppress gensim NumPy warnings
import argparse
import os
import time
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau
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
)

'''
    This file trains an RNN (LSTM) model for text generation using the
    prepared joke data. It handles the following steps:

    1) Load the preprocessed data from prepare_data.py
    2) Build an LSTM-based neural network
    3) Train the model on the joke corpus
    4) Save the trained model and tokenizer
    5) Evaluates perplexity and loss
    5) Generates a few output examples

    This was the first model we attempted for this project. As you will see
    from our perplexity and loss scores, this model had poor performance,
    leading to us deciding to explore the transfer learning alternative
    using gpt2. We really struggled to train a model big/complex enough for our task
    on the hardware (our laptops) that were available to us.

    If you do not want to build a new model and retrain from scratch
    (this could take hours), you can alternatively perform evaluation and text generation
    on our best model.

    Instructions on how to run this model can be seen in the main function below!
'''

# Don't change!
MODEL_DIR='lstm_rnn_models'

# Our optimal base training parameters, determined through trial and error
EPOCHS = 10 # We experimented with 2 and 3 for the majority, but went with 10 for our final attempt
LEARNING_RATE = 0.001 # This script automatically adjusts this if loss minimally improves between epochs
LSTM_UNITS = 128 # We started with a more lightweight 64 units, bumped this up to 128 for the final version
DROPOUT_RATE = 0.2 # This stayed constant through most of our trials

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
    optimizer = Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
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
    # Use .h5 format (compatibility issues with Keras)
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
        patience=3,
        verbose=1,
        mode='min'
    )

    # ADjust the learning rate for the next epoch if loss does not improve
    reduce_lr = ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=2,
        min_lr=0.0001,
        verbose=1
    )
    
    # Train!
    history = model.fit(
        data_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )
    
    return history

def sample_with_temperature(preds, temperature=1.0):
    '''
    Sample from predictions along with a temperature factor
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
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    return np.random.choice(len(preds), p=preds)

def generate_text(model, tokenizer, index_to_embedding, seed_text, vocab_size, num_words=50, ngram=NGRAM, temperature=0.8):
    '''
    Generates text using the trained model with temperature sampling
    Args:
        model: trained Keras model
        tokenizer: fitted tokenizer from prepare_data
        index_to_embedding: mapping from word index to embedding
        seed_text (str): initial text to start generation
        vocab_size (int): number of vocab words in the corpus
        num_words (int): number of words to generate
        ngram (int): ngram size used in training
        temperature (float): adjust the randomness/spicyness of output
    Returns:
        A generated joke as a string
    '''
    from prepare_data import tokenize_line, SENTENCE_BEGIN, SENTENCE_END
    
    # Tokenize seed text
    tokens = tokenize_line(seed_text.lower(), ngram, by_char=False, space_char="_")
    generated_text = seed_text
    
    # Keep generating the next word until we reach our word limit
    for i in range(num_words):

        # Pad with begin token if needed
        if len(tokens) < ngram - 1:
            context = [SENTENCE_BEGIN] * (ngram - 1 - len(tokens)) + tokens
        else:
            context = tokens[-(ngram-1):]
        
        # Look up token index
        encoded_context = tokenizer.texts_to_sequences([context])[0]
        
        # If empty sequence, try using OOV token
        if not encoded_context:
            encoded_context = [1] * (ngram - 1)  # Use OOV tokens
        
        # Filter out padding tokens (0) and ensure we have enough context
        encoded_context = [idx for idx in encoded_context if idx > 0]
        
        # Handle unknown words and pad with OOV token
        if len(encoded_context) < ngram - 1:
            encoded_context = [1] * (ngram - 1 - len(encoded_context)) + encoded_context
    
        encoded_context = encoded_context[-(ngram-1):]
        
        # Get our embeddings from our indices
        embeddings = []
        for idx in encoded_context:
            if idx in index_to_embedding:
                embeddings.append(index_to_embedding[idx])
            elif idx == 1:  # OOV token
                # Use a zero vector for OOV
                if len(embeddings) > 0:
                    embeddings.append(np.zeros_like(embeddings[0]))
                # Default length otherwise
                else:
                    embeddings.append(np.zeros(100))
            else:
                # Unknown index, use OOV
                if len(embeddings) > 0:
                    embeddings.append(np.zeros_like(embeddings[0]))
                else:
                    embeddings.append(np.zeros(100))
        
        if len(embeddings) < ngram - 1:
            break
        
        # Convert our embeddings to np array
        embedding_sequence = np.array([embeddings])
        
        # Predict next word with temperature sampling
        predictions = model.predict(embedding_sequence, verbose=0)
        predicted_index = sample_with_temperature(predictions[0], temperature)
        
        # Handle padding token (index 0) - skip it
        if predicted_index == 0:
            probs = predictions[0].copy()
            probs[0] = 0
            probs = probs / np.sum(probs)
            predicted_index = sample_with_temperature(probs, temperature)
        
        # Ensure the predicted index is a valid word we can look up
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
        
        # Stop if we've genered the end or begin token
        if predicted_word == SENTENCE_END or predicted_word == SENTENCE_BEGIN:
            break
        
        # Handle space character replacement
        if predicted_word == "_":
            generated_text += " "
        elif predicted_word != SENTENCE_BEGIN:
            generated_text += " " + predicted_word
        
        tokens.append(predicted_word)
    
    return generated_text

def evaluate_model_perplexity(model, data_gen, steps):
    """
    Computes loss and perplexity on a given dataset using the
    same generator format used for training.

    Args:
        model: trained Keras model
        data_gen: generator yielding (X, y) batches
        steps: number of batches to evaluate
    
    Returns:
        (loss, perplexity)
    """
    # Keras evaluate returns [loss, accuracy]
    results = model.evaluate(
        data_gen,
        steps=steps,
        verbose=1
    )

    loss = results[0]
    perplexity = np.exp(loss)

    return loss, perplexity

'''
    Main program/entry point to run evaluation and text generation on our model

    RUN INSTRUCTIONS
    Note: If you haven't already, create/activate a new python virtual environment at the
    root of this repository.

    OPTION 1: Build and train a new model
    1) From the repository root, run the command:
        python src/lstm_rnn.py --train

    OPTION 2: Use an existing model
    1) From the repository root, run the command:
        python src/lstm_rnn.py

    After the model was built and trained or loaded from a file, the script will perform
    perplexity and loss evaluation and then print out three samples of jokes!
'''
if __name__ == "__main__":

    # Read program arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', 
                       help='Force retraining even if model exists')
    args = parser.parse_args()
    model_path = f"{MODEL_DIR}/final_model.h5"

    # Load training data
    kaggle_jokes = read_kjokes_data()
    rJokes_scores, rJokes = read_rjokes_data("train.tsv", max_jokes=10000)  # Limit number of jokes
    all_jokes = kaggle_jokes + rJokes

        # Create embeddings
    word_embeddings = create_word_embeddings(all_jokes, save=True, fp=MODEL_DIR)
    
    # Encode words, limit vocab size for more efficient training
    encoded_words, word_tokenizer = encode_as_indices(all_jokes, max_vocab_size=10000)
    vocab_size = len(word_tokenizer.word_index) + 1
    print(f"\nVocabulary size: {vocab_size} (limited from full vocabulary)")
    
    # Generate training samples
    training_samples = generate_ngram_training_samples(encoded_words, NGRAM)
    training_samples_xy = isolate_labels(training_samples, NGRAM)
    print(f"Total training samples: {len(training_samples_xy[0])}")

    # If a model already exists and the user has not prompted retraining, load existing best model
    if os.path.exists(model_path) and not args.train:
        print("\n" + "=" * 50)
        print("Loading in existing model...")
        print("=" * 50)

        # Load existing model
        model = load_model(model_path)

        # Load in tokenizer
        tokenizer_path = f"{MODEL_DIR}/tokenizer.pkl"
        with open(tokenizer_path, 'rb') as f:
            word_tokenizer = pickle.load(f)

        # Load in word embeddings
        word_to_embedding, word_index_to_embedding = read_embeddings(
            f"{MODEL_DIR}/word_embeddings.txt", 
            word_tokenizer
        )
        vocab_size = len(word_tokenizer.word_index) + 1

    # Retrain from scratch otherwise
    else:
        print("\n" + "=" * 50)
        print("Building and Training a new model...")
        print("=" * 50)
        
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
        
        # Build model
        model = build_model(timesteps, embedding_dim, vocab_size)
        model.summary()
        
        print("\n" + "=" * 50)
        print("Beginning Model Training...")
        print("=" * 50)
        
        # Train model
        start = time.time()
        history = train_model(model, data_gen, steps_per_epoch, epochs=EPOCHS)
        end = time.time()
        elapsed = end - start
        minutes = int(elapsed // 60)
        seconds = elapsed % 60

        print("\n" + "=" * 50)
        print(f"Training complete! Training took {minutes}m {seconds}s")
        print("=" * 50)
        
        # Save final model (use .h5 for compatibility)
        model.save(f"{MODEL_DIR}/final_model.h5")
        print(f"\nModel saved to {MODEL_DIR}/final_model.h5")
        
        # Save tokenizer for later use
        with open(f"{MODEL_DIR}/tokenizer.pkl", 'wb') as f:
            pickle.dump(word_tokenizer, f)
        print(f"Tokenizer saved to {MODEL_DIR}/tokenizer.pkl")

    print("\n" + "=" * 50)
    print("Computing Perplexity and Loss!")
    print("=" * 50)
    
    eval_gen = custom_data_generator(
        training_samples_xy[0],
        training_samples_xy[1],
        BATCH_SIZE,
        word_index_to_embedding,
        vocab_size
    )

    eval_steps = len(training_samples_xy[0]) // BATCH_SIZE
    loss, ppl = evaluate_model_perplexity(model, eval_gen, eval_steps)

    print(f"Perplexity: {ppl}")
    print(f"Loss: {loss}")

    print("\n" + "=" * 50)
    print("Generating some jokes!")
    print("=" * 50)
    
    # Test text generation WITH different temperatures
    seed_texts = ["why did the", "a man walks into", "what do you call"]
    
    temperature = 0.7
    print(f"\nGenerating with temperature={temperature} (more focused):")
    for seed in seed_texts:
        generated = generate_text(
            model, 
            word_tokenizer, 
            word_index_to_embedding, 
            seed,
            vocab_size,
            num_words=25,
            temperature=temperature
        )
        print(f"\nSeed: '{seed}'")
        print(f"Generated: {generated}")
    
    print("\n" + "-" * 50)

    temperature = 1.2
    print(f"Generating with temperature={temperature} (balanced):")
    for seed in seed_texts:
        generated = generate_text(
            model, 
            word_tokenizer, 
            word_index_to_embedding, 
            seed,
            vocab_size,
            num_words=25,
            temperature=temperature
        )
        print(f"\nSeed: '{seed}'")
        print(f"Generated: {generated}")
    
    print("\n" + "=" * 50)

    temperature = 2.0
    print(f"Generating with temperature={temperature} (spicy):")
    for seed in seed_texts:
        generated = generate_text(
            model, 
            word_tokenizer, 
            word_index_to_embedding, 
            seed,
            vocab_size,
            num_words=25,
            temperature=temperature
        )
        print(f"\nSeed: '{seed}'")
        print(f"Generated: {generated}")
    print("\n" + "=" * 50)