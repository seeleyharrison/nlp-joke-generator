
'''
    This file prepares/processes our joke data into inputs for our
    RNN. It handles the following steps:

    1) Read in and clean both the kaggle and rJokes corpuses
    2) Tokenizes each dataset by our NGRAM parameter
    3) Generates word embeddings from both corpuses
    4) Encode tokens into indices
    5) Generates training samples and splits into x and y
    6) Creates a data generator for efficiency
    7) Map word embeddings to tokens and their index
'''

from gensim.models import Word2Vec
import pandas as pd
import re
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from gensim.models import KeyedVectors
from keras.utils import to_categorical
import numpy as np
nltk.download('punkt')

DATA_DIR = "../../data"
MODEL_DIR = "../models"
SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"
NGRAM = 3
BATCH_SIZE = 3

'''
    Tokenize a single string. Glue on the appropriate number of 
    sentence begin tokens and sentence end tokens (ngram - 1), except
    for the case when ngram == 1, when there will be one sentence begin
    and one sentence end token.
    Args:
        line (str): text to tokenize
        ngram (int): ngram preparation number
        by_char (bool): default value True, if True, tokenize by character, if
        False, tokenize by whitespace
        space_char (str): if by_char is True, use this character to separate to replace spaces
        sentence_begin (str): sentence begin token value
        sentence_end (str): sentence end token value

    Returns:
        list of strings - a single line tokenized
'''
def tokenize_line(line: str, ngram: int, 
                   by_char: bool = True, 
                   space_char: str = ' ',
                   sentence_begin: str=SENTENCE_BEGIN, 
                   sentence_end: str=SENTENCE_END):
  inner_pieces = None
  if by_char:
    line = line.replace(' ', space_char)
    inner_pieces = list(line)
  else:
    # otherwise use nltk's word tokenizer
    inner_pieces = nltk.word_tokenize(line)

  if ngram == 1:
    tokens = [sentence_begin] + inner_pieces + [sentence_end]
  else:
    tokens = ([sentence_begin] * (ngram - 1)) + inner_pieces + ([sentence_end] * (ngram - 1))
  # always count the unigrams
  return tokens

'''
    Reads in all jokes from the rJokes dataset and returns back a list of
    tokenized lines for each joke and the scores for each line. 
    Tokenizes based on the NGRAM parameter.
    Right now this only tokenizes by words, not characters.
    Returns (tuple):
        scores for each joke
        list of lists - each inner list is a single joke tokenized
'''
def read_rjokes_data(file):
    
    with open(f"{DATA_DIR}/{file}", 'r', encoding='utf-8') as f:  # Add encoding='utf-8'
        joke_lines = f.readlines()

    scores = []
    tokens = []
    for joke in joke_lines:
       parts = joke.split('\t', 1)
       scores.append(parts[0])
       joke_text = parts[1].replace('\t', '').replace('\n', '').replace('\r', '').strip()
       tokens.append(tokenize_line(joke_text.lower(), NGRAM, by_char=False, space_char="_"))

    return (scores, tokens)

'''
    Reads in all jokes from the kaggle dataset and returns back a list of
    tokenized lines for each joke. Tokenizes based on the NGRAM parameter.
    Right now this only tokenizes by words, not characters.
    Returns:
        list of lists - each inner list is a single joke tokenized
'''
def read_kjokes_data():
    df = pd.read_csv(f"{DATA_DIR}/one-million-reddit-jokes.csv", nrows=1000)
    df['all_text'] = df['title'] + ' ' + df['selftext']

    # Remove posts with redacted text ([removed], [deleted], etc)
    df = df[~df['all_text'].str.contains("[removed]", regex=False)]
    df = df[~df['all_text'].str.contains("[deleted]", regex=False)]
    text_list = df['all_text'].tolist()
    tokens = []

    for i in range(len(text_list)):

        # Clean away unwanted characters (should only have valid english characters, numbers, and punctuation)
        text_list[i] = re.sub(r'&?#?x[0-9a-fA-F]+;?', '', text_list[i])
        text_list[i] = re.sub(r'&[a-zA-Z]+;', '', text_list[i])
        text_list[i] = re.sub(r'[^a-zA-Z0-9\s\.,!?;:\'\"-]', '', text_list[i])
        text_list[i] = re.sub(r'\s+', ' ', text_list[i]).strip()

        # Tokenize each line
        tokens.append(tokenize_line(text_list[i].lower(), NGRAM, by_char=False, space_char="_"))

    return tokens

'''
    Creates word embeddings from the given sets of tokens. If flagged,
    saves the word embeddings to a file to be used later in training
    Returns
        Word2Vec word model
'''
def create_word_embeddings(tokens, save=True, fp="../models"):
   model = Word2Vec(sentences=tokens, window=5, min_count=1, sg=1)
   model.wv.save_word2vec_format(f"{fp}/word_embeddings.txt", binary=False)
   return model

'''
    Encodes words as indices and returns the tokenizer that performs
    this aciton.
    Returns (tuple):
        A list of lists, each inner list is the a joke encoded as indices
        The tokenizer object that achieves this goal
'''
def encode_as_indices(tokens):
    word_tokenizer = Tokenizer()
    word_tokenizer.fit_on_texts(tokens)
    return (word_tokenizer.texts_to_sequences(tokens), word_tokenizer)

'''
    Takes the encoded data (list of lists) and 
    generates the training samples out of it.
    Return:
        list of lists in the format [[x1, x2, ... , x(n-1), y], ...]
'''
def generate_ngram_training_samples(encoded: list, ngram: int) -> list:
    samples = []

    for line in encoded:
        length = len(line)
        i = 0
        while (i + ngram <= length):
            samples.append(line[i:i + ngram])
            i += 1

    return samples

'''
    Splits the training samples into x and y, essentially labels
    each training sample
    Returns:
        A list where the first index are the sequences and the second
        is the label for each sequence
'''
def isolate_labels(training_sequences: list, ngram: int) -> list:
    sequences = []
    labels = []

    for sequence in training_sequences:

        if ngram > 1:
            label = sequence[-1]
            sequence = sequence[:-1]
        else:
            label = sequence[0]
        
        sequences.append(sequence)
        labels.append(label)

    return [sequences, labels]

'''
    Loads, parses, and maps back word embeddings trained in earlier to word index.
    Parameters:
        filename (str): path to file
        Tokenizer: tokenizer used to tokenize the data (needed to get the word to index mapping)
    Returns:
        (dict): mapping from word to its embedding vector
        (dict): mapping from index to its embedding vector

'''
def read_embeddings(filename: str, tokenizer: Tokenizer):
    embeddings = KeyedVectors.load_word2vec_format(filename, binary=False)

    word_to_embedding = {word: embeddings[word] for word in embeddings.key_to_index}
    index_to_embedding = {index: embeddings[word] for word, index in tokenizer.word_index.items() if word in embeddings}

    return (word_to_embedding, index_to_embedding)


'''
    Returns data generator to be used by feed_forward
    https://wiki.python.org/moin/Generators
    https://realpython.com/introduction-to-python-generators/
    
    Yields batches of embeddings and labels to go with them.
    Use one hot vectors to encode the labels 
    (see the to_categorical function)
    
    Returns data generator to be used by feed_forward
'''
def data_generator(X: list, y: list, num_sequences_per_batch: int, index_2_embedding: dict, vocab_size: int):
    num_samples = len(X)

    while True:
        for start in range(0, num_samples, num_sequences_per_batch):
            end = start + num_sequences_per_batch
            X_batch_raw = X[start:end]
            y_batch = np.array(y[start:end])

            # Convert token indices to flattened embedding vectors
            X_batch = []
            for ngram in X_batch_raw:
                # Get embeddings for each token index in the (n-1)-gram
                embeddings = [index_2_embedding[idx] for idx in ngram]
                # Flatten them into a single vector
                flat_vector = np.concatenate(embeddings)
                X_batch.append(flat_vector)
            
            X_batch = np.array(X_batch)
            y_batch_categorical = to_categorical(y_batch, num_classes=vocab_size)

            yield X_batch, y_batch_categorical

def custom_data_generator(X: list, y: list, num_sequences_per_batch: int, index_2_embedding: dict, vocab_size: int):
    '''
    Returns data generator for LSTM training
    Yields batches of 3D embeddings (batch_size, timesteps, embedding_dim) and labels
    Args:
        X: list of sequences (each sequence is a list of token indices)
        y: list of labels (target token indices)
        num_sequences_per_batch: batch size
        index_2_embedding: mapping from token index to embedding vector
        vocab_size: size of vocabulary for one-hot encoding
    Yields:
        X_batch: (batch_size, timesteps, embedding_dim)
        y_batch_categorical: (batch_size, vocab_size)
    '''
    num_samples = len(X)

    while True:
        for start in range(0, num_samples, num_sequences_per_batch):
            end = start + num_sequences_per_batch
            X_batch_raw = X[start:end]
            y_batch = np.array(y[start:end])

            # Convert token indices to 3D embedding tensor
            X_batch = []
            for ngram in X_batch_raw:
                # Get embeddings for each token index in the (n-1)-gram
                embeddings = [index_2_embedding[idx] for idx in ngram]
                # Stack them to create (timesteps, embedding_dim)
                X_batch.append(np.array(embeddings))
            
            X_batch = np.array(X_batch)  # Shape: (batch_size, timesteps, embedding_dim)
            y_batch_categorical = to_categorical(y_batch, num_classes=vocab_size)

            yield X_batch, y_batch_categorical

if __name__ == "__main__":
    kaggle_jokes = read_kjokes_data()
    rJokes_scores, rJokes = read_rjokes_data("lightweight.tsv")
    all_jokes = kaggle_jokes + rJokes
    word_embeddings = create_word_embeddings(all_jokes)

    print("Word Embedding for 'girlfriend' is: ")
    print(word_embeddings.wv["girlfriend"])

    encoded_words, word_tokenizer = encode_as_indices(all_jokes)
    vocab_size = len(word_tokenizer.word_index) + 1
    print(f"Number of words we have in our corpus: {vocab_size}")

    training_samples = generate_ngram_training_samples(encoded_words, NGRAM)
    print("First 5 training samples")
    print(training_samples[0:5])

    training_samples_xy = isolate_labels(training_samples, NGRAM)
    print("First 5 training samples x")
    print(training_samples[0][0:3])
    print("First 5 training samples y")
    print(training_samples[1][0:3])

    word_to_embedding, word_index_to_embedding = read_embeddings(f"{MODEL_DIR}/word_embeddings.txt", word_tokenizer)
    print(f"Word embedding of 'girlfriend': {word_to_embedding['must']}")
    print(f"Word embedding of word index 1: {word_index_to_embedding[1]}")

    word_data_generator = data_generator(training_samples_xy[0], training_samples_xy[1], BATCH_SIZE, word_index_to_embedding, vocab_size)
    print(f"First batch of training samples: {next(word_data_generator)}")
        