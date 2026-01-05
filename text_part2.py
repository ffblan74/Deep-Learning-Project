import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, GRU
from tensorflow.keras.models import Model
import string
import os


# Step 1: texte preprocessing
def load_flickr_descriptions(filename):
    """
    Reads the Flickr8k token file and returns a dictionary.
    Format: image_id -> list of 5 descriptions
    """
    mapping = {}
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read()
            
        for line in text.split('\n'):
            
            # Skip empty lines
            if len(line) < 2:
                continue
                
            # Split the line by whitespace
            tokens = line.split()
            
            # The first token is the image ID (ex : 1000268201_693b08cb0e.jpg#0)
            image_id = tokens[0]
            
            # The rest is the description
            image_desc = ' '.join(tokens[1:])
            
            # Remove the tag part (#0, #1, etc.) from the ID
            image_id = image_id.split('#')[0]
            
            # Add to dictionary
            if image_id not in mapping:
                mapping[image_id] = []
            
            mapping[image_id].append(image_desc)
            
        print(f"Loaded {len(mapping)} images from {filename}")
        return mapping

    except FileNotFoundError:
        print(f"ERROR: File '{filename}' not found.")
        return {}
    
def clean_descriptions(descriptions_dict):
    """
    Cleans the text: lowercase, punctuation removal, adds <start>/<end> tags.
    """
    translation_table = str.maketrans('', '', string.punctuation)
    clean_descriptions_dict = {}

    for image_id, desc_list in descriptions_dict.items():
        new_list = []
        for desc in desc_list:
            # Convert to lowercase
            desc = desc.lower()
            # Remove punctuation
            desc = desc.translate(translation_table)
            # Keep only alphabetic words (remove isolated numbers)
            words = [word for word in desc.split() if len(word) > 1 and word.isalpha()]
            desc = ' '.join(words)
            #Add tags (Crucial for the RNN sequence)
            desc = '<start> ' + desc + ' <end>'
            new_list.append(desc)
        clean_descriptions_dict[image_id] = new_list
        
    return clean_descriptions_dict

def create_tokenizer(all_descriptions_list):
    """
    Creates the vocabulary dictionary (Tokenization).
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_descriptions_list)
    return tokenizer

# Step 2: pre-trained word embeddings (GloVe)
def load_glove_embeddings(glove_file_path, word_index, embedding_dim=200):
    """
    Loads GloVe vectors to avoid learning word meanings from scratch.
    """
    print(f"Loading GloVe from {glove_file_path}...")
    embeddings_index = {}
    
    # Read the GloVe file
    try:
        with open(glove_file_path, encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
    except FileNotFoundError:
        print("GLOVE file not found. Using empty embedding (training from scratch).")
        return None

    # Create the weight matrix for our specific vocabulary
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    hits = 0
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            hits += 1
            
    print(f"GloVe: {hits} words found out of {vocab_size} in the vocabulary.")
    return embedding_matrix


# Step 3: RNN / LSTM
def build_text_model(vocab_size, max_length, embedding_dim=200, embedding_matrix=None):
    """
    Builds the Text branch of the project.
    Input: Sequence of words.
    Output: Encoding vector (256).
    """
    
    # Input Layer
    text_input = Input(shape=(max_length,), name="text_input")
    
    # Embedding Layer (With or without GloVe)
    if embedding_matrix is not None:
        # Case we use GloVe
        embedding_layer = Embedding(vocab_size, 
                                    embedding_dim, 
                                    weights=[embedding_matrix], 
                                    trainable=False, # trainable=False means we freeze weights to keep GloVe knowledge
                                    mask_zero=True)(text_input)
    else:
        # Case we learning from scratch (if no GloVe file found)
        embedding_layer = Embedding(vocab_size, 
                                    embedding_dim, 
                                    mask_zero=True)(text_input)
    

    x = Dropout(0.5)(embedding_layer)
    
    # LSTM Layer
    # x = LSTM(256)(x)
    x = GRU(256)(x)
    # Sequence Encoding (Output)
    encoding_output = Dense(256, activation='relu', name="text_vector_output")(x)
    
    model = Model(inputs=text_input, outputs=encoding_output, name="Text_RNN_Branch")
    return model



# Quick test (To verify the file works independently)
if __name__ == "__main__":
    print("--- Starting Part 2: Text Processing ---")
    
    # 1. Load Real Data (Flickr8k)
    filename = 'Flickr8k.token.txt' 
    
    if os.path.exists(filename):
        # A. Load and Clean descriptions
        descriptions = load_flickr_descriptions(filename)
        clean_desc = clean_descriptions(descriptions)
        all_sentences = [desc for desc_list in clean_desc.values() for desc in desc_list]
        
        # B. Create Tokenizer
        my_tokenizer = create_tokenizer(all_sentences)
        vocab_size = len(my_tokenizer.word_index) + 1
        print(f"Vocabulary Size: {vocab_size}")
        
        # C. Calculate max length
        max_length = max(len(d.split()) for d in all_sentences)
        print(f"Max sentence length: {max_length}")
        
        # --- LOAD GLOVE ---
        glove_path = 'glove.6B.300d.txt' # This file should be in the same directory
        
        if os.path.exists(glove_path):
            print("GLOVE file found. Loading matrix...")
            embedding_matrix = load_glove_embeddings(glove_path, my_tokenizer.word_index, embedding_dim=300)
        else:
            print("GLOVE file not found. Training embeddings from scratch.")
            embedding_matrix = None
            
        # D. Build Model (Passing the matrix)
        model = build_text_model(vocab_size, max_length, embedding_dim=300, embedding_matrix=embedding_matrix)
        
        model.summary()
        print("Model ready for Fusion.")
        
    else:
        print(f"File {filename} is missing. Download the dataset.")