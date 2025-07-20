import torch
import torch.nn as nn # Import nn for the Embedding layer
import tiktoken
from pathlib import Path

def get_text_data_and_tiktoken_tokenize(file_path: str, encoding_name: str = "cl100k_base"):
    try:
        # 1. Get text data from a text file
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"Successfully loaded {len(texts)} lines from '{file_path}'")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None, None, None, None, None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None, None, None, None, None

    # Load the tiktoken encoder
    try:
        tokenizer = tiktoken.get_encoding(encoding_name)
        print(f"Successfully loaded tiktoken encoding: '{encoding_name}'")
    except ValueError:
        print(f"Error: Encoding '{encoding_name}' not found. "
              "Please choose from available encodings (e.g., 'cl100k_base', 'gpt2', 'p50k_base', 'r50k_base').")
        return None, None, None, None, None
    except Exception as e:
        print(f"An error occurred while loading tiktoken encoding: {e}")
        return None, None, None, None, None

    # Perform BPE subword tokenization
    encoded_texts = []
    decoded_texts = []
    print("\nPerforming tokenization on loaded texts:")

    for i, text in enumerate(texts):
        encoded_ids = tokenizer.encode(text) # Or tokenizer.encode(text, allowed_special="all")
        encoded_texts.append(encoded_ids)

        # Decode back to verify
        decoded = tokenizer.decode(encoded_ids)
        decoded_texts.append(decoded)

        if i < 3:  # Print first 3 for demonstration
            print(f"Original: '{text}'")
            print(f"IDs: {encoded_ids}")
            print(f"Decoded: '{decoded}'\n")

    # Get vocabulary size
    vocab_size = tokenizer.n_vocab
    print(f"Vocabulary size of '{tokenizer.name}' encoding: {vocab_size}")

    # Define embedding dimension (you can choose any appropriate size)
    embedding_dim = 768 # Common embedding dimension for many models (e.g., BERT-like)
    print(f"Using embedding dimension: {embedding_dim}")

    # Initialize the embedding layer
    # This layer will map each token ID to a dense vector of size embedding_dim
    token_embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    print("Initialized `torch.nn.Embedding` layer.")

    # Convert encoded texts to tensors and get embeddings
    # This assumes you want to process each text individually.
    # For batch processing, you would typically pad sequences to a common length.
    embedded_texts = []
    print("\nGenerating token embeddings:")
    for i, encoded_ids in enumerate(encoded_texts):
        # Convert list of IDs to a PyTorch tensor
        input_tensor = torch.tensor(encoded_ids, dtype=torch.long) # dtype must be long for embedding layer

        # Get embeddings for the current text
        # The output shape will be (sequence_length, embedding_dim)
        embeddings = token_embedding_layer(input_tensor)
        embedded_texts.append(embeddings)

        if i < 3: # Print first 3 for demonstration
            print(f"Original text index {i}:")
            print(f"  Number of tokens: {len(encoded_ids)}")
            print(f"  Shape of embeddings: {embeddings.shape}") # Should be (num_tokens, embedding_dim)
            print(f"  Example embedding for first token (first 5 dimensions): {embeddings[0][:5].tolist()}...")
            print("-" * 20)

    return tokenizer, encoded_texts, decoded_texts, token_embedding_layer, embedded_texts

if __name__ == "__main__":
    # Create a dummy file for demonstration purposes
    dummy_file_path = r"D:\Story.txt"
    with open(dummy_file_path, "w", encoding="utf-8") as f:
        f.write("Hello world, this is a test sentence.\n")
        f.write("Another line to demonstrate tokenization and embedding.\n")
        f.write("GenAI is fascinating.\n")
        f.write("More text here.\n")

    text_file_path = dummy_file_path # Use the dummy file for execution

    bpe_tokenizer, tokenized_data_ids, original_decoded_data, embedding_layer, embedded_data = \
        get_text_data_and_tiktoken_tokenize(
            text_file_path,
            encoding_name="cl100k_base"
        )

    if bpe_tokenizer and tokenized_data_ids and embedding_layer and embedded_data:
        print("\n--- Final Summary ---")
        print(f"Number of tokenized texts: {len(tokenized_data_ids)}")
        print(f"Example of tokenized IDs (first text): {tokenized_data_ids[0]}")
        print(f"Example of decoded text (first text): '{original_decoded_data[0]}'")
        print(f"Vocabulary size of '{bpe_tokenizer.name}' encoding: {bpe_tokenizer.n_vocab}")
        print(f"Embedding layer: {embedding_layer}")
        print(f"Number of embedded texts: {len(embedded_data)}")
        print(f"Shape of embeddings for the first text: {embedded_data[0].shape}")

    # Clean up the dummy file
    Path(dummy_file_path).unlink(missing_ok=True)