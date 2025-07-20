import torch
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
        return None, None, None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None, None, None

    # Load the tiktoken encoder
    try:
        tokenizer = tiktoken.get_encoding(encoding_name)
        print(f"Successfully loaded tiktoken encoding: '{encoding_name}'")
    except ValueError:
        print(f"Error: Encoding '{encoding_name}' not found. "
              "Please choose from available encodings (e.g., 'cl100k_base', 'gpt2', 'p50k_base', 'r50k_base').")
        return None, None, None
    except Exception as e:
        print(f"An error occurred while loading tiktoken encoding: {e}")
        return None, None, None

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
            # Note: tiktoken does not directly expose `tokens` (the string representation of subwords)
            # in the same way `tokenizers` library does without manually decoding each ID.
            # If you need the subword strings, you'd decode each ID individually or use another library.
            print(f"Decoded: '{decoded}'\n")

    return tokenizer, encoded_texts, decoded_texts

if __name__ == "__main__":
    text_file_path = r"D:\Story.txt"

    bpe_tokenizer, tokenized_data_ids, original_decoded_data = get_text_data_and_tiktoken_tokenize(
        text_file_path,
        encoding_name="cl100k_base"
    )

    if bpe_tokenizer and tokenized_data_ids:
        print("\n--- Summary ---")
        print(f"Number of tokenized texts: {len(tokenized_data_ids)}")
        print(f"Example of tokenized IDs (first text): {tokenized_data_ids[0]}")
        print(f"Example of decoded text (first text): '{original_decoded_data[0]}'")

        vocab_size = bpe_tokenizer.n_vocab
        print(f"Vocabulary size of '{bpe_tokenizer.name}' encoding: {vocab_size}")

        