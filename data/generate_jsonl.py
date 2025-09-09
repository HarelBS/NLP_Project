# This file utilized the https://huggingface.co/datasets/mandarjoshi/trivia_qa dataset.
# It generates out of it 2 JSONL files with the following format:
# {"prompt": "Q: What is the capital of France? A:", "generation": "Paris"}
# The JSONL files are saved in this script's directory with the filenames "trivia_qa_train.jsonl" and "trivia_qa_test.jsonl".

import json
import datasets
import unicodedata
import pathlib
from transformers import AutoTokenizer

ROWS = 5000


def normalize_text(text):
    """Normalize Unicode characters to ASCII equivalents"""
    # Normalize Unicode characters
    text = unicodedata.normalize("NFKD", text)

    # Replace common Unicode quotes and dashes with ASCII equivalents
    text = text.replace("\u2018", "'")  # Left single quote
    text = text.replace("\u2019", "'")  # Right single quote
    text = text.replace("\u201c", '"')  # Left double quote
    text = text.replace("\u201d", '"')  # Right double quote
    text = text.replace("\u2013", "-")  # En dash
    text = text.replace("\u2014", "--")  # Em dash
    text = text.replace("\u2026", "...")  # Ellipsis
    text = text.replace("\u00a3", "£")  # Pound sterling symbol

    return text


def get_single_token_answer(text, tokenizer):
    """Get a single token answer from the text using the Pythia tokenizer"""
    # Normalize the text first
    normalized_text = normalize_text(text)

    # Tokenize the text
    tokens = tokenizer.tokenize(normalized_text)

    if not tokens:
        # If no tokens, return the original text
        return normalized_text

    # Return the first token (remove any special characters that might be added)
    first_token = tokens[0]

    # Remove any special characters that the tokenizer might add
    if first_token.startswith("Ġ"):  # Pythia tokenizer prefix
        first_token = first_token[1:]

    return first_token


def main():
    """Main function to generate JSONL files from the TriviaQA dataset and made-up QA data"""
    # Load the Pythia tokenizer
    print("Loading Pythia tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped")

    # Load only the rows we need directly from the dataset
    train_dataset = datasets.load_dataset(
        "mandarjoshi/trivia_qa", "rc.nocontext", split=f"train[:{ROWS}]"
    )

    print(f"Training set: {len(train_dataset)} rows")

    with open(pathlib.Path(__file__).parent / "trivia_qa_train.jsonl", "w") as f:
        for i in train_dataset:
            f.write(
                json.dumps(
                    {
                        "prompt": "Q: " + normalize_text(i["question"]) + " A:",
                        "generation": i["answer"]["value"],
                    }
                )
                + "\n"
            )

    # Process the made-up QA data
    print("Processing made-up QA data...")

    # Load the made-up QA JSON file
    with open(pathlib.Path(__file__).parent / "made_up_QA.json", "r") as f:
        made_up_data = json.load(f)

    print(f"Made-up QA data: {len(made_up_data)} questions")

    # Create made_up_ver1.jsonl with answer1
    with open(pathlib.Path(__file__).parent / "made_up_ver1.jsonl", "w") as f:
        for item in made_up_data:
            f.write(
                json.dumps(
                    {
                        "prompt": "Q: " + normalize_text(item["question"]) + " A:",
                        "generation": get_single_token_answer(
                            item["answer1"], tokenizer
                        ),
                    }
                )
                + "\n"
            )

    # Create made_up_ver2.jsonl with answer2
    with open(pathlib.Path(__file__).parent / "made_up_ver2.jsonl", "w") as f:
        for item in made_up_data:
            f.write(
                json.dumps(
                    {
                        "prompt": "Q: " + normalize_text(item["question"]) + " A:",
                        "generation": get_single_token_answer(
                            item["answer2"], tokenizer
                        ),
                    }
                )
                + "\n"
            )

    print("Successfully created all JSONL files:")
    print("- trivia_qa_train.jsonl")
    print("- made_up_ver1.jsonl")
    print("- made_up_ver2.jsonl")


if __name__ == "__main__":
    main()
