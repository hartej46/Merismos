import pandas as pd
from pathlib import Path

# Define file paths based on your directory structure
DATA_DIR = Path("dataset")
ARXIV_DIR = DATA_DIR / "arxiv_csv"
CODE_DIR = DATA_DIR / "code_csv"
SHAKESPEARE_FILE = DATA_DIR / "input.txt"
OUTPUT_FILE = DATA_DIR / "combined_corpus.txt"

# The special boundary token for the Transformer to recognize context switches
SEPARATOR = "\n<|endoftext|>\n"

def extract_text_from_csv_dir(directory: Path, text_column: str) -> list:
    """Reads all CSVs in a directory and extracts the specified text column."""
    extracted_texts = []
    
    if not directory.exists():
        print(f"Warning: Directory {directory} not found.")
        return extracted_texts

    # Iterate through all CSV files in the folder
    for csv_file in directory.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if text_column in df.columns:
                # Drop empty rows and convert the column to a list of strings
                texts = df[text_column].dropna().astype(str).tolist()
                extracted_texts.extend(texts)
            else:
                print(f"Warning: Column '{text_column}' not found in {csv_file.name}")
        except Exception as e:
            print(f"Failed to read {csv_file.name}: {e}")
            
    return extracted_texts

def build_dataset():
    all_documents = []

    # 1. Load Tiny Shakespeare
    if SHAKESPEARE_FILE.exists():
        with open(SHAKESPEARE_FILE, 'r', encoding='utf-8') as f:
            all_documents.append(f.read())
        print("Loaded Tiny Shakespeare.")

    # 2. Extract ArXiv text
    # NOTE: Change 'abstract' to the exact column header used in your ArXiv CSVs
    arxiv_docs = extract_text_from_csv_dir(ARXIV_DIR, text_column="question")
    all_documents.extend(arxiv_docs)
    print(f"Loaded {len(arxiv_docs)} ArXiv documents.")

    # 3. Extract Source Code text
    # NOTE: Change 'content' to the exact column header used in your Code CSVs
    code_docs = extract_text_from_csv_dir(CODE_DIR, text_column="text")
    all_documents.extend(code_docs)
    print(f"Loaded {len(code_docs)} Code documents.")

    # 4. Concatenate everything with the separator token
    print("Concatenating corpus...")
    final_corpus = SEPARATOR.join(all_documents)

    # 5. Save the unified dataset to disk
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(final_corpus)
    
    print(f"\nSuccess! Combined {len(all_documents)} total documents.")
    print(f"Total corpus size: {len(final_corpus) / (1024*1024):.2f} MB")
    print(f"Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    build_dataset()