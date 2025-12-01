import os
import pandas as pd

# ==============================
# Configuration / Paths / Labels
# ==============================

# Assuming this file is inside src/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
ECONOMY_DIR = os.path.join(DATA_DIR, "Economy")
SPORTS_DIR = os.path.join(DATA_DIR, "Sports")

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

LABEL_ECONOMY = "Economy"
LABEL_SPORTS = "Sports"


# ==============================
# File loading
# ==============================

def read_file_text(file_path: str, encoding: str = "utf-8") -> str:
    """
    Read the full content of a file and return it as a single string.
    No cleaning or preprocessing is applied here.
    """
    with open(file_path, "r", encoding=encoding, errors="ignore") as f:
        text = f.read()
    return text


def load_from_folder(folder_path: str, label_name: str):
    """
    Traverse a folder, read all text-like files, and attach a label.
    Returns a list of dictionaries: [{filename, text, label}, ...]
    """
    documents = []

    for filename in os.listdir(folder_path):
        # Accept both .txt and .html files (dataset uses .html)
        if filename.lower().endswith((".txt", ".html")):
            file_path = os.path.join(folder_path, filename)
            text = read_file_text(file_path)

            doc = {
                "filename": filename,
                "text": text,
                "label": label_name
            }
            documents.append(doc)

    return documents


# ==============================
# Data loader (reusable)
# ==============================

def load_dataset() -> pd.DataFrame:
    """
    Load all documents from:
      data/economy/ -> label = 'Economy'
      data/sports/  -> label = 'Sports'
    Combine them into a single pandas DataFrame with columns:
      filename, text, label
    """
    economy_docs = load_from_folder(ECONOMY_DIR, LABEL_ECONOMY)
    sports_docs = load_from_folder(SPORTS_DIR, LABEL_SPORTS)

    all_docs = economy_docs + sports_docs

    df = pd.DataFrame(all_docs, columns=["filename", "text", "label"])
    return df


# ==============================
# Dataset summary (for Phase 2)
# ==============================

def save_dataset_summary(df: pd.DataFrame, summary_path: str | None = None):
    """
    Save a simple verified dataset summary into a .txt file:
      - number of docs per class
      - total docs
      - a few example rows
    """
    if summary_path is None:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        summary_path = os.path.join(OUTPUTS_DIR, "dataset_summary.txt")

    total_docs = len(df)
    economy_count = (df["label"] == LABEL_ECONOMY).sum()
    sports_count = (df["label"] == LABEL_SPORTS).sum()

    sample_rows = df.head(3)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("==== DATASET SUMMARY ====\n\n")
        f.write(f"Number of Economy documents: {economy_count}\n")
        f.write(f"Number of Sports documents : {sports_count}\n")
        f.write(f"Total documents            : {total_docs}\n\n")

        f.write("==== SAMPLE ROWS ====\n\n")
        for _, row in sample_rows.iterrows():
            preview = row["text"][:150].replace("\n", " ")
            f.write(f"filename: {row['filename']}\n")
            f.write(f"label   : {row['label']}\n")
            f.write(f"text    : {preview}...\n\n")

    print(f"Dataset summary saved to: {summary_path}")


# ==============================
# Simple self-test (optional)
# ==============================

if __name__ == "__main__":
    df = load_dataset()

    # Print first rows to inspect structure
    print(df.head())
    print()

    # Print basic statistics: number of docs per class
    print(df["label"].value_counts())
    print()

    # Save verified dataset summary into txt file
    save_dataset_summary(df)