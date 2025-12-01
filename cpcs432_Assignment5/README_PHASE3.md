# Phase 3 - Arabic Text Preprocessing & Feature Extraction

## Overview
Phase 3 implements a complete Arabic text preprocessing pipeline with TF-IDF feature extraction for the CPCS432 Arabic text classification project.

## Project Structure

```
cpcs432_Assignment5/
├── data/
│   ├── Economy/          # Economy category HTML files
│   └── Sports/           # Sports category HTML files
├── src/
│   ├── preprocess.py     # Arabic preprocessing classes & TF-IDF vectorizer
│   ├── train.py          # Main training/preprocessing pipeline
│   ├── predict.py        # Prediction module (future use)
│   └── utils.py          # Utility functions for data loading
├── outputs/
│   ├── tfidf_vectorizer.pkl          # Trained TF-IDF vectorizer
│   ├── processed_documents.pkl       # Preprocessed token lists
│   ├── tfidf_vectors.pkl             # TF-IDF feature vectors
│   ├── labels.pkl                    # Document labels
│   ├── preprocessing_stats.json      # Detailed statistics
│   ├── preprocessing_report.json     # Complete report
│   └── preprocessing_notes.txt       # Human-readable notes
└── README_PHASE3.md      # This file
```

## Files Description

### src/preprocess.py
Contains the core preprocessing functionality:
- **ArabicPreprocessor**: Complete Arabic text preprocessing pipeline
  - Text normalization (Alef, Taa Marbuta, Yaa, diacritics)
  - Punctuation removal
  - Tokenization
  - Stopword removal (comprehensive Arabic stopwords list)
  - Light stemming (prefix/suffix removal)
  - Token filtering

- **TfidfVectorizer**: TF-IDF feature extraction
  - Vocabulary building
  - IDF calculation
  - TF-IDF transformation
  - Save/load functionality

- **Helper functions**:
  - `read_html_file()`: Read HTML files with multiple encoding support
  - `load_data_from_directory()`: Load all documents from data directory

### src/train.py
Main training pipeline module:
- **run_preprocessing_pipeline()**: Complete preprocessing workflow
  - Loads documents from data directory
  - Preprocesses all documents
  - Extracts TF-IDF features
  - Saves all artifacts to outputs/
  - Generates detailed reports

- **main()**: Entry point for running the pipeline

### src/utils.py
Utility functions for data handling:
- Configuration and paths
- File loading helpers
- Dataset summary generation

## Usage

### Running Phase 3 Pipeline

From the project root directory:

```bash
cd cpcs432_Assignment5/src
python train.py
```

Or from Python:

```python
from src.train import run_preprocessing_pipeline

stats = run_preprocessing_pipeline()
print(f"Processed {stats['total_documents']} documents")
print(f"Vocabulary size: {stats['vocabulary_size']}")
```

### Using the Preprocessor

```python
from src.preprocess import ArabicPreprocessor

# Initialize preprocessor
preprocessor = ArabicPreprocessor()

# Preprocess text
arabic_text = "هذا نص عربي للاختبار"
tokens = preprocessor.preprocess(arabic_text)
print(tokens)
```

### Using the TF-IDF Vectorizer

```python
from src.preprocess import TfidfVectorizer

# Create and train vectorizer
vectorizer = TfidfVectorizer()
tfidf_vectors = vectorizer.fit_transform(processed_documents)

# Save for later use
vectorizer.save('outputs/tfidf_vectorizer.pkl')

# Load saved vectorizer
vectorizer = TfidfVectorizer.load('outputs/tfidf_vectorizer.pkl')
```

## Preprocessing Pipeline

### Step-by-Step Process:

1. **Text Normalization**
   - Unify Alef variations (أ، إ، آ → ا)
   - Remove diacritics (tashkeel)
   - Normalize Taa Marbuta (ة → ه)
   - Normalize Yaa (ى → ي)

2. **Punctuation Removal**
   - Remove all non-Arabic characters
   - Keep only Arabic letters and spaces

3. **Tokenization**
   - Split text into words

4. **Stopword Removal**
   - Remove common Arabic stopwords
   - Comprehensive list of ~400+ stopwords

5. **Light Stemming**
   - Remove common prefixes (ال، و، ف، ب، etc.)
   - Remove common suffixes (ها، ان، ات، etc.)

6. **Token Filtering**
   - Remove very short tokens (< 2 characters)

### Theoretical Justification:

- **Normalization**: Reduces vocabulary size and improves term matching (NLP1 concept)
- **Stopword Removal**: Focuses on meaningful content by removing high-frequency, low-information words (NLP1)
- **Stemming**: Improves recall by reducing inflected words to approximate root forms (NLP1)
- **TF-IDF**: Balances term frequency with document frequency to identify important discriminative terms (NLP3/IR concept)

## Output Files

After running the pipeline, the following files are generated in `outputs/`:

1. **tfidf_vectorizer.pkl** - Trained vectorizer for future predictions
2. **processed_documents.pkl** - List of preprocessed token lists
3. **tfidf_vectors.pkl** - TF-IDF feature vectors for all documents
4. **labels.pkl** - Document category labels
5. **preprocessing_stats.json** - Detailed statistics for each document
6. **preprocessing_report.json** - Complete report with justifications
7. **preprocessing_notes.txt** - Human-readable summary

## Requirements

- Python 3.7+
- Standard library only (no external dependencies for preprocessing)
- pickle, json, re, math, collections, pathlib

## Next Steps (Future Phases)

- Phase 4: Train classification models (Naive Bayes, SVM, etc.)
- Phase 5: Model evaluation and comparison
- Phase 6: Prediction on new documents

## Notes

- The preprocessing pipeline is designed for Arabic text classification
- All files support UTF-8 encoding with fallback to other Arabic encodings
- The vectorizer uses sparse dictionary representation for efficiency
- Stopwords list is comprehensive and includes Modern Standard Arabic and dialectal variations

## Author
CPCS432 Assignment 5 - Phase 3

