"""
Training Module - Phase 3
CPCS432 Assignment 5

This module handles the complete training pipeline:
1. Load Arabic documents from data directory (using Phase 2 utilities)
2. Preprocess documents (normalize, tokenize, remove stopwords, stem)
3. Extract TF-IDF features
4. Save all artifacts and reports for future use
"""

import os
import json
import pickle
from pathlib import Path

# Import Phase 2 utilities for data loading
from utils import load_dataset, OUTPUTS_DIR

# Import Phase 3 preprocessing components
from preprocess import (
    ArabicPreprocessor,
    TfidfVectorizer,
    ARABIC_STOPWORDS
)


def run_preprocessing_pipeline():
    """
    Run complete preprocessing and feature extraction pipeline.
    
    This function:
    - Loads all documents from the data directory (using Phase 2 utilities)
    - Preprocesses them using Arabic NLP techniques
    - Extracts TF-IDF features
    - Saves all artifacts to outputs directory
    
    Returns:
        dict: Summary statistics of the preprocessing
    """
    
    print("="*70)
    print("Arabic Text Preprocessing & Feature Extraction")
    print("Phase 3 - CPCS432 Assignment 5")
    print("Integrating with Phase 2 Data Loading")
    print("="*70)
    
    # Setup paths
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    # Load data using Phase 2 utilities
    print("\n[INFO] Loading dataset using Phase 2 utilities...")
    df = load_dataset()
    
    print(f"[INFO] Loaded {len(df)} documents from Phase 2")
    print(f"[INFO] Categories: {df['label'].unique().tolist()}")
    print(f"[INFO] Distribution:")
    print(df['label'].value_counts().to_string())
    
    # Extract documents, labels, and filepaths from DataFrame
    documents = df['text'].tolist()
    labels = df['label'].tolist()
    filepaths = df['filename'].tolist()
    
    if not documents:
        print("\n[ERROR] No documents found! Make sure 'data' directory exists with HTML files.")
        return None
    
    # Initialize preprocessor
    print("\n" + "="*70)
    print("PREPROCESSING PIPELINE")
    print("="*70)
    
    preprocessor = ArabicPreprocessor()
    
    # Preprocess all documents
    processed_docs = []
    preprocessing_stats = []
    
    for idx, (doc, label, filepath) in enumerate(zip(documents, labels, filepaths)):
        # Show progress for first 10, then every 100th document
        show_progress = idx < 10 or (idx + 1) % 100 == 0 or idx == len(documents) - 1
        
        if show_progress:
            print(f"\n[{idx+1}/{len(documents)}] Processing: {filepath}")
            print(f"  Category: {label}")
            print(f"  Raw text length: {len(doc)} characters")
        
        tokens = preprocessor.preprocess(doc)
        processed_docs.append(tokens)
        
        stats = {
            'filepath': filepath,
            'category': label,
            'raw_length': len(doc),
            'token_count': len(tokens),
            'unique_tokens': len(set(tokens)),
            'sample_tokens': tokens[:15]
        }
        preprocessing_stats.append(stats)
        
        if show_progress:
            print(f"  Tokens: {len(tokens)}")
            print(f"  Unique tokens: {len(set(tokens))}")
            if tokens:
                print(f"  Sample: {' '.join(tokens[:10])}...")
    
    # Feature extraction
    print("\n" + "="*70)
    print("FEATURE EXTRACTION (TF-IDF)")
    print("="*70)
    
    vectorizer = TfidfVectorizer()
    tfidf_vectors = vectorizer.fit_transform(processed_docs)
    
    print(f"\n[INFO] TF-IDF vectors created:")
    print(f"  Number of documents: {len(tfidf_vectors)}")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary)}")
    
    # Save all artifacts
    print("\n" + "="*70)
    print("SAVING ARTIFACTS")
    print("="*70)
    
    # Save vectorizer
    vectorizer_path = os.path.join(OUTPUTS_DIR, 'tfidf_vectorizer.pkl')
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print("\n[✓] Saved: outputs/tfidf_vectorizer.pkl")
    
    # Save processed documents
    processed_docs_path = os.path.join(OUTPUTS_DIR, 'processed_documents.pkl')
    with open(processed_docs_path, 'wb') as f:
        pickle.dump(processed_docs, f)
    print("[✓] Saved: outputs/processed_documents.pkl")
    
    # Save TF-IDF vectors
    vectors_path = os.path.join(OUTPUTS_DIR, 'tfidf_vectors.pkl')
    with open(vectors_path, 'wb') as f:
        pickle.dump(tfidf_vectors, f)
    print("[✓] Saved: outputs/tfidf_vectors.pkl")
    
    # Save preprocessing stats
    stats_path = os.path.join(OUTPUTS_DIR, 'preprocessing_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(preprocessing_stats, f, ensure_ascii=False, indent=2)
    print("[✓] Saved: outputs/preprocessing_stats.json")
    
    # Save labels
    labels_path = os.path.join(OUTPUTS_DIR, 'labels.pkl')
    with open(labels_path, 'wb') as f:
        pickle.dump(labels, f)
    print("[✓] Saved: outputs/labels.pkl")
    
    # Save original DataFrame with preprocessing info (Phase 2 + Phase 3 integration)
    df['processed_tokens'] = processed_docs
    df['token_count'] = [len(tokens) for tokens in processed_docs]
    df['unique_token_count'] = [len(set(tokens)) for tokens in processed_docs]
    
    df_path = os.path.join(OUTPUTS_DIR, 'dataset_with_preprocessing.pkl')
    df.to_pickle(df_path)
    print("[✓] Saved: outputs/dataset_with_preprocessing.pkl (Phase 2 + Phase 3 combined)")
    
    # Create validation report
    print("\n" + "="*70)
    print("VALIDATION & INSPECTION")
    print("="*70)
    
    # Show top TF-IDF terms for first document
    if tfidf_vectors:
        print("\n[Sample] Top 15 TF-IDF terms from first document:")
        first_vector = tfidf_vectors[0]
        
        # Reverse vocabulary for lookup
        idx_to_term = {idx: term for term, idx in vectorizer.vocabulary.items()}
        
        # Sort by TF-IDF score
        sorted_terms = sorted(first_vector.items(), key=lambda x: x[1], reverse=True)
        
        for idx, (term_idx, score) in enumerate(sorted_terms[:15], 1):
            term = idx_to_term[term_idx]
            print(f"  {idx}. {term}: {score:.4f}")
    
    # Create detailed report
    report = {
        'project': 'CPCS432 - Phase 3 (Integrated with Phase 2)',
        'phase2_integration': {
            'data_source': 'Phase 2 load_dataset() from utils.py',
            'dataframe_columns': ['filename', 'text', 'label'],
            'original_document_count': len(df)
        },
        'preprocessing_steps': [
            '1. Text Normalization (unify Alef, remove diacritics, normalize Taa Marbuta and Yaa)',
            '2. Remove punctuation and non-Arabic symbols',
            '3. Tokenize text into words',
            f'4. Remove stopwords (using {len(ARABIC_STOPWORDS)} Arabic stopwords)',
            '5. Apply light stemming (remove common prefixes and suffixes)',
            '6. Filter tokens (remove very short tokens)'
        ],
        'theoretical_justification': {
            'normalization': 'Unifying character variations reduces vocabulary size and improves matching (NLP1 concept)',
            'stopword_removal': 'Removes high-frequency, low-information words to focus on meaningful content (NLP1)',
            'stemming': 'Reduces inflected words to root form, improving recall in IR tasks (NLP1)',
            'tfidf': 'Balances term frequency with document frequency to identify important terms (NLP3/IR concept)'
        },
        'statistics': {
            'total_documents': len(documents),
            'vocabulary_size': len(vectorizer.vocabulary),
            'categories': list(set(labels)),
            'documents_per_category': {label: labels.count(label) for label in set(labels)}
        },
        'sample_vocabulary': list(vectorizer.vocabulary.keys())[:50]
    }
    
    report_path = os.path.join(OUTPUTS_DIR, 'preprocessing_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n[✓] Saved: outputs/preprocessing_report.json")
    
    # Create text report
    notes_path = os.path.join(OUTPUTS_DIR, 'preprocessing_notes.txt')
    with open(notes_path, 'w', encoding='utf-8') as f:
        f.write("Arabic Text Preprocessing & Feature Extraction\n")
        f.write("Phase 3 - CPCS432 Assignment 5\n")
        f.write("Integrated with Phase 2 Data Loading\n")
        f.write("="*70 + "\n\n")
        
        f.write("PHASE 2 INTEGRATION:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Data loaded using Phase 2 utilities (utils.load_dataset())\n")
        f.write(f"Original documents: {len(df)}\n")
        f.write(f"Categories: {', '.join(report['statistics']['categories'])}\n\n")
        
        f.write("PREPROCESSING STEPS:\n")
        f.write("-" * 70 + "\n")
        for step in report['preprocessing_steps']:
            f.write(f"{step}\n")
        
        f.write("\n\nTHEORETICAL JUSTIFICATION:\n")
        f.write("-" * 70 + "\n")
        for key, value in report['theoretical_justification'].items():
            f.write(f"{key.upper()}: {value}\n\n")
        
        f.write("\nSTATISTICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total documents: {report['statistics']['total_documents']}\n")
        f.write(f"Vocabulary size: {report['statistics']['vocabulary_size']}\n")
        f.write(f"Categories: {', '.join(report['statistics']['categories'])}\n")
        f.write("\nDocuments per category:\n")
        for cat, count in report['statistics']['documents_per_category'].items():
            f.write(f"  - {cat}: {count}\n")
        
        # Add sample of top terms
        if tfidf_vectors and len(tfidf_vectors) > 0:
            f.write("\n\nSAMPLE: Top 15 TF-IDF terms from first document:\n")
            f.write("-" * 70 + "\n")
            first_vector = tfidf_vectors[0]
            idx_to_term = {idx: term for term, idx in vectorizer.vocabulary.items()}
            sorted_terms = sorted(first_vector.items(), key=lambda x: x[1], reverse=True)
            for idx, (term_idx, score) in enumerate(sorted_terms[:15], 1):
                term = idx_to_term[term_idx]
                f.write(f"{idx}. {term}: {score:.4f}\n")
    
    print("[✓] Saved: outputs/preprocessing_notes.txt")
    
    print("\n" + "="*70)
    print("PHASE 3 COMPLETE! (Integrated with Phase 2)")
    print("="*70)
    print("\nDeliverables saved in 'outputs/' directory:")
    print("  1. tfidf_vectorizer.pkl - Fitted vectorizer for prediction")
    print("  2. processed_documents.pkl - Preprocessed token lists")
    print("  3. tfidf_vectors.pkl - TF-IDF feature vectors")
    print("  4. preprocessing_stats.json - Detailed statistics per document")
    print("  5. preprocessing_report.json - Complete report with justification")
    print("  6. preprocessing_notes.txt - Human-readable notes")
    print("  7. labels.pkl - Document labels")
    print("  8. dataset_with_preprocessing.pkl - Combined Phase 2 + Phase 3 data")
    print("\n[Phase 2 → Phase 3 Integration Complete]")
    print("  ✓ Used Phase 2 data loading utilities")
    print("  ✓ Applied Phase 3 preprocessing pipeline")
    print("  ✓ Combined dataset saved for future phases")
    print("\n" + "="*70)
    
    return report['statistics']


def main():
    """Main entry point for training pipeline"""
    stats = run_preprocessing_pipeline()
    
    if stats:
        print("\n[SUCCESS] Phase 3 preprocessing pipeline completed successfully!")
        print(f"Total documents processed: {stats['total_documents']}")
        print(f"Vocabulary size: {stats['vocabulary_size']}")
    else:
        print("\n[FAILED] Phase 3 preprocessing pipeline failed!")


if __name__ == "__main__":
    main()

