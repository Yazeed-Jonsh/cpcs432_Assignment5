"""
Prediction Module - Phase 6
CPCS432 Assignment 5

This module handles prediction on new/test data:
1. Loads trained models (TF-IDF, DictVectorizer, Naive Bayes).
2. Reads raw files from a test directory.
3. Preprocesses and vectorizes the text.
4. Generates predictions and saves them to a CSV file.
"""

import os
import pickle
import pandas as pd
from preprocess import ArabicPreprocessor, TfidfVectorizer
from utils import OUTPUTS_DIR, read_file_text

def load_models():
    """
    Load all necessary pickle artifacts for prediction.
    Returns:
        tuple: (tfidf_vectorizer, dict_vectorizer, model)
    """
    print("[INFO] Loading trained models...")
    
    try:
        # 1. Load Phase 3 TF-IDF Vectorizer (Custom Class)
        tfidf_path = os.path.join(OUTPUTS_DIR, 'tfidf_vectorizer.pkl')
        with open(tfidf_path, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
            
        # 2. Load Phase 4 DictVectorizer (Sklearn)
        dv_path = os.path.join(OUTPUTS_DIR, 'dict_vectorizer.pkl')
        with open(dv_path, 'rb') as f:
            dict_vectorizer = pickle.load(f)
            
        # 3. Load Phase 4 Naive Bayes Model
        model_path = os.path.join(OUTPUTS_DIR, 'nb_model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        print("[✓] Models loaded successfully.")
        return tfidf_vectorizer, dict_vectorizer, model
        
    except FileNotFoundError as e:
        print(f"[ERROR] Missing model file: {e}")
        print("Ensure you have run both train.py (Phase 3) and NBmodel.py (Phase 4).")
        return None, None, None

def predict_directory(test_dir, output_csv="predictions.csv"):
    """
    Predict labels for all files in a directory and save to CSV.
    
    Args:
        test_dir (str): Path to directory containing test files (txt/html)
        output_csv (str): Path to save the output CSV
    """
    # 1. Load Models
    tfidf_vec, dict_vec, model = load_models()
    if not model:
        return

    # 2. Load and Preprocess Test Data
    print(f"\n[INFO] Processing test files from: {test_dir}")
    
    if not os.path.exists(test_dir):
        print(f"[ERROR] Test directory not found: {test_dir}")
        return

    filenames = []
    processed_docs = []
    preprocessor = ArabicPreprocessor()
    
    files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.txt', '.html'))]
    
    if not files:
        print("[WARN] No .txt or .html files found in directory.")
        return

    print(f"Found {len(files)} files. Starting preprocessing...")
    
    for idx, filename in enumerate(files):
        file_path = os.path.join(test_dir, filename)
        
        # Read text
        text = read_file_text(file_path)
        
        # Preprocess (Same pipeline as training)
        tokens = preprocessor.preprocess(text)
        
        filenames.append(filename)
        processed_docs.append(tokens)
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(files)} files...")

    # 3. Feature Extraction
    print("\n[INFO] Extracting features...")
    
    # Step A: Convert tokens to TF-IDF dictionaries (using Phase 3 vocabulary)
    # Note: transform() handles unseen words by ignoring them
    tfidf_dicts = tfidf_vec.transform(processed_docs)
    
    # Step B: Convert dictionaries to Sparse Matrix (using Phase 4 mapping)
    X_test = dict_vec.transform(tfidf_dicts)
    
    # 4. Prediction
    print(f"[INFO] Predicting classes for {X_test.shape[0]} documents...")
    predictions = model.predict(X_test)
    
    # 5. Save Results
    print("\n[INFO] Saving results...")
    
    results_df = pd.DataFrame({
        'filename': filenames,
        'label': predictions
    })
    
    # Save to CSV
    save_path = os.path.join(OUTPUTS_DIR, output_csv)
    results_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    
    print(f"[✓] Predictions saved to: {save_path}")
    print("\nSample Predictions:")
    print(results_df.head())

if __name__ == "__main__":
    # Example Usage:
    # Update this path to point to your actual test set folder
    # If you don't have a test folder yet, create 'data/Test' and put some files in it.
    
    # Assuming the predict.py file is in src/
    # The root of the project is one level up from src/
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TEST_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'Test')
    
    print("="*60)
    print("CPCS432 - Phase 6: Prediction")
    print("="*60)
    
    # Check if directory exists, otherwise ask user
    if not os.path.exists(TEST_DATA_DIR):
        print(f"[NOTE] Default test directory '{TEST_DATA_DIR}' does not exist.")
        # Try to run interactively or fallback
        try:
            user_path = input("Please enter the full path to your test folder: ").strip()
            if user_path:
                predict_directory(user_path)
            else:
                print("No path provided. Exiting.")
        except EOFError:
            print("Non-interactive mode detected. Please ensure 'data/Test' exists.")
    else:
        predict_directory(TEST_DATA_DIR)

