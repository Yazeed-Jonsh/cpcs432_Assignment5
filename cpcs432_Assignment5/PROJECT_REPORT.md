# Arabic Text Classification Project Report
## CPCS432 Assignment 5 - Machine Learning Pipeline

### 1. Executive Summary

This report documents the design, implementation, and evaluation of a machine learning system developed for **CPCS432 Assignment 5**. The objective of the project is to classify Arabic text documents into one of two distinct categories: **Economy (الاقتصاد)** and **Sports (الرياضة)**.

The solution is implemented in Python, adhering to a modular architecture that separates concerns into three distinct phases: **Preprocessing (Phase 3)**, **Model Training (Phase 4)**, and **Prediction (Phase 6)**. By utilizing custom Arabic Natural Language Processing (NLP) techniques and a **Complement Naive Bayes** classifier, the model achieves a validation accuracy of **99.33%**, demonstrating exceptional performance in distinguishing between the two topics.

---

### 2. Introduction

Text classification is a fundamental task in Natural Language Processing (NLP), enabling the automated organization of unstructured data. While English NLP tools are mature, processing **Arabic text** presents unique challenges due to its rich morphology, complex orthography, and the prevalence of dialectal variations.

This project addresses these challenges by building a custom preprocessing pipeline tailored specifically for the Arabic language. The system is designed to handle:
*   **Orthographic Variations**: Different forms of Alef, Yaa, and Taa Marbuta.
*   **Diacritics (Tashkeel)**: Optional vowel marks that change the byte representation of words.
*   **Morphology**: Complex affixation where articles, prepositions, and pronouns attach to the word stem.

The goal is to build a robust classifier that can ingest raw HTML or text files and accurately predict their topic label.

---

### 3. Project Structure and Organization

The project follows a standard data science directory structure, ensuring reproducibility and ease of navigation.

```text
CPCS432_Assignment5/
├── data/                       # Raw Dataset
│   ├── Economy/                # 850+ Economy HTML documents
│   └── Sports/                 # 1300+ Sports HTML documents
│   └── Test/                   # Directory for inference/testing
├── src/                        # Source Code
│   ├── preprocess.py           # Core NLP and Vectorizer classes
│   ├── train.py                # Phase 3: Preprocessing execution
│   ├── NBmodel.py              # Phase 4: Model training & evaluation
│   ├── predict.py              # Phase 6: Inference on new data
│   └── utils.py                # Helper functions (I/O, Config)
├── outputs/                    # Generated Artifacts
│   ├── tfidf_vectorizer.pkl    # Fitted feature extractor
│   ├── nb_model.pkl            # Trained Classifier
│   ├── predictions.csv         # Final output results
│   └── ... (logs & reports)
└── PROJECT_REPORT.md           # This documentation
```

---

### 4. Technical Implementation: Phase 3 (Preprocessing)

Phase 3 is arguably the most critical component of the pipeline. Raw text data is noisy and high-dimensional. This phase transforms it into a structured format suitable for machine learning.

#### 4.1. Data Ingestion & Encoding
Reading Arabic text files often leads to encoding errors due to legacy standards (e.g., `windows-1256` vs `utf-8`). The `utils.read_file_text` function implements a robust fallback mechanism, attempting multiple encodings to ensure no data is lost during ingestion.

#### 4.2. The Arabic Preprocessor (`src/preprocess.py`)
A custom class, `ArabicPreprocessor`, was developed to handle the linguistic nuances of Arabic. It executes the following pipeline sequentially:

1.  **Text Normalization**:
    *   **Alef Normalization**: Unifies `أ`, `إ`, `آ` into bare `ا`. This is crucial because users often omit Hamzas in writing.
    *   **Taa Marbuta**: Converts `ة` to `ه` (Ha) to handle common spelling inconsistencies.
    *   **Yaa**: Converts final `ى` (Alif Maqsura) to `ي` (Yaa).
    *   **Diacritic Removal**: Strips all Tashkeel (Fatha, Damma, Kasra, etc.) as they are rarely used in standard news text.

2.  **Noise Removal**:
    *   Removes all non-Arabic characters, including English letters, numbers, punctuation, and HTML tags.

3.  **Stopword Removal**:
    *   Filters out a comprehensive list of over **400 Arabic stopwords** (functional words like "في", "هذا", "الذي", "عن"). These words appear frequently but carry little discriminative power for topic classification.

4.  **Light Stemming**:
    *   Unlike aggressive root extraction (Khoja stemmer), this project uses **Light Stemming**.
    *   It removes common prefixes (`ال`, `و`, `ف`, `ب`) and suffixes (`ات`, `ون`, `ين`, `ها`).
    *   **Benefit**: This preserves the semantic difference between words that might share a root but have different meanings, while still reducing vocabulary size.

#### 4.3. Feature Extraction (TF-IDF)
The `TfidfVectorizer` class converts the cleaned tokens into numerical vectors.
*   **TF (Term Frequency)**: Measures how often a word appears in a document.
*   **IDF (Inverse Document Frequency)**: Penalizes words that appear in *every* document (common words) and boosts words that are unique to specific documents.
*   **Result**: A dictionary representation for each document, where keys are word indices and values are their "importance" scores.

---

### 5. Technical Implementation: Phase 4 (Model Training)

Phase 4 involves training the predictive model using the artifacts generated in Phase 3.

#### 5.1. Algorithm Selection: Complement Naive Bayes
The project utilizes the **Complement Naive Bayes (CNB)** algorithm (`sklearn.naive_bayes.ComplementNB`).
*   **Why Naive Bayes?** It is highly efficient for high-dimensional text data and works well with small to medium-sized datasets.
*   **Why Complement NB?** The dataset is slightly imbalanced (Sports has ~1380 docs, Economy has ~850). Standard Multinomial Naive Bayes can be biased towards the majority class. CNB is specifically designed to correct this bias by estimating parameters from the *complement* of each class.

#### 5.2. Training Workflow (`src/NBmodel.py`)
1.  **Artifact Loading**: The script loads the TF-IDF dictionaries and labels saved by Phase 3.
2.  **Vectorization**: A `DictVectorizer` transforms the list of dictionaries into a sparse matrix `X`.
3.  **Data Splitting**: The data is split 80/20:
    *   **Training Set (80%)**: Used to learn the probability distributions of words given a class.
    *   **Validation Set (20%)**: Used to evaluate the model on unseen data.
    *   **Stratification**: Ensures the ratio of Sports/Economy remains consistent in both splits.
4.  **Smoothing**: Laplace smoothing (`alpha=1.0`) is applied to handle "zero-frequency" issues (words appearing in testing that were never seen in training).

---

### 6. Technical Implementation: Phase 6 (Prediction)

Phase 6 represents the deployment stage of the pipeline (`src/predict.py`).

#### 6.1. Pipeline Consistency
A common pitfall in ML deployment is **Training-Serving Skew**. To avoid this, Phase 6 loads the *exact same* vectorizers (`tfidf_vectorizer.pkl` and `dict_vectorizer.pkl`) used in training. This ensures that a word like "ملعب" is mapped to the exact same feature index during prediction as it was during training.

#### 6.2. Batch Processing
The prediction script is designed to process entire directories of new files. It:
1.  Iterates through the target folder.
2.  Applies the full `ArabicPreprocessor` chain.
3.  Vectorizes the text.
4.  Predicts labels using the loaded model.
5.  Exports a CSV file (`filename`, `label`) for easy consumption by downstream systems.

---

### 7. Results and Evaluation

The model performance was evaluated using the 20% held-out validation set (approx. 450 documents).

#### 7.1. Quantitative Metrics
The model achieved outstanding results:

*   **Accuracy**: **99.33%**
*   **Precision**: > 0.99 for both classes.
*   **Recall**: > 0.98 for both classes.
*   **F1-Score**: 0.99

This indicates the model is both highly accurate and unbiased. It rarely confuses Economy for Sports or vice versa.

#### 7.2. Qualitative Error Analysis
An inspection of the few misclassified examples reveals that errors mostly occur in "crossover" articles.
*   *Example*: An article discussing the "budget of a football club" might contain heavy financial terminology ("millyun", "sarrf", "mizaniyah") alongside sports terms.
*   Because Naive Bayes treats words independently (Bag of Words assumption), a preponderance of "money" words might tip the probability towards Economy, even if the subject is Sports.

---

### 8. Conclusion and Future Work

#### 8.1. Summary
This project successfully delivers a high-performance Arabic text classifier. The high accuracy validates the effectiveness of the chosen preprocessing steps—specifically normalization and light stemming—which are essential for handling Arabic morphology. The modular code structure ensures the system is maintainable and scalable.
The current solution stands as a solid baseline for any future Arabic NLP tasks in this domain.
