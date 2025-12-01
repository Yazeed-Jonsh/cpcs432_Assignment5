"""
Arabic Text Preprocessing & Feature Extraction
Phase 3 - CPCS432 Assignment 5

This module contains:
- ArabicPreprocessor: Arabic text preprocessing pipeline
- TfidfVectorizer: TF-IDF feature extraction

Note: This module is integrated with Phase 2's data loading utilities.
The main pipeline (train.py) uses utils.load_dataset() from Phase 2.
The load_data_from_directory() function below is kept for standalone use.
"""

import re
import math
import pickle
from collections import Counter
from pathlib import Path


# ==============================
# Arabic Stopwords
# ==============================

ARABIC_STOPWORDS = {
'و','او','أو','ثم','بل','لكن','كما','لذلك','اذ','اذا','إذ','إلا','حتى','اما','أم','أمّا',
'في','من','على','علي','عليه','عليها','عن','مع','الى','إلى','إليه','إليها','ل','له','لها',
'ما','ماذا','متى','كيف','لماذا','اي','أي','أين','هنا','هناك','هذا','هذه','هؤلاء','ذلك',
'تلك','هكذا','الذي','التي','الذين','اللذين','اللتي','أمام','خلف','فوق','تحت','عند','عندما',
'حيث','خلال','بعد','قبل','ضمن','بسبب','جراء','وفق','حسب','بحسب','منذ','بين','وسط','بجانب',
'انا','أنت','انت','أنتم','انتم','هما','هم','هن','هو','هي','نحن','إياهم','إياها','إياكما',
'كان','كانت','كنت','كانوا','يكون','تكون','نكون','يكونون','اصبح','أصبح','أصبحت','صار','تصير',
'أمسى','ليس','ليست','لن','لم','قد','قد','قد','قد','ظل','لايزال','مازال','بات','صار','أصبح',
'كل','أيضا','أيضاً','ايضا','أي','بعض','كثير','قليل','مثل','ضمن','نحو','غير','دون','سوى',
'سوا','سواء','تماما','تماماً','مرة','مرات','هو','هي','وهو','وهي','به','بها','فيه','فيها',
'منه','منها','اليوم','الآن','أمس','غدا','غداً','صباح','مساء','ليلا','نهارا','سنوات','سنة',
'شهر','أشهر','اسبوع','اسابيع','وقت','لحظة','حاليا','حالياً','الوقت','الفترة','احد','اخرى',
'أي شخص','احدهم','إحداها','كلا','كلتا','كذلك','كما','أيضا','أيضاً','فقط','خاصة','خصوصاً',
'لانه','لان','لأن','ذلك','تلك','اذن','إذن','طيب','حسنا','تمام','اي','أية','إيا','إياك','إياكم',
'ان','أن','إن','إنّ','الى','إلى','ال','او','أو','ا','و','نعم','لا','كلا','أجل','بالتأكيد',
'ربما','قد','غالبا','غالباً','عادة','عادةً','تقريباً','تقريبا','كذا','كده','كذاك','كدا','كدة',
'هناك','هنا','هنالك','كلّ','كله','كلها','كامل','كاملة','شيء','أشياء','كل شيء','أياً كان',
'أحد','احد','احدى','إحدى','إحداهن','اياً','او','لو','حتى','إلا','مازال','لاتزال','لا يزال',
'ليس','ليست','أينما','كيفما','حيثما','مهما','ممن','عنما','كلما','إذما','إذًا','إذاً',
'أيا','ايا','ايّا','إياها','إياه','بهذا','بذلك','بهذه','بهؤلاء','لهذا','لهذه','لذلك','لذا',
'الذي','التي','اللذان','اللتان','اللذين','اللتين','أولئك','أولائك','هؤلاء','ذو','ذات','ذوي',
'ذا','ذي','ذين','اللهم','أياً','أيّا','أية','أيتها','أيتها','أياه','أياهم','أياها','أياهما',
'فان','حري',
'قال','يقول','أشار','يشير','ذكر','يذكر','أوضح','يوضح','أضاف','يضيف','تابع','يتابع','أكد','يؤكد',
'ذكر','يذكر','صرح','يصرح','أعلن','يعلن','أعلنت','يتم','تم','يحدث','حدث','سوف','سيت','سيتم',
'خلال','ضمن','حسب','بحسب','وفق','ضد','أمام','وراء','من جديد','مرة أخرى','من جهة','من جانب','من طرف',
'أعلى','أدنى','أكبر','أصغر','أكثر','أقل','كثيراً','قليلاً','نوعاً ما','إجمالاً','عموماً','فعلياً',
'اوه','اوو','طيب','تمام','حسنًا','يعني',' basically ',' literally ',' kinda ',' sorta ',
'شيء','أشياء','شيئا','شيئاً','أي شيء','ماشي','كذا','هيك','كذاك','هيّا','يلا','هيا','على حد',
'من ناحية','من جانب','من طرف','من جهة','من ثم','من قبل','من بعد','كذلك','أيضاً','بالمقابل','بالنظر',
'هذا','هذه','هاتان','هذان','هؤلاء','ذلك','تلك','ذلكم','ذلكن','هنالك','هناك','هنا','هذه','هاهنا',
'الآن','اليوم','غدا','أمس','ليلاً','نهاراً','صباحاً','مساءً','أي وقت','أي مكان','طوال','كافة',
'إجمالاً','فعلياً','تماماً','نوعاً ما','إلى حد ما','بشكل عام','بشكل كامل','بشكل كبير'
}


# ==============================
# Arabic Preprocessor Class
# ==============================

class ArabicPreprocessor:
    """
    Arabic text preprocessing class with complete preprocessing pipeline.
    
    Preprocessing steps:
    1. Text Normalization (unify Alef variations, remove diacritics, normalize Taa Marbuta and Yaa)
    2. Remove punctuation and non-Arabic symbols
    3. Tokenize text into words
    4. Remove stopwords (using Arabic stopwords list)
    5. Apply light stemming (remove common prefixes and suffixes)
    6. Filter tokens (remove very short tokens)
    
    Theoretical Justification:
    - Normalization: Unifying character variations reduces vocabulary size and improves matching (NLP1 concept)
    - Stopword removal: Removes high-frequency, low-information words to focus on meaningful content (NLP1)
    - Stemming: Reduces inflected words to root form, improving recall in IR tasks (NLP1)
    """
    
    def __init__(self):
        self.stopwords = ARABIC_STOPWORDS
    
    def normalize_arabic(self, text):
        """
        Normalize Arabic text by:
        - Removing diacritics (Tashkeel)
        - Normalizing Alef variations to ا
        - Normalizing Taa Marbuta ة to ه
        - Normalizing Yaa ى to ي
        """
        # Remove diacritics (Tashkeel)
        text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
        
        # Normalize Alef variations
        text = re.sub(r'[إأآا]', 'ا', text)
        
        # Normalize Taa Marbuta
        text = re.sub(r'ة', 'ه', text)
        
        # Normalize Yaa
        text = re.sub(r'ى', 'ي', text)
        
        return text
    
    def remove_punctuation(self, text):
        """
        Remove punctuation and non-Arabic symbols.
        Keep only Arabic letters and spaces.
        """
        # Keep only Arabic letters and spaces
        text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def tokenize(self, text):
        """Tokenize Arabic text into words"""
        return [token for token in text.split() if token]
    
    def remove_stopwords(self, tokens):
        """Remove Arabic stopwords from token list"""
        return [token for token in tokens if token not in self.stopwords]
    
    def light_stem(self, word):
        """
        Apply light stemming for Arabic.
        Removes common prefixes and suffixes to reduce words to approximate stems.
        """
        # Remove common prefixes
        prefixes = ['ال', 'و', 'ف', 'ب', 'ك', 'ل', 'لل']
        for prefix in prefixes:
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                word = word[len(prefix):]
                break
        
        # Remove common suffixes
        suffixes = ['ها', 'ان', 'ات', 'ون', 'ين', 'ه', 'ك', 'ت', 'ن', 'ي']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                word = word[:-len(suffix)]
                break
        
        return word
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline for Arabic text.
        
        Args:
            text (str): Raw Arabic text
            
        Returns:
            list: List of preprocessed tokens
        """
        # Step 1: Normalize
        text = self.normalize_arabic(text)
        
        # Step 2: Remove punctuation
        text = self.remove_punctuation(text)
        
        # Step 3: Tokenize
        tokens = self.tokenize(text)
        
        # Step 4: Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Step 5: Apply light stemming
        tokens = [self.light_stem(token) for token in tokens]
        
        # Step 6: Remove very short tokens
        tokens = [token for token in tokens if len(token) > 1]
        
        return tokens


# ==============================
# TF-IDF Vectorizer Class
# ==============================

class TfidfVectorizer:
    """
    TF-IDF Vectorizer for Arabic text.
    
    TF-IDF (Term Frequency-Inverse Document Frequency) balances term frequency 
    with document frequency to identify important terms (NLP3/IR concept).
    
    This vectorizer:
    - Builds vocabulary from training documents
    - Calculates IDF scores for each term
    - Transforms documents into TF-IDF feature vectors
    """
    
    def __init__(self):
        self.vocabulary = {}
        self.idf = {}
        self.is_fitted = False
    
    def fit(self, documents):
        """
        Fit the vectorizer on training documents.
        
        Args:
            documents (list): List of preprocessed token lists
        """
        print("\n[INFO] Fitting TF-IDF vectorizer...")
        
        # Build vocabulary from all terms
        all_terms = set()
        for doc in documents:
            all_terms.update(doc)
        
        # Create vocabulary mapping: term -> index
        self.vocabulary = {term: idx for idx, term in enumerate(sorted(all_terms))}
        
        # Calculate IDF for each term
        num_docs = len(documents)
        df = Counter()  # Document frequency for each term
        
        for doc in documents:
            unique_terms = set(doc)
            for term in unique_terms:
                df[term] += 1
        
        # IDF = log(N / df(term))
        for term in self.vocabulary:
            self.idf[term] = math.log(num_docs / df[term])
        
        self.is_fitted = True
        print(f"[INFO] Vocabulary size: {len(self.vocabulary)}")
        
    def transform(self, documents):
        """
        Transform documents to TF-IDF vectors.
        
        Args:
            documents (list): List of preprocessed token lists
            
        Returns:
            list: List of TF-IDF vectors (sparse dictionaries)
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        vectors = []
        
        for doc in documents:
            # Calculate TF (term frequency)
            tf = Counter(doc)
            total_terms = len(doc)
            
            # Calculate TF-IDF
            vector = {}
            for term in tf:
                if term in self.vocabulary:
                    tf_score = tf[term] / total_terms
                    tfidf_score = tf_score * self.idf[term]
                    vector[self.vocabulary[term]] = tfidf_score
            
            vectors.append(vector)
        
        return vectors
    
    def fit_transform(self, documents):
        """
        Fit and transform in one step.
        
        Args:
            documents (list): List of preprocessed token lists
            
        Returns:
            list: List of TF-IDF vectors
        """
        self.fit(documents)
        return self.transform(documents)
    
    def save(self, filepath):
        """Save vectorizer to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"[✓] Saved vectorizer to: {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load vectorizer from file"""
        with open(filepath, 'rb') as f:
            vectorizer = pickle.load(f)
        print(f"[✓] Loaded vectorizer from: {filepath}")
        return vectorizer


# ==============================
# Helper Functions
# ==============================

def read_html_file(filepath):
    """
    Read and extract text from HTML file.
    Tries multiple encodings for Arabic files.
    
    Args:
        filepath (str or Path): Path to HTML file
        
    Returns:
        str: Extracted text content
    """
    # Try multiple encodings for Arabic files
    encodings = ['utf-8', 'windows-1256', 'iso-8859-6', 'cp1256', 'latin-1']
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                content = f.read()
            
            # Simple HTML tag removal
            text = re.sub(r'<[^>]+>', ' ', content)
            text = re.sub(r'\s+', ' ', text)
            
            return text.strip()
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            print(f"[ERROR] Failed to read {filepath}: {e}")
            return ""
    
    print(f"[ERROR] Could not decode {filepath} with any known encoding")
    return ""


def load_data_from_directory(base_path='data'):
    """
    Load all HTML files from data directory (recursively).
    
    Args:
        base_path (str): Base directory path containing category folders
        
    Returns:
        tuple: (documents, labels, filepaths)
            - documents: list of raw text strings
            - labels: list of category labels
            - filepaths: list of file paths
    """
    print(f"\n[INFO] Loading data from '{base_path}' directory...")
    
    documents = []
    labels = []
    filepaths = []
    
    base_dir = Path(base_path)
    
    if not base_dir.exists():
        print(f"[ERROR] Directory '{base_path}' does not exist!")
        return documents, labels, filepaths
    
    # Process each category folder
    for category_dir in base_dir.iterdir():
        if category_dir.is_dir():
            category_name = category_dir.name
            print(f"\n[INFO] Processing category: {category_name}")
            
            # Process HTML files in category (recursively search all subfolders)
            html_files = list(category_dir.rglob('*.html'))  # rglob for recursive search
            print(f"[INFO] Found {len(html_files)} HTML files")
            
            for html_file in html_files:
                text = read_html_file(html_file)
                if text:
                    documents.append(text)
                    labels.append(category_name)
                    filepaths.append(str(html_file))
    
    print(f"\n[INFO] Total documents loaded: {len(documents)}")
    return documents, labels, filepaths


# ==============================
# Self-test / Example Usage
# ==============================

if __name__ == "__main__":
    print("="*70)
    print("Arabic Text Preprocessor - Phase 3")
    print("="*70)
    
    # Example: Test preprocessing on sample text
    preprocessor = ArabicPreprocessor()
    
    sample_text = "هذا نص عربي تجريبي للاختبار والتحقق من عمل المعالج"
    print(f"\nOriginal text: {sample_text}")
    
    tokens = preprocessor.preprocess(sample_text)
    print(f"Preprocessed tokens: {tokens}")
    print(f"Number of tokens: {len(tokens)}")

