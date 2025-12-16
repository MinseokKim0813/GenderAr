import pandas as pd
import numpy as np
import joblib  # <--- Added to save the model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Global variable for analyzer
analyzer = None

# ---------------------------------------------------------
# SECTION: Linguistic Facts (Feature Engineering)
# ---------------------------------------------------------
def extract_features(name):
    """
    Refined Feature Extraction based on Feedback:
    Prioritizes analyses where POS is Noun, Proper Noun, or Adjective.
    """
    if not isinstance(name, str) or not name:
        return {}

    features = {}

    # --- 1. Orthographic & Surface Features ---
    features['suffix_ta_marbuta'] = name.endswith('ة')
    features['suffix_alif_maqsura'] = name.endswith('ى')
    features['suffix_hamza'] = name.endswith('ء')
    features['last_char'] = name[-1] if len(name) > 0 else ''
    features['last_2_chars'] = name[-2:] if len(name) > 1 else ''

    # --- 2. Deep Morphological Features (via CAMeL Tools) ---
    if analyzer:
        analyses = analyzer.analyze(name)
    else:
        analyses = []

    selected_analysis = None
    
    # FEEDBACK IMPLEMENTATION: Filter for Nouns/Adjectives
    # We prioritize Noun, Proper Noun, or Adjective. 
    # If found, use that analysis. If multiple, take the first match.
    valid_pos = ['noun', 'noun_prop', 'adj']
    
    if analyses:
        # Try to find a valid POS match first
        for analysis in analyses:
            if analysis.get('pos', '') in valid_pos:
                selected_analysis = analysis
                break
        
        # Fallback: If no Noun/Adj found, but analyses exist, 
        # we mark it as "No Valid Morphology" for the hybrid logic later,
        # but for training, we might optionally default to the first one 
        # or leave it empty. 
        # Here, we will return None for selected_analysis to indicate 
        # "Morphological approach failed to find a valid meaning".
        if selected_analysis is None:
             features['valid_morphology_found'] = False
        else:
             features['valid_morphology_found'] = True
    else:
        features['valid_morphology_found'] = False

    if selected_analysis:
        for key, value in selected_analysis.items():
            exclude_keys = [
                'source', 'bw', 'gloss', 'stem', 'stemgloss', 'catib6', 
                'gen', 'form_gen'
            ]
            
            if key not in exclude_keys:
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    features[f'morph_{key}'] = 'UNKNOWN'
                else:
                    features[f'morph_{key}'] = value
    else:
        features['morph_status'] = 'NO_ANALYSIS'

    return features

# ... [Load Data function remains the same] ...
def load_and_process_data():
    # (Copy your original load_and_process_data function here)
    print("Loading Dataset...", flush=True)
    try:
        df = pd.read_csv('muslim_names_cleaned.csv')
    except FileNotFoundError:
        print("Error: 'muslim_names_cleaned.csv' not found.", flush=True)
        return pd.DataFrame()
    df = df.dropna(subset=['arabic_name', 'gender'])
    df = df.drop_duplicates(subset=['arabic_name'])
    df['arabic_name'] = df['arabic_name'].str.strip()
    return df
    
if __name__ == "__main__":
    print("--- Script Started ---", flush=True)
    
    # 1. Setup CAMeL Tools
    print("Initializing CAMeL Tools...", flush=True)
    try:
        from camel_tools.morphology.database import MorphologyDB
        from camel_tools.morphology.analyzer import Analyzer
        try:
            db = MorphologyDB.builtin_db()
        except:
            db = MorphologyDB('calima-msa-r13')
        analyzer = Analyzer(db)
    except Exception as e:
        print(f"CRITICAL ERROR: CAMeL Tools setup failed. {e}", flush=True)
        exit()
    
    # 2. Load Data
    df = load_and_process_data()
    if df.empty: exit()
    
    # 3. Extract Features
    print("Extracting features...", flush=True)
    X_raw = df['arabic_name'].tolist()
    y = df['gender'].tolist()
    
    X_features = [extract_features(name) for name in X_raw]
    
    # 4. Vectorize
    print("Vectorizing...", flush=True)
    vectorizer = DictVectorizer(sparse=True)
    X_vectors = vectorizer.fit_transform(X_features)
    
    # ---------------------------------------------------------
    # NEW SECTION: Evaluation Step
    # ---------------------------------------------------------
    print("\n--- Evaluating Model Performance ---")
    
    # Split data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectors, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train a temporary model just for evaluation
    clf_eval = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
    clf_eval.fit(X_train, y_train)
    
    # Predict on the unseen test set
    y_pred = clf_eval.predict(X_test)
    
    # Print Metrics
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report (Precision, Recall, F1-Score):")
    print(classification_report(y_test, y_pred))
    
    # ---------------------------------------------------------
    # SECTION: Final Training & Saving
    # ---------------------------------------------------------
    print("--- Retraining on Full Dataset for Production ---", flush=True)
    clf_final = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
    clf_final.fit(X_vectors, y) # Train on EVERYTHING for the saved file

    print("Saving Model and Vectorizer...", flush=True)
    joblib.dump(clf_final, 'logreg_gender_model.pkl')
    joblib.dump(vectorizer, 'logreg_vectorizer.pkl')
    print("Saved to 'logreg_gender_model.pkl' and 'logreg_vectorizer.pkl'")