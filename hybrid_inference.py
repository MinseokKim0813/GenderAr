import joblib
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer

# ---------------------------------------------------------
# 1. LOAD MODELS
# ---------------------------------------------------------
print("Loading Morphological Model...", flush=True)
logreg_model = joblib.load('logreg_gender_model.pkl')
vectorizer = joblib.load('logreg_vectorizer.pkl')

print("Loading CAMeL Tools...", flush=True)
try:
    db = MorphologyDB.builtin_db()
except:
    # Fallback if default isn't set, though you likely have r13 now
    db = MorphologyDB('calima-msa-r13')
analyzer = Analyzer(db)

print("Loading Deep Learning Model (AraBERT)...", flush=True)
DL_MODEL_PATH = "./final_arabert_gender_model"
tokenizer = AutoTokenizer.from_pretrained(DL_MODEL_PATH)
dl_model = AutoModelForSequenceClassification.from_pretrained(DL_MODEL_PATH)
dl_model.eval()

# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------
def get_morph_features(name):
    """
    Extracts features for LogReg. Returns (features_dict, is_valid_morphology).
    """
    if not isinstance(name, str) or not name:
        return {}, False

    features = {}
    
    # Surface features
    features['suffix_ta_marbuta'] = name.endswith('ة')
    features['suffix_alif_maqsura'] = name.endswith('ى')
    features['suffix_hamza'] = name.endswith('ء')
    features['last_char'] = name[-1] if len(name) > 0 else ''
    features['last_2_chars'] = name[-2:] if len(name) > 1 else ''

    # Deep features
    analyses = analyzer.analyze(name)
    selected_analysis = None
    
    # Filter for Noun/PropNoun/Adj
    valid_pos = ['noun', 'noun_prop', 'adj']
    
    if analyses:
        for analysis in analyses:
            if analysis.get('pos', '') in valid_pos:
                selected_analysis = analysis
                break
    
    is_valid_morphology = False
    if selected_analysis:
        is_valid_morphology = True
        for key, value in selected_analysis.items():
            exclude_keys = ['source', 'bw', 'gloss', 'stem', 'stemgloss', 'catib6', 'gen', 'form_gen']
            if key not in exclude_keys:
                features[f'morph_{key}'] = value if value is not None else 'UNKNOWN'
    else:
        features['morph_status'] = 'NO_ANALYSIS'
        
    features['valid_morphology_found'] = is_valid_morphology
    return features, is_valid_morphology

def predict_dl(name):
    """Predicts gender using AraBERT."""
    inputs = tokenizer(name, return_tensors="pt", truncation=True, max_length=16)
    with torch.no_grad():
        outputs = dl_model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_idx = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][pred_idx].item()
    
    # Ensure this map matches your fine_tuning.py labels
    label_map = {0: "Male", 1: "Female"} 
    return label_map[pred_idx], confidence

def predict_gender_hybrid(name):
    """
    Hybrid Logic:
    1. Try Morphological (LogReg) if valid noun/adj found.
    2. Else, fallback to Deep Learning (AraBERT).
    """
    features, is_valid = get_morph_features(name)
    
    if is_valid:
        vec = vectorizer.transform(features)
        pred = logreg_model.predict(vec)[0]
        probs = logreg_model.predict_proba(vec).max()
        # LogReg usually returns the label directly (e.g. 'Male' or 'Female')
        return pred, probs, "Morphological"
    else:
        pred, confidence = predict_dl(name)
        return pred, confidence, "Deep Learning"

# ---------------------------------------------------------
# 3. TEST SET EVALUATION LOGIC
# ---------------------------------------------------------
def load_test_set():
    print("\nPreparing Test Set...", flush=True)
    try:
        df = pd.read_csv('muslim_names_cleaned.csv')
    except FileNotFoundError:
        print("Error: 'muslim_names_cleaned.csv' not found.")
        exit()
        
    # Standardize Cleaning
    df = df.dropna(subset=['arabic_name', 'gender'])
    df = df.drop_duplicates(subset=['arabic_name'])
    df['arabic_name'] = df['arabic_name'].str.strip()
    
    # Standardize Labels to Title Case for comparison (Male/Female)
    df['gender'] = df['gender'].astype(str).str.strip().str.capitalize()
    
    # Replicate the Split Logic (80 Train / 10 Val / 10 Test)
    # This ensures we test on the same data intended for testing
    X_raw = df['arabic_name'].tolist()
    y_raw = df['gender'].tolist()
    
    # First split: 80% Train, 20% Temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )
    # Second split: Split Temp into 50% Val, 50% Test (10% total each)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Test Set Size: {len(X_test)} names")
    return X_test, y_test

# ---------------------------------------------------------
# 4. EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    
    # A. Quick Single Instance Test (Optional, just to verify it works)
    print("\n" + "="*40)
    print("SINGLE INSTANCE CHECK")
    print("="*40)
    demo_name = "حمزة"
    g, p, m = predict_gender_hybrid(demo_name)
    print(f"Name: {demo_name} -> {g} ({p:.2%}) via {m}")

    # B. Full Test Set Evaluation
    print("\n" + "="*40)
    print("RUNNING FULL TEST SET EVALUATION")
    print("="*40)
    
    test_names, test_labels = load_test_set()
    
    predictions = []
    methods_used = []
    
    # Iterate and Predict
    # (Using a simple loop; for massive datasets batching is better, 
    # but for ~1.5k names this is fine)
    for i, name in enumerate(test_names):
        pred, prob, method = predict_gender_hybrid(name)
        
        # Normalization: Ensure prediction is Title Case (Male/Female)
        pred = pred.capitalize() 
        
        predictions.append(pred)
        methods_used.append(method)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(test_names)}...", flush=True)

    # C. Calculate Metrics
    print("\n" + "="*40)
    print("PERFORMANCE REPORT")
    print("="*40)
    
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(test_labels, predictions, digits=4))
    
    # D. Hybrid Usage Stats
    morph_count = methods_used.count("Morphological")
    dl_count = methods_used.count("Deep Learning")
    print("\nHybrid Pipeline Statistics:")
    print(f"Names handled by Morphological Logic: {morph_count} ({morph_count/len(test_names):.1%})")
    print(f"Names handled by Deep Learning (Fallback): {dl_count} ({dl_count/len(test_names):.1%})")