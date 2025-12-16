import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split # Import train_test_split
from tqdm import tqdm

# 1. Setup Device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Load Pre-trained AraBERT Model
# We use 'aubmindlab/bert-base-arabertv02' as specified in your paper
model_name = "aubmindlab/bert-base-arabertv02"
print("Loading AraBERT model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# 3. Define Helper Function to Extract Embeddings
def get_embedding(text):
    """
    Tokenizes the text and returns the mean of the last hidden state 
    as the vector representation.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    
    with torch.no_grad(): # Disable gradient calculation for inference
        outputs = model(**inputs)
    
    # We take the last hidden state: [batch_size, seq_len, hidden_dim]
    last_hidden_state = outputs.last_hidden_state
    
    # Strategy: Mean Pooling (Average of all tokens to get one vector per name)
    # Alternatively, you could use outputs.pooler_output for the [CLS] token
    mean_embedding = last_hidden_state.mean(dim=1).cpu().numpy()
    
    return mean_embedding[0] # Return 1D array

# --- ASSUMING YOU HAVE YOUR DATA SPLIT HERE ---
# Ideally, X_train, y_train, X_test, y_test are lists or Series.
# Example format:
# X_train = ["أحمد", "فاطمة", ...]
# y_train = ["Male", "Female", ...]

# For the sake of this code running, let's assume 'train_df' and 'test_df' 
# are pandas DataFrames with 'arabic_name' and 'gender' columns.

def run_geometric_approach(train_df, test_df):
    
    # 4. Compute Centroids (Training Phase)
    print("Computing embeddings for Training set...")
    male_vectors = []
    female_vectors = []

    # Iterate through training data
    for _, row in tqdm(train_df.iterrows(), total=train_df.shape[0]):
        name = row['arabic_name']
        gender = row['gender']
        
        try:
            vec = get_embedding(str(name))
            
            if gender.lower() == 'male':
                male_vectors.append(vec)
            elif gender.lower() == 'female':
                female_vectors.append(vec)
        except Exception as e:
            print(f"error processing name '{name}': {e}") # Uncommented for debugging
            continue

    # Calculate Centroids (Equation 5 in your paper)
    # Stack lists into numpy arrays and calculate mean along axis 0
    if not male_vectors:
        raise ValueError("no male samples processed for centroid calculation. check your training data or get_embedding function.")
    V_male_centroid = np.mean(np.vstack(male_vectors), axis=0)

    if not female_vectors:
        raise ValueError("no female samples processed for centroid calculation. check your training data or get_embedding function.")
    V_female_centroid = np.mean(np.vstack(female_vectors), axis=0)
    
    print("Centroids computed.")

    # 5. Inference (Testing Phase)
    print("Running Inference on Test set...")
    y_true = []
    y_pred = []

    # Reshape centroids for cosine_similarity function [1, dim]
    V_male_centroid = V_male_centroid.reshape(1, -1)
    V_female_centroid = V_female_centroid.reshape(1, -1)

    for _, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
        name = row['arabic_name']
        true_gender = row['gender']
        
        try:
            # Get vector for new name
            v_new = get_embedding(str(name)).reshape(1, -1)
            
            # Calculate Cosine Similarity (Equation 6)
            score_male = cosine_similarity(v_new, V_male_centroid)[0][0]
            score_female = cosine_similarity(v_new, V_female_centroid)[0][0]
            
            # Argmax classification
            if score_male > score_female:
                pred_gender = 'male'
            else:
                pred_gender = 'female'
                
            y_true.append(true_gender)
            y_pred.append(pred_gender)
        except Exception as e:
            print(f"error processing name '{name}' during inference: {e}") # Uncommented for debugging
            continue

    # 6. Evaluation
    print("\n--- Results for Word Embedding (Centroid) Approach ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['female', 'male']))

# --- Data Loading and Splitting ---
if __name__ == "__main__":
    print("Loading data from 'muslim_names_cleaned.csv'...")
    try:
        data_df = pd.read_csv('muslim_names_cleaned.csv')
    except FileNotFoundError:
        print("error: 'muslim_names_cleaned.csv' not found. please make sure the file is in the same directory.")
        exit()
    
    # Ensure necessary columns exist
    if 'arabic_name' not in data_df.columns or 'gender' not in data_df.columns:
        print("error: 'muslim_names_cleaned.csv' must contain 'arabic_name' and 'gender' columns.")
        exit()

    # Drop rows with missing values in critical columns
    data_df.dropna(subset=['arabic_name', 'gender'], inplace=True)
    
    # Split data into training and testing sets
    # We'll use a 80/20 split, but you can adjust test_size as needed
    train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=42, stratify=data_df['gender'])
    
    print(f"data loaded: {len(data_df)} samples total.")
    print(f"training set size: {len(train_df)} samples.")
    print(f"testing set size: {len(test_df)} samples.")

    # Call the main function with your dataframes
    run_geometric_approach(train_df, test_df)