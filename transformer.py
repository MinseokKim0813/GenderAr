import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --------------------------------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------------------------------
MODEL_NAME = "aubmindlab/bert-base-arabertv02"
DATASET_NAME = "takiuddinahmed/muslim-names-dataset"
MAX_LENGTH = 16
SEED = 42

# --------------------------------------------------------------------------
# 2. DATA LOADING (Identical to Training Script)
# --------------------------------------------------------------------------
print("Loading and preparing data...")
dataset = load_dataset(DATASET_NAME, split="train")
df = dataset.to_pandas()
df = df[['arabic_name', 'gender']]
df = df.dropna()
df = df.drop_duplicates(subset=['arabic_name', 'gender'])

label_map = {'male': 0, 'female': 1}
df['label'] = df['gender'].map(label_map)
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

# --------------------------------------------------------------------------
# 3. SPLITTING (Must match Training Script exactly for fair comparison)
# --------------------------------------------------------------------------
# 80% Train, 20% Temp
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df['label'])
# Split Temp into 50% Val / 50% Test
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED, stratify=temp_df['label'])

# We only need the TEST dataset for this evaluation
test_dataset = Dataset.from_pandas(test_df)

# --------------------------------------------------------------------------
# 4. TOKENIZATION
# --------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["arabic_name"], 
        padding="max_length", 
        truncation=True, 
        max_length=MAX_LENGTH
    )

tokenized_test = test_dataset.map(tokenize_function, batched=True)
tokenized_test = tokenized_test.remove_columns(["arabic_name", "gender", "__index_level_0__"])
tokenized_test.set_format("torch")

# --------------------------------------------------------------------------
# 5. MODEL SETUP (Pre-trained Body, Random Head)
# --------------------------------------------------------------------------
print("Loading model with UNTRAINED classification head...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=2
)

# --------------------------------------------------------------------------
# 6. EVALUATION
# --------------------------------------------------------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# We define a Trainer just for the convenient evaluate() method
trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics,
)

print("Running baseline evaluation (No Training)...")
# Run evaluation on the test set
baseline_results = trainer.evaluate(tokenized_test)

print("\n------------------------------------------------")
print("BASELINE (UNTRAINED) RESULTS")
print("------------------------------------------------")
print(f"Accuracy:  {baseline_results['eval_accuracy']:.4f}")
print(f"Macro F1:  {baseline_results['eval_f1']:.4f}")
print(f"Precision: {baseline_results['eval_precision']:.4f}")
print(f"Recall:    {baseline_results['eval_recall']:.4f}")
print("------------------------------------------------")
print("Note: Low scores are expected. The classifier head has not learned the task yet.")