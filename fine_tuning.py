import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt  # <--- Added for plotting
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --------------------------------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------------------------------
MODEL_NAME = "aubmindlab/bert-base-arabertv02"
DATASET_NAME = "takiuddinahmed/muslim-names-dataset"
MAX_LENGTH = 16
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-5
SEED = 42

# --------------------------------------------------------------------------
# 2. DATA LOADING & CLEANING
# --------------------------------------------------------------------------
print(f"Loading dataset: {DATASET_NAME}...")
dataset = load_dataset(DATASET_NAME, split="train")

df = dataset.to_pandas()
df = df[['arabic_name', 'gender']]

# Cleaning
initial_count = len(df)
df = df.dropna()
df = df.drop_duplicates(subset=['arabic_name', 'gender'])
print(f"Data cleaned. Rows dropped: {initial_count - len(df)}. Total rows: {len(df)}")

# Label Encoding
label_map = {'male': 0, 'female': 1}
df['label'] = df['gender'].map(label_map)
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

# --------------------------------------------------------------------------
# 3. SPLITTING
# --------------------------------------------------------------------------
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED, stratify=temp_df['label'])

print(f"Splits created - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# --------------------------------------------------------------------------
# 4. TOKENIZATION
# --------------------------------------------------------------------------
print("Tokenizing data...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["arabic_name"], 
        padding="max_length", 
        truncation=True, 
        max_length=MAX_LENGTH
    )

tokenized_datasets = DatasetDict({
    'train': train_dataset.map(tokenize_function, batched=True),
    'validation': val_dataset.map(tokenize_function, batched=True),
    'test': test_dataset.map(tokenize_function, batched=True)
})

tokenized_datasets = tokenized_datasets.remove_columns(["arabic_name", "gender", "__index_level_0__"])
tokenized_datasets.set_format("torch")

# --------------------------------------------------------------------------
# 5. MODEL SETUP
# --------------------------------------------------------------------------
print("Initializing Model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=2,
    id2label={0: "male", 1: "female"},
    label2id={"male": 0, "female": 1}
)

# --------------------------------------------------------------------------
# 6. METRICS FUNCTION
# --------------------------------------------------------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# --------------------------------------------------------------------------
# 7. TRAINING
# --------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    evaluation_strategy="epoch", 
    save_strategy="epoch",       
    load_best_model_at_end=True, 
    learning_rate=LEARNING_RATE
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=compute_metrics,
)

print("Starting Training...")
trainer.train()

# --------------------------------------------------------------------------
# 8. EVALUATION
# --------------------------------------------------------------------------
print("Evaluating on Test Set...")
test_results = trainer.evaluate(tokenized_datasets['test'])

print("\n------------------------------------------------")
print("FINAL TEST RESULTS")
print("------------------------------------------------")
print(f"Accuracy:  {test_results['eval_accuracy']:.4f}")
print(f"Macro F1:  {test_results['eval_f1']:.4f}")
print("------------------------------------------------")

model.save_pretrained("./final_arabert_gender_model")
tokenizer.save_pretrained("./final_arabert_gender_model")

# --------------------------------------------------------------------------
# 9. PLOTTING HISTORY
# --------------------------------------------------------------------------
print("Generating training plots...")

# Extract logs from trainer memory
history = trainer.state.log_history

# Extract Validation Accuracy (recorded at end of every epoch)
val_acc = [x['eval_accuracy'] for x in history if 'eval_accuracy' in x]
val_epochs = [x['epoch'] for x in history if 'eval_accuracy' in x]

# Extract Training Loss (recorded every logging_steps)
train_loss = [x['loss'] for x in history if 'loss' in x]
train_epochs = [x['epoch'] for x in history if 'loss' in x]

# Extract Validation Loss (if available)
val_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]

# Create the plot
plt.figure(figsize=(12, 5))

# Subplot 1: Accuracy over Epochs
plt.subplot(1, 2, 1)
plt.plot(val_epochs, val_acc, marker='o', linestyle='-', color='b', label='Validation Accuracy')
plt.title('Accuracy Evolution')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

# Subplot 2: Loss over Epochs
plt.subplot(1, 2, 2)
plt.plot(train_epochs, train_loss, label='Training Loss', color='orange', alpha=0.7)
if val_loss:
    plt.plot(val_epochs, val_loss, marker='o', linestyle='--', label='Validation Loss', color='red')
plt.title('Loss Evolution')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('training_metrics.png')
print("Graph saved as 'training_metrics.png'")