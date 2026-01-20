# GenderAr: Arabic Name Gender Classification

A comprehensive machine learning system for classifying the gender of Arabic names. This project compares multiple methodologies—ranging from linguistic rules to deep learning—and introduces a Hybrid Cascade System that achieves 91.68% accuracy by combining the structural precision of morphology with the robustness of transformers.

The comprehensive research report is accessible through this [link](https://drive.google.com/file/d/12Fqo6CqU5zPI6X3DGB6CRB2hLpGwGlxB/view)

## Overview

Gender identification of Arabic names is a complex task due to the lack of strict orthographic rules and the ambiguity of names that share forms with particles (e.g., "Ali" vs. "on").

This project implements and evaluates four distinct approaches:

- **Morphological Analysis (Logistic Regression)**: Uses engineered linguistic features (suffixes, templates) via CAMeL Tools.
- **Word Embedding Centroids**: A geometric approach using cosine similarity within pre-trained AraBERT vector space.
- **Fine-tuned AraBERT**: Transfer learning using bert-base-arabertv02.
- **Hybrid Inference (Cascade)**: A pipeline that intelligently prioritizes morphological rules and falls back to deep learning for ambiguous cases.

## 🏆 Key Results

Based on a test set of ~1,300 names (from a cleaned dataset of 13,622), the Hybrid approach significantly outperformed individual baselines.

| Approach | Accuracy | F1-Score | Notes |
|----------|----------|----------|-------|
| **Hybrid (Cascade)** | **91.68%** | 0.91 | Best Performance |
| Morphological (LogReg) | 76.69% | 0.77 | High precision, low coverage |
| Fine-tuned AraBERT | 74.73% | 0.74 | Robust but data-hungry |
| Word Embedding | 65.00% | 0.65 | Shows semantic gender bias |

## 🧠 Methodology & Analysis

### 1. The Morphological Approach (High Precision)

We utilize CAMeL Tools to extract 36 distinct features, including:

- **Root & Pattern**: Derivational templates.
- **Suffixes**: Teh Marbuta, Alif Maqsura.
- **N-Grams**: Analysis shows the Last 2 Characters are the most predictive feature.

### 2. The Deep Learning Approach (High Recall)

We fine-tuned `aubmindlab/bert-base-arabertv02` for 5 epochs. While powerful, the model suffered from rapid overfitting due to the dataset size (~13k samples), peaking at an accuracy of ~74%.

### 3. The Hybrid Strategy (Best of Both Worlds)

The project analysis revealed a critical "blind spot": 94% of names were not identified as Proper Nouns by standard analyzers (e.g., the name "Ali" parsed as the preposition "on").

**The Solution:**

1. **Filter**: The system checks if CAMeL Tools identifies the word as a Noun, Proper Noun, or Adjective.
2. **Branch**:
   - **If Valid Noun**: Use the Morphological Classifier (Higher precision for structured names).
   - **If Ambiguous/Particle**: Fall back to Fine-tuned AraBERT (Contextual handling of ambiguous names).

## Features

- 🎯 **State-of-the-Art Accuracy**: 91.68% on the Muslim Names Dataset.
- 🔄 **Smart Fallback System**: Automatically detects when linguistic analysis fails.
- 📊 **Comprehensive Metrics**: Tracks Precision, Recall, and F1 across gender classes.
- 🔤 **Feature Engineering**: Extracts n-grams and morphological templates.
- 📈 **Visualization**: Generates training curves and confusion matrices.

## Installation

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (highly recommended for AraBERT training)

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd GenderAr
```

2. **Create a virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **CAMeL Tools Data:**
You may need to install the CAMeL Tools data resources:
```bash
camel_data -i all
```

## Usage

**Note**: Model weights are not included in the repo (5GB+). You must run the training scripts locally to generate them.

### 1. Train the Models

You must train both the morphological and transformer models before running the hybrid inference.

**Step A: Fine-tune AraBERT**
```bash
python fine_tuning.py
```
**Output**: Saves model to `./final_arabert_gender_model/`

**Step B: Train Logistic Regression**
```bash
python logistic_reg_without_gen.py
```
**Output**: Saves `logreg_gender_model.pkl` and vectorizer.

### 2. Run Hybrid Inference

Once models are trained, run the cascade system:
```bash
python hybrid_inference.py
```

This script will load both models and apply the fallback logic described in the Methodology.

### 3. Experimental Scripts

- `word_embedding.py`: Runs the geometric centroid experiment.
- `transformer.py`: Runs the baseline (untrained) transformer evaluation.

## Dataset

The project utilizes the **Muslim Names Dataset** (sourced from HuggingFace).

- **Total Cleaned Size**: 13,622 names.
- **Distribution**: Balanced (Male: 50.4%, Female: 49.6%).
- **Splits**: 80% Train, 10% Validation, 10% Test.

## Project Structure

```
GenderAr/
├── fine_tuning.py              # Transformer training pipeline
├── logistic_reg_without_gen.py # Feature extraction & LogReg training
├── hybrid_inference.py         # The main 91% accuracy pipeline
├── word_embedding.py           # Centroid-based experiment
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Limitations

- **Dataset Size**: While sufficient for linear models, 13k names is relatively small for Transformers, leading to early overfitting.
- **Binary Classification**: The system currently classifies strictly into Male/Female, not accounting for unisex names without context.
