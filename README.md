# GenderAr: Arabic Name Gender Classification

A comprehensive machine learning system for classifying the gender of Arabic names using multiple approaches, including fine-tuned transformer models, morphological analysis, and hybrid inference.

## Overview

This project implements several methods for Arabic name gender classification:
- **Fine-tuned AraBERT**: Transformer-based deep learning model
- **Morphological Analysis**: Logistic regression with linguistic features using CAMeL Tools
- **Hybrid Inference**: Intelligent combination of morphological and deep learning approaches
- **Word Embedding Centroid**: Geometric approach using AraBERT embeddings

## Features

- 🎯 Multiple classification approaches for robust predictions
- 🔄 Hybrid inference system that intelligently selects the best method
- 📊 Comprehensive evaluation metrics and performance reports
- 🔤 Morphological feature extraction using CAMeL Tools
- 🤖 Fine-tuned AraBERT model for deep learning-based classification
- 📈 Training visualization and metrics tracking

## Installation

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended for training)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd GenderAr
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure CAMeL Tools is properly configured (for morphological analysis):
```bash
# CAMeL Tools should be installed via requirements.txt
# The system will attempt to use the builtin database or fallback to 'calima-msa-r13'
```

### Windows Setup

For Windows users, follow these additional steps:

1. **Open PowerShell or Command Prompt**:
   - Navigate to the project directory:
   ```powershell
   cd path\to\GenderAr
   ```

2. **Create a virtual environment** (recommended):
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   
   If you get an execution policy error, run:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
   Then try activating again.

3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

4. **Windows-specific considerations**:
   - **CAMeL Tools**: May require additional setup. If you encounter issues, try:
     ```powershell
     pip install camel-tools --no-cache-dir
     ```
   - **CUDA for PyTorch** (if using GPU):
     - Install CUDA Toolkit from [NVIDIA](https://developer.nvidia.com/cuda-downloads)
     - PyTorch will detect CUDA automatically if properly installed
     - Verify CUDA availability:
     ```powershell
     python -c "import torch; print(torch.cuda.is_available())"
     ```
   - **Path separators**: All scripts use forward slashes (`/`) which work on Windows, but if you encounter path issues, ensure you're using the correct format

5. **Verify installation**:
   ```powershell
   python -c "import transformers; import torch; import camel_tools; print('All dependencies installed successfully!')"
   ```

## Project Structure

```
GenderAr/
├── fine_tuning.py              # AraBERT fine-tuning script
├── hybrid_inference.py          # Hybrid inference system
├── logistic_reg_without_gen.py # Morphological feature-based logistic regression
├── transformer.py               # Baseline transformer evaluation
├── word_embedding.py            # Word embedding centroid approach
├── final_arabert_gender_model/  # Saved fine-tuned AraBERT model
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
├── logreg_gender_model.pkl      # Saved logistic regression model
├── logreg_vectorizer.pkl        # Saved feature vectorizer
├── muslim_names_cleaned.csv     # Dataset
├── training_metrics.png         # Training visualization
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Usage

> **Note**: All commands work in both bash (Linux/Mac) and PowerShell (Windows). On Windows, you can use either PowerShell or Command Prompt.

### 1. Fine-tuning AraBERT Model

Train the transformer model on the Arabic names dataset:

**Linux/Mac:**
```bash
python fine_tuning.py
```

**Windows (PowerShell):**
```powershell
python fine_tuning.py
```

This script will:
- Load the Muslim names dataset from HuggingFace
- Fine-tune the AraBERT model
- Save the trained model to `./final_arabert_gender_model/`
- Generate training metrics visualization

### 2. Training Morphological Model

Train the logistic regression model with morphological features:

```bash
python logistic_reg_without_gen.py
```

This script will:
- Extract morphological features using CAMeL Tools
- Train a logistic regression classifier
- Save the model and vectorizer to `.pkl` files
- Evaluate performance on a test set

### 3. Hybrid Inference

Run the hybrid inference system that combines both approaches:

```bash
python hybrid_inference.py
```

The hybrid system:
- Uses morphological analysis when valid noun/adjective analysis is found
- Falls back to deep learning (AraBERT) for other cases
- Evaluates on the test set and provides detailed metrics

### 4. Word Embedding Approach

Run the geometric centroid-based classification:

```bash
python word_embedding.py
```

This approach:
- Computes embeddings for all training names
- Calculates centroids for male and female classes
- Classifies test names using cosine similarity

### 5. Baseline Evaluation

Evaluate the untrained transformer baseline:

```bash
python transformer.py
```

## Models and Approaches

### 1. Fine-tuned AraBERT
- **Model**: `aubmindlab/bert-base-arabertv02`
- **Approach**: Transfer learning with fine-tuning
- **Features**: Contextual embeddings from transformer architecture
- **Output**: Binary classification (Male/Female)

### 2. Morphological Logistic Regression
- **Features**: 
  - Orthographic features (suffixes, last characters)
  - Deep morphological features from CAMeL Tools
  - Part-of-speech filtering (Noun, Proper Noun, Adjective)
- **Model**: Logistic Regression with DictVectorizer
- **Advantage**: Interpretable linguistic features

### 3. Hybrid System
- **Strategy**: 
  1. Attempt morphological analysis first
  2. Use morphological model if valid analysis found
  3. Fallback to AraBERT for ambiguous cases
- **Benefit**: Combines strengths of both approaches

### 4. Word Embedding Centroid
- **Method**: Mean pooling of AraBERT embeddings
- **Classification**: Cosine similarity to class centroids
- **Use Case**: Geometric interpretation of name embeddings

## Configuration

Key parameters in `fine_tuning.py`:
- `MODEL_NAME`: "aubmindlab/bert-base-arabertv02"
- `MAX_LENGTH`: 16 (maximum sequence length)
- `BATCH_SIZE`: 32
- `EPOCHS`: 5
- `LEARNING_RATE`: 2e-5
- `SEED`: 42 (for reproducibility)

## Dataset

The project uses the **Muslim Names Dataset** from HuggingFace:
- **Source**: `takiuddinahmed/muslim-names-dataset`
- **Format**: CSV with `arabic_name` and `gender` columns
- **Split**: 80% train, 10% validation, 10% test

## Dependencies

See `requirements.txt` for the complete list. Key dependencies include:
- `transformers` - HuggingFace transformers library
- `torch` - PyTorch for deep learning
- `scikit-learn` - Machine learning utilities
- `camel-tools` - Arabic morphological analysis
- `datasets` - Dataset loading and processing
- `pandas`, `numpy` - Data manipulation
- `matplotlib`, `seaborn` - Visualization

## Results

The hybrid inference system provides:
- Overall accuracy metrics
- Per-class precision, recall, and F1-scores
- Statistics on method usage (morphological vs. deep learning)
- Detailed classification reports

Training metrics are visualized in `training_metrics.png`, showing:
- Validation accuracy over epochs
- Training and validation loss curves

## Notes

- The morphological approach requires valid noun/adjective analyses from CAMeL Tools
- Names without valid morphological analysis automatically use the deep learning fallback
- All models use stratified train/test splits for balanced evaluation
- The fine-tuned model is saved in a format compatible with HuggingFace transformers

## License

[Add your license information here]

## Citation

If you use this project in your research, please cite:
[Add citation information here]

## Contributing

[Add contribution guidelines here]

## Contact

[Add contact information here]

