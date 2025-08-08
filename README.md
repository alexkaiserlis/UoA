# NER for Legal Documents

A comprehensive Named Entity Recognition (NER) system for Greek legal documents, developed as part of a research project at the University of the Aegean.

## Project Overview

This project focuses on developing and evaluating NER models specifically trained on Greek legal text data. It includes data preprocessing, model training, evaluation, and comparison scripts.

## Project Structure

```
├── 01_DATASETS/                    # Dataset management and preprocessing
│   ├── GREEK_LEGAL_NER/           # Main NER dataset
│   │   ├── *.py                   # Data processing scripts
│   │   ├── *.json                 # Training/validation/test data
│   │   └── NER MODEL/             # Trained models
│   ├── ETHNIKO_TYPOGRAFEIO/       # Legal documents from Greek National Printing Office
│   ├── HELLASVOC/                 # Greek legal vocabulary dataset
│   └── RAW_DATA/                  # Raw unprocessed data
│
├── 02_ANALYSIS_SCRIPTS/           # Data analysis and comparison tools
│   ├── COMPARISON/                # Dataset comparison utilities
│   ├── DATASET_ANALYSIS/          # Statistical analysis of datasets
│   └── ENTITY_COUNTING/           # Entity statistics and counting
│
├── 03_ML_MODELS/                  # Machine Learning models and training
│   ├── NER_MODELS/               # Main NER model implementations
│   │   ├── *.py                  # Training and evaluation scripts
│   │   ├── config.py             # Model configurations
│   │   ├── experiments/          # Training experiments and results
│   │   └── logs/                 # Training logs
│   ├── BASELINE_MODELS/          # Baseline model implementations
│   └── EXPERIMENTAL/             # Experimental model architectures
│
├── 04_RESULTS/                   # Training results and analysis
│   ├── COMPARISON_RESULTS/       # Model comparison results
│   ├── CRF_RESULTS/             # CRF model specific results
│   └── FOCAL_LOSS_RESULTS/      # Focal loss experiments
│
├── 05_SCRAPERS/                 # Web scraping utilities
│   ├── ET_SCRAPER/              # Ethniko Typografeio scraper
│   ├── FEK_SCRAPER/             # FEK (Government Gazette) scraper
│   └── HELLASVOC_SCRAPER/       # Hellasvoc dataset scraper
│
├── 06_CONFIGS/                  # Configuration files
│   ├── MODEL_CONFIGS/           # Model-specific configurations
│   └── VSCODE/                  # VS Code settings
│
├── 07_DOCUMENTATION/            # Project documentation
│   ├── ANALYSIS_REPORTS/        # Analysis and evaluation reports
│   └── README_FILES/            # Various README files
│
└── FINAL_VOCABULARY_PACKAGE/    # Legal vocabulary processing tools
    ├── scripts/                 # Vocabulary processing scripts
    └── IR/                      # Information Retrieval tools
```

## Key Features

### 🤖 Machine Learning Models
- **Transformer-based NER**: RoBERTa models fine-tuned for Greek legal text
- **CRF Enhancement**: Conditional Random Field layer for sequence labeling
- **Focal Loss**: Advanced loss function for handling class imbalance
- **Multi-run Experiments**: Statistical validation with multiple training runs

### 📊 Analysis Tools
- Dataset comparison and statistics
- Entity distribution analysis
- Model performance evaluation
- Cross-dataset validation

### 🔄 Data Processing
- IOB format conversion
- Data cleaning and preprocessing
- Dataset combination utilities
- Entity validation tools

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- Transformers library
- scikit-learn
- pandas
- numpy

### Installation
1. Clone the repository:
```bash
git clone https://github.com/alexkaiserlis/UoA.git
cd UoA
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Training a Model
```bash
cd 03_ML_MODELS/NER_MODELS
python main_script.py --preset CRF_ENHANCED
```

#### Multi-run Experiments
```bash
# Run 6 experiments with different seeds for statistical validation
python main_script.py --preset CRF_ENHANCED --num-runs 6 --base-seed 42
```

#### Data Analysis
```bash
cd 02_ANALYSIS_SCRIPTS/DATASET_ANALYSIS
python greek_legal_ner_analysis.py
```

## Model Configurations

The project includes several pre-configured model setups:

- **BASELINE**: Standard RoBERTa model for NER
- **CRF_ENHANCED**: RoBERTa + CRF layer for improved sequence labeling
- **FOCAL_LOSS**: Using focal loss to handle class imbalance
- **AUGMENTED**: Data augmentation techniques

## Dataset Information

### Greek Legal NER Dataset
- **Entities**: PERSON, ORGANIZATION, LOCATION, LAW, etc.
- **Format**: IOB tagging scheme
- **Language**: Greek
- **Domain**: Legal documents

### Data Sources
- Greek National Printing Office (Ethniko Typografeio)
- Government Gazette (FEK)
- Legal vocabulary databases

## Results and Evaluation

The project includes comprehensive evaluation metrics:
- F1-score per entity type
- Overall F1-score
- Precision and Recall
- Confusion matrices
- Statistical significance testing

## Multi-run Feature

For statistical reliability, the system supports multi-run experiments:
- Multiple training runs with different seeds
- Aggregated statistics (mean, std, min, max)
- Individual and combined results
- Robust evaluation methodology

See [MULTI_RUN_GUIDE.md](03_ML_MODELS/NER_MODELS/MULTI_RUN_GUIDE.md) for detailed usage instructions.
