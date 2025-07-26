# Models Directory Documentation

This directory contains the models, training infrastructure, and evaluation frameworks for the Legal NLP + Explainability Toolkit project. The models are specifically designed for legal document analysis, including multi-label clause extraction, abstractive summarization, and explainable AI components.

## Overview

The legal NLP pipeline supports:
- **Multi-label Classification**: 41 CUAD legal clause types with clean name mapping
- **Abstractive Summarization**: Legal document summarization with domain-specific optimization
- **Explainability**: SHAP, LIME, and attention-based interpretability for legal practitioners
- **Comprehensive Evaluation**: Legal domain-specific metrics and performance analysis

## Directory Structure

```
models/
├── README.md                           # This documentation
├── evaluation_report.txt               # Model performance evaluation
├── clause_performance_analysis.csv     # Per-clause performance metrics
├── shap_analysis_report.txt            # SHAP explainability analysis
├── shap_summary_plot.png              # SHAP visualization
├── bert/                              # BERT model for legal clause extraction
│   ├── config.json                    # Model configuration
│   ├── pytorch_model.bin              # Fine-tuned BERT weights
│   ├── tokenizer.json                 # Legal-optimized tokenizer
│   ├── training_results.json          # Training history and metrics
│   ├── evaluation_results.json        # Test set evaluation
│   └── model_info.json               # Model metadata
├── t5/                               # T5 model for legal summarization
│   ├── config.json                   # T5 configuration
│   ├── pytorch_model.bin             # Fine-tuned T5 weights
│   ├── tokenizer.json                # Legal-enhanced tokenizer
│   ├── training_results.json         # Training history
│   └── model_info.json              # Model metadata
└── fine_tuning/                      # Training and fine-tuning scripts
    ├── clause_extraction.py          # Multi-label BERT fine-tuning
    └── summarization.py              # T5 legal summarization fine-tuning
```

## Models

### 1. BERT Legal Clause Extractor (`bert/`)

**Purpose**: Multi-label classification for detecting 41 legal clause types from the CUAD dataset.

**Key Features**:
- Multi-label BERT architecture supporting simultaneous detection of multiple clause types
- Clean clause name mapping for human-readable outputs (e.g., "Agreement Date" vs verbose CUAD questions)
- Legal domain optimization with clause-specific evaluation metrics
- Integration with explainability tools (SHAP, LIME, attention visualization)

**Performance Metrics**:
- F1 Micro/Macro/Weighted scores for multi-label evaluation
- Per-clause performance analysis with support statistics
- Hamming loss and Jaccard similarity for multi-label assessment
- Confusion matrices and classification reports

**Supported Clause Types** (41 total):
```
Agreement Date, Anti-Assignment, Audit Rights, Cap on Liability, 
Change of Control, Competitive Restriction Exception, Covenant Not To Sue,
Document Name, Effective Date, Exclusivity, Expiration Date, Governing Law,
Insurance, IP Ownership Assignment, License Grant, Liquidated Damages,
Minimum Commitment, Most Favored Nation, Non-Compete, Non-Disparagement,
Post-Termination Services, Price Restrictions, Revenue/Profit Sharing,
Termination for Convenience, Third Party Beneficiary, Uncapped Liability,
and more...
```

### 2. T5 Legal Summarizer (`t5/`)

**Purpose**: Abstractive summarization of legal contracts and clauses with domain-specific optimization.

**Key Features**:
- Legal-specific token enhancement for better domain understanding
- Contract-aware preprocessing with legal phrase normalization
- ROUGE evaluation metrics for summarization quality assessment
- Configurable compression ratios and length constraints
- Batch processing capabilities for document collections

**Legal Optimizations**:
- Legal phrase normalization (shall → must, hereof → of this agreement)
- Contract-specific tokens (`<legal_entity>`, `<termination_clause>`, etc.)
- Legal document structure awareness
- Optimized beam search for coherent legal summaries

## Fine-tuning Scripts

### [`clause_extraction.py`](fine_tuning/clause_extraction.py)