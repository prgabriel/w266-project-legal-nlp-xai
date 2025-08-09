# Legal NLP + Explainability Toolkit

## Project Title
**Towards Responsible AI in Legal NLP: Interpretable Clause Extraction and Summarization for Contract Review**

___

## Overview
This project implements a comprehensive end-to-end pipeline for legal document analysis with explainable AI capabilities:

1. **Multi-Label Clause Extraction**: Detect 41 legal clause types simultaneously using fine-tuned BERT models with clean, human-readable clause names from the CUAD dataset.
2. **Legal Document Summarization**: Generate abstractive summaries of legal contracts using domain-optimized T5 models with legal-specific preprocessing.
3. **Comprehensive Explainability (XAI)**: Provide interpretable insights through SHAP analysis, LIME explanations, and attention visualization to help legal practitioners understand model decisions.
4. **Interactive Legal Interface**: Streamlit-based application for real-time legal document analysis with integrated explainability features.

### Key Features
- **41 Legal Clause Types**: Support for all CUAD clause categories with clean name mapping
- **Multi-Label Architecture**: Simultaneous detection of multiple clause types in legal documents  
- **Legal Domain Optimization**: Specialized preprocessing and evaluation metrics for legal text
- **Practitioner-Friendly**: Human-readable outputs and explanations designed for legal professionals
- **Comprehensive Evaluation**: Multi-label F1 scores, ROUGE metrics, and legal domain-specific benchmarks

___

## Project Structure

| Folder/File | Description |
|-------------|-------------|
| **`data/`** | **CUAD dataset and processed legal document files** |
| `├── raw/` | Original CUAD JSON files and raw legal documents |
| `├── processed/` | Multi-label CSV files for 41 clause types + metadata.json |
| `└── README.md` | Data structure and preprocessing documentation |
| **`models/`** | **Fine-tuned models and comprehensive training infrastructure** |
| `├── bert/` | Multi-label BERT for legal clause extraction |
| `├── t5/` | Legal-optimized T5 for document summarization |
| `├── fine_tuning/` | Training scripts with legal domain optimization |
| `│   ├── clause_extraction.py` | Multi-label BERT fine-tuning pipeline |
| `│   └── summarization.py` | T5 legal summarization training |
| `├── evaluation_report.txt` | Comprehensive model performance analysis |
| `├── clause_performance_analysis.csv` | Per-clause metrics and statistics |
| `├── shap_analysis_report.txt` | SHAP explainability analysis |
| `└── README.md` | Detailed model documentation and usage |
| **`notebooks/`** | **Jupyter notebooks for analysis and experimentation** |
| `├── data_exploration.ipynb` | CUAD dataset analysis with clean clause names |
| `├── model_training.ipynb` | Interactive model training and validation |
| `├── evaluation.ipynb` | Comprehensive multi-label evaluation framework |
| `└── explainability_analysis.ipynb` | SHAP, LIME, and attention visualization |
| **`scripts/`** | **Utility scripts for processing and analysis** |
| `├── data_preprocessing.py` | CUAD to multi-label conversion pipeline |
| `├── evaluation_metrics.py` | Legal domain-specific evaluation functions |
| `├── shap_analysis.py` | Standalone SHAP analysis with visualization |
| `└── utils.py` | Common utilities and helper functions |
| **`app/`** | **Interactive Streamlit application** |
| `├── app.py` | Main application with integrated explainability |
| `├── components/` | Modular application components |
| `│   ├── clause_extractor.py` | Multi-label clause detection interface |
| `│   ├── explainer.py` | SHAP and LIME explanation generation |
| `│   └── summarizer.py` | Legal document summarization interface |
| `└── static/styles.css` | Application styling and UI components |
| **`tests/`** | **Comprehensive test suite** |
| `├── test_app.py` | Application functionality tests |
| `├── test_models.py` | Model performance and integration tests |
| `└── test_scripts.py` | Utility script validation tests |
| `requirements.txt` | **Complete Python dependencies with versions** |
| `README.md` | **This comprehensive project documentation** |
| `LICENSE` | MIT License for open-source distribution |

## Architecture Overview

```
Legal Document Input
         ↓
    Text Preprocessing
    (Legal normalization)
         ↓
    ┌─────────────────┐    ┌─────────────────┐
    │   BERT Model    │    │   T5 Model      │
    │ (Multi-label    │    │ (Abstractive    │
    │  Clause         │    │  Legal          │
    │  Extraction)    │    │  Summarization) │
    └─────────────────┘    └─────────────────┘
         ↓                          ↓
    41 Clause Types            Legal Summary
    (Clean Names)              (Plain English)
         ↓                          ↓
    ┌─────────────────────────────────────────┐
    │         Explainability Layer            │
    │  SHAP • LIME • Attention Visualization  │
    └─────────────────────────────────────────┘
         ↓
    Interactive Streamlit Interface
    (Legal Practitioner Dashboard)
```

## Setup Instructions

### Prerequisites
- Python 3.8+ (tested with Python 3.9)
- CUDA-compatible GPU (recommended for training, optional for inference)
- 16GB+ RAM for full dataset processing

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/prgabriel/w266-project-legal-nlp-xai.git
   cd w266-project-legal-nlp-xai
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv legal-nlp-env
   source legal-nlp-env/bin/activate  # On Windows: legal-nlp-env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (required for summarization):
   ```python
   python -c "import nltk; nltk.download('punkt')"
   ```

### Quick Start

1. **Launch the Interactive Application**:
   ```bash
   streamlit run app/app.py
   ```
   Access the interface at `http://localhost:8501`

2. **Run SHAP Analysis** (standalone):
   ```bash
   python scripts/shap_analysis.py
   ```

3. **Train Models** (optional):
   ```bash
   # Multi-label clause extraction
   python models/fine_tuning/clause_extraction.py
   
   # Legal document summarization  
   python models/fine_tuning/summarization.py
   ```

___

## Data Sources & Processing

### CUAD Dataset Integration
- **Source**: [Contract Understanding Atticus Dataset (CUAD)](https://atticusprojectai.org/cuad)
- **Scale**: 500+ legal contracts with 41 clause type annotations
- **Processing**: Multi-label conversion with clean clause name mapping
- **Format**: Individual CSV files per clause type + consolidated metadata

### Legal Clause Types (41 Total)
```
Core Contract Elements:
• Agreement Date • Document Name • Parties • Effective Date
• Expiration Date • Governing Law • Termination for Convenience

Financial & Commercial Terms:
• Revenue/Profit Sharing • Price Restrictions • Minimum Commitment
• Most Favored Nation • Liquidated Damages • Cap on Liability

Intellectual Property:
• License Grant • IP Ownership Assignment • Joint IP Ownership
• Irrevocable or Perpetual License • Non-Transferable License

Restrictions & Obligations:
• Non-Compete • Non-Disparagement • No-Solicit of Employees
• No-Solicit of Customers • Anti-Assignment • Exclusivity

Legal Protections:
• Insurance • Audit Rights • Covenant Not to Sue
• Third Party Beneficiary • Uncapped Liability • Post-Termination Services

And 16 additional specialized clause types...
```

### Clean Name Mapping
All verbose CUAD questions are mapped to human-readable clause names:
- `"Highlight the parts related to Agreement Date..."` → `"Agreement Date"`
- `"Highlight the parts related to Anti-Assignment..."` → `"Anti-Assignment"`

___

## Model Architecture & Performance

### Multi-Label BERT Clause Extractor

**Architecture**: `bert-base-uncased` fine-tuned for 41-class multi-label classification
<!-- ```python
Model Performance (Test Set):
• F1 Micro: 0.847 (overall multi-label performance)
• F1 Macro: 0.782 (average per-clause performance)  
• F1 Weighted: 0.831 (support-weighted performance)
• Hamming Loss: 0.124 (fraction of incorrect labels)

Top Performing Clauses:
• Agreement Date: F1=0.924 (high-frequency, clear patterns)
• Document Name: F1=0.891 (structural identification)
• Parties: F1=0.873 (entity recognition strength)

Challenging Clauses:
• Most Favored Nation: F1=0.451 (complex legal concept)
• Joint IP Ownership: F1=0.523 (rare, nuanced language)
• Volume Restriction: F1=0.578 (context-dependent terms)
``` -->

### Legal T5 Summarization Model

**Architecture**: `t5-base` enhanced with legal-specific tokens and preprocessing
```python
Summarization Performance:
• ROUGE-1: 0.6054 (unigram overlap with reference)
• ROUGE-2: 0.5620 (bigram overlap quality)
• ROUGE-L: 0.5983 (longest common subsequence)
• Avg Compression Ratio: 0.1907 (4:1 summarization)

Legal Enhancements:
• Contract phrase normalization (shall→must, hereof→of this agreement)
• Legal entity tokens (<legal_entity>, <termination_clause>)
• Clause relationship preservation in summaries
• Domain-specific beam search optimization
```

___

## Explainability Features

### SHAP (SHapley Additive exPlanations)
- **Global Interpretability**: Feature importance across all 41 clause types
- **Local Explanations**: Token-level attributions for individual predictions
- **Visualization**: Summary plots, waterfall charts, and force plots
- **Legal Focus**: Clause-specific explanation reports for practitioners

### LIME (Local Interpretable Model-agnostic Explanations)  
- **Alternative Methodology**: Complementary explanations to SHAP
- **Perturbation-Based**: Understanding model sensitivity to text changes
- **Comparative Analysis**: Cross-validation with SHAP explanations

### Attention Visualization
- **Multi-Head Analysis**: BERT attention pattern interpretation
- **Token Relationships**: Understanding model focus areas
- **Legal Phrase Detection**: Attention on key legal terminology
- **Layer-wise Analysis**: Deep vs. shallow attention patterns

### Generated Reports
- `shap_analysis_report.txt`: Comprehensive SHAP analysis with legal insights
- `shap_summary_plot.png`: Visual summary of feature importance
- `clause_performance_analysis.csv`: Per-clause explainability metrics

___

## Evaluation Framework

### Multi-Label Classification Metrics
```python
Legal Domain Evaluation:
• F1 Scores: Micro/Macro/Weighted for comprehensive assessment
• Per-Clause Analysis: Individual performance with support statistics  
• Hamming Loss: Multi-label prediction accuracy
• Jaccard Similarity: Set-based label similarity
• Confusion Matrices: Per-clause error analysis
• Classification Reports: Precision/recall breakdown
```

### Summarization Quality Assessment
```python
Content Preservation Metrics:
• ROUGE-1/2/L: N-gram overlap with human references
• Compression Ratios: Summary length optimization
• Legal Coherence: Domain-specific evaluation
• Factual Accuracy: Legal information preservation
• Readability Scores: Plain-English accessibility
```

### Explainability Validation
```python
Interpretability Assessment:
• SHAP Value Consistency: Explanation stability across runs
• Attention-SHAP Correlation: Multi-method validation
• Legal Expert Evaluation: Practitioner feedback on explanations
• Faithfulness Metrics: Explanation-prediction alignment
```

___

## Interactive Application Features

### Streamlit Legal Dashboard
- **Document Upload**: PDF, TXT, and direct text input support
- **Real-time Analysis**: Instant clause detection and summarization
- **Interactive Visualizations**: SHAP plots and attention heatmaps
- **Export Capabilities**: Analysis reports and visualizations
- **Legal Professional UI**: Domain-specific interface design

### Core Functionalities
1. **Multi-Label Clause Detection**: Visual clause type identification
2. **Plain-English Summarization**: Legal jargon translation
3. **Explainable Predictions**: Interactive SHAP and LIME explanations
4. **Comparative Analysis**: Multiple document comparison
5. **Export & Reporting**: PDF reports for legal review

___

## Research & Development

### Notebooks for Analysis
- **`data_exploration.ipynb`**: CUAD dataset analysis with statistical insights
- **`model_training.ipynb`**: Interactive fine-tuning with hyperparameter optimization
- **`evaluation.ipynb`**: Comprehensive model evaluation and benchmarking
- **`explainability_analysis.ipynb`**: Advanced SHAP/LIME analysis and visualization

### Training Pipeline
```bash
# Complete training workflow
python scripts/data_preprocessing.py      # CUAD multi-label conversion
python models/fine_tuning/clause_extraction.py  # BERT fine-tuning
python models/fine_tuning/summarization.py      # T5 optimization
python scripts/evaluation_metrics.py            # Performance assessment
python scripts/shap_analysis.py                 # Explainability analysis
```

### Hyperparameter Optimization
- **Learning Rates**: 2e-5 (BERT), 5e-5 (T5) - legal domain optimized
- **Batch Sizes**: 8 (clause extraction), 4 (summarization) - memory balanced
- **Early Stopping**: Patience=2 to prevent overfitting on legal text
- **Sequence Lengths**: 512 (clauses), 1024 (summarization) - legal document optimized

___

## Contributing

We welcome contributions to advance legal NLP and explainable AI! 

### Development Workflow
1. **Fork the repository** and create a feature branch
2. **Follow coding standards**: Type hints, docstrings, and legal domain naming
3. **Add comprehensive tests**: Model performance and explainability validation
4. **Update documentation**: Include legal domain context and examples
5. **Submit pull request** with detailed description of legal NLP improvements

### Priority Areas for Contribution
- **Additional Legal Datasets**: Beyond CUAD for broader legal domain coverage
- **Multilingual Legal Support**: International contract analysis capabilities
- **Advanced Explainability**: Novel interpretability methods for legal AI
- **Legal Domain Metrics**: Specialized evaluation frameworks for legal NLP
- **Practitioner Feedback Integration**: Legal professional validation workflows

### Code Quality Standards
```python
# Example contribution structure
def extract_legal_clauses(document: str, 
                         clause_types: List[str],
                         threshold: float = 0.5) -> Dict[str, List[Dict]]:
    """
    Extract legal clauses with explainable confidence scores.
    
    Args:
        document: Legal contract text for analysis
        clause_types: List of clean clause names to detect
        threshold: Confidence threshold for clause detection
        
    Returns:
        Dictionary mapping clause types to detected instances with explanations
        
    Example:
        >>> clauses = extract_legal_clauses(contract_text, ["Agreement Date", "Termination"])
        >>> print(f"Found {len(clauses['Agreement Date'])} agreement date clauses")
    """
```

___

## Deployment & Production

### Model Serving Options
- **Local Streamlit**: Development and small-scale analysis
- **Docker Containers**: Scalable deployment with legal document processing
- **Cloud Integration**: AWS/Azure deployment for enterprise legal teams
- **API Endpoints**: RESTful services for legal software integration

### Performance Optimization
- **Model Quantization**: Reduced memory footprint for production deployment
- **Batch Processing**: Efficient handling of multiple legal documents
- **Caching Strategies**: Optimized repeated analysis workflows
- **GPU Acceleration**: CUDA optimization for large document collections

___

## License & Legal Considerations

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Important Legal Disclaimers
- **Not Legal Advice**: This tool provides technical analysis only and does not constitute legal advice
- **Professional Review Required**: All AI-generated insights should be validated by qualified legal professionals  
- **Data Privacy**: Ensure compliance with confidentiality requirements when processing legal documents
- **Model Limitations**: AI models may miss nuanced legal interpretations requiring human expertise

___

## Acknowledgements & References

### Academic & Research Contributions
- **CUAD Dataset**: Atticus Project for comprehensive legal clause annotations
- **Transformers Library**: Hugging Face for state-of-the-art NLP models
- **SHAP Framework**: Lundberg & Lee for unified explainability methodology
- **Legal NLP Research**: Stanford CodeX and other legal technology initiatives

### Open Source Community
- **PyTorch**: Deep learning framework powering our legal AI models
- **Streamlit**: Interactive application framework for legal professional interfaces
- **Scikit-learn**: Machine learning utilities and evaluation metrics
- **Legal Domain Experts**: Practitioners providing feedback on real-world utility

### Special Recognition
- UC Berkeley W266 Natural Language Processing course for academic foundation
- Legal technology community for domain expertise and validation
- Open-source contributors advancing responsible AI in legal applications

___

## Contact & Support

### Project Maintainer
**Perry Gabriel**  
- [pgabriel@berkeley.edu](mailto:pgabriel@berkeley.edu)  
- UC Berkeley - Master of Information and Data Science  
- Research Focus: Legal NLP + Explainable AI

### Getting Help
- **GitHub Issues**: Technical problems and feature requests
- **Discussions**: Legal NLP research questions and use cases  
- **Documentation**: Comprehensive guides in project wiki
- **Academic Collaboration**: Research partnerships and publications

### Citation
If you use this toolkit in your research, please cite:
```bibtex
@software{gabriel2024_legal_nlp_xai,
  author = {Gabriel, Perry},
  title = {Legal NLP + Explainability Toolkit: Interpretable Clause Extraction and Summarization},
  year = {2025},
  url = {https://github.com/prgabriel/w266-project-legal-nlp-xai},
  note = {UC Berkeley W266 Final Project - Responsible AI in Legal NLP}
}
```

---

**Advancing Responsible AI in Legal Technology**  
*Empowering Legal Professionals with Interpretable AI Tools*
