"""
Comprehensive Utility Functions for Legal NLP + Explainability Toolkit
Common functions for data handling, preprocessing, and project management
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import torch
import re
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer
import logging
from pathlib import Path
import yaml
from datetime import datetime
import hashlib
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# =========================================================================
# PROJECT STRUCTURE UTILITIES
# =========================================================================

class ProjectStructure:
    """Manages project paths and directory structure"""
    
    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize project structure manager
        
        Args:
            project_root: Root directory of the project. Auto-detected if None.
        """
        if project_root is None:
            # Auto-detect project root
            current_file = os.path.abspath(__file__)
            self.project_root = os.path.dirname(os.path.dirname(current_file))
        else:
            self.project_root = os.path.abspath(project_root)
        
        # Define standard directories
        self.dirs = {
            'data': os.path.join(self.project_root, 'data'),
            'data_raw': os.path.join(self.project_root, 'data', 'raw'),
            'data_processed': os.path.join(self.project_root, 'data', 'processed'),
            'models': os.path.join(self.project_root, 'models'),
            'models_bert': os.path.join(self.project_root, 'models', 'bert'),
            'models_t5': os.path.join(self.project_root, 'models', 't5'),
            'models_fine_tuning': os.path.join(self.project_root, 'models', 'fine_tuning'),
            'notebooks': os.path.join(self.project_root, 'notebooks'),
            'scripts': os.path.join(self.project_root, 'scripts'),
            'app': os.path.join(self.project_root, 'app'),
            'tests': os.path.join(self.project_root, 'tests'),
            'logs': os.path.join(self.project_root, 'logs')
        }
        
        logger.info(f"ProjectStructure initialized with root: {self.project_root}")
    
    def get_path(self, path_key: str, *subpaths) -> str:
        """
        Get full path for a directory or file
        
        Args:
            path_key: Key from self.dirs
            *subpaths: Additional path components
            
        Returns:
            Full path string
        """
        base_path = self.dirs.get(path_key, self.project_root)
        return os.path.join(base_path, *subpaths)
    
    def ensure_dirs(self, *path_keys) -> None:
        """Ensure directories exist, creating them if necessary"""
        for key in path_keys:
            if key in self.dirs:
                os.makedirs(self.dirs[key], exist_ok=True)
                logger.debug(f"Ensured directory exists: {self.dirs[key]}")
    
    def get_all_paths(self) -> Dict[str, str]:
        """Get dictionary of all defined paths"""
        return self.dirs.copy()

# Global project structure instance
PROJECT = ProjectStructure()

# =========================================================================
# DATA LOADING AND SAVING UTILITIES
# =========================================================================

def load_data(file_path: str, 
              file_type: Optional[str] = None,
              encoding: str = 'utf-8',
              **kwargs) -> Any:
    """
    Load data from various file formats with automatic type detection
    
    Args:
        file_path: Path to the file to load
        file_type: Force specific file type ('json', 'csv', 'pickle', 'txt', 'yaml')
        encoding: File encoding for text files
        **kwargs: Additional arguments passed to specific loaders
        
    Returns:
        Loaded data in appropriate format
        
    Examples:
        >>> data = load_data('data/processed/metadata.json')
        >>> df = load_data('data/processed/train_data.csv')
        >>> model = load_data('models/bert/model.pkl')
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Auto-detect file type from extension if not specified
    if file_type is None:
        file_type = os.path.splitext(file_path)[1].lower().lstrip('.')
    
    logger.info(f"Loading {file_type.upper()} file: {file_path}")
    
    try:
        if file_type == 'json':
            with open(file_path, 'r', encoding=encoding) as f:
                return json.load(f, **kwargs)
                
        elif file_type == 'csv':
            return pd.read_csv(file_path, encoding=encoding, **kwargs)
            
        elif file_type in ['pkl', 'pickle']:
            with open(file_path, 'rb') as f:
                return pickle.load(f, **kwargs)
                
        elif file_type == 'txt':
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
                
        elif file_type in ['yaml', 'yml']:
            with open(file_path, 'r', encoding=encoding) as f:
                return yaml.safe_load(f)
                
        elif file_type == 'parquet':
            return pd.read_parquet(file_path, **kwargs)
            
        elif file_type == 'xlsx':
            return pd.read_excel(file_path, **kwargs)
            
        else:
            # Default to text loading
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
                
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        raise

def save_data(data: Any, 
              file_path: str,
              file_type: Optional[str] = None,
              encoding: str = 'utf-8',
              create_dirs: bool = True,
              **kwargs) -> None:
    """
    Save data to various file formats with automatic type detection
    
    Args:
        data: Data to save
        file_path: Path where to save the file
        file_type: Force specific file type
        encoding: File encoding for text files
        create_dirs: Whether to create parent directories if they don't exist
        **kwargs: Additional arguments passed to specific savers
        
    Examples:
        >>> save_data(metadata_dict, 'data/processed/metadata.json')
        >>> save_data(df, 'data/processed/results.csv', index=False)
        >>> save_data(model, 'models/trained_model.pkl')
    """
    # Create parent directories if needed
    if create_dirs:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Auto-detect file type from extension if not specified
    if file_type is None:
        file_type = os.path.splitext(file_path)[1].lower().lstrip('.')
    
    logger.info(f"Saving {file_type.upper()} file: {file_path}")
    
    try:
        if file_type == 'json':
            with open(file_path, 'w', encoding=encoding) as f:
                json.dump(data, f, indent=2, ensure_ascii=False, **kwargs)
                
        elif file_type == 'csv':
            if isinstance(data, pd.DataFrame):
                data.to_csv(file_path, encoding=encoding, index=False, **kwargs)
            else:
                pd.DataFrame(data).to_csv(file_path, encoding=encoding, index=False, **kwargs)
                
        elif file_type in ['pkl', 'pickle']:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f, **kwargs)
                
        elif file_type == 'txt':
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(str(data))
                
        elif file_type in ['yaml', 'yml']:
            with open(file_path, 'w', encoding=encoding) as f:
                yaml.dump(data, f, default_flow_style=False, **kwargs)
                
        elif file_type == 'parquet':
            if isinstance(data, pd.DataFrame):
                data.to_parquet(file_path, **kwargs)
            else:
                pd.DataFrame(data).to_parquet(file_path, **kwargs)
                
        elif file_type == 'xlsx':
            if isinstance(data, pd.DataFrame):
                data.to_excel(file_path, index=False, **kwargs)
            else:
                pd.DataFrame(data).to_excel(file_path, index=False, **kwargs)
                
        else:
            # Default to text saving
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(str(data))
                
        logger.info(f"Successfully saved to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving file {file_path}: {e}")
        raise

# =========================================================================
# LEGAL TEXT PREPROCESSING UTILITIES
# =========================================================================

def preprocess_text(text: str, 
                   for_legal: bool = True,
                   normalize_whitespace: bool = True,
                   remove_special_chars: bool = False,
                   lowercase: bool = False,
                   max_length: Optional[int] = None) -> str:
    """
    Preprocess text for legal NLP models with domain-specific optimizations
    
    Args:
        text: Input text to preprocess
        for_legal: Apply legal domain-specific preprocessing
        normalize_whitespace: Normalize whitespace and line breaks
        remove_special_chars: Remove special characters (preserve legal punctuation)
        lowercase: Convert to lowercase (not recommended for legal text)
        max_length: Truncate text to maximum length
        
    Returns:
        Preprocessed text string
        
    Examples:
        >>> text = preprocess_text(contract_text, for_legal=True)
        >>> text = preprocess_text(raw_text, normalize_whitespace=True, max_length=512)
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Legal domain-specific preprocessing
    if for_legal:
        # Normalize common legal abbreviations and phrases
        legal_normalizations = {
            r'\bhereof\b': 'of this agreement',
            r'\bherein\b': 'in this agreement',
            r'\bwhereas\b': 'given that',
            r'\bthereof\b': 'of that',
            r'\bwhereby\b': 'by which',
            r'\bshall\b': 'must',
            r'\bInc\.\b': 'Incorporated',
            r'\bLLC\b': 'Limited Liability Company',
            r'\bCorp\.\b': 'Corporation',
            r'\b&\b': 'and'
        }
        
        for pattern, replacement in legal_normalizations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Normalize whitespace
    if normalize_whitespace:
        # Preserve paragraph breaks but normalize excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)      # Normalize spaces and tabs
        text = text.strip()
    
    # Remove special characters (preserve legal punctuation)
    if remove_special_chars:
        if for_legal:
            # Keep legal punctuation: periods, commas, semicolons, colons, parentheses, quotes
            text = re.sub(r'[^\w\s.,;:()\'""\-]', '', text)
        else:
            # Standard special character removal
            text = re.sub(r'[^\w\s]', '', text)
    
    # Convert to lowercase (generally not recommended for legal text)
    if lowercase:
        text = text.lower()
    
    # Truncate to maximum length
    if max_length and len(text) > max_length:
        text = text[:max_length].rsplit(' ', 1)[0]  # Truncate at word boundary
        logger.debug(f"Text truncated to {len(text)} characters")
    
    return text

def extract_legal_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract common legal entities from text using regex patterns
    
    Args:
        text: Input legal text
        
    Returns:
        Dictionary with entity types as keys and lists of found entities as values
    """
    entities = {
        'dates': [],
        'companies': [],
        'monetary_amounts': [],
        'percentages': [],
        'parties': []
    }
    
    # Date patterns
    date_patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or MM-DD-YYYY
        r'\b\w+ \d{1,2}, \d{4}\b',              # Month DD, YYYY
        r'\b\d{1,2} \w+ \d{4}\b'                # DD Month YYYY
    ]
    
    for pattern in date_patterns:
        entities['dates'].extend(re.findall(pattern, text))
    
    # Company name patterns
    company_patterns = [
        r'\b\w+(?:\s+\w+)*\s+(?:Inc\.|LLC|Corp\.|Corporation|Company|Co\.|Ltd\.)\b',
        r'\b\w+(?:\s+\w+)*\s+(?:Incorporated|Limited)\b'
    ]
    
    for pattern in company_patterns:
        entities['companies'].extend(re.findall(pattern, text, re.IGNORECASE))
    
    # Monetary amounts
    money_pattern = r'\$[\d,]+(?:\.\d{2})?'
    entities['monetary_amounts'] = re.findall(money_pattern, text)
    
    # Percentages
    percentage_pattern = r'\d+(?:\.\d+)?%'
    entities['percentages'] = re.findall(percentage_pattern, text)
    
    # Remove duplicates and clean
    for key in entities:
        entities[key] = list(set(entities[key]))
    
    return entities

def clean_clause_name(clause_name: str) -> str:
    """
    Extract clean clause name from verbose CUAD question format
    
    Args:
        clause_name: Verbose CUAD clause question
        
    Returns:
        Clean, human-readable clause name
        
    Examples:
        >>> clean_name = clean_clause_name('Highlight the parts related to "Agreement Date"...')
        >>> # Returns: "Agreement Date"
    """
    # Try to extract text between first set of quotes
    quote_match = re.search(r'"([^"]*)"', clause_name)
    if quote_match:
        return quote_match.group(1)
    
    # Fallback: extract first few words after "related to"
    related_match = re.search(r'related to ([^"]*?) that should', clause_name)
    if related_match:
        return related_match.group(1).strip()
    
    # Final fallback: take first 50 characters
    return clause_name[:50].strip()

# =========================================================================
# DATA SPLITTING AND SAMPLING UTILITIES
# =========================================================================

def split_data(data: Union[pd.DataFrame, np.ndarray, List], 
               train_size: float = 0.8,
               val_size: Optional[float] = None,
               test_size: Optional[float] = None,
               random_state: int = 42,
               stratify: Optional[Union[np.ndarray, pd.Series]] = None) -> Tuple:
    """
    Split data into training, validation, and test sets with flexible sizing
    
    Args:
        data: Input data to split
        train_size: Proportion for training set (0.0-1.0)
        val_size: Proportion for validation set (if None, uses remaining after train/test)
        test_size: Proportion for test set (if None, uses 1-train_size or calculates from val_size)
        random_state: Random seed for reproducibility
        stratify: Array for stratified splitting (for classification tasks)
        
    Returns:
        Tuple of (train, val, test) or (train, test) depending on val_size
        
    Examples:
        >>> train, val, test = split_data(df, train_size=0.7, val_size=0.2)  # 70/20/10 split
        >>> train, test = split_data(data, train_size=0.8)  # 80/20 split
    """
    if isinstance(data, pd.DataFrame):
        data_len = len(data)
        indices = np.arange(data_len)
    elif isinstance(data, (list, np.ndarray)):
        data_len = len(data)
        indices = np.arange(data_len)  
        data = np.array(data)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
    
    # Calculate sizes
    if val_size is None and test_size is None:
        # Simple train/test split
        test_size = 1.0 - train_size
        val_size = 0.0
    elif val_size is None:
        # Calculate val_size from remaining
        val_size = 1.0 - train_size - test_size
        if val_size < 0:
            raise ValueError("train_size + test_size cannot exceed 1.0")
    elif test_size is None:
        # Calculate test_size from remaining
        test_size = 1.0 - train_size - val_size
        if test_size < 0:
            raise ValueError("train_size + val_size cannot exceed 1.0")
    
    # Validate sizes
    total_size = train_size + val_size + test_size
    if not np.isclose(total_size, 1.0, atol=1e-6):
        raise ValueError(f"Sizes must sum to 1.0, got {total_size}")
    
    logger.info(f"Splitting data: train={train_size:.1%}, val={val_size:.1%}, test={test_size:.1%}")
    
    if val_size == 0:
        # Simple train/test split
        if isinstance(data, pd.DataFrame):
            train_data, test_data = train_test_split(
                data, train_size=train_size, random_state=random_state, stratify=stratify
            )
        else:
            train_idx, test_idx = train_test_split(
                indices, train_size=train_size, random_state=random_state, stratify=stratify
            )
            train_data, test_data = data[train_idx], data[test_idx]
        
        return train_data, test_data
    
    else:
        # Three-way split
        # First split: separate out test set
        temp_size = train_size + val_size
        temp_train_size = train_size / temp_size
        
        if isinstance(data, pd.DataFrame):
            temp_data, test_data = train_test_split(
                data, train_size=temp_size, random_state=random_state, stratify=stratify
            )
            train_data, val_data = train_test_split(
                temp_data, train_size=temp_train_size, random_state=random_state + 1
            )
        else:
            temp_idx, test_idx = train_test_split(
                indices, train_size=temp_size, random_state=random_state, stratify=stratify
            )
            train_idx, val_idx = train_test_split(
                temp_idx, train_size=temp_train_size, random_state=random_state + 1
            )
            train_data, val_data, test_data = data[train_idx], data[val_idx], data[test_idx]
        
        return train_data, val_data, test_data

def create_stratified_sample(data: pd.DataFrame, 
                           stratify_column: str,
                           sample_size: Union[int, float],
                           random_state: int = 42) -> pd.DataFrame:
    """
    Create stratified sample maintaining class distribution
    
    Args:
        data: Input DataFrame
        stratify_column: Column name to stratify on
        sample_size: Number of samples (int) or fraction (float)
        random_state: Random seed
        
    Returns:
        Stratified sample DataFrame
    """
    if isinstance(sample_size, float) and 0 < sample_size <= 1.0:
        return data.groupby(stratify_column, group_keys=False).apply(
            lambda x: x.sample(frac=sample_size, random_state=random_state)
        ).reset_index(drop=True)
    elif isinstance(sample_size, int):
        # Calculate samples per class
        class_counts = data[stratify_column].value_counts()
        n_classes = len(class_counts)
        samples_per_class = sample_size // n_classes
        
        sampled_dfs = []
        for class_value in class_counts.index:
            class_data = data[data[stratify_column] == class_value]
            n_samples = min(samples_per_class, len(class_data))
            sampled_dfs.append(class_data.sample(n=n_samples, random_state=random_state))
        
        return pd.concat(sampled_dfs, ignore_index=True)
    else:
        raise ValueError("sample_size must be float (0-1) or positive int")

# =========================================================================
# MULTI-LABEL UTILITIES
# =========================================================================

def create_multilabel_matrix(labels: List[List[str]], 
                           all_labels: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Convert list of label lists to multi-label binary matrix
    
    Args:
        labels: List of lists containing labels for each sample
        all_labels: Complete list of possible labels (if None, inferred from data)
        
    Returns:
        Tuple of (binary_matrix, label_names)
        
    Examples:
        >>> labels = [['Agreement Date', 'Parties'], ['License Grant'], ['Agreement Date']]
        >>> matrix, label_names = create_multilabel_matrix(labels)
    """
    mlb = MultiLabelBinarizer()
    
    if all_labels is not None:
        mlb.fit([all_labels])
    
    binary_matrix = mlb.fit_transform(labels)
    label_names = list(mlb.classes_)
    
    logger.info(f"Created multi-label matrix: {binary_matrix.shape[0]} samples, {binary_matrix.shape[1]} labels")
    
    return binary_matrix, label_names

def convert_multilabel_to_single(multilabel_matrix: np.ndarray, 
                                strategy: str = 'most_frequent') -> np.ndarray:
    """
    Convert multi-label matrix to single-label for specific tasks
    
    Args:
        multilabel_matrix: Binary multi-label matrix
        strategy: 'most_frequent', 'random', or 'first'
        
    Returns:
        Single-label array
    """
    single_labels = []
    
    for row in multilabel_matrix:
        positive_indices = np.where(row == 1)[0]
        
        if len(positive_indices) == 0:
            # No positive labels - assign negative class
            single_labels.append(-1)
        elif len(positive_indices) == 1:
            # Single positive label
            single_labels.append(positive_indices[0])
        else:
            # Multiple positive labels - apply strategy
            if strategy == 'most_frequent':
                # This would require additional frequency information
                single_labels.append(positive_indices[0])
            elif strategy == 'random':
                single_labels.append(np.random.choice(positive_indices))
            elif strategy == 'first':
                single_labels.append(positive_indices[0])
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
    
    return np.array(single_labels)

# =========================================================================
# METRICS CALCULATION UTILITIES
# =========================================================================

def calculate_metrics(predictions: Union[np.ndarray, List], 
                     labels: Union[np.ndarray, List],
                     metric_types: List[str] = ['precision', 'recall', 'f1'],
                     average: str = 'macro',
                     multilabel: bool = False) -> Dict[str, float]:
    """
    Calculate various evaluation metrics for model performance
    
    Args:
        predictions: Model predictions
        labels: True labels
        metric_types: List of metrics to calculate
        average: Averaging strategy ('macro', 'micro', 'weighted', 'binary')
        multilabel: Whether this is multi-label classification
        
    Returns:
        Dictionary of calculated metrics
        
    Examples:
        >>> metrics = calculate_metrics(y_pred, y_true, ['precision', 'recall', 'f1'])
        >>> ml_metrics = calculate_metrics(y_pred, y_true, multilabel=True)
    """
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, accuracy_score,
        hamming_loss, jaccard_score
    )
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    metrics = {}
    
    try:
        if multilabel:
            # Multi-label specific metrics
            if 'hamming_loss' in metric_types:
                metrics['hamming_loss'] = hamming_loss(labels, predictions)
            
            if 'jaccard_score' in metric_types:
                metrics['jaccard_score'] = jaccard_score(labels, predictions, average=average)
            
            if 'subset_accuracy' in metric_types:
                metrics['subset_accuracy'] = accuracy_score(labels, predictions)
        
        # Standard classification metrics
        if 'precision' in metric_types:
            metrics['precision'] = precision_score(labels, predictions, average=average, zero_division=0)
        
        if 'recall' in metric_types:
            metrics['recall'] = recall_score(labels, predictions, average=average, zero_division=0)
        
        if 'f1' in metric_types:
            metrics['f1'] = f1_score(labels, predictions, average=average, zero_division=0)
        
        if 'accuracy' in metric_types and not multilabel:
            metrics['accuracy'] = accuracy_score(labels, predictions)
        
        logger.info(f"Calculated {len(metrics)} metrics")
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        # Return zero metrics as fallback
        for metric_type in metric_types:
            metrics[metric_type] = 0.0
    
    return metrics

def calculate_confusion_matrix_metrics(y_true: np.ndarray, 
                                     y_pred: np.ndarray) -> Dict[str, Union[int, float]]:
    """
    Calculate detailed metrics from confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with TP, TN, FP, FN and derived metrics
    """
    from sklearn.metrics import confusion_matrix
    
    # Handle binary case
    if y_true.ndim == 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        return {
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'positive_predictive_value': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            'negative_predictive_value': float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        }
    else:
        # Multi-class case - return aggregated metrics
        cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
        
        # Calculate per-class metrics and average
        total_metrics = {
            'true_positives': 0,
            'true_negatives': 0, 
            'false_positives': 0,
            'false_negatives': 0
        }
        
        for i in range(cm.shape[0]):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - (tp + fp + fn)
            
            total_metrics['true_positives'] += tp
            total_metrics['false_positives'] += fp
            total_metrics['false_negatives'] += fn
            total_metrics['true_negatives'] += tn
        
        return total_metrics

# =========================================================================
# MODEL AND TOKENIZER UTILITIES
# =========================================================================

def load_tokenizer(model_name_or_path: str, 
                  cache_dir: Optional[str] = None,
                  **kwargs) -> AutoTokenizer:
    """
    Load tokenizer with error handling and caching
    
    Args:
        model_name_or_path: Model name or path to tokenizer
        cache_dir: Directory to cache tokenizer files
        **kwargs: Additional arguments for tokenizer
        
    Returns:
        Loaded tokenizer
    """
    try:
        logger.info(f"Loading tokenizer: {model_name_or_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            **kwargs
        )
        
        logger.info(f"Successfully loaded tokenizer with vocab size: {tokenizer.vocab_size}")
        return tokenizer
        
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        raise

def get_model_info(model_path: str) -> Dict[str, Any]:
    """
    Extract model information from saved model files
    
    Args:
        model_path: Path to model directory
        
    Returns:
        Dictionary with model information
    """
    info = {
        'model_path': model_path,
        'exists': os.path.exists(model_path),
        'files': [],
        'size_mb': 0,
        'config': {}
    }
    
    if info['exists']:
        # List all files in model directory
        for root, dirs, files in os.walk(model_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                info['files'].append({
                    'name': file,
                    'size_mb': file_size / (1024 * 1024),
                    'path': file_path
                })
                info['size_mb'] += file_size / (1024 * 1024)
        
        # Try to load config if available
        config_path = os.path.join(model_path, 'config.json')
        if os.path.exists(config_path):
            try:
                info['config'] = load_data(config_path)
            except:
                pass
    
    return info

# =========================================================================
# LOGGING AND DEBUGGING UTILITIES
# =========================================================================

def setup_logging(log_level: str = 'INFO', 
                 log_file: Optional[str] = None,
                 format_string: Optional[str] = None) -> None:
    """
    Setup comprehensive logging configuration
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional file to save logs
        format_string: Custom format string for logs
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory if saving to file
    if log_file:
        PROJECT.ensure_dirs('logs')
        if not os.path.isabs(log_file):
            log_file = PROJECT.get_path('logs', log_file)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=format_string,
        filename=log_file,
        filemode='a' if log_file else None
    )
    
    # Also log to console if file specified  
    if log_file:
        console = logging.StreamHandler()
        console.setLevel(getattr(logging, log_level.upper()))
        formatter = logging.Formatter(format_string)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
    
    logger.info(f"Logging configured: level={log_level}, file={log_file}")

def create_experiment_id() -> str:
    """Create unique experiment ID based on timestamp and random hash"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_hash = hashlib.md5(str(np.random.random()).encode()).hexdigest()[:8]
    return f"{timestamp}_{random_hash}"

def log_system_info() -> None:
    """Log system and environment information"""
    import platform
    import psutil
    
    logger.info("=== SYSTEM INFORMATION ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"CPU cores: {psutil.cpu_count()}")
    logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # GPU information if available
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.device_count()} devices")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("CUDA not available")
    
    logger.info(f"Project root: {PROJECT.project_root}")

# =========================================================================
# CONFIGURATION MANAGEMENT
# =========================================================================

class ConfigManager:
    """Manage configuration files and settings"""
    
    def __init__(self, config_file: str = 'config.yaml'):
        """
        Initialize configuration manager
        
        Args:
            config_file: Name of configuration file in project root
        """
        self.config_file = os.path.join(PROJECT.project_root, config_file)
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        if os.path.exists(self.config_file):
            return load_data(self.config_file)
        else:
            # Create default configuration
            default_config = {
                'model': {
                    'bert_model_name': 'bert-base-uncased',
                    't5_model_name': 't5-base',
                    'max_length': 512,
                    'batch_size': 8
                },
                'training': {
                    'learning_rate': 2e-5,
                    'num_epochs': 3,
                    'warmup_steps': 500,
                    'weight_decay': 0.01
                },
                'data': {
                    'train_size': 0.7,
                    'val_size': 0.2,
                    'test_size': 0.1
                },
                'paths': PROJECT.get_all_paths()
            }
            
            self.save_config(default_config)
            return default_config
    
    def save_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Save configuration to file"""
        if config is None:
            config = self.config
        
        save_data(config, self.config_file)
        logger.info(f"Configuration saved to {self.config_file}")
        
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'model.bert_model_name')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to config value
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to parent dictionary
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set final value
        config[keys[-1]] = value
        logger.debug(f"Set config {key_path} = {value}")

# Global configuration manager
CONFIG = ConfigManager()

# =========================================================================
# MAIN EXECUTION FOR TESTING
# =========================================================================

if __name__ == "__main__":
    # Demo of utility functions
    logger.info("Legal NLP Utilities - Demo")
    
    # Test project structure
    logger.info("=== PROJECT STRUCTURE ===")
    logger.info(f"Project root: {PROJECT.project_root}")
    for key, path in PROJECT.get_all_paths().items():
        logger.info(f"{key}: {path}")
    
    # Test configuration
    logger.info("\n=== CONFIGURATION ===")
    logger.info(f"BERT model: {CONFIG.get('model.bert_model_name')}")
    logger.info(f"Learning rate: {CONFIG.get('training.learning_rate')}")
    
    # Test text preprocessing
    logger.info("\n=== TEXT PREPROCESSING ===")
    sample_text = """
    This Agreement shall commence on January 1, 2024 and shall continue 
    for a period of three years, unless terminated earlier hereof. The parties
    hereby agree that TechCorp Inc. shall provide services.
    """
    
    processed = preprocess_text(sample_text, for_legal=True)
    logger.info(f"Original: {sample_text[:100]}...")
    logger.info(f"Processed: {processed[:100]}...")
    
    # Test entity extraction
    entities = extract_legal_entities(sample_text)
    logger.info(f"Entities found: {entities}")
    
    # Test data splitting
    logger.info("\n=== DATA SPLITTING ===")
    sample_data = np.random.randn(100, 5)
    train, val, test = split_data(sample_data, train_size=0.7, val_size=0.2)
    logger.info(f"Split sizes: train={len(train)}, val={len(val)}, test={len(test)}")
    
    # Test metrics calculation
    logger.info("\n=== METRICS CALCULATION ===")
    y_true = np.random.randint(0, 2, 50)
    y_pred = np.random.randint(0, 2, 50)
    metrics = calculate_metrics(y_pred, y_true)
    logger.info(f"Sample metrics: {metrics}")
    
    logger.info("\nUtilities demo completed successfully!")